# =============================================================================
# File: test_VAE_SYNBADD_dice.py
# Purpose: Inference/evaluation entrypoint for a specific experiment variant.
# Prerequisites: Trained checkpoints and fold-specific dataset metadata.
# Usually Run Next: Run comparison analysis scripts or aggregate fold metrics.
# Model Components Used: Uses model modules imported in this file; latest subjectwise artifact generation uses test_model_subjectwise.py.
# =============================================================================

import pandas as pd
import os
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import re
from torch.autograd import Variable

import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import ants
import wandb

fold_index = 2
csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
save_path = f"/scratch/anirudhkaushik/SYNBADDQualitative{fold_index}/Images/"
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(csv_file)
df = df[df[f'fold_{fold_index}'] == 'test'].reset_index(drop=True)

df['Acq Date'] = pd.to_datetime(df['Acq Date'])
df = df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
subject_groups = df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
subjects = list(subject_groups.index.levels[0])

def get_metrics(target_img, synbadd_img, brain_mask=None):
    target_img_data = target_img
    synbadd_img_data = synbadd_img
    
    if brain_mask is not None:
        binary_mask = brain_mask > 0
        target_masked = target_img_data * binary_mask
        synbadd_masked = synbadd_img_data * binary_mask
        
        if np.sum(binary_mask) > 0:
            ssim_value = ssim(target_masked, synbadd_masked, data_range=1, multichannel=False)
            psnr_value = psnr(target_masked, synbadd_masked, data_range=1)
        else:
            ssim_value = ssim(target_img_data, synbadd_img_data, data_range=1, multichannel=False)
            psnr_value = psnr(target_img_data, synbadd_img_data, data_range=1)
    else:
        ssim_value = ssim(target_img_data, synbadd_img_data, data_range=1, multichannel=False)
        psnr_value = psnr(target_img_data, synbadd_img_data, data_range=1)
        
    return ssim_value, psnr_value

def load_nii(imgPath, padding=(128, 128, 128)):
    img_data = nib.load(imgPath).get_fdata()
    if img_data.ndim == 4:
        img_data = img_data.squeeze(0)
    img_data, _ = pad(img_data, shape=padding)
    img_data = preprocess_Normalizition(img_data)
    return img_data

def pad(array, shape):
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices

def preprocess_Normalizition(img, method="min_max"):
    img = img.astype(np.float32)
    if method == "min_max":
        front_img = img[img > 0]
        min_value = np.min(front_img)
        max_value = np.max(front_img)
        img = (img - min_value) / (max_value - min_value)
        img[img < 0] = 0.0
    elif method == "mean":
        front_img = img[img > 0]
        mean_value = np.mean(front_img)
        std_value = np.std(front_img)
        img = (img - mean_value) / std_value
    return img

def dice_score(pred, target):
    intersection = np.sum(pred * target)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)

def extract_region(seg_data, region_ids):
    mask = np.zeros_like(seg_data)
    for region_id in region_ids:
        mask[seg_data == region_id] = 1
    return mask


def evaluate_dice_metrics(generator, vae_model, regression_model, transformer_model, discriminator_model, test_dataloader, criterion_lddmm, device, opt, run_name, fold_index, regions=None):
    if regions is None:
        regions = {
            'hippocampus': [17, 53],
            'parahippocampal': [1016, 2016],
            'third_ventricle': [14],
            'amygdala': [18, 54],
            'entorhinal_cortex': [1006, 2006],
            'fusiform_gyrus': [1007, 2007],
            'inferior_temporal_gyrus': [1009, 2009],
            'middle_temporal_gyrus': [1015, 2015],
        }
    
    generator.eval()
    vae_model.eval()
    if opt.use_transformer:
        transformer_model.eval()
    else:
        regression_model.eval()
    if opt.use_discriminator:
        discriminator_model.eval()
    
    ssim_values = []
    psnr_values = []
    synbadd_dice_scores = {region: [] for region in regions}
    per_image_metrics = []
    
    brain_mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_brain_masks/"
    save_path = f"/scratch/anirudhkaushik/SYNBADDQualitative{fold_index}/Images/"
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        with tqdm(total=len(test_dataloader.dataset), desc="Evaluating Dice Metrics") as progress_bar:
            for i, batch in enumerate(test_dataloader):
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_timepoint_index, target_svf = [batch[k].to(device) for k in ["stacked_svf", "latest_scan", "target_scan", "class_label", "reg_label", "target_timepoint", "latest_timepoint", "target_timepoint_index", "target_svf"]]
                
                # Get segmentations from dataloader
                target_seg = batch["target_seg"].to(device)
                baseline_seg = batch["baseline_seg"].to(device)
                target_uids = batch["target_uid"].cpu().numpy()
                
                stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf = [Variable(v.type(torch.float32)).to(device) for v in [stacked_svf, latest_scan, target_scan, class_labels, reg_labels, target_timepoint, latest_scan_timepoint, target_svf]]
                
                target_labels = torch.cat([class_labels, reg_labels, target_timepoint], dim=1)
                
                if opt.use_past_data or opt.use_future_data:
                    batch_size, temporal_dim, c_dim, h, w, d = stacked_svf.shape
                    stacked_svf = stacked_svf.view(-1, *stacked_svf.shape[2:])
                    
                    with torch.no_grad():
                        contextualized_scan, logvar, skips = vae_model(stacked_svf, "encode", None)
                        contextualized_scan, logvar, skips = contextualized_scan.detach(), logvar.detach(), [s.detach() for s in skips]
                    
                    for skip_ind in range(len(skips)):
                        s_b, s_c, s_h, s_w, s_d = skips[skip_ind].shape
                        skips[skip_ind] = skips[skip_ind].view(batch_size, temporal_dim, s_c, s_h, s_w, s_d)
                        skips[skip_ind] = skips[skip_ind].mean(dim=1)
                    
                    contextualized_scan = contextualized_scan.view(batch_size, temporal_dim, -1)
                    if opt.use_transformer:
                        contextualized_scan = transformer_model(contextualized_scan, target_timepoint_index)
                    else:
                        contextualized_scan = regression_model(contextualized_scan)
                    with torch.no_grad():
                        contextualized_scan = vae_model(contextualized_scan, "decode", skips)
                    
                    stacked_svf = stacked_svf.view(batch_size, temporal_dim, c_dim, h, w, d)
                else:
                    contextualized_scan = latest_scan
                
                y_synthetic, synthetic_field = generator(latest_scan, contextualized_scan, target_labels)
                
                for j, (target, synthetic, field, target_segmentation, baseline_segmentation, uid) in enumerate(zip(
                    target_scan.cpu().numpy(), 
                    y_synthetic.cpu().numpy(), 
                    synthetic_field.cpu().numpy(),
                    target_seg.cpu().numpy(),
                    baseline_seg.cpu().numpy(),
                    target_uids
                )):
                    try:
                        # Access file paths from the uid to get image IDs for brain mask loading and logs
                        uid_to_path = test_dataloader.dataset.get_file_path_mapping()
                        file_path = uid_to_path.get(uid)
                        if file_path is None:
                            continue
                            
                        # Extract image ID from file path
                        image_id_match = re.search(r'S\d+_I\d+', file_path)
                        if image_id_match:
                            image_id = image_id_match.group(0)
                        else:
                            image_id = f"uid_{uid}"
                            
                        # Load brain mask if available (for masked metrics)
                        brain_mask_files = [f for f in os.listdir(brain_mask_dir) if image_id in f]
                        brain_mask_file = None
                        for k in brain_mask_files:
                            if '_brain_mask.nii.gz' in k:
                                brain_mask_file = k
                                break
                        
                        target_img_data = target[0]  
                        synbadd_img_data = synthetic[0]
                        
                        # Convert to ANTs format for registration (using backward warping approach)
                        target_ants = ants.from_numpy(target_img_data, origin=None, spacing=None, direction=None)
                        synbadd_ants = ants.from_numpy(synbadd_img_data, origin=None, spacing=None, direction=None)
                        target_seg_ants = ants.from_numpy(target_segmentation, origin=None, spacing=None, direction=None)
                        
                        # Register synthetic image to target image (backward warping)
                        print(f"Registering synthetic image for {image_id}...")
                        synbadd_reg = ants.registration(
                            fixed=synbadd_ants,
                            moving=target_ants,
                            type_of_transform='SyN'
                        )
                        
                        # Apply transformation to target segmentation
                        print(f"Applying transformation to target segmentation {image_id}...")
                        synbadd_mask_warped = ants.apply_transforms(
                            fixed=synbadd_ants,
                            moving=target_seg_ants,
                            transformlist=synbadd_reg['fwdtransforms'],
                            interpolator='nearestNeighbor'
                        )
                        
                        brain_mask_data = None
                        if brain_mask_file:
                            brain_mask_nib = nib.load(os.path.join(brain_mask_dir, brain_mask_file))
                            brain_mask_data = brain_mask_nib.get_fdata()
                            
                            # Also warp brain mask if available
                            brain_mask_ants = ants.from_numpy(brain_mask_data, origin=None, spacing=None, direction=None)
                            synbadd_brain_mask_warped = ants.apply_transforms(
                                fixed=synbadd_ants,
                                moving=brain_mask_ants,
                                transformlist=synbadd_reg['fwdtransforms'],
                                interpolator='nearestNeighbor'
                            )
                            brain_mask_data = synbadd_brain_mask_warped.numpy()
                        
                        # Convert warped mask back to numpy
                        synbadd_mask_data = synbadd_mask_warped.numpy()
                        
                        image_metrics = {
                            "image_id": image_id,
                            "subject": image_id.split('_')[0] if '_' in image_id else f"subject_{j}"
                        }
                        
                        # Calculate Dice scores for each region
                        for region_name, region_ids in regions.items():
                            target_region = extract_region(target_segmentation, region_ids)
                            synbadd_region = extract_region(synbadd_mask_data, region_ids)
                            
                            synbadd_dice = dice_score(synbadd_region, target_region)
                            synbadd_dice_scores[region_name].append(synbadd_dice)
                            
                            image_metrics[f"dice_synbadd_{region_name}"] = synbadd_dice
                            print(f"  {region_name.capitalize()} - SynBADD Dice: {synbadd_dice:.4f}")
                        
                        # Calculate image quality metrics
                        if brain_mask_data is not None:
                            synbadd_ssim_value, synbadd_psnr_value = get_metrics(target_img_data, synbadd_img_data, brain_mask_data)
                        else:
                            synbadd_ssim_value, synbadd_psnr_value = get_metrics(target_img_data, synbadd_img_data)
                            
                        ssim_values.append(synbadd_ssim_value)
                        psnr_values.append(synbadd_psnr_value)
                        
                        image_metrics["ssim_synbadd"] = synbadd_ssim_value
                        image_metrics["psnr_synbadd"] = synbadd_psnr_value
                        image_metrics["used_brain_mask"] = brain_mask_data is not None
                        
                        per_image_metrics.append(image_metrics)
                        
                        print(f"Image ID: {image_id}, SynBADD SSIM: {synbadd_ssim_value:.4f}, SynBADD PSNR: {synbadd_psnr_value:.4f}")
                        
                        # Log to wandb
                        wandb.log(image_metrics)
                    
                    except Exception as e:
                        print(f"Error processing batch {i}, item {j}: {str(e)}")
                        continue
                
                progress_bar.update(latest_scan.shape[0])
    
    # Create DataFrame of per-image metrics
    per_image_df = pd.DataFrame(per_image_metrics)
    
    # Calculate brain mask usage statistics
    mask_used_count = per_image_df['used_brain_mask'].sum() if 'used_brain_mask' in per_image_df.columns else 0
    total_images = len(per_image_df)
    
    wandb.log({
        "brain_mask_usage_count": mask_used_count,
        "brain_mask_usage_percent": (mask_used_count/total_images*100) if total_images > 0 else 0
    })
    
    # Save the per-image metrics
    per_image_df.to_csv(f'{save_path}/synbadd_per_image_metrics_fold_{fold_index}.csv', index=False)
    
    # Create and log a table of image metrics
    image_metrics_table = wandb.Table(
        columns=["image_id", "synbadd_psnr", "synbadd_ssim"]
    )
    
    for _, row in per_image_df.iterrows():
        image_metrics_table.add_data(
            row["image_id"], 
            row["psnr_synbadd"], 
            row["ssim_synbadd"]
        )
    
    wandb.log({"image_metrics_table": image_metrics_table})
    
    # Log summary metrics
    wandb.log({
        "avg_psnr_synbadd": np.mean(psnr_values),
        "avg_ssim_synbadd": np.mean(ssim_values),
        "metric_note": "Brain-masked PSNR & SSIM (when masks available)"
    })
    
    # Create results DataFrame for dice scores
    results_df = pd.DataFrame(columns=['Region', 'SynBADD'])
    
    # Calculate and log average Dice scores for each region
    print("\nAverage Dice Scores:")
    for region_name in regions:
        avg_synbadd_dice = np.mean(synbadd_dice_scores[region_name])
        
        results_df = results_df._append({
            'Region': region_name.capitalize(),
            'SynBADD': avg_synbadd_dice
        }, ignore_index=True)
        
        print(f"{region_name.capitalize()}: {avg_synbadd_dice:.4f}")
        
        wandb.log({
            f"dice_{region_name}_synbadd": avg_synbadd_dice
        })
    
    # Save results to CSV
    results_df.to_csv(f'{save_path}/synbadd_dice_score_results_fold_{fold_index}.csv', index=False)
    
    # Log results as table to wandb
    wandb.log({"dice_score_results": wandb.Table(dataframe=results_df)})
    
    # Create bar chart visualization of dice scores
    plt.figure(figsize=(14, 8))
    plt.bar(results_df['Region'], results_df['SynBADD'], color='orange', alpha=0.7)
    plt.xlabel('Brain Region')
    plt.ylabel('Average Dice Score')
    plt.title('SynBADD Dice Scores by Brain Region')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = f'{save_path}/synbadd_dice_scores_fold_{fold_index}.png'
    plt.savefig(plot_path)
    plt.close()
    
    wandb.log({"synbadd_dice_scores": wandb.Image(plot_path)})
    
    # Create summary data dictionary
    summary_data = {
        "PSNR (SynBADD)": np.mean(psnr_values),
        "SSIM (SynBADD)": np.mean(ssim_values),
    }
    
    # Add dice scores to summary
    for region_name in regions:
        summary_data[f"Dice {region_name.capitalize()} (SynBADD)"] = np.mean(synbadd_dice_scores[region_name])
    
    # Create and save summary DataFrame
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(f'{save_path}/synbadd_summary_metrics_fold_{fold_index}.csv', index=False)
    
    wandb.log({"summary_metrics": wandb.Table(dataframe=summary_df)})
    
    # Create histograms of dice scores for each region
    for region_name in regions:
        plt.figure(figsize=(10, 6))
        plt.hist(synbadd_dice_scores[region_name], bins=20, alpha=0.7, color='orange')
        plt.xlabel('Dice Score')
        plt.ylabel('Number of Images')
        plt.title(f'Distribution of SynBADD Dice Scores for {region_name.capitalize()}')
        hist_path = f'{save_path}/synbadd_dice_hist_{region_name}_fold_{fold_index}.png'
        plt.savefig(hist_path)
        plt.close()
        
        wandb.log({f"dice_hist_{region_name}": wandb.Image(hist_path)})
    
    return results_df, summary_df, per_image_df