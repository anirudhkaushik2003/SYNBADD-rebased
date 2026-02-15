# =============================================================================
# File: adni_avg_impute_baseline.py
# Purpose: Analysis/comparison script producing fold-level metrics, visualizations, and W and B logging.
# Prerequisites: Pre-generated synthetic outputs plus baseline/reference assets in expected directories.
# Usually Run Next: Aggregate results across folds and prepare final reporting.
# Model Components Used: Primarily consumes generated outputs; model classes are generally not trained here.
# =============================================================================

import pandas as pd
import os
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import re
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import ants
import wandb
import sys

fold_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
save_path = f"/scratch/anirudhkaushik/SYNBADDQualitative{fold_index}/Images/"
os.makedirs(save_path, exist_ok=True)

# Read and prepare the DataFrame
df = pd.read_csv(csv_file)
df = df[df[f'fold_{fold_index}'] == 'test'].reset_index(drop=True)
df['Acq Date'] = pd.to_datetime(df['Acq Date'])
df = df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')

def get_metrics(target_img, averaged_img, brain_mask=None, averaged_brain_mask=None):
    target_img_data = target_img
    averaged_img_data = averaged_img
    
    if brain_mask is not None and averaged_brain_mask is not None:
        binary_mask = brain_mask > 0
        averaged_binary_mask = averaged_brain_mask > 0
        target_masked = target_img_data * binary_mask
        averaged_masked = averaged_img_data * averaged_binary_mask
        
        if np.sum(binary_mask) > 0:
            ssim_value = ssim(target_masked, averaged_masked, data_range=1, multichannel=False)
            psnr_value = psnr(target_masked, averaged_masked, data_range=1)
        else:
            ssim_value = ssim(target_img_data, averaged_img_data, data_range=1, multichannel=False)
            psnr_value = psnr(target_img_data, averaged_img_data, data_range=1)
    else:
        ssim_value = ssim(target_img_data, averaged_img_data, data_range=1, multichannel=False)
        psnr_value = psnr(target_img_data, averaged_img_data, data_range=1)
        
    return ssim_value, psnr_value

def load_nii(imgPath, padding=(128, 128, 128)):
    img_data = nib.load(imgPath).get_fdata()
    if img_data.ndim == 4:
        img_data = img_data.squeeze(0)
    img_data, _ = pad(img_data, shape=padding)
    img_data = preprocess_Normalizition(img_data)
    return img_data

def load_avg_impute_for_target(target_path, target_visit, avg_dir):
    """
    Your test() saves:
      syn_image_name = basename(target_path).replace(".nii.gz", f"_{opt.target_timepoint}_synthetic.nii.gz")
    This loader mirrors that convention.
    """
    base = os.path.basename(target_path)
    fname = base.replace(".nii.gz", f"_all_synthetic.nii.gz")
    fpath = os.path.join(avg_dir, fname)
    if not os.path.exists(fpath):
        return None
    return load_nii(fpath)

def pad(array, shape):
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape, offsets)])
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

def calculate_volume(mask_data, voxel_size=(2, 2, 2)):
    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    return np.sum(mask_data) * voxel_volume

def calculate_roi_volumes(mask_data, regions):
    volumes = {}
    for region_name, region_ids in regions.items():
        region_mask = extract_region(mask_data, region_ids)
        volumes[region_name] = calculate_volume(region_mask)
    return volumes

def calculate_volume_difference(target_volumes, predicted_volumes):
    diff = {}
    for region_name in target_volumes:
        diff[region_name] = predicted_volumes[region_name] - target_volumes[region_name]
        diff[f"{region_name}_percent"] = (diff[region_name] / target_volumes[region_name]) * 100 if target_volumes[region_name] > 0 else 0
    return diff

def get_all_other_scans(subject, target_visit, df):
    """Get all scans for the subject except the target visit"""
    # Get all visits for the subject
    subject_scans = df[df['Subject'] == subject]
    
    # Exclude the target visit
    other_scans = subject_scans[subject_scans['Visit'] != target_visit]
    
    return other_scans

def compute_averaged_impute(target_subject, target_visit, df):
    """Compute averaged impute from all other timepoints for the subject"""
    
    # Get all other scans for this subject
    other_scans = get_all_other_scans(target_subject, target_visit, df)
    
    if len(other_scans) == 0:
        return None, None
    
    # Lists to store all images
    all_images = []
    scan_info = []
    
    # Load all other timepoint data
    for _, scan in other_scans.iterrows():
        try:
            # Load image
            img_data = load_nii(scan['File_Path'])
            all_images.append(img_data)
            
            scan_id = re.search(r'S\d+_I\d+', scan['File_Path']).group(0)
            scan_info.append({
                'visit': scan['Visit'],
                'date': scan['Acq Date'],
                'image_id': scan_id
            })
            
        except Exception as e:
            print(f"Error loading scan {scan['File_Path']}: {str(e)}")
            continue
    
    if len(all_images) == 0:
        return None, None
    
    # Compute average image only (masks will be warped from target)
    averaged_img = np.mean(all_images, axis=0)
    
    return averaged_img, scan_info

def plot_volume_trajectories(volume_df, region="hippocampus", subjects_to_plot=5, individual_plots=True):
    volume_df['Acq Date'] = pd.to_datetime(volume_df['Acq Date'])
    
    subjects_list = volume_df['Subject'].unique()
    if len(subjects_list) > subjects_to_plot:
        subjects_list = subjects_list[:subjects_to_plot]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects_list)))
    
    plt.figure(figsize=(14, 8))
    for i, subject in enumerate(subjects_list):
        subject_data = volume_df[volume_df['Subject'] == subject].sort_values(by='Acq Date')
        
        plt.plot(subject_data['Acq Date'], subject_data[f'target_{region}_volume'], 
                 'o--', color=colors[i], alpha=0.7, label=f"{subject} (Target)")
        plt.plot(subject_data['Acq Date'], subject_data[f'averaged_{region}_volume'], 
                 '-', color=colors[i], alpha=0.5, label=f"{subject} (Averaged)")
        plt.plot(subject_data['Acq Date'], subject_data[f'synbadd_{region}_volume'], 
                 '-', color=colors[i], linewidth=2, label=f"{subject} (SynBADD)")
    
    plt.xlabel('Acquisition Date')
    plt.ylabel(f'{region.capitalize()} Volume')
    plt.title(f'{region.capitalize()} Volume Trajectory Across Subjects')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_path}/{region}_volume_trajectory_group.png')
    
    wandb.log({f"{region}_volume_trajectory_group": wandb.Image(f'{save_path}/{region}_volume_trajectory_group.png')})
    
    plt.close()
    
    if individual_plots:
        for i, subject in enumerate(subjects_list):
            subject_data = volume_df[volume_df['Subject'] == subject].sort_values(by='Acq Date')
            
            plt.figure(figsize=(10, 6))
            plt.plot(subject_data['Acq Date'], subject_data[f'target_{region}_volume'], 
                     'o--', color='black', alpha=0.7, label="Target")
            plt.plot(subject_data['Acq Date'], subject_data[f'averaged_{region}_volume'], 
                     '-', color='blue', alpha=0.7, label="Averaged")
            plt.plot(subject_data['Acq Date'], subject_data[f'synbadd_{region}_volume'], 
                     '-', color='orange', linewidth=2, label="SynBADD")
            
            plt.xlabel('Acquisition Date')
            plt.ylabel(f'{region.capitalize()} Volume')
            plt.title(f'{region.capitalize()} Volume Trajectory for Subject {subject}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_path}/{region}_volume_trajectory_{subject}.png')
            
            wandb.log({f"{region}_volume_trajectory_{subject}": wandb.Image(f'{save_path}/{region}_volume_trajectory_{subject}.png')})
            
            plt.close()

def main():
    synbadd_run_name_list = [
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
        "SYNBADD_128_6_5e-05_0.0_2025-05-24_02-29-18_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold3",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_15-09-16_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold4",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-35-52_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold5"
    ]
    synbadd_run_name = synbadd_run_name_list[fold_index - 1]
    
    project_name = f"NEW Averaged_impute_dice_scores"
    
    wandb.init(
        project=project_name,
        name=f"Averaged_Impute_Fold_{fold_index}",
        config={
            "fold_index": fold_index,
            "model_name": synbadd_run_name
        }
    )
    
    averaged_mask_dir = f"/scratch/anirudhkaushik/Averaged_Impute_masks_{fold_index}/"
    synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/{synbadd_run_name}"
    avg_impute_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/FINAL_Avg_impute_fold_{fold_index}/"
    mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_Affine/"
    brain_mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_brain_masks/"
    
    synbadd_mask_dir = f"{synbadd_dir}_mask"
    
    os.makedirs(averaged_mask_dir, exist_ok=True)
    os.makedirs(synbadd_mask_dir, exist_ok=True)
    
    all_synbadd_files = os.listdir(synbadd_dir)
    all_synbadd_files_iter = [f for f in all_synbadd_files if "_synthetic.nii.gz" in f]
    all_mask_files = os.listdir(mask_dir)
    all_brain_mask_files = os.listdir(brain_mask_dir)
    
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
    
    averaged_dice_scores = {region: [] for region in regions}
    synbadd_dice_scores = {region: [] for region in regions}
    
    volume_data = []
    
    # Process all target timepoints
    for i, file in enumerate(tqdm(all_synbadd_files_iter)):
        if not file.endswith('.nii.gz'):
            continue
        
        image_id = re.search(r'S\d+_I\d+', file)
        if not image_id:
            continue
        image_id = image_id.group(0)
        try:
            # Get target scan info
            target_info = df[df['File_Path'].str.contains(image_id)].iloc[0]
            target_subject = target_info['Subject']
            target_date = target_info['Acq Date']
            target_visit = target_info['Visit']
            target_path = target_info['File_Path']

            averaged_img_data = load_avg_impute_for_target(
                target_path=target_path,
                target_visit=target_visit,
                avg_dir=avg_impute_dir
            )
            if averaged_img_data is None:
                print(f"Skipping {image_id}: Avg-impute file not found in {avg_impute_dir}")
                continue

            # We didn’t “build” an average from other scans in this script anymore,
            # so 'scan_info' isn’t meaningful. You can either drop it or stub it:
            scan_info = []
            
            # Get SynBADD files
            synbadd_files = [f for f in all_synbadd_files if image_id in f]
            if len(synbadd_files) < 2:
                print(f"Skipping {image_id}: Missing SynBADD files")
                continue
            
            synbadd_file_path = next(f for f in synbadd_files if "_synthetic.nii.gz" in f)
            synbadd_field_path = next(f for f in synbadd_files if "_synthetic_field.nii.gz" in f)
            
            # Get target mask files
            mask_files = [f for f in all_mask_files if image_id in f]
            seg_file = next((f for f in mask_files if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in f), None)
            if not seg_file:
                print(f"Skipping {image_id}: Missing segmentation file")
                continue
            
            brain_mask_file = next((f for f in all_brain_mask_files if image_id in f and '_mask_resized_IPL_padded.nii.gz' in f), None)
            
            # Load target images
            target_img_data = load_nii(target_path)
            synbadd_img_data = load_nii(os.path.join(synbadd_dir, synbadd_file_path))
            
            # Load target masks
            mask_nib = nib.load(os.path.join(mask_dir, seg_file))
            mask_data = mask_nib.get_fdata()
            
            brain_mask_data = None
            if brain_mask_file:
                brain_mask_nib = nib.load(os.path.join(brain_mask_dir, brain_mask_file))
                brain_mask_data = brain_mask_nib.get_fdata()
            
            # Warp target mask to averaged space
            averaged_mask_path = os.path.join(averaged_mask_dir, f"{image_id}_averaged_mask.nii.gz")
            averaged_brain_mask_path = os.path.join(averaged_mask_dir, f"{image_id}_averaged_brain_mask.nii.gz")
            
            # Check if warped masks already exist for averaged, if not create them
            if not os.path.exists(averaged_mask_path):
                # Convert to ANTs format
                target_ants = ants.from_numpy(target_img_data)
                averaged_ants = ants.from_numpy(averaged_img_data)
                mask_ants = ants.from_numpy(mask_data)
                
                brain_mask_ants = None
                if brain_mask_data is not None:
                    brain_mask_ants = ants.from_numpy(brain_mask_data)
                
                # Register target to averaged space
                averaged_reg = ants.registration(
                    fixed=averaged_ants,
                    moving=target_ants,
                    type_of_transform='SyN'
                )
                
                # Warp masks from target to averaged space
                averaged_mask_warped = ants.apply_transforms(
                    fixed=averaged_ants,
                    moving=mask_ants,
                    transformlist=averaged_reg['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                
                # Handle brain masks if they exist
                averaged_brain_mask_warped = None
                if brain_mask_ants is not None:
                    averaged_brain_mask_warped = ants.apply_transforms(
                        fixed=averaged_ants,
                        moving=brain_mask_ants,
                        transformlist=averaged_reg['fwdtransforms'],
                        interpolator='nearestNeighbor'
                    )
                
                # Save warped masks
                averaged_mask_data = averaged_mask_warped.numpy()
                nib.save(nib.Nifti1Image(averaged_mask_data, affine=np.eye(4)), averaged_mask_path)
                
                averaged_brain_mask_data = None
                if averaged_brain_mask_warped is not None:
                    averaged_brain_mask_data = averaged_brain_mask_warped.numpy()
                    nib.save(nib.Nifti1Image(averaged_brain_mask_data, affine=np.eye(4)), averaged_brain_mask_path)
            else:
                # Load existing warped masks
                averaged_mask_data = nib.load(averaged_mask_path).get_fdata()
                averaged_brain_mask_data = None
                if os.path.exists(averaged_brain_mask_path):
                    averaged_brain_mask_data = nib.load(averaged_brain_mask_path).get_fdata()
            
            # Warp target mask to SynBADD space
            synbadd_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_mask.nii.gz")
            synbadd_brain_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_brain_mask.nii.gz")
            
            # Check if warped masks already exist for SynBADD, if not create them
            if not os.path.exists(synbadd_mask_path):
                # Convert to ANTs format
                target_ants = ants.from_numpy(target_img_data)
                synbadd_ants = ants.from_numpy(synbadd_img_data)
                mask_ants = ants.from_numpy(mask_data)
                
                brain_mask_ants = None
                if brain_mask_data is not None:
                    brain_mask_ants = ants.from_numpy(brain_mask_data)
                
                # Register target to SynBADD space
                synbadd_reg = ants.registration(
                    fixed=synbadd_ants,
                    moving=target_ants,
                    type_of_transform='SyN'
                )
                
                # Warp masks
                synbadd_mask_warped = ants.apply_transforms(
                    fixed=synbadd_ants,
                    moving=mask_ants,
                    transformlist=synbadd_reg['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                
                # Handle brain masks if they exist
                synbadd_brain_mask_warped = None
                if brain_mask_ants is not None:
                    synbadd_brain_mask_warped = ants.apply_transforms(
                        fixed=synbadd_ants,
                        moving=brain_mask_ants,
                        transformlist=synbadd_reg['fwdtransforms'],
                        interpolator='nearestNeighbor'
                    )
                
                # Save warped masks
                synbadd_mask_data = synbadd_mask_warped.numpy()
                nib.save(nib.Nifti1Image(synbadd_mask_data, affine=np.eye(4)), synbadd_mask_path)
                
                if synbadd_brain_mask_warped is not None:
                    synbadd_brain_mask_data = synbadd_brain_mask_warped.numpy()
                    nib.save(nib.Nifti1Image(synbadd_brain_mask_data, affine=np.eye(4)), synbadd_brain_mask_path)
            else:
                # Load existing warped masks
                synbadd_mask_data = nib.load(synbadd_mask_path).get_fdata()

            # Calculate SSIM and PSNR metrics
            averaged_ssim, averaged_psnr = get_metrics(
                target_img=target_img_data,
                averaged_img=averaged_img_data,
                brain_mask=brain_mask_data,
                averaged_brain_mask=averaged_brain_mask_data
            )
            
            synbadd_ssim, synbadd_psnr = get_metrics(
                target_img=target_img_data,
                averaged_img=synbadd_img_data,
                brain_mask=brain_mask_data,
                averaged_brain_mask=brain_mask_data  # For SynBADD, use same brain mask
            )
            
            # Log metrics to wandb
            wandb.log({
                "averaged_ssim": averaged_ssim,
                "averaged_psnr": averaged_psnr,
                "synbadd_ssim": synbadd_ssim,
                "synbadd_psnr": synbadd_psnr,
                "ssim_diff": synbadd_ssim - averaged_ssim,
                "psnr_diff": synbadd_psnr - averaged_psnr
            })
            
            # Calculate volumes and differences
            target_volumes = calculate_roi_volumes(mask_data, regions)
            averaged_volumes = calculate_roi_volumes(averaged_mask_data, regions)
            synbadd_volumes = calculate_roi_volumes(synbadd_mask_data, regions)
            
            averaged_volume_diff = calculate_volume_difference(target_volumes, averaged_volumes)
            synbadd_volume_diff = calculate_volume_difference(target_volumes, synbadd_volumes)
            
            # Store results
            image_data = {
                "image_id": image_id,
                "Subject": target_subject,
                "visit": target_visit,
                "Acq Date": target_date,
                "num_scans_averaged": len(scan_info),
                "scans_used": str([s['visit'] for s in scan_info]),
                "averaged_ssim": averaged_ssim,
                "averaged_psnr": averaged_psnr,
                "synbadd_ssim": synbadd_ssim,
                "synbadd_psnr": synbadd_psnr,
                "ssim_diff": synbadd_ssim - averaged_ssim,
                "psnr_diff": synbadd_psnr - averaged_psnr
            }

            for region in regions:
                image_data[f"target_{region}_volume"] = target_volumes[region]
                image_data[f"averaged_{region}_volume"] = averaged_volumes[region]
                image_data[f"synbadd_{region}_volume"] = synbadd_volumes[region]
                
                image_data[f"averaged_{region}_volume_diff"] = averaged_volume_diff[region]
                image_data[f"averaged_{region}_volume_diff_percent"] = averaged_volume_diff[f"{region}_percent"]
                
                image_data[f"synbadd_{region}_volume_diff"] = synbadd_volume_diff[region]
                image_data[f"synbadd_{region}_volume_diff_percent"] = synbadd_volume_diff[f"{region}_percent"]
            
            volume_data.append(image_data)
            
            # Calculate Dice scores
            for region_name, region_ids in regions.items():
                target_region = extract_region(mask_data, region_ids)
                averaged_region = extract_region(averaged_mask_data, region_ids)
                synbadd_region = extract_region(synbadd_mask_data, region_ids)
                
                averaged_dice = dice_score(averaged_region, target_region)
                synbadd_dice = dice_score(synbadd_region, target_region)
                
                averaged_dice_scores[region_name].append(averaged_dice)
                synbadd_dice_scores[region_name].append(synbadd_dice)
                
                wandb.log({
                    f"dice_{region_name}_averaged": averaged_dice,
                    f"dice_{region_name}_synbadd": synbadd_dice,
                    f"dice_diff_{region_name}": synbadd_dice - averaged_dice
                })
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Create DataFrame and save results
    volume_df = pd.DataFrame(volume_data)
    volume_df.to_csv(f'{save_path}/roi_volumes_averaged_fold_{fold_index}.csv', index=False)

    # Calculate and save metrics summary
    metrics_summary = {
        'Averaged_SSIM': volume_df['averaged_ssim'].mean(),
        'Averaged_PSNR': volume_df['averaged_psnr'].mean(),
        'SynBADD_SSIM': volume_df['synbadd_ssim'].mean(),
        'SynBADD_PSNR': volume_df['synbadd_psnr'].mean(),
        'SSIM_Diff': volume_df['ssim_diff'].mean(),
        'PSNR_Diff': volume_df['psnr_diff'].mean()
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(f'{save_path}/metrics_summary_averaged_fold_{fold_index}.csv', index=False)
    wandb.log({"metrics_summary": wandb.Table(dataframe=metrics_df)})
    
    # Plot metrics comparison
    plt.figure(figsize=(10, 6))
    metrics_labels = ['SSIM', 'PSNR']
    averaged_values = [metrics_summary['Averaged_SSIM'], metrics_summary['Averaged_PSNR']]
    synbadd_values = [metrics_summary['SynBADD_SSIM'], metrics_summary['SynBADD_PSNR']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    plt.bar(x - width/2, averaged_values, width, label='Averaged Impute', color='blue', alpha=0.7)
    plt.bar(x + width/2, synbadd_values, width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.title('Comparison of Image Quality Metrics')
    plt.xticks(x, metrics_labels)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/metrics_comparison_averaged_fold_{fold_index}.png')
    wandb.log({"metrics_comparison": wandb.Image(f'{save_path}/metrics_comparison_averaged_fold_{fold_index}.png')})
    plt.close()

    wandb.log({"volume_data": wandb.Table(dataframe=volume_df)})
    
    # Plot trajectories
    plot_volume_trajectories(volume_df, region="hippocampus", subjects_to_plot=5, individual_plots=True)
    for region in regions:
        if region != "hippocampus":
            plot_volume_trajectories(volume_df, region=region, subjects_to_plot=5, individual_plots=False)
    
    # Calculate and save volume summary
    volume_summary = []
    for region in regions:
        avg_averaged_diff = volume_df[f"averaged_{region}_volume_diff"].abs().mean()
        avg_averaged_diff_pct = volume_df[f"averaged_{region}_volume_diff_percent"].abs().mean()
        
        avg_synbadd_diff = volume_df[f"synbadd_{region}_volume_diff"].abs().mean()
        avg_synbadd_diff_pct = volume_df[f"synbadd_{region}_volume_diff_percent"].abs().mean()
        
        volume_summary.append({
            'Region': region.capitalize(),
            'Averaged_Diff': avg_averaged_diff,
            'Averaged_Diff_Percent': avg_averaged_diff_pct,
            'SynBADD_Diff': avg_synbadd_diff,
            'SynBADD_Diff_Percent': avg_synbadd_diff_pct
        })
    
    volume_summary = pd.DataFrame(volume_summary)
    volume_summary.to_csv(f'{save_path}/volume_diff_summary_averaged_fold_{fold_index}.csv', index=False)
    wandb.log({"volume_diff_summary": wandb.Table(dataframe=volume_summary)})
    
    # Plot volume differences
    plt.figure(figsize=(14, 8))
    index = np.arange(len(regions))
    bar_width = 0.35
    
    plt.bar(index, volume_summary['Averaged_Diff_Percent'], bar_width, 
            label='Averaged Impute', color='blue', alpha=0.7)
    plt.bar(index + bar_width, volume_summary['SynBADD_Diff_Percent'], bar_width, 
            label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Volume Difference (%)')
    plt.title('Comparison of Averaged Impute vs SynBADD Volume Differences by Brain Region')
    plt.xticks(index + bar_width/2, volume_summary['Region'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/volume_diff_comparison_averaged_fold_{fold_index}.png')
    wandb.log({"volume_diff_comparison": wandb.Image(f'{save_path}/volume_diff_comparison_averaged_fold_{fold_index}.png')})
    plt.close()
    
    # Calculate and save dice summary
    dice_summary = []
    for region in regions:
        if len(averaged_dice_scores[region]) > 0:
            avg_averaged_dice = np.mean(averaged_dice_scores[region])
            avg_synbadd_dice = np.mean(synbadd_dice_scores[region])
            
            dice_summary.append({
                'Region': region.capitalize(),
                'Averaged_Dice': avg_averaged_dice,
                'SynBADD_Dice': avg_synbadd_dice,
                'Dice_Diff': avg_synbadd_dice - avg_averaged_dice
            })
    
    if len(dice_summary) > 0:
        dice_summary = pd.DataFrame(dice_summary)
        dice_summary.to_csv(f'{save_path}/dice_summary_averaged_fold_{fold_index}.csv', index=False)
        wandb.log({"dice_summary": wandb.Table(dataframe=dice_summary)})
        
        # Plot dice scores
        plt.figure(figsize=(14, 8))
        index = np.arange(len(dice_summary))
        plt.bar(index, dice_summary['Averaged_Dice'], bar_width, 
                label='Averaged Impute', color='blue', alpha=0.7)
        plt.bar(index + bar_width, dice_summary['SynBADD_Dice'], bar_width, 
                label='SynBADD', color='orange', alpha=0.7)
        
        plt.xlabel('Brain Region')
        plt.ylabel('Average Dice Score')
        plt.title('Comparison of Averaged Impute vs SynBADD Dice Scores by Brain Region')
        plt.xticks(index + bar_width/2, dice_summary['Region'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{save_path}/dice_comparison_averaged_fold_{fold_index}.png')
        wandb.log({"dice_comparison": wandb.Image(f'{save_path}/dice_comparison_averaged_fold_{fold_index}.png')})
        plt.close()
    
    # Log summary statistics
    print("\n" + "="*50)
    print(f"AVERAGED IMPUTE ANALYSIS COMPLETE - FOLD {fold_index}")
    print("="*50)
    print(f"\nNumber of subjects processed: {len(volume_df)}")
    print(f"Average number of scans used for averaging: {volume_df['num_scans_averaged'].mean():.2f}")
    print(f"\nMETRICS COMPARISON:")
    print(f"  Averaged Impute SSIM: {metrics_summary['Averaged_SSIM']:.4f}")
    print(f"  SynBADD SSIM: {metrics_summary['SynBADD_SSIM']:.4f}")
    print(f"  SSIM Improvement: {(metrics_summary['SynBADD_SSIM'] - metrics_summary['Averaged_SSIM']):.4f}")
    print(f"  Averaged Impute PSNR: {metrics_summary['Averaged_PSNR']:.2f}")
    print(f"  SynBADD PSNR: {metrics_summary['SynBADD_PSNR']:.2f}")
    print(f"  PSNR Improvement: {(metrics_summary['SynBADD_PSNR'] - metrics_summary['Averaged_PSNR']):.2f}")
    
    wandb.finish()
    
    print(f"\nResults saved to {save_path}")
    print("="*50)

if __name__ == "__main__":
    main()