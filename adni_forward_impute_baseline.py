# =============================================================================
# File: adni_forward_impute_baseline.py
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

def get_metrics(target_img, forward_img, brain_mask=None, forward_brain_mask=None):
    target_img_data = target_img
    forward_img_data = forward_img
    
    if brain_mask is not None:
        binary_mask = brain_mask > 0
        forward_binary_mask = forward_brain_mask > 0
        target_masked = target_img_data * binary_mask
        forward_masked = forward_img_data * forward_binary_mask
        
        if np.sum(binary_mask) > 0:
            ssim_value = ssim(target_masked, forward_masked, data_range=1, multichannel=False)
            psnr_value = psnr(target_masked, forward_masked, data_range=1)
        else:
            ssim_value = ssim(target_img_data, forward_img_data, data_range=1, multichannel=False)
            psnr_value = psnr(target_img_data, forward_img_data, data_range=1)
    else:
        ssim_value = ssim(target_img_data, forward_img_data, data_range=1, multichannel=False)
        psnr_value = psnr(target_img_data, forward_img_data, data_range=1)
        
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
    slices = tuple([slice(offset, l + offset) for offset, l, offset in zip(offsets, array.shape)])
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

def get_most_recent_scan(subject, target_visit, df):
    """Find the most recent scan before the target date for the given subject"""
    # Define visit ordering
    visit_ordering = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
    
    # Get all visits for the subject
    subject_scans = df[df['Subject'] == subject]
    
    # Get target visit from subject_scans where Acq Date matches target_date
    # Get current visit index
    target_visit_idx = visit_ordering.index(target_visit)
    
    # Look for the most recent visit before target_visit
    for visit_idx in range(target_visit_idx - 1, -1, -1):
        prev_visit = visit_ordering[visit_idx]
        prev_scan = subject_scans[subject_scans['Visit'] == prev_visit]
        if len(prev_scan) > 0:
            return prev_scan.iloc[0]
    
    return None

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
        plt.plot(subject_data['Acq Date'], subject_data[f'forward_{region}_volume'], 
                 '-', color=colors[i], alpha=0.5, label=f"{subject} (Forward)")
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
            plt.plot(subject_data['Acq Date'], subject_data[f'forward_{region}_volume'], 
                     '-', color='blue', alpha=0.7, label="Forward")
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
        "SYNBADD_128_6_5e-05_0.0_2025-04-16_04-59-48_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold1",
        "SYNBADD_128_6_5e-05_0.0_2025-04-16_14-29-00_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold2",
        "SYNBADD_128_6_5e-05_0.0_2025-04-17_06-36-19_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold3",
        "SYNBADD_128_6_5e-05_0.0_2025-04-18_00-45-42_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold4",
        "SYNBADD_128_6_5e-05_0.0_2025-04-18_10-33-39_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold5",
    ]
    synbadd_run_name = synbadd_run_name_list[fold_index - 1]
    
    # project_name = f"ADNI_FOLD{fold_index}_Baseline_ROI_Volumes"
    # project_name = f"Forward_impute_dice_scores"
    project_name = f"Averaged_impute_dice_scores"
    
    wandb.init(
        project=project_name,
        # name=synbadd_run_name,
        name=f"Averaged_Impute_Fold_{fold_index}",
        config={
            "fold_index": fold_index,
            "model_name": synbadd_run_name
        }
    )
    
    forward_mask_dir = f"/scratch/anirudhkaushik/Forward_Impute_masks_{fold_index}/"
    synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/{synbadd_run_name}"
    mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_Affine/"
    brain_mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_brain_masks/"
    
    synbadd_mask_dir = f"{synbadd_dir}_mask"
    
    os.makedirs(forward_mask_dir, exist_ok=True)
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
    
    forward_dice_scores = {region: [] for region in regions}
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

            # search by subject and find all visits for that subject then find most recent available visit
            visit_ordering = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']       
            all_visits = df[df['Subject'] == target_subject]['Visit'].unique()
            # most recent visit since target visit
            all_visits = sorted(all_visits, key=lambda x: visit_ordering.index(x) if x in visit_ordering else len(visit_ordering))
            most_recent_visit = all_visits[all_visits.index(target_visit) - 1]
            
            # search by subject and most recent visit to find the row and use Image Data ID column
            forward_image_id = df[(df['Subject'] == target_subject) & (df['Visit'] == most_recent_visit)]['Image Data ID'].values[0]
            forward_image_path = df[(df['Subject'] == target_subject) & (df['Visit'] == most_recent_visit)]['File_Path'].values[0]

            # Find most recent previous scan
            prev_scan = get_most_recent_scan(target_subject, target_visit, df)
            if prev_scan is None:
                print(f"Skipping {image_id}: No previous scan found")
                continue
                
            prev_image_id = re.search(r'S\d+_I\d+', prev_scan['File_Path']).group(0)
            
            # Get SynBADD files
            synbadd_files = [f for f in all_synbadd_files if image_id in f]
            if len(synbadd_files) < 2:
                print(f"Skipping {image_id}: Missing SynBADD files")
                continue
            
            synbadd_file_path = next(f for f in synbadd_files if "_synthetic.nii.gz" in f)
            synbadd_field_path = next(f for f in synbadd_files if "_synthetic_field.nii.gz" in f)
            
            # Get mask files
            mask_files = [f for f in all_mask_files if image_id in f]
            forward_mask_files = [f for f in all_mask_files if forward_image_id in f]
            seg_file = next((f for f in mask_files if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in f), None)
            forward_seg_file = next((f for f in forward_mask_files if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in f), None)
            if not seg_file:
                print(f"Skipping {image_id}: Missing segmentation file")
                continue
            
            brain_mask_file = next((f for f in all_brain_mask_files if image_id in f and '_mask_resized_IPL_padded.nii.gz' in f), None)
            forward_brain_mask_file = next((f for f in all_brain_mask_files if forward_image_id in f and '_mask_resized_IPL_padded.nii.gz' in f), None)
            
            # Load images
            target_img_data = load_nii(target_path)
            forward_img_data = load_nii(prev_scan['File_Path'])
            synbadd_img_data = load_nii(os.path.join(synbadd_dir, synbadd_file_path))
            
            # Load masks
            mask_nib = nib.load(os.path.join(mask_dir, seg_file))
            mask_data = mask_nib.get_fdata()
            forward_mask_nib = nib.load(os.path.join(mask_dir, forward_seg_file))
            forward_mask_data = forward_mask_nib.get_fdata()
            
            brain_mask_data = None
            if brain_mask_file:
                brain_mask_nib = nib.load(os.path.join(brain_mask_dir, brain_mask_file))
                brain_mask_data = brain_mask_nib.get_fdata()

            forward_brain_mask_data = None
            if forward_brain_mask_file:
                forward_brain_mask_nib = nib.load(os.path.join(brain_mask_dir, forward_brain_mask_file))
                forward_brain_mask_data = forward_brain_mask_nib.get_fdata()
            
            # Define paths for warped masks
            synbadd_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_mask.nii.gz")
            synbadd_brain_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_brain_mask.nii.gz")
            
            # Check if warped masks already exist, if not create them
            if not os.path.exists(synbadd_mask_path):
                # Convert to ANTs format
                target_ants = ants.from_numpy(target_img_data)
                synbadd_ants = ants.from_numpy(synbadd_img_data)
                mask_ants = ants.from_numpy(mask_data)
                
                brain_mask_ants = None
                if brain_mask_data is not None:
                    brain_mask_ants = ants.from_numpy(brain_mask_data)
                
                # Register images
                
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
            forward_ssim, forward_psnr = get_metrics(
                target_img=target_img_data,
                forward_img=forward_img_data,
                brain_mask=brain_mask_data,
                forward_brain_mask=forward_brain_mask_data
            )
            
            synbadd_ssim, synbadd_psnr = get_metrics(
                target_img=target_img_data,
                forward_img=synbadd_img_data,
                brain_mask=brain_mask_data,
                forward_brain_mask=brain_mask_data  # For SynBADD, use same brain mask
            )
            
            # Log metrics to wandb
            wandb.log({
                "forward_ssim": forward_ssim,
                "forward_psnr": forward_psnr,
                "synbadd_ssim": synbadd_ssim,
                "synbadd_psnr": synbadd_psnr,
                "ssim_diff": synbadd_ssim - forward_ssim,
                "psnr_diff": synbadd_psnr - forward_psnr
            })
            
            
            
            # Calculate volumes and differences
            target_volumes = calculate_roi_volumes(mask_data, regions)
            forward_volumes = calculate_roi_volumes(forward_mask_data, regions)
            synbadd_volumes = calculate_roi_volumes(synbadd_mask_data, regions)
            
            forward_volume_diff = calculate_volume_difference(target_volumes, forward_volumes)
            synbadd_volume_diff = calculate_volume_difference(target_volumes, synbadd_volumes)
            
            # Store results
            image_data = {
                "image_id": image_id,
                "Subject": target_subject,
                "visit": target_visit,
                "Acq Date": target_date,
                "prev_image_id": prev_image_id,
                "prev_visit": prev_scan['Visit'],
                "prev_date": prev_scan['Acq Date'],
                "forward_ssim": forward_ssim,
                "forward_psnr": forward_psnr,
                "synbadd_ssim": synbadd_ssim,
                "synbadd_psnr": synbadd_psnr,
                "ssim_diff": synbadd_ssim - forward_ssim,
                "psnr_diff": synbadd_psnr - forward_psnr
            }

            for region in regions:
                image_data[f"target_{region}_volume"] = target_volumes[region]
                image_data[f"forward_{region}_volume"] = forward_volumes[region]
                image_data[f"synbadd_{region}_volume"] = synbadd_volumes[region]
                
                image_data[f"forward_{region}_volume_diff"] = forward_volume_diff[region]
                image_data[f"forward_{region}_volume_diff_percent"] = forward_volume_diff[f"{region}_percent"]
                
                image_data[f"synbadd_{region}_volume_diff"] = synbadd_volume_diff[region]
                image_data[f"synbadd_{region}_volume_diff_percent"] = synbadd_volume_diff[f"{region}_percent"]
            
            volume_data.append(image_data)
            
            # Calculate Dice scores
            for region_name, region_ids in regions.items():
                target_region = extract_region(mask_data, region_ids)
                forward_region = extract_region(forward_mask_data, region_ids)
                synbadd_region = extract_region(synbadd_mask_data, region_ids)
                
                forward_dice = dice_score(forward_region, target_region)
                synbadd_dice = dice_score(synbadd_region, target_region)
                
                forward_dice_scores[region_name].append(forward_dice)
                synbadd_dice_scores[region_name].append(synbadd_dice)
                
                wandb.log({
                    f"dice_{region_name}_forward": forward_dice,
                    f"dice_{region_name}_synbadd": synbadd_dice,
                    f"dice_diff_{region_name}": synbadd_dice - forward_dice
                })
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    
    # Create DataFrame and save results
    volume_df = pd.DataFrame(volume_data)
    volume_df.to_csv(f'{save_path}/roi_volumes_fold_{fold_index}.csv', index=False)

    # Calculate and save metrics summary
    metrics_summary = {
        'Forward_SSIM': volume_df['forward_ssim'].mean(),
        'Forward_PSNR': volume_df['forward_psnr'].mean(),
        'SynBADD_SSIM': volume_df['synbadd_ssim'].mean(),
        'SynBADD_PSNR': volume_df['synbadd_psnr'].mean(),
        'SSIM_Diff': volume_df['ssim_diff'].abs().mean(),
        'PSNR_Diff': volume_df['psnr_diff'].abs().mean()
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(f'{save_path}/metrics_summary_fold_{fold_index}.csv', index=False)
    wandb.log({"metrics_summary": wandb.Table(dataframe=metrics_df)})
    
    # Plot metrics comparison
    plt.figure(figsize=(10, 6))
    metrics_labels = ['SSIM', 'PSNR']
    forward_values = [metrics_summary['Forward_SSIM'], metrics_summary['Forward_PSNR']]
    synbadd_values = [metrics_summary['SynBADD_SSIM'], metrics_summary['SynBADD_PSNR']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    plt.bar(x - width/2, forward_values, width, label='Forward Impute', color='blue', alpha=0.7)
    plt.bar(x + width/2, synbadd_values, width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Average Value')
    plt.title('Comparison of Image Quality Metrics')
    plt.xticks(x, metrics_labels)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/metrics_comparison_fold_{fold_index}.png')
    wandb.log({"metrics_comparison": wandb.Image(f'{save_path}/metrics_comparison_fold_{fold_index}.png')})
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
        avg_forward_diff = volume_df[f"forward_{region}_volume_diff"].abs().mean()
        avg_forward_diff_pct = volume_df[f"forward_{region}_volume_diff_percent"].abs().mean()
        
        avg_synbadd_diff = volume_df[f"synbadd_{region}_volume_diff"].abs().mean()
        avg_synbadd_diff_pct = volume_df[f"synbadd_{region}_volume_diff_percent"].abs().mean()
        
        volume_summary.append({
            'Region': region.capitalize(),
            'Forward_Diff': avg_forward_diff,
            'Forward_Diff_Percent': avg_forward_diff_pct,
            'SynBADD_Diff': avg_synbadd_diff,
            'SynBADD_Diff_Percent': avg_synbadd_diff_pct
        })
    
    volume_summary = pd.DataFrame(volume_summary)
    volume_summary.to_csv(f'{save_path}/volume_diff_summary_fold_{fold_index}.csv', index=False)
    wandb.log({"volume_diff_summary": wandb.Table(dataframe=volume_summary)})
    
    # Plot volume differences
    plt.figure(figsize=(14, 8))
    index = np.arange(len(regions))
    bar_width = 0.35
    
    plt.bar(index, volume_summary['Forward_Diff_Percent'], bar_width, 
            label='Forward Impute', color='blue', alpha=0.7)
    plt.bar(index + bar_width, volume_summary['SynBADD_Diff_Percent'], bar_width, 
            label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Volume Difference (%)')
    plt.title('Comparison of Forward Impute vs SynBADD Volume Differences by Brain Region')
    plt.xticks(index + bar_width/2, volume_summary['Region'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/volume_diff_comparison_fold_{fold_index}.png')
    wandb.log({"volume_diff_comparison": wandb.Image(f'{save_path}/volume_diff_comparison_fold_{fold_index}.png')})
    plt.close()
    
    # Calculate and save dice summary
    dice_summary = []
    for region in regions:
        avg_forward_dice = np.mean(forward_dice_scores[region])
        avg_synbadd_dice = np.mean(synbadd_dice_scores[region])
        
        dice_summary.append({
            'Region': region.capitalize(),
            'Forward_Dice': avg_forward_dice,
            'SynBADD_Dice': avg_synbadd_dice,
            'Dice_Diff': avg_synbadd_dice - avg_forward_dice
        })
    
    dice_summary = pd.DataFrame(dice_summary)
    dice_summary.to_csv(f'{save_path}/dice_summary_fold_{fold_index}.csv', index=False)
    wandb.log({"dice_summary": wandb.Table(dataframe=dice_summary)})
    
    # Plot dice scores
    plt.figure(figsize=(14, 8))
    plt.bar(index, dice_summary['Forward_Dice'], bar_width, 
            label='Forward Impute', color='blue', alpha=0.7)
    plt.bar(index + bar_width, dice_summary['SynBADD_Dice'], bar_width, 
            label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Dice Score')
    plt.title('Comparison of Forward Impute vs SynBADD Dice Scores by Brain Region')
    plt.xticks(index + bar_width/2, dice_summary['Region'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/dice_comparison_fold_{fold_index}.png')
    wandb.log({"dice_comparison": wandb.Image(f'{save_path}/dice_comparison_fold_{fold_index}.png')})
    plt.close()
    
    wandb.finish()
    
    print("ROI volume analysis complete!")
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()