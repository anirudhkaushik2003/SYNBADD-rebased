# =============================================================================
# File: temp_adni_baseline_comparison_analysis.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
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

df = pd.read_csv(csv_file)
df = df[df[f'fold_{fold_index}'] == 'test'].reset_index(drop=True)

df['Acq Date'] = pd.to_datetime(df['Acq Date'])
df = df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
subject_groups = df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
subjects = list(subject_groups.index.levels[0])

def get_metrics(target_img, model_img, brain_mask=None, model_brain_mask=None):
    target_img_data = target_img
    model_img_data = model_img
    
    if brain_mask is not None:
        binary_mask = brain_mask > 0
        model_binary_mask = model_brain_mask > 0
        target_masked = target_img_data * binary_mask
        model_masked = model_img_data * model_binary_mask
        
        if np.sum(binary_mask) > 0:
            ssim_value = ssim(target_masked, model_masked, data_range=1, multichannel=False)
            psnr_value = psnr(target_masked, model_masked, data_range=1)
        else:
            ssim_value = ssim(target_img_data, model_img_data, data_range=1, multichannel=False)
            psnr_value = psnr(target_img_data, model_img_data, data_range=1)
    else:
        ssim_value = ssim(target_img_data, model_img_data, data_range=1, multichannel=False)
        psnr_value = psnr(target_img_data, model_img_data, data_range=1)
        
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

def calculate_trajectory_alignment(volume_df, regions):
    """Calculate overall trajectory alignment for all subjects"""
    trajectory_alignment = {}
    
    for region in regions:
        noder_alignments = []
        synbadd_alignments = []
        
        # Calculate for each subject
        for subject in volume_df['Subject'].unique():
            subject_data = volume_df[volume_df['Subject'] == subject].sort_values(by='Acq Date')
            
            if len(subject_data) < 2:  # Need at least 2 timepoints for trajectory
                continue
                
            # Calculate absolute differences at each timepoint
            noder_diffs = np.abs(subject_data[f'target_{region}_volume'] - subject_data[f'noder_{region}_volume'])
            synbadd_diffs = np.abs(subject_data[f'target_{region}_volume'] - subject_data[f'synbadd_{region}_volume'])
            
            # Average across timepoints for this subject
            noder_alignments.append(np.mean(noder_diffs))
            synbadd_alignments.append(np.mean(synbadd_diffs))
        
        # Average across all subjects
        trajectory_alignment[f'{region}_noder'] = np.mean(noder_alignments) if noder_alignments else 0
        trajectory_alignment[f'{region}_synbadd'] = np.mean(synbadd_alignments) if synbadd_alignments else 0
        trajectory_alignment[f'{region}_improvement'] = trajectory_alignment[f'{region}_noder'] - trajectory_alignment[f'{region}_synbadd']
    
    # Calculate overall alignment (average across all regions)
    overall_noder = np.mean([trajectory_alignment[f'{region}_noder'] for region in regions])
    overall_synbadd = np.mean([trajectory_alignment[f'{region}_synbadd'] for region in regions])
    
    trajectory_alignment['overall_noder'] = overall_noder
    trajectory_alignment['overall_synbadd'] = overall_synbadd
    trajectory_alignment['overall_improvement'] = overall_noder - overall_synbadd
    
    return trajectory_alignment

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
        plt.plot(subject_data['Acq Date'], subject_data[f'noder_{region}_volume'], 
                 '-', color=colors[i], alpha=0.5, label=f"{subject} (NODER)")
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
            plt.plot(subject_data['Acq Date'], subject_data[f'noder_{region}_volume'], 
                     '-', color='blue', alpha=0.7, label="NODER")
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
    
    return

def main():
    synbadd_run_name_list = [
        # "SYNBADD_128_6_5e-05_0.0_2025-04-16_04-59-48_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold1",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-42-58_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold2",
        "SYNBADD_128_6_5e-05_0.0_2025-05-24_02-29-18_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold3",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_15-09-16_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold4",
        "SYNBADD_128_6_5e-05_0.0_2025-05-22_21-35-52_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold5"
    ]
    synbadd_run_name = synbadd_run_name_list[fold_index - 1]
    
    project_name = f"ADNI_FOLD{fold_index}_Baseline_ROI_Volumes"
    
    wandb.init(
        project="Final Dice Scores",
        name=synbadd_run_name,
        config={
            "fold_index": fold_index,
            "model_name": synbadd_run_name
        }
    )
    
    noder_dir = f"/scratch/anirudhkaushik/NODER_result_{fold_index}/"
    synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/{synbadd_run_name}"
    mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_Affine/"
    brain_mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_brain_masks/"
    
    noder_mask_dir = f"{noder_dir}_mask"
    synbadd_mask_dir = f"{synbadd_dir}_mask"
    
    os.makedirs(noder_mask_dir, exist_ok=True)
    os.makedirs(synbadd_mask_dir, exist_ok=True)
    
    all_noder_files = os.listdir(noder_dir)
    all_synbadd_files = os.listdir(synbadd_dir)
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
    
    noder_dice_scores = {region: [] for region in regions}
    synbadd_dice_scores = {region: [] for region in regions}
    
    volume_data = []
    
    for i, file in enumerate(tqdm(all_noder_files)):
        if not file.endswith('.nii.gz'):
            continue
        
        image_id = re.search(r'S\d+_I\d+', file)
        if not image_id:
            continue
        image_id = image_id.group(0)
        try:
            file_path = df[df['File_Path'].str.contains(image_id)]['File_Path'].values[0]
            subject = df[df['File_Path'].str.contains(image_id)]['Subject'].values[0]
            acq_date = df[df['File_Path'].str.contains(image_id)]['Acq Date'].values[0]
            visit = df[df['File_Path'].str.contains(image_id)]['Visit'].values[0]
            
            synbadd_files = [f for f in all_synbadd_files if image_id in f]
            if len(synbadd_files) < 2:
                print(f"Skipping {image_id}: Missing SynBADD files")
                continue
            
            for k in synbadd_files:
                if "_synthetic.nii.gz" in k:
                    synbadd_file_path = k
                elif "_synthetic_field.nii.gz" in k:
                    synbadd_field_path = k
    
            mask_files = [f for f in all_mask_files if image_id in f]
            seg_file = None
            for k in mask_files:
                if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in k:
                    seg_file = k
                    break
            
            if not seg_file:
                print(f"Skipping {image_id}: Missing segmentation file")
                continue
                
            brain_mask_files = [f for f in all_brain_mask_files if image_id in f]
            brain_mask_file = None
            for k in brain_mask_files:
                if '_mask_resized_IPL_padded.nii.gz' in k:
                    brain_mask_file = k
                    break
            
            target_img_data = load_nii(file_path)
            noder_img_data = load_nii(os.path.join(noder_dir, file))
            synbadd_img_data = load_nii(os.path.join(synbadd_dir, synbadd_file_path))
            
            mask_nib = nib.load(os.path.join(mask_dir, seg_file))
            mask_data = mask_nib.get_fdata()
            
            brain_mask_data = None
            if brain_mask_file:
                brain_mask_nib = nib.load(os.path.join(brain_mask_dir, brain_mask_file))
                brain_mask_data = brain_mask_nib.get_fdata()
            
            noder_mask_path = os.path.join(noder_mask_dir, f"{image_id}_mask.nii.gz")
            synbadd_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_mask.nii.gz")
            noder_brain_mask_path = os.path.join(noder_mask_dir, f"{image_id}_brain_mask.nii.gz")
            synbadd_brain_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_brain_mask.nii.gz")
            
            noder_brain_mask_data = None
            synbadd_brain_mask_data = None
            
            if os.path.exists(noder_mask_path) and os.path.exists(synbadd_mask_path):
                noder_mask_nib = nib.load(noder_mask_path)
                noder_mask_data = noder_mask_nib.get_fdata()
                
                synbadd_mask_nib = nib.load(synbadd_mask_path)
                synbadd_mask_data = synbadd_mask_nib.get_fdata()
                
                if os.path.exists(noder_brain_mask_path) and os.path.exists(synbadd_brain_mask_path):
                    noder_brain_mask_nib = nib.load(noder_brain_mask_path)
                    noder_brain_mask_data = noder_brain_mask_nib.get_fdata()
                    
                    synbadd_brain_mask_nib = nib.load(synbadd_brain_mask_path)
                    synbadd_brain_mask_data = synbadd_brain_mask_nib.get_fdata()
            else:
                target_ants = ants.from_numpy(target_img_data)
                noder_ants = ants.from_numpy(noder_img_data)
                synbadd_ants = ants.from_numpy(synbadd_img_data)
                mask_ants = ants.from_numpy(mask_data)
                
                brain_mask_ants = None
                if brain_mask_data is not None:
                    brain_mask_ants = ants.from_numpy(brain_mask_data)
                
                noder_reg = ants.registration(
                    fixed=noder_ants,
                    moving=target_ants,
                    type_of_transform='SyN'
                )
                
                synbadd_reg = ants.registration(
                    fixed=synbadd_ants,
                    moving=target_ants,
                    type_of_transform='SyN'
                )
                
                noder_mask_warped = ants.apply_transforms(
                    fixed=noder_ants,
                    moving=mask_ants,
                    transformlist=noder_reg['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                
                synbadd_mask_warped = ants.apply_transforms(
                    fixed=synbadd_ants,
                    moving=mask_ants,
                    transformlist=synbadd_reg['fwdtransforms'],
                    interpolator='nearestNeighbor'
                )
                
                noder_brain_mask_warped = None
                synbadd_brain_mask_warped = None
                if brain_mask_ants is not None:
                    noder_brain_mask_warped = ants.apply_transforms(
                        fixed=noder_ants,
                        moving=brain_mask_ants,
                        transformlist=noder_reg['fwdtransforms'],
                        interpolator='nearestNeighbor'
                    )
                    
                    synbadd_brain_mask_warped = ants.apply_transforms(
                        fixed=synbadd_ants,
                        moving=brain_mask_ants,
                        transformlist=synbadd_reg['fwdtransforms'],
                        interpolator='nearestNeighbor'
                    )
                
                noder_mask_data = noder_mask_warped.numpy()
                synbadd_mask_data = synbadd_mask_warped.numpy()
                
                noder_mask_nib = nib.Nifti1Image(noder_mask_data, affine=np.eye(4))
                nib.save(noder_mask_nib, noder_mask_path)
                
                synbadd_mask_nib = nib.Nifti1Image(synbadd_mask_data, affine=np.eye(4))
                nib.save(synbadd_mask_nib, synbadd_mask_path)
                
                if noder_brain_mask_warped is not None and synbadd_brain_mask_warped is not None:
                    noder_brain_mask_data = noder_brain_mask_warped.numpy()
                    synbadd_brain_mask_data = synbadd_brain_mask_warped.numpy()
                    
                    noder_brain_mask_nib = nib.Nifti1Image(noder_brain_mask_data, affine=np.eye(4))
                    nib.save(noder_brain_mask_nib, os.path.join(noder_mask_dir, f"{image_id}_brain_mask.nii.gz"))
                    
                    synbadd_brain_mask_nib = nib.Nifti1Image(synbadd_brain_mask_data, affine=np.eye(4))
                    nib.save(synbadd_brain_mask_nib, os.path.join(synbadd_mask_dir, f"{image_id}_brain_mask.nii.gz"))
                    
                    if brain_mask_data is not None:
                        brain_mask_nib = nib.Nifti1Image(brain_mask_data, affine=np.eye(4))
                        nib.save(brain_mask_nib, os.path.join(noder_mask_dir, f"{image_id}_orig_brain_mask.nii.gz"))
                        nib.save(brain_mask_nib, os.path.join(synbadd_mask_dir, f"{image_id}_orig_brain_mask.nii.gz"))
            
            # Calculate SSIM and PSNR
            noder_ssim, noder_psnr = get_metrics(
                target_img_data, noder_img_data, 
                brain_mask_data, noder_brain_mask_data
            )
            
            synbadd_ssim, synbadd_psnr = get_metrics(
                target_img_data, synbadd_img_data,
                brain_mask_data, synbadd_brain_mask_data
            )
            
            target_volumes = calculate_roi_volumes(mask_data, regions)
            noder_volumes = calculate_roi_volumes(noder_mask_data, regions)
            synbadd_volumes = calculate_roi_volumes(synbadd_mask_data, regions)
            
            noder_volume_diff = calculate_volume_difference(target_volumes, noder_volumes)
            synbadd_volume_diff = calculate_volume_difference(target_volumes, synbadd_volumes)
            
            image_data = {
                "image_id": image_id,
                "Subject": subject,
                "visit": visit,
                "Acq Date": acq_date,
                "noder_ssim": noder_ssim,
                "noder_psnr": noder_psnr,
                "synbadd_ssim": synbadd_ssim,
                "synbadd_psnr": synbadd_psnr,
            }
            
            for region in regions:
                image_data[f"target_{region}_volume"] = target_volumes[region]
                image_data[f"noder_{region}_volume"] = noder_volumes[region]
                image_data[f"synbadd_{region}_volume"] = synbadd_volumes[region]
                
                image_data[f"noder_{region}_volume_diff"] = noder_volume_diff[region]
                image_data[f"noder_{region}_volume_diff_percent"] = noder_volume_diff[f"{region}_percent"]
                
                image_data[f"synbadd_{region}_volume_diff"] = synbadd_volume_diff[region]
                image_data[f"synbadd_{region}_volume_diff_percent"] = synbadd_volume_diff[f"{region}_percent"]
            
            volume_data.append(image_data)
            
            # Log individual metrics to wandb
            wandb.log({
                f"ssim_noder": noder_ssim,
                f"ssim_synbadd": synbadd_ssim,
                f"ssim_improvement": synbadd_ssim - noder_ssim,
                f"psnr_noder": noder_psnr,
                f"psnr_synbadd": synbadd_psnr,
                f"psnr_improvement": synbadd_psnr - noder_psnr,
            })
            
        except Exception as e:
            print(f"Error processing {image_id} file: {str(e)}")
            continue
    
    volume_df = pd.DataFrame(volume_data)
    volume_df.to_csv(f'{save_path}/roi_volumes_fold_{fold_index}.csv', index=False)
    
    wandb.log({"volume_data": wandb.Table(dataframe=volume_df)})
    
    # Calculate trajectory alignment for all subjects
    trajectory_alignment = calculate_trajectory_alignment(volume_df, regions)
    
    # Log trajectory alignment metrics to wandb
    for key, value in trajectory_alignment.items():
        wandb.log({f"trajectory_alignment_{key}": value})
    
    # Save trajectory alignment results
    trajectory_df = pd.DataFrame([trajectory_alignment])
    trajectory_df.to_csv(f'{save_path}/trajectory_alignment_fold_{fold_index}.csv', index=False)
    wandb.log({"trajectory_alignment": wandb.Table(dataframe=trajectory_df)})
    
    plot_volume_trajectories(volume_df, region="hippocampus", subjects_to_plot=5, individual_plots=True)
    
    for region in regions:
        if region != "hippocampus":
            plot_volume_trajectories(volume_df, region=region, subjects_to_plot=5, individual_plots=False)
    
    for i, file in enumerate(tqdm(all_noder_files)):
        if not file.endswith('.nii.gz'):
            continue
        
        image_id = re.search(r'S\d+_I\d+', file)
        if not image_id:
            continue
        image_id = image_id.group(0)
        try:
            noder_mask_path = os.path.join(noder_mask_dir, f"{image_id}_mask.nii.gz")
            synbadd_mask_path = os.path.join(synbadd_mask_dir, f"{image_id}_mask.nii.gz")
            
            if not os.path.exists(noder_mask_path) or not os.path.exists(synbadd_mask_path):
                continue
                
            mask_files = [f for f in all_mask_files if image_id in f]
            seg_file = None
            for k in mask_files:
                if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in k:
                    seg_file = k
                    break
            
            if not seg_file:
                continue
            
            mask_nib = nib.load(os.path.join(mask_dir, seg_file))
            mask_data = mask_nib.get_fdata()
            
            noder_mask_nib = nib.load(noder_mask_path)
            noder_mask_data = noder_mask_nib.get_fdata()
            
            synbadd_mask_nib = nib.load(synbadd_mask_path)
            synbadd_mask_data = synbadd_mask_nib.get_fdata()
            
            for region_name, region_ids in regions.items():
                target_region = extract_region(mask_data, region_ids)
                noder_region = extract_region(noder_mask_data, region_ids)
                synbadd_region = extract_region(synbadd_mask_data, region_ids)
                
                noder_dice = dice_score(noder_region, target_region)
                synbadd_dice = dice_score(synbadd_region, target_region)
                
                noder_dice_scores[region_name].append(noder_dice)
                synbadd_dice_scores[region_name].append(synbadd_dice)
                
                wandb.log({
                    f"dice_{region_name}_noder": noder_dice,
                    f"dice_{region_name}_synbadd": synbadd_dice,
                    f"dice_diff_{region_name}": synbadd_dice - noder_dice
                })
                
        except Exception as e:
            print(f"Error processing dice scores for {image_id}: {str(e)}")
            continue
    
    # Calculate and log average SSIM/PSNR metrics
    avg_noder_ssim = volume_df['noder_ssim'].mean()
    avg_synbadd_ssim = volume_df['synbadd_ssim'].mean()
    avg_noder_psnr = volume_df['noder_psnr'].mean()
    avg_synbadd_psnr = volume_df['synbadd_psnr'].mean()
    
    wandb.log({
        "avg_ssim_noder": avg_noder_ssim,
        "avg_ssim_synbadd": avg_synbadd_ssim,
        "avg_ssim_improvement": avg_synbadd_ssim - avg_noder_ssim,
        "avg_psnr_noder": avg_noder_psnr,
        "avg_psnr_synbadd": avg_synbadd_psnr,
        "avg_psnr_improvement": avg_synbadd_psnr - avg_noder_psnr,
    })
    
    volume_summary = []
    for region in regions:
        avg_noder_diff = volume_df[f"noder_{region}_volume_diff"].abs().mean()
        avg_noder_diff_pct = volume_df[f"noder_{region}_volume_diff_percent"].abs().mean()
        
        avg_synbadd_diff = volume_df[f"synbadd_{region}_volume_diff"].abs().mean()
        avg_synbadd_diff_pct = volume_df[f"synbadd_{region}_volume_diff_percent"].abs().mean()
        
        volume_summary.append({
            'Region': region.capitalize(),
            'NODER_Diff': avg_noder_diff,
            'NODER_Diff_Percent': avg_noder_diff_pct,
            'SynBADD_Diff': avg_synbadd_diff,
            'SynBADD_Diff_Percent': avg_synbadd_diff_pct
        })
    
    volume_summary = pd.DataFrame(volume_summary)
    volume_summary.to_csv(f'{save_path}/volume_diff_summary_fold_{fold_index}.csv', index=False)
    wandb.log({"volume_diff_summary": wandb.Table(dataframe=volume_summary)})
    
    plt.figure(figsize=(14, 8))
    index = np.arange(len(regions))
    bar_width = 0.35
    
    plt.bar(index, volume_summary['NODER_Diff_Percent'], bar_width, label='NODER', color='blue', alpha=0.7)
    plt.bar(index + bar_width, volume_summary['SynBADD_Diff_Percent'], bar_width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Volume Difference (%)')
    plt.title('Comparison of NODER vs SynBADD Volume Differences by Brain Region')
    plt.xticks(index + bar_width/2, volume_summary['Region'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/volume_diff_comparison_fold_{fold_index}.png')
    wandb.log({"volume_diff_comparison": wandb.Image(f'{save_path}/volume_diff_comparison_fold_{fold_index}.png')})
    plt.close()
    
    dice_summary = []
    for region in regions:
        avg_noder_dice = np.mean(noder_dice_scores[region])
        avg_synbadd_dice = np.mean(synbadd_dice_scores[region])
        
        dice_summary.append({
            'Region': region.capitalize(),
            'NODER_Dice': avg_noder_dice,
            'SynBADD_Dice': avg_synbadd_dice,
            'Dice_Diff': avg_synbadd_dice - avg_noder_dice
        })
    
    dice_summary = pd.DataFrame(dice_summary)
    dice_summary.to_csv(f'{save_path}/dice_summary_fold_{fold_index}.csv', index=False)
    wandb.log({"dice_summary": wandb.Table(dataframe=dice_summary)})
    
    plt.figure(figsize=(14, 8))
    plt.bar(index, dice_summary['NODER_Dice'], bar_width, label='NODER', color='blue', alpha=0.7)
    plt.bar(index + bar_width, dice_summary['SynBADD_Dice'], bar_width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Dice Score')
    plt.title('Comparison of NODER vs SynBADD Dice Scores by Brain Region')
    plt.xticks(index + bar_width/2, dice_summary['Region'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/dice_comparison_fold_{fold_index}.png')
    wandb.log({"dice_comparison": wandb.Image(f'{save_path}/dice_comparison_fold_{fold_index}.png')})
    plt.close()
    
    # Create SSIM/PSNR comparison plots
    ssim_psnr_summary = pd.DataFrame({
        'Metric': ['SSIM', 'PSNR'],
        'NODER': [avg_noder_ssim, avg_noder_psnr],
        'SynBADD': [avg_synbadd_ssim, avg_synbadd_psnr]
    })
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ssim_psnr_summary))
    width = 0.35
    
    plt.bar(x - width/2, ssim_psnr_summary['NODER'], width, label='NODER', color='blue', alpha=0.7)
    plt.bar(x + width/2, ssim_psnr_summary['SynBADD'], width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('SSIM and PSNR Comparison: NODER vs SynBADD')
    plt.xticks(x, ssim_psnr_summary['Metric'])
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/ssim_psnr_comparison_fold_{fold_index}.png')
    wandb.log({"ssim_psnr_comparison": wandb.Image(f'{save_path}/ssim_psnr_comparison_fold_{fold_index}.png')})
    plt.close()
    
    # Create trajectory alignment visualization
    trajectory_regions = [region for region in regions.keys()]
    noder_alignments = [trajectory_alignment[f'{region}_noder'] for region in trajectory_regions]
    synbadd_alignments = [trajectory_alignment[f'{region}_synbadd'] for region in trajectory_regions]
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(trajectory_regions))
    width = 0.35
    
    plt.bar(x - width/2, noder_alignments, width, label='NODER', color='blue', alpha=0.7)
    plt.bar(x + width/2, synbadd_alignments, width, label='SynBADD', color='orange', alpha=0.7)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Average Trajectory Alignment Error')
    plt.title('Trajectory Alignment Comparison: NODER vs SynBADD\n(Lower is Better)')
    plt.xticks(x, [region.capitalize() for region in trajectory_regions], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{save_path}/trajectory_alignment_comparison_fold_{fold_index}.png')
    wandb.log({"trajectory_alignment_comparison": wandb.Image(f'{save_path}/trajectory_alignment_comparison_fold_{fold_index}.png')})
    plt.close()
    
    wandb.finish()
    
    print("ROI volume analysis complete!")
    print(f"Results saved to {save_path}")
    print(f"Average SSIM - NODER: {avg_noder_ssim:.4f}, SynBADD: {avg_synbadd_ssim:.4f}")
    print(f"Average PSNR - NODER: {avg_noder_psnr:.4f}, SynBADD: {avg_synbadd_psnr:.4f}")
    print(f"Overall Trajectory Alignment - NODER: {trajectory_alignment['overall_noder']:.4f}, SynBADD: {trajectory_alignment['overall_synbadd']:.4f}")

if __name__ == "__main__":
    main()