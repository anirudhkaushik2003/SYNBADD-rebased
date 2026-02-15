# =============================================================================
# File: adni_ablations_comparision_analysis.py
# Purpose: Ablation comparison script for multiple SYNBADD model variants on the same fold. Computes ROI Dice and image-quality summaries and logs comparison artifacts.
# Prerequisites: Generated outputs for each ablation model directory from test_model_subjectwise.py for the same fold.
# Usually Run Next: Use produced CSV/plots/W and B metrics to select strongest variant, then include in baseline comparisons.
# Model Components Used: Consumes generated outputs from multiple run directories; no model class training/inference occurs here beyond registration-based analysis.
# =============================================================================

import pandas as pd
import os
import nibabel as nib
import numpy as np
# from pytorch_msssim import ssim, ms_ssim
from skimage.metrics import structural_similarity as ssim
# psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
import re
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import ants
import sys
import wandb  # Added wandb import

fold_index = int(sys.argv[1])
csv_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
df = pd.read_csv(csv_file)
df = df[df[f'fold_{fold_index}'] == 'test'].reset_index(drop=True)

# Create save path directory for visualizations
save_path = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/Results"
os.makedirs(save_path, exist_ok=True)

# Initialize WandB
project_name = f"ADNI_FOLD{fold_index}_ROI_DiceScores_TransformerAblations"
wandb.init(
    project=project_name,
    name=f"SYNBADD_Models_Comparison_Fold{fold_index}",
    config={
        "fold_index": fold_index
    }
)

df['Acq Date'] = pd.to_datetime(df['Acq Date'])
df = df.sort_values(by=['Subject', 'Visit', 'Acq Date']).drop_duplicates(subset=['Subject', 'Visit'], keep='last')
# Grouping subjects
subject_groups = df.groupby('Subject').apply(lambda x: x.sort_values(by='Acq Date'))
subjects = list(subject_groups.index.levels[0])

# calculate ssim and ms_ssim    
def get_metrics(target_img, noder_img):
    target_img_data = target_img
    noder_img_data = noder_img

    # target_img_data = (target_img_data - target_img_data.min())/ (target_img_data.max() - target_img_data.min())
    # noder_img_data = (noder_img_data - noder_img_data.min())/ (noder_img_data.max() - noder_img_data.min())
    # target_img_data = target_img_data.astype(np.float32)
    # noder_img_data = noder_img_data.astype(np.float32)


    ssim_value = ssim(target_img_data, noder_img_data, data_range=1, multichannel=False)
    psnr_value = psnr(target_img_data, noder_img_data, data_range=1)
    return ssim_value, psnr_value

def load_nii(imgPath,padding = (128, 128, 128)):
    img_data = nib.load(imgPath).get_fdata()
    if img_data.ndim == 4:
        print(img_data.shape)
        img_data = img_data.squeeze(0)
    img_data,_ = pad(img_data,shape=padding)
    img_data = preprocess_Normalizition(img_data)

    return img_data

def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices

def preprocess_Normalizition(img,method="min_max"):
    img=img.astype(np.float32)
    if method=="min_max":
        front_img=img[img>0]
        min_value=np.min(front_img)
        max_value=np.max(front_img)
        img=(img-min_value)/(max_value-min_value)
        img[img<0]=0.0
    elif method=="mean":
        front_img=img[img>0]
        mean_value=np.mean(front_img)
        std_value=np.std(front_img)
        img=(img-mean_value)/std_value
    return img


# Define function to calculate Dice score
def dice_score(pred, target):
    intersection = np.sum(pred * target)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)  # Add small epsilon to avoid division by zero

# Function to extract specific region from segmentation mask
def extract_region(seg_data, region_ids):
    """
    Extract binary mask for specific region(s)
    
    Args:
        seg_data: 3D numpy array of segmentation data
        region_ids: List of region IDs to extract
        
    Returns:
        Binary mask where the specified regions are 1, everything else is 0
    """
    mask = np.zeros_like(seg_data)
    for region_id in region_ids:
        mask[seg_data == region_id] = 1
    return mask

# Update your existing code with the following additions:
# synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/SYNBADD_128_6_5e-05_0.0_2025-04-18_00-45-42_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold4/"
# synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/SYNBADD_128_6_5e-05_0.0_2025-04-18_10-33-39_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold5/"
# synbadd_dir = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/SYNBADD_128_6_5e-05_0.0_2025-04-17_06-36-19_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold3/"
# synbadd_dir_full = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/SYNBADD_128_6_5e-05_0.0_2025-04-16_14-29-00_ADNI_all_longitudinal_FinalVAETransformer_Vanilla_Fold2/"
synbadd_dir_main = f"/scratch/anirudhkaushik/SYNBADD_Qualitative_{fold_index}/"
var1_ssim_values = []
var1_psnr_values = []
var2_ssim_values = []
var2_psnr_values = []
var3_ssim_values = []
var3_psnr_values = []


mask_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered_segs/"
all_mask_files = os.listdir(mask_dir)

# Define regions of interest and their corresponding IDs in the DKT atlas
# Updated with all regions from the reference code
regions = {
    'hippocampus': [17, 53],  # Left and right hippocampus
    'parahippocampal': [1016, 2016],  # Left and right parahippocampal gyrus
    'third_ventricle': [14],  # Third ventricle
    'amygdala': [18, 54],  # Left and right amygdala
    'entorhinal_cortex': [1006, 2006],  # Left and right entorhinal cortex
    'fusiform_gyrus': [1007, 2007],  # Left and right fusiform gyrus
    'inferior_temporal_gyrus': [1009, 2009],  # Left and right inferior temporal gyrus
    'middle_temporal_gyrus': [1015, 2015],  # Left and right middle temporal gyrus
}

# Initialize dictionaries to store Dice scores
var1_dice_scores = {region: [] for region in regions}
var2_dice_scores = {region: [] for region in regions}
var3_dice_scores = {region: [] for region in regions}

# Create a list to store per-image metrics for wandb logging
per_image_metrics = []

# all_model_dirs = os.listdir(synbadd_dir_main) # models are in this directory
all_model_dirs = [
"SYNBADD_128_6_5e-05_0.0_2025-05-22_13-44-20_ADNI_all_longitudinal_CrossAttentionTransformer+DAFT_Past+Future_Fold1",
"SYNBADD_128_6_5e-05_0.0_2025-05-23_17-20-56_ADNI_all_longitudinal_CrossAttentionGatedTransformer+DAFT_Past+Future_Fold1",
"SYNBADD_128_6_5e-05_0.0_2025-05-24_02-23-23_ADNI_all_longitudinal_GatedCrossAttentionTransformer+DAFT_Past+Future_Fold1",
]

var1_files = os.listdir(os.path.join(synbadd_dir_main, all_model_dirs[0]))
var2_files = os.listdir(os.path.join(synbadd_dir_main, all_model_dirs[1]))
var3_files = os.listdir(os.path.join(synbadd_dir_main, all_model_dirs[2]))

# Log model names to wandb
model_names = {
    "model1": all_model_dirs[0],
    "model2": all_model_dirs[1],
    "model3": all_model_dirs[2]
}
wandb.config.update(model_names)

total_files_placeholder = [f for f in var1_files if "_synthetic.nii.gz" in f]

var_names = [
    "CrossAttentionTransformer",
"CrossAttentionGatedTransformer",
"GatedCrossAttentionTransformer",
]

for i, file in enumerate(tqdm(total_files_placeholder)):
    if not file.endswith('.nii.gz'):
        continue
    
    image_id = re.search(r'S\d+_I\d+', file)
    if not image_id:
        continue
    image_id = image_id.group(0)
    try:
        # Find corresponding image in df File_Path using contains
        file_path = df[df['File_Path'].str.contains(image_id)]['File_Path'].values[0]
        # Find subject id in df Subject
        subject_id = df[df['File_Path'].str.contains(image_id)]['Subject'].values[0]
        # Find image id by subject and Visit 'sc'
        baseline_image_file_path = df[(df['Subject'] == subject_id) & (df['Visit'] == 'sc')]['File_Path'].values[0]
        baseline_image_id = df[(df['Subject'] == subject_id) & (df['Visit'] == 'sc')]['Image Data ID'].values[0]

        
        # Find corresponding image in synbadd
        var1_file_path = [f for f in var1_files if image_id in f]
        if len(var1_file_path) < 2:
            print(f"Skipping {image_id}: Missing SynBADD files")
            continue
        
        for k in var1_file_path:
            if "_synthetic.nii.gz" in k:
                var1_img_path = k
            elif "_synthetic_field.nii.gz" in k:
                var1_field_path = k


        # Find corresponding image in synbadd
        var2_file_path = [f for f in var2_files if image_id in f]
        if len(var2_file_path) < 2:
            print(f"Skipping {image_id}: Missing SynBADD files")
            continue
        
        for k in var2_file_path:
            if "_synthetic.nii.gz" in k:
                var2_img_path = k
            elif "_synthetic_field.nii.gz" in k:
                var2_field_path = k


        # Find corresponding image in synbadd
        var3_file_path = [f for f in var3_files if image_id in f]
        if len(var3_file_path) < 2:
            print(f"Skipping {image_id}: Missing SynBADD files")
            continue
        
        for k in var3_file_path:
            if "_synthetic.nii.gz" in k:
                var3_img_path = k
            elif "_synthetic_field.nii.gz" in k:
                var3_field_path = k

        # Find corresponding mask
        mask_files = [f for f in all_mask_files if image_id in f]
        seg_file = None
        for k in mask_files:
            if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in k:
                seg_file = k
                break
        
        if not seg_file:
            print(f"Skipping {image_id}: Missing segmentation file")
            continue

        # Find baseline mask
        baseline_mask_files = [f for f in all_mask_files if baseline_image_id in f]
        baseline_seg_file = None
        for k in baseline_mask_files:
            if '_aparc.DKTatlas+aseg.deep_resized_IPL_padded.nii.gz' in k:
                baseline_seg_file = k
                break
        if not baseline_seg_file:
            print(f"Skipping {image_id}: Missing baseline segmentation file")
            continue


        # Load images
        target_img_data = load_nii(file_path)
        baseline_img_data = load_nii(baseline_image_file_path)
        var1_img_data = load_nii(os.path.join(synbadd_dir_main, all_model_dirs[0], var1_img_path))
        var2_img_data = load_nii(os.path.join(synbadd_dir_main, all_model_dirs[1], var2_img_path))
        var3_img_data = load_nii(os.path.join(synbadd_dir_main, all_model_dirs[2], var3_img_path))

        # Load segmentation mask
        mask_nib = nib.load(os.path.join(mask_dir, seg_file))
        mask_data = mask_nib.get_fdata()
        baseline_mask_nib = nib.load(os.path.join(mask_dir, baseline_seg_file))
        baseline_mask_data = baseline_mask_nib.get_fdata()
        
        # Convert numpy arrays to ANTs images for registration
        target_ants = ants.from_numpy(target_img_data, origin=None, spacing=None, direction=None)
        baseline_ants = ants.from_numpy(baseline_img_data, origin=None, spacing=None, direction=None)
        var1_ants = ants.from_numpy(var1_img_data, origin=None, spacing=None, direction=None)
        var2_ants = ants.from_numpy(var2_img_data, origin=None, spacing=None, direction=None)
        var3_ants = ants.from_numpy(var3_img_data, origin=None, spacing=None, direction=None)
        mask_ants = ants.from_numpy(mask_data, origin=None, spacing=None, direction=None)
        
        print(f"Registering {var_names[0]} image {image_id}...")
        var1_reg = ants.registration(
            fixed=var1_ants,
            moving=target_ants,
            type_of_transform='SyN'  # Using SyN for deformable registration
        )
        
        print(f"Registering {var_names[1]} image {image_id}...")
        var2_reg = ants.registration(
            fixed=var2_ants,
            moving=target_ants,
            type_of_transform='SyN'  # Using SyN for deformable registration
        )

        print(f"Registering {var_names[2]} image {image_id}...")
        var3_reg = ants.registration(
            fixed=var3_ants,
            moving=target_ants,
            type_of_transform='SyN'  # Using SyN for deformable registration
        )
        
        # Apply transformations to segmentation mask (with nearest neighbor interpolation)
        print(f"Applying transformations to mask {image_id}...")
        var1_mask_warped = ants.apply_transforms(
            fixed=var1_ants,
            moving=mask_ants,
            transformlist=var1_reg['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        var2_mask_warped = ants.apply_transforms(
            fixed=var2_ants,
            moving=mask_ants,
            transformlist=var2_reg['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        var3_mask_warped = ants.apply_transforms(
            fixed=var3_ants,
            moving=mask_ants,
            transformlist=var3_reg['fwdtransforms'],
            interpolator='nearestNeighbor'
        )

        # Convert warped masks back to numpy arrays
        var1_mask_data = var1_mask_warped.numpy()
        var2_mask_data = var2_mask_warped.numpy()
        var3_mask_data = var3_mask_warped.numpy()

        
        # Calculate Dice scores for each region of interest
        print(f"Calculating Dice scores for {image_id}...")
        for region_name, region_ids in regions.items():
            # Extract region from target mask
            target_region = extract_region(mask_data, region_ids)
            
            var1_region = extract_region(var1_mask_data, region_ids)
            
            var2_region = extract_region(var2_mask_data, region_ids)

            var3_region = extract_region(var3_mask_data, region_ids)
            
            # Calculate Dice scores
            var1_dice = dice_score(var1_region, target_region)
            var2_dice = dice_score(var2_region, target_region)
            var3_dice = dice_score(var3_region, target_region)
            
            # Store Dice scores
            var1_dice_scores[region_name].append(var1_dice)
            var2_dice_scores[region_name].append(var2_dice)
            var3_dice_scores[region_name].append(var3_dice)

            print(f"  {region_name.capitalize()} - {var_names[0]} Dice: {var1_dice:.4f}, {var_names[1]} Dice: {var2_dice:.4f}, {var_names[2]} Dice: {var3_dice:.4f}")
        
        # Calculate image quality metrics
        var1_ssim_value, var1_psnr_value = get_metrics(target_img_data, var1_img_data)
        var2_ssim_value, var2_psnr_value = get_metrics(target_img_data, var2_img_data)
        var3_ssim_value, var3_psnr_value = get_metrics(target_img_data, var3_img_data)

        # Append metrics to lists
        var1_ssim_values.append(var1_ssim_value)
        var1_psnr_values.append(var1_psnr_value)
        var2_ssim_values.append(var2_ssim_value)
        var2_psnr_values.append(var2_psnr_value)
        var3_ssim_values.append(var3_ssim_value)
        var3_psnr_values.append(var3_psnr_value)

        
        print(f"Image ID: {image_id}, {var_names[0]} SSIM: {var1_ssim_value:.4f}, {var_names[0]} PSNR: {var1_psnr_value:.4f}, {var_names[1]} SSIM: {var2_ssim_value:.4f}, {var_names[1]} PSNR: {var2_psnr_value:.4f}, {var_names[2]} SSIM: {var3_ssim_value:.4f}, {var_names[2]} PSNR: {var3_psnr_value:.4f}")
        
    except:
        print(f"Error processing {image_id} file")
        continue
    # Visualization code from your original script can remain here
    # ...

# Create pandas dataframe of per-image metrics for easier analysis
per_image_df = pd.DataFrame(per_image_metrics)

# Save the per-image metrics
per_image_df.to_csv(f'{save_path}/per_image_metrics_fold_{fold_index}.csv', index=False)

# Log the per-image metrics DataFrame as a table in wandb
image_metrics_table = wandb.Table(
    columns=["image_id", "subject"] + 
    [f"psnr_{model}" for model in var_names] + 
    [f"ssim_{model}" for model in var_names]
)

for _, row in per_image_df.iterrows():
    image_metrics_table.add_data(
        row["image_id"], 
        row["subject"],
        *[row[f"psnr_{model}"] for model in var_names],
        *[row[f"ssim_{model}"] for model in var_names]
    )

wandb.log({"image_metrics_table": image_metrics_table})

print(f"Results for fold: {fold_index}")
# print min, max and avg psnr and ssim

print(f"{var_names[0]} Min PSNR: {min(var1_psnr_values)}, Max PSNR: {max(var1_psnr_values)}, Avg PSNR: {np.mean(var1_psnr_values)}")
print(f"{var_names[1]} Min PSNR: {min(var2_psnr_values)}, Max PSNR: {max(var2_psnr_values)}, Avg PSNR: {np.mean(var2_psnr_values)}")
print(f"{var_names[2]} Min PSNR: {min(var3_psnr_values)}, Max PSNR: {max(var3_psnr_values)}, Avg PSNR: {np.mean(var3_psnr_values)}")

print(f"{var_names[0]} Min SSIM: {min(var1_ssim_values)}, Max SSIM: {max(var1_ssim_values)}, Avg SSIM: {np.mean(var1_ssim_values)}")
print(f"{var_names[1]} Min SSIM: {min(var2_ssim_values)}, Max SSIM: {max(var2_ssim_values)}, Avg SSIM: {np.mean(var2_ssim_values)}")
print(f"{var_names[2]} Min SSIM: {min(var3_ssim_values)}, Max SSIM: {max(var3_ssim_values)}, Avg SSIM: {np.mean(var3_ssim_values)}")

# Log summary image quality metrics to wandb
wandb.log({
    f"avg_psnr_{var_names[0]}": np.mean(var1_psnr_values),
    f"avg_ssim_{var_names[0]}": np.mean(var1_ssim_values),
    f"avg_psnr_{var_names[1]}": np.mean(var2_psnr_values),
    f"avg_ssim_{var_names[1]}": np.mean(var2_ssim_values),
    f"avg_psnr_{var_names[2]}": np.mean(var3_psnr_values),
    f"avg_ssim_{var_names[2]}": np.mean(var3_ssim_values)
})

# After processing all images, calculate average Dice scores
print("\nAverage Dice Scores:")

# Create a DataFrame to store all results for easier comparison
results_df = pd.DataFrame(columns=['Region'] + var_names)

for region_name in regions:
    avg_var1_dice = np.mean(var1_dice_scores[region_name])
    avg_var2_dice = np.mean(var2_dice_scores[region_name])
    avg_var3_dice = np.mean(var3_dice_scores[region_name])

    print(f"{region_name.capitalize()}:")
    print(f"  {var_names[0]}: {avg_var1_dice:.4f}")
    print(f"  {var_names[1]}: {avg_var2_dice:.4f}")
    print(f"  {var_names[2]}: {avg_var3_dice:.4f}")
    
    # Add to DataFrame
    results_df = pd.concat([
        results_df,
        pd.DataFrame([{
            'Region': region_name.capitalize(),
            var_names[0]: avg_var1_dice,
            var_names[1]: avg_var2_dice,
            var_names[2]: avg_var3_dice
        }])
    ], ignore_index=True)
    
    # Log region-specific metrics to wandb
    wandb.log({
        f"dice_{region_name}_{var_names[0]}": avg_var1_dice,
        f"dice_{region_name}_{var_names[1]}": avg_var2_dice,
        f"dice_{region_name}_{var_names[2]}": avg_var3_dice
    })

# Save results to CSV
results_df.to_csv(f'{save_path}/dice_score_results_fold_{fold_index}.csv', index=False)

# Log results dataframe as table to wandb
wandb.log({"dice_score_results": wandb.Table(dataframe=results_df)})

# Create bar chart to visualize comparison
plt.figure(figsize=(14, 8))
bar_width = 0.25
index = np.arange(len(regions))

plt.bar(index, results_df[var_names[0]], bar_width, label=var_names[0], color='blue', alpha=0.7)
plt.bar(index + bar_width, results_df[var_names[1]], bar_width, label=var_names[1], color='orange', alpha=0.7)
plt.bar(index + 2*bar_width, results_df[var_names[2]], bar_width, label=var_names[2], color='green', alpha=0.7)

plt.xlabel('Brain Region')
plt.ylabel('Average Dice Score')
plt.title('Comparison of Models Dice Scores by Brain Region')
plt.xticks(index + bar_width, results_df['Region'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
comparison_path = f'{save_path}/dice_score_comparison_fold_{fold_index}.png'
plt.savefig(comparison_path)
plt.close()

# Log comparison chart to wandb
wandb.log({"dice_score_comparison": wandb.Image(comparison_path)})

# Create a plot for PSNR comparison
plt.figure(figsize=(10, 6))
data = [np.mean(var1_psnr_values), np.mean(var2_psnr_values), np.mean(var3_psnr_values)]
plt.bar(var_names, data, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Average PSNR')
plt.title('Average PSNR Comparison Between Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
psnr_path = f'{save_path}/psnr_comparison_fold_{fold_index}.png'
plt.savefig(psnr_path)
plt.close()

# Log PSNR comparison to wandb
wandb.log({"psnr_comparison": wandb.Image(psnr_path)})

# Create a plot for SSIM comparison
plt.figure(figsize=(10, 6))
data = [np.mean(var1_ssim_values), np.mean(var2_ssim_values), np.mean(var3_ssim_values)]
plt.bar(var_names, data, color=['blue', 'orange', 'green'])
plt.xlabel('Model')
plt.ylabel('Average SSIM')
plt.title('Average SSIM Comparison Between Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
ssim_path = f'{save_path}/ssim_comparison_fold_{fold_index}.png'
plt.savefig(ssim_path)
plt.close()

# Log SSIM comparison to wandb
wandb.log({"ssim_comparison": wandb.Image(ssim_path)})

# Create histograms of dice scores for each region and model
for region_name in regions:
    plt.figure(figsize=(12, 6))
    plt.hist(var1_dice_scores[region_name], bins=20, alpha=0.7, label=var_names[0])
    plt.hist(var2_dice_scores[region_name], bins=20, alpha=0.7, label=var_names[1])
    plt.hist(var3_dice_scores[region_name], bins=20, alpha=0.7, label=var_names[2])
    plt.xlabel('Dice Score')
    plt.ylabel('Number of Images')
    plt.title(f'Distribution of Dice Scores for {region_name.capitalize()}')
    plt.legend()
    hist_path = f'{save_path}/dice_score_hist_{region_name}_fold_{fold_index}.png'
    plt.savefig(hist_path)
    plt.close()
    
    # Log histogram to wandb
    wandb.log({f"dice_hist_{region_name}": wandb.Image(hist_path)})

# Create a summary table with key metrics
summary_data = {
    f"PSNR ({var_names[0]})": np.mean(var1_psnr_values),
    f"PSNR ({var_names[1]})": np.mean(var2_psnr_values),
    f"PSNR ({var_names[2]})": np.mean(var3_psnr_values),
    f"SSIM ({var_names[0]})": np.mean(var1_ssim_values),
    f"SSIM ({var_names[1]})": np.mean(var2_ssim_values),
    f"SSIM ({var_names[2]})": np.mean(var3_ssim_values),
}

# Add dice scores for each region to summary
for region_name in regions:
    summary_data[f"Dice {region_name.capitalize()} ({var_names[0]})"] = np.mean(var1_dice_scores[region_name])
    summary_data[f"Dice {region_name.capitalize()} ({var_names[1]})"] = np.mean(var2_dice_scores[region_name])
    summary_data[f"Dice {region_name.capitalize()} ({var_names[2]})"] = np.mean(var3_dice_scores[region_name])

# Convert to DataFrame and save
summary_df = pd.DataFrame([summary_data])
summary_df.to_csv(f'{save_path}/summary_metrics_fold_{fold_index}.csv', index=False)

# Log summary as a table to wandb
wandb.log({"summary_metrics": wandb.Table(dataframe=summary_df)})

# Finish the wandb run
wandb.finish()

print(f"\nResults saved to {save_path}/dice_score_results_fold_{fold_index}.csv")
print(f"Visualizations saved in {save_path}")
print(f"WandB logging completed for project {project_name}")