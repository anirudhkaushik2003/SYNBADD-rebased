# =============================================================================
# File: scripts/generate_SVF.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import ADNIVxmTraining
from voxelmorph.torch.networks import VxmDense
from collections import OrderedDict
import csv


class CustomConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        
        # Preserve important attributes from the original datasets
        self.subjects = []
        for dataset in datasets:
            self.subjects.extend(dataset.subjects)
        
        # You can preserve other attributes as needed
        # self.other_attribute = []
        # for dataset in datasets:
        #     self.other_attribute.extend(dataset.other_attribute)
    
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        dataset_idx = next(i for i, offset in enumerate(self.cumulative_lengths[1:], 1) if idx < offset)
        sample_idx = idx - self.cumulative_lengths[dataset_idx-1]
        return self.datasets[dataset_idx-1][sample_idx]
    

# === Configuration ===
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "/scratch/anirudhkaushik/svf_outputs"
os.makedirs(save_dir, exist_ok=True)

metadata_csv_path = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"
csv_output_path = os.path.join("/scratch/anirudhkaushik/", "svf_metadata.csv")

# Load metadata dataframe for lookup
metadata_df = pd.read_csv(metadata_csv_path)

timepoints = ['sc', 'm06', 'm12', 'm18', 'm24', 'm36']
# === Load dataset & model ===
dataset_train = ADNIVxmTraining(metadata_csv_path, subset='train', fold=1, aug_p=0.0, timepoints=timepoints, target_timepoint="m18", use_val_dataset=False)
dataset_test = ADNIVxmTraining(metadata_csv_path, subset='test', fold=1, aug_p=0.0, timepoints=timepoints, target_timepoint="m18", use_val_dataset=False)

# combine train and test datasets
dataset = CustomConcatDataset([dataset_train, dataset_test])

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

vxm_path = "/scratch/anirudhkaushik/vxm_checkpoints/best_vxm_model_128.pth"
vxm_model = VxmDense(inshape=(128, 128, 128), nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32]], int_downsize=1)
vxm_model.to(device)
state_dict = torch.load(vxm_path)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    new_state_dict[new_key] = v
vxm_model.load_state_dict(new_state_dict)
vxm_model.eval()
for param in vxm_model.parameters():
    param.requires_grad = False

# === Timepoint list for filename construction ===

# === Prepare new metadata DataFrame ===
svf_metadata_rows = []

# === Start SVF generation ===
with torch.no_grad():
    for i, batch in enumerate(tqdm(dataloader, desc="Generating SVFs")):
        latest = batch["latest_scan"].to(device)
        stacked = batch["stacked_scan"].to(device)
        masks = batch["mask"]
        subject_id = dataset.subjects[i]

        T = stacked.shape[1]
        for t in range(T):
            if t == 0 or masks[0, t].item() == 0:
                continue  # skip sc→sc and missing scans

            target_timepoint = timepoints[t]
            target = stacked[0, t].unsqueeze(0)
            _, svf = vxm_model(latest, target)

            svf_filename = f"{subject_id}_sc_{target_timepoint}.pt"
            svf_path = os.path.join(save_dir, svf_filename)
            torch.save(svf.squeeze(0).cpu(), svf_path)

            # === Add matching metadata row from target scan ===
            matched_row = metadata_df[
                (metadata_df["Subject"] == subject_id) & 
                (metadata_df["Visit"] == target_timepoint)
            ]

            if not matched_row.empty:
                row = matched_row.iloc[0].to_dict()
                row["SVF_Path"] = svf_path
                svf_metadata_rows.append(row)
            else:
                print(f"Warning: No metadata found for {subject_id} at {target_timepoint}")

# === Save final SVF metadata DataFrame ===
svf_metadata_df = pd.DataFrame(svf_metadata_rows)
svf_metadata_df.to_csv(csv_output_path, index=False)
print(f"Saved SVF metadata to {csv_output_path}")