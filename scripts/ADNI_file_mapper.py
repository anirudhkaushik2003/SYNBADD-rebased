# =============================================================================
# File: scripts/ADNI_file_mapper.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import pandas as pd

# Define paths
CSV_PATH = "/scratch/anirudhkaushik/ADNI1_Complete_3Yr_1.5T_2_17_2025.csv"
PREPROCESSED_DIR = "/scratch/anirudhkaushik/ADNI_Resized_Registered/"
OUTPUT_CSV_PATH = "/scratch/anirudhkaushik/ADNI1_Complete_Updated_Registered.csv"

# Load the CSV file
df = pd.read_csv(CSV_PATH)

# Dictionary to store image ID to file path mapping
image_id_to_path = {}

# Extract image IDs from the CSV
image_ids = df["Image Data ID"].astype(str).tolist()

# Walk through the directory and match filenames with image IDs
for root, _, files in os.walk(PREPROCESSED_DIR):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            for image_id in image_ids:
                if image_id in file:
                    image_id_to_path[image_id] = os.path.join(root, file)

# Map file paths to the CSV
df["File_Path"] = df["Image Data ID"].astype(str).map(image_id_to_path)

# drop rows with missing file paths
df = df.dropna(subset=["File_Path"])

# print number of rows in each label of Visit, ex number of rows in Visit = sc, Visit = m12, etc
print(df["Visit"].value_counts())

# Save the updated CSV
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Updated CSV with file paths saved to {OUTPUT_CSV_PATH}")