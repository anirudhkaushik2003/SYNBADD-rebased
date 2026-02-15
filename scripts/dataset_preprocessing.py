# =============================================================================
# File: scripts/dataset_preprocessing.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file
def process_csv(input_file, output_file):
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Extract subject IDs
    subjects = df['Subject'].unique()
    
    # Create a dictionary to map subjects to fold assignments
    subject_fold_mapping = {}
    
    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # For each fold, assign subjects to train/val/test sets
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        # Split the training set further into train and validation
        train_subjects = subjects[train_idx]
        np.random.shuffle(train_subjects)
        
        # Use 80% of training data for training and 20% for validation
        split_point = int(len(train_subjects) * 0.9)
        actual_train_subjects = train_subjects[:split_point]
        val_subjects = train_subjects[split_point:]
        test_subjects = subjects[test_idx]
        
        # Update the mapping dictionary
        for subject in actual_train_subjects:
            if subject not in subject_fold_mapping:
                subject_fold_mapping[subject] = {}
            subject_fold_mapping[subject][f'fold_{fold+1}'] = 'train'
            
        for subject in val_subjects:
            if subject not in subject_fold_mapping:
                subject_fold_mapping[subject] = {}
            subject_fold_mapping[subject][f'fold_{fold+1}'] = 'val'
            
        for subject in test_subjects:
            if subject not in subject_fold_mapping:
                subject_fold_mapping[subject] = {}
            subject_fold_mapping[subject][f'fold_{fold+1}'] = 'test'
    
    # Create new columns for each fold
    for fold in range(1, 6):
        df[f'fold_{fold}'] = df['Subject'].map(lambda x: subject_fold_mapping[x].get(f'fold_{fold}', None))
    
    # Save to a new CSV
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "/scratch/anirudhkaushik/ADNI1_Complete_Updated_Registered.csv"  # Replace with your input file path
    output_file = "/scratch/anirudhkaushik/ADNI1_Complete_DataSplits.csv"  # Replace with your output file path
    
    process_csv(input_file, output_file)