#!/bin/bash

# Base directory containing ADNI subjects
BASE_DIR="/scratch/anirudhkaushik/final_ADNI"
OUTPUT_DIR="/scratch/anirudhkaushik/ADNI_preprocessed"
LOG_FILE="processing_log.txt"

# Ensure FreeSurfer is set up
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to check and apply missing preprocessing steps
process_subject() {
    local subject_dir=$1
    local subject_id=$(basename "$subject_dir")
    
    echo "Processing subject: $subject_id"
    
    # Locate the most recent preprocessed image dynamically
    local input_img=""
    best_folder=$(find "$subject_dir" -mindepth 2 -maxdepth 2 -type d -name "*GradWarp*B1_Correction*N3*Scaled*" | sort | tail -n 1)
    
    if [[ -n "$best_folder" ]]; then
        input_img=$(find "$best_folder" -type f -name "*.nii" | sort | tail -n 1)
    fi
    
    if [[ -z "$input_img" ]]; then
        echo "No suitable preprocessed image found for $subject_id. Skipping."
        echo "$subject_id - No valid input image" >> "$LOG_FILE"
        return
    fi

    echo "Using input image: $input_img"
    
    # Extract metadata from the input filename to maintain ADNI naming convention
    local filename=$(basename "$input_img")
    local scan_date=$(echo "$filename" | grep -oP '\d{8}_\d{6}\.\d+')
    local scan_id=$(echo "$filename" | grep -oP 'S\d+')
    local image_id=$(echo "$filename" | grep -oP 'I\d+')
    
    # Generate standardized output filename in the new directory
    local output_file="$OUTPUT_DIR/ADNI_${subject_id}_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_${scan_date}_${scan_id}_${image_id}.nii.gz"
    
    # Apply B1 Correction if missing
    if [[ ! -d "$subject_dir"/*B1_Correction* ]]; then
        echo "Applying B1 Correction..."
        mri_nu_correct.mni --i "$input_img" --o "$output_file"
        input_img="$output_file"
    fi

    # Apply N3 Correction if missing
    if [[ ! -d "$subject_dir"/*N3* ]]; then
        echo "Applying N3 Correction..."
        mri_nu_correct.mni --i "$input_img" --o "$output_file"
        input_img="$output_file"
    fi

    # Apply Scaling if missing
    if [[ ! -d "$subject_dir"/*Scaled* ]]; then
        echo "Applying Scaling..."
        mri_convert --scale 1.0 "$input_img" "$output_file"
        input_img="$output_file"
    fi

    # Apply Skull Stripping if missing
    if [[ ! -d "$subject_dir"/*SkullStripped* ]]; then
        echo "Applying Skull Stripping..."
        mri_watershed "$input_img" "$output_file"
        input_img="$output_file"
    fi

    echo "$subject_id - Processing complete" >> "$LOG_FILE"
}

# Loop through each subject directory and process them
for subject_dir in "$BASE_DIR"/*; do
    if [[ -d "$subject_dir" ]]; then
        process_subject "$subject_dir"
    fi
done

echo "Processing complete. Log saved to $LOG_FILE."
