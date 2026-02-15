# =============================================================================
# File: scripts/subject-level-registration-ADNI.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import os
import pandas as pd
import ants
import multiprocessing
import logging
from tqdm import tqdm

# --- FILE PATHS ---
csv_path = "/scratch/anirudhkaushik/ADNI1_Complete_Updated.csv"
output_dir = "/scratch/anirudhkaushik/ADNI_Resized_Registered/"
log_file = "/scratch/anirudhkaushik/registration_log.log"
error_log_file = "/scratch/anirudhkaushik/registration_errors.log"
updated_csv_path = "/scratch/anirudhkaushik/ADNI1_Complete_Updated_Registered.csv"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# --- SETUP LOGGING ---
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
error_logger = logging.getLogger("error_logger")
error_handler = logging.FileHandler(error_log_file)
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

# --- 1. LOAD CSV FILE ---
df = pd.read_csv(csv_path)

# Convert 'Acq Date' to datetime format for proper sorting
df['Acq Date'] = pd.to_datetime(df['Acq Date'], format='%m/%d/%Y')

# --- 2. GROUP BY SUBJECT & SORT BY DATE ---
df = df.sort_values(by=['Subject', 'Acq Date'])

# Drop duplicate visits (keep latest scan for each visit)
df = df.drop_duplicates(subset=['Subject', 'Visit'], keep='last')

# Dictionary to store updated file paths
updated_paths = {}

# Get unique subjects
subjects = df['Subject'].unique()


# --- FUNCTION TO PROCESS EACH SUBJECT ---
def process_subject(subject):
    """
    Processes all scans for a given subject:
    - Applies N4 bias correction
    - Registers each scan to the SC (Screening) scan using rigid transformation
    - Saves the registered scan (with the same filename in a new directory)
    - Updates the file path in the new CSV
    """
    subject_scans = df[df['Subject'] == subject]

    # Get SC (Screening) scan as the fixed target
    sc_scan = subject_scans[subject_scans['Visit'] == 'sc']
    if sc_scan.empty:
        error_logger.error(f"Skipping {subject} (No SC scan found)")
        return subject, "skipped"

    target_path = sc_scan.iloc[0]['File_Path']
    target_filename = os.path.basename(target_path)  # Keep original filename
    new_target_path = os.path.join(output_dir, target_filename)

    target_image = ants.image_read(target_path)

    # Apply N4 Bias Correction to target image
    target_image = ants.n4_bias_field_correction(target_image)

    # Save the SC scan (if not already in output dir)
    ants.image_write(target_image, new_target_path)
    updated_paths[sc_scan.iloc[0]['Image Data ID']] = new_target_path  # Update CSV

    for _, row in subject_scans.iterrows():
        moving_path = row['File_Path']
        visit = row['Visit']

        # Skip SC scan itself (already saved)
        if visit == 'sc':
            continue

        try:
            # Read and apply N4 Bias Field Correction
            moving_image = ants.image_read(moving_path)
            moving_image = ants.n4_bias_field_correction(moving_image)

            # Perform Rigid Registration
            mytx = ants.registration(fixed=target_image, moving=moving_image, type_of_transform='Rigid')
            warped_image = ants.apply_transforms(fixed=target_image, moving=moving_image, transformlist=mytx['fwdtransforms'])

            # Save registered image (keeping original filename)
            output_filename = os.path.basename(moving_path)
            output_path = os.path.join(output_dir, output_filename)
            ants.image_write(warped_image, output_path)

            # Log successful registration
            logging.info(f"Registered {subject} {visit} -> {output_path}")

            # Update file path in dictionary
            updated_paths[row['Image Data ID']] = output_path

        except Exception as e:
            error_logger.error(f"Error processing {subject} {visit}: {e}")

    return subject, "processed"


# --- 3. RUN MULTIPROCESSING WITH REAL-TIME PROGRESS UPDATES ---
def tqdm_update(result):
    """Updates progress bar when a subject is processed."""
    pbar.update(1)


if __name__ == "__main__":
    num_workers = multiprocessing.cpu_count()  # Use 8 cores or max available

    print(f"Starting parallel processing with {num_workers} workers...\n")

    with tqdm(total=len(subjects), desc="Processing Subjects", position=0, leave=True) as pbar:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for subject in subjects:
                pool.apply_async(process_subject, args=(subject,), callback=tqdm_update)
            pool.close()
            pool.join()

    print("Registration pipeline complete!")

    # --- 4. UPDATE CSV FILE ---
    df['File_Path'] = df['Image Data ID'].map(updated_paths)

    # Save updated CSV file
    df.to_csv(updated_csv_path, index=False)
    print(f"Updated CSV saved to {updated_csv_path}")

    logging.info(f"Updated CSV saved to {updated_csv_path}")
