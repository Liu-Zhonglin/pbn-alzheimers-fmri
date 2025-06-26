import os
import glob
import shutil
import pandas as pd


# ==============================================================================
# --- Data Preparation Script for SAILoR Pipeline ---
# ==============================================================================
# This script reads the final preprocessed data from the 'PBN_Ready_Data'
# and 'denoised' directories, organizes it by cohort (AD, MCI, Normal),
# renames the files to a simple format, and copies them into a clean
# 'SAILoR_Input' directory, ready for the main analysis pipeline.

def main():
    """
    Main function to organize and prepare the data.
    """
    print("--- Starting Data Preparation for SAILoR ---")

    # --- 1. Define Source and Destination Directories ---
    # Source directory for the binarised data
    binarised_source_dir = "/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/PBN_Ready_Data"

    # Source directory for the continuous, denoised time series data
    timeseries_source_dir = "/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/fMRI Preprocessing for PBN Analysis/denoised"

    # The clean, final destination directory for the SAILoR pipeline
    destination_dir = "/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/SAILoR_Input"

    print(f"Binarised data source: {binarised_source_dir}")
    print(f"Time series data source: {timeseries_source_dir}")
    print(f"Destination: {destination_dir}")

    # List of cohorts to process
    cohorts = ["AD", "MCI", "Normal"]

    # --- 2. Create Destination Directories ---
    # Ensure the main destination and cohort subdirectories exist
    for cohort in cohorts:
        os.makedirs(os.path.join(destination_dir, cohort), exist_ok=True)

    # --- 3. Process Each Cohort ---
    total_subjects_processed = 0
    for cohort in cohorts:
        print(f"\n--- Processing Cohort: {cohort} ---")

        # Define the specific source directories for this cohort
        binarised_cohort_path = os.path.join(binarised_source_dir, cohort)
        timeseries_cohort_path = os.path.join(timeseries_source_dir, cohort)

        # Find all the binarised data files for the current cohort
        search_pattern = os.path.join(binarised_cohort_path, "*_PBN_ready.csv")
        binarised_files = glob.glob(search_pattern)

        if not binarised_files:
            print(f"No binarised files found for cohort {cohort}. Skipping.")
            continue

        print(f"Found {len(binarised_files)} subjects in this cohort.")

        # --- 4. Loop Through Each Subject's Files ---
        for binarised_file_path in binarised_files:

            # Extract the base filename to identify the subject
            base_filename = os.path.basename(binarised_file_path)

            # The subject ID is the part of the filename before the first underscore
            # e.g., "sub-007S4272" from "sub-007S4272_task-rest..."
            subject_id = base_filename.split('_')[0]

            # --- Construct the corresponding time series file path ---
            # The original time series filename has a different suffix
            timeseries_filename = base_filename.replace("_PBN_ready.csv", "_denoised.csv")
            timeseries_file_path = os.path.join(timeseries_cohort_path, timeseries_filename)

            # --- Define the new, clean destination file paths ---
            dest_cohort_dir = os.path.join(destination_dir, cohort)
            new_binarised_path = os.path.join(dest_cohort_dir, f"{subject_id}_binarised.csv")
            new_timeseries_path = os.path.join(dest_cohort_dir, f"{subject_id}_timeseries.csv")

            # --- 5. Verify, Copy, and Rename ---
            # Check if both source files actually exist before copying
            if os.path.exists(binarised_file_path) and os.path.exists(timeseries_file_path):
                print(f"  -> Preparing files for subject: {subject_id}")

                # Copy the binarised data file
                shutil.copy(binarised_file_path, new_binarised_path)

                # Copy the denoised time series file
                shutil.copy(timeseries_file_path, new_timeseries_path)

                total_subjects_processed += 1
            else:
                print(f"  -> WARNING: Missing a file for subject {subject_id}. Skipping.")
                if not os.path.exists(timeseries_file_path):
                    print(f"     - Missing: {timeseries_file_path}")

    print("\n" + "=" * 50)
    print("--- Data Preparation Finished ---")
    print(f"Successfully organized data for {total_subjects_processed} subjects.")
    print(f"Your data is now ready in: {destination_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()