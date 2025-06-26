# =========================================================================
# DICOM to BIDS Conversion and Finalization Script for ADNI Data
# =========================================================================
# This script loops through a directory of downloaded ADNI subjects,
# uses dcm2bids to convert them, and then finalizes the dataset by
# renaming functional scans and creating the dataset_description.json file.
#
# This version uses command-line arguments for flexibility and includes
# a robust finalization step to ensure correct BIDS naming.
#
# Example Usage from your terminal:
# python run_dcm2bids.py \
#   /path/to/your/raw_data_folder \
#   /path/to/your/bids_output_folder \
#   /path/to/your/adni_config.json
# =========================================================================

import os
import subprocess
import sys
import re
import json
import glob
import argparse


# =========================================================================
# SCRIPT LOGIC - DO NOT MODIFY
# =========================================================================

def run_command(command):
    """Runs a command in the shell and prints real-time output."""
    print(f"\nExecuting: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\nError: Command failed with exit code {process.returncode}")
    except FileNotFoundError:
        print(f"\nError: The command '{command[0]}' was not found.")
        print("Please ensure 'dcm2bids' is installed and in your system's PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn exception occurred: {e}")


def finalize_filenames(bids_dir):
    """
    Ensures functional scans are correctly named with the 'task-rest' entity.
    This is a robust "safety net" in case the dcm2bids config matching fails.
    """
    print("\n--- Finalizing functional filenames to add _task-rest ---")
    # Search pattern for any bold file that does NOT already have "task-rest"
    search_pattern = os.path.join(bids_dir, 'sub-*', 'func', '*_bold.*')
    files_to_rename = glob.glob(search_pattern)

    # Filter out files that already have the task entity
    files_needing_rename = [f for f in files_to_rename if '_task-' not in f]

    if not files_needing_rename:
        print("All functional files are already correctly named. Skipping.")
        return

    renamed_count = 0
    for old_path in files_needing_rename:
        directory, filename = os.path.split(old_path)
        # Construct the new filename by inserting '_task-rest' before '_bold'
        new_filename = filename.replace('_bold', '_task-rest_bold')
        new_path = os.path.join(directory, new_filename)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
            renamed_count += 1
        except OSError as e:
            print(f"Error renaming file {old_path}: {e}")

    print(f"File renaming complete. {renamed_count} files were renamed.")


def create_dataset_description(bids_dir):
    """Creates the dataset_description.json file in the BIDS root directory."""
    print("\n--- Creating dataset_description.json ---")
    json_content = {
        "Name": "ADNI Resting State fMRI",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "Authors": [
            "Zhonglin Liu"
        ],
        "HowToAcknowledge": "Data were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf"
    }

    json_path = os.path.join(bids_dir, 'dataset_description.json')
    try:
        with open(json_path, 'w') as f:
            json.dump(json_content, f, indent=4)
        print(f"Successfully created {json_path}")
    except IOError as e:
        print(f"Error creating file {json_path}: {e}")


def main():
    """Main function to find subjects, run dcm2bids, and finalize the dataset."""

    parser = argparse.ArgumentParser(
        description="ADNI DICOM to BIDS Conversion and Finalization Script.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('source_dir', type=str, help="Full path to the directory with raw ADNI subject folders.")
    parser.add_argument('bids_dir', type=str, help="Full path to the directory where the BIDS dataset will be created.")
    parser.add_argument('config_file', type=str, help="Full path to the dcm2bids configuration JSON file.")

    args = parser.parse_args()

    source_data_dir = args.source_dir
    bids_output_dir = args.bids_dir
    config_file = args.config_file

    print("--- ADNI DICOM to BIDS Conversion ---")

    if not os.path.exists(source_data_dir):
        print(f"Error: Source data directory not found at '{source_data_dir}'")
        sys.exit(1)

    if not os.path.exists(config_file):
        print(f"Error: dcm2bids config file not found at '{config_file}'")
        sys.exit(1)

    os.makedirs(bids_output_dir, exist_ok=True)

    subject_folders = [d for d in os.listdir(source_data_dir) if
                       os.path.isdir(os.path.join(source_data_dir, d)) and not d.startswith('.')]

    if not subject_folders:
        print(f"Error: No subject folders found in '{source_data_dir}'.")
        sys.exit(1)

    print(f"\nFound {len(subject_folders)} potential subjects in '{source_data_dir}'.")

    for subject_folder_name in sorted(subject_folders):
        participant_id = re.sub(r'[^0-9a-zA-Z]+', '', subject_folder_name)

        print(f"\n\n==================================================")
        print(f"Processing Subject: {subject_folder_name} (BIDS ID: {participant_id})")
        print(f"==================================================")

        subject_dicom_dir = os.path.join(source_data_dir, subject_folder_name)

        command = [
            'dcm2bids',
            '-d', subject_dicom_dir,
            '-p', participant_id,
            '-c', config_file,
            '-o', bids_output_dir,
            '--clobber'
        ]
        run_command(command)

    print("\n\n==================================================")
    print("--- Finalizing BIDS Dataset ---")
    print("==================================================")

    # This is the crucial post-processing step
    finalize_filenames(bids_output_dir)
    create_dataset_description(bids_output_dir)

    print("\n\n--- BIDS CONVERSION AND FINALIZATION COMPLETE ---")
    print(f"Your BIDS dataset is ready at: {bids_output_dir}")
    print("You can now run fMRIPrep on this directory.")


if __name__ == '__main__':
    main()