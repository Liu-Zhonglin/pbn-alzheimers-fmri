# run_all_pbn_inference.py
import os
import glob
import subprocess


def main():
    """
    Finds all subjects across all cohorts and runs the PBN inference engine on them
    using the user-configured robust parameters.
    """
    print("--- Starting ROBUST Batch PBN Inference for All Subjects ---")
    print("Using configured parameters: max_inputs=6, TOP_N_REGULATORS=12, ntrees=1500")

    # --- 1. Define Base Directories ---
    base_pipeline_dir = "/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline"
    base_input_path = os.path.join(base_pipeline_dir, "SAILoR_Input")
    base_output_path = os.path.join(base_pipeline_dir, "SAILoR_Output")

    # This is the engine script we will call for each subject
    pbn_engine_script = os.path.join(os.path.dirname(__file__), "pbn_engine.py")

    # --- 2. Find all binarised subject files to process ---
    # The glob pattern `**` will search recursively through AD, MCI, and Normal folders.
    search_pattern = os.path.join(base_input_path, "**", "*_binarised.csv")
    subjects_to_process = glob.glob(search_pattern, recursive=True)

    if not subjects_to_process:
        print("\nERROR: No '_binarised.csv' files found in the SAILoR_Input directory.")
        return

    print(f"\nFound {len(subjects_to_process)} subjects to process.")

    # --- 3. Loop through each subject and run the PBN engine ---
    for i, bin_path in enumerate(subjects_to_process):
        subject_id = os.path.basename(bin_path).replace("_binarised.csv", "")
        cohort = os.path.basename(os.path.dirname(bin_path))

        print("\n" + "=" * 80)
        print(f"Processing Subject {i + 1}/{len(subjects_to_process)}: {subject_id} (Cohort: {cohort})")
        print("=" * 80)

        # Define the final output path for the PBN model
        output_json_path = os.path.join(base_output_path, cohort, f"{subject_id}_pbn_model.json")

        # This is a crucial feature: if the output already exists, skip it.
        # This allows you to stop and resume the batch job without re-doing work.
        if os.path.exists(output_json_path):
            print(f"Output PBN file already exists. Skipping: {output_json_path}")
            continue

        # Construct the command to run the engine for this subject
        command = [
            "python",
            pbn_engine_script,
            "--binarisedPath", bin_path,
            "--outputJsonPath", output_json_path
        ]

        try:
            # Execute the command. This will take a manageable, but still significant,
            # amount of time for each subject.
            subprocess.run(command, check=True, capture_output=False, text=True, encoding='utf-8')
            print(f"\nSuccessfully created PBN model for {subject_id}.")
            print(f"Output saved to: {output_json_path}")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: PBN inference failed for subject {subject_id}.")
            print("--- STDERR ---")
            print(e.stderr)

    print("\n--- All Batch PBN Inference Finished ---")


if __name__ == "__main__":
    main()