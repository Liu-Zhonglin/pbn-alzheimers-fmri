# run_all_subjects.py (Final, Robust Version with CORRECT Data Harmonization)
import os
import subprocess
import pandas as pd


def prepare_truncated_data(base_dir, groups):
    """
    Scans all subjects, correctly handling wide and long formats, to find the
    minimum time series length. Then, creates a new directory with all data
    files truncated to that length.
    """
    original_input_dir = os.path.join(base_dir, 'SAILoR_Input')
    truncated_input_dir = os.path.join(base_dir, 'SAILoR_Input_Truncated')

    print("=" * 60)
    print("      STEP 1: DATA HARMONIZATION (TRUNCATION)")
    print("=" * 60)

    # --- Part 1: Find the minimum length across all subjects ---
    min_length = float('inf')
    all_files_to_process = []
    print(f"Scanning all subjects in {original_input_dir} to find minimum time dimension...")

    for group in groups:
        group_path = os.path.join(original_input_dir, group)
        if not os.path.isdir(group_path):
            continue
        for filename in os.listdir(group_path):
            if filename.endswith('_timeseries.csv'):
                filepath = os.path.join(group_path, filename)

                # --- MODIFIED LOGIC: Correctly read the number of time points ---
                # For timeseries files (wide format), length is number of columns - 1
                df = pd.read_csv(filepath)
                # The first column is 'ROI', so we subtract it from the count
                current_length = len(df.columns) - 1
                # --- END OF MODIFICATION ---

                if current_length < min_length:
                    min_length = current_length
                all_files_to_process.append((group, filename))

    if min_length == float('inf'):
        print("[ERROR] No time series files found. Aborting.")
        return None

    print(f"--- Minimum time series length found across all subjects: {min_length} ---")

    # --- Part 2: Create new truncated files ---
    print(f"Creating harmonized dataset in: {truncated_input_dir}")
    for group, filename in all_files_to_process:
        os.makedirs(os.path.join(truncated_input_dir, group), exist_ok=True)

        # --- MODIFIED LOGIC: Correctly truncate both file types ---

        # 1. Process the WIDE timeseries file by truncating COLUMNS
        original_ts_path = os.path.join(original_input_dir, group, filename)
        truncated_ts_path = os.path.join(truncated_input_dir, group, filename)
        df_ts = pd.read_csv(original_ts_path)
        # Keep the 'ROI' column plus the first 'min_length' time point columns
        columns_to_keep = ['ROI'] + list(df_ts.columns[1:min_length + 1])
        df_ts_truncated = df_ts[columns_to_keep]
        df_ts_truncated.to_csv(truncated_ts_path, index=False)

        # 2. Process the LONG binarised file by truncating ROWS
        binarised_filename = filename.replace('_timeseries.csv', '_binarised.csv')
        original_bin_path = os.path.join(original_input_dir, group, binarised_filename)
        truncated_bin_path = os.path.join(truncated_input_dir, group, binarised_filename)
        if os.path.exists(original_bin_path):
            df_bin = pd.read_csv(original_bin_path, header=None)
            # .head() correctly truncates rows for the long format
            df_bin_truncated = df_bin.head(min_length)
            df_bin_truncated.to_csv(truncated_bin_path, index=False, header=False)
        # --- END OF MODIFICATION ---

    print("--- Data harmonization complete. ---")
    return truncated_input_dir


def run_pbn_inference(base_dir, groups, harmonized_input_dir):
    """
    Runs the PBN engine on the harmonized (truncated) data.
    (This function does not need any changes)
    """
    output_dir_parent = os.path.join(base_dir, 'SAILoR_Output')
    engine_script_path = os.path.join(base_dir, 'SAILoR_Inference/hybrid_pbn_engine.py')

    print("\n" + "=" * 60)
    print("      STEP 2: PBN INFERENCE")
    print("=" * 60)

    for group in groups:
        print(f"\n--- Processing Group: {group} ---")

        input_dir = os.path.join(harmonized_input_dir, group)
        output_dir = os.path.join(output_dir_parent, group)

        if not os.path.isdir(input_dir):
            print(f"[WARNING] Input directory for group '{group}' not found. Skipping.")
            continue

        all_files = os.listdir(input_dir)
        subject_ids = sorted([f.replace('_timeseries.csv', '') for f in all_files if f.endswith('_timeseries.csv')])

        if not subject_ids:
            print(f"[INFO] No subjects found in group '{group}'. Skipping.")
            continue

        print(f"Found {len(subject_ids)} subjects in group '{group}'.")

        for i, subject_id in enumerate(subject_ids):
            print("\n" + "-" * 50)
            print(f"Processing Subject {i + 1}/{len(subject_ids)} in Group '{group}': {subject_id}")
            print("-" * 50)

            timeseries_path = os.path.join(input_dir, f'{subject_id}_timeseries.csv')
            binarised_path = os.path.join(input_dir, f'{subject_id}_binarised.csv')
            output_json_path = os.path.join(output_dir, f'{subject_id}_hybrid_pbn_model.json')

            command = [
                'python', engine_script_path,
                '--timeSeriesPath', timeseries_path,
                '--binarisedPath', binarised_path,
                '--outputJsonPath', output_json_path
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            for line in iter(process.stdout.readline, ''):
                print(f"  [Engine] {line.strip()}")

            process.wait()

            if process.returncode == 0:
                print(f"  [SUCCESS] Finished processing {subject_id}.")
            else:
                print(
                    f"  [FAILURE] An error occurred while processing {subject_id}. (Return code: {process.returncode})")


def main():
    BASE_DIR = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline'
    GROUPS = ['AD', 'MCI', 'Normal']

    # Step 1: Create the harmonized dataset
    harmonized_input_dir = prepare_truncated_data(BASE_DIR, GROUPS)

    # Step 2: Run the inference on the new, clean data
    if harmonized_input_dir:
        run_pbn_inference(BASE_DIR, GROUPS, harmonized_input_dir)
        print("\n" + "=" * 60)
        print("          ALL BATCH PROCESSING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()