import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
import glob
import os
import warnings
from collections import defaultdict

# Suppress FutureWarning messages for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

print("--- Group-Specific and Unified Reference Network Generation ---")

# ==============================================================================
# === STEP 1: FIND AND GROUP ALL HARMONIZED TIME SERIES DATA ===================
# ==============================================================================
print("\n--- Step 1: Finding and Grouping Local Harmonized Time Series Data ---")

# This path MUST point to the folder containing the AD, MCI, and Normal subfolders.
base_data_path = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/pbn_data/'

# --- FIX: Use the recursive wildcard '**' to find all .csv files in all subdirectories ---
timeseries_data_pattern = os.path.join(base_data_path, '**', '*.csv')
csv_filenames = sorted(glob.glob(timeseries_data_pattern, recursive=True))

if not csv_filenames:
    raise FileNotFoundError(
        f"CRITICAL ERROR: No .csv files found in '{base_data_path}' or its subdirectories. Please check the path and pattern.")

print(f"Found {len(csv_filenames)} total harmonized time series files to process.")

# ==============================================================================
# === STEP 2: CALCULATE CONNECTIVITY AND ORGANIZE BY GROUP =====================
# ==============================================================================
print("\n--- Step 2: Calculating Connectivity for Each Subject by Group ---")
connectome_measure = ConnectivityMeasure(kind='correlation', standardize="zscore_sample")

# Use a defaultdict to automatically handle the creation of new group lists
grouped_matrices = defaultdict(list)

for i, csv_file in enumerate(csv_filenames):
    subject_id = os.path.basename(csv_file).split('_')[0]

    try:
        # Determine the subject's group from its parent folder name
        group_name = os.path.basename(os.path.dirname(csv_file))
        print(f"  Processing subject {i + 1}/{len(csv_filenames)} ({subject_id}) -> Group: {group_name}")

        # Load the time series data. The previous script added headers, so we use header=0.
        timeseries_data = pd.read_csv(csv_file, header=0).values

        # Calculate the correlation matrix for this subject
        correlation_matrix = connectome_measure.fit_transform([timeseries_data])[0]

        # Add the matrix to the correct group list
        grouped_matrices[group_name].append(correlation_matrix)

    except Exception as e:
        print(f"    -> ERROR: Failed to process subject {os.path.basename(csv_file)}. Reason: {e}")

# Also create a combined list for the unified "All_Subjects" group
all_subject_matrices = [matrix for group_list in grouped_matrices.values() for matrix in group_list]
if all_subject_matrices:
    grouped_matrices['All_Subjects'] = all_subject_matrices

# ==============================================================================
# === STEP 3: COMPUTE AND SAVE GROUP-SPECIFIC & UNIFIED NETWORKS ==============
# ==============================================================================
print("\n--- Step 3: Compute and Save Group-Average Reference Networks ---")

if not grouped_matrices:
    raise ValueError("CRITICAL ERROR: Could not successfully process ANY subjects. Cannot generate reference networks.")

# Define the final parent output directory
output_dir = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks'
os.makedirs(output_dir, exist_ok=True)
print(f"Group-specific and unified reference networks will be saved to: {output_dir}")

# Loop through each group we found (e.g., 'AD', 'MCI', 'Normal', 'All_Subjects')
for group_name, matrices in grouped_matrices.items():

    # Create a specific output folder for this group
    group_output_dir = os.path.join(output_dir, group_name)
    os.makedirs(group_output_dir, exist_ok=True)

    print(f"\nProcessing group: '{group_name}' ({len(matrices)} subjects)")

    # Calculate the mean correlation matrix for this specific group
    mean_correlation_matrix = np.mean(matrices, axis=0)
    np.fill_diagonal(mean_correlation_matrix, 0)
    print(f"  -> Group-average matrix (18x18) computed.")

    # Generate 10 thresholded networks for this group
    threshold_percentiles = np.arange(95, 45, -5)
    for i, p in enumerate(threshold_percentiles):
        threshold_value = np.percentile(np.abs(mean_correlation_matrix), p)
        adjacency_matrix = (np.abs(mean_correlation_matrix) > threshold_value).astype(int)

        output_filename = f'reference_network_{group_name}_top_{100 - p}pct.csv'
        full_output_path = os.path.join(group_output_dir, output_filename)

        df = pd.DataFrame(adjacency_matrix)
        df.to_csv(full_output_path, header=False, index=False)
        print(f"  -> Saved {output_filename}")

print(f"\n\nProcess complete. All reference networks have been saved.")
print(
    "REMINDER: For your PBN inference engine, use the networks from the 'All_Subjects' folder as your unbiased prior knowledge.")