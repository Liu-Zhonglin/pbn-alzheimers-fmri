import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg  # Used for Welch's ANOVA and Games-Howell
import warnings

# --- Configuration ---

# Set the base directory where the 'Influence_Matrices' folder is located.
# The script assumes it's in the current working directory.
BASE_INPUT_DIR = Path("./Influence_Matrices")

# Set the significance level for the False Discovery Rate
ALPHA = 0.05

# --- ROI Mapping ---

# This dictionary maps the node indices from the matrices to their anatomical names.
ROI_MAP = {
    'x_0': 'Angular_L', 'x_1': 'Angular_R',
    'x_2': 'Frontal_Med_Orb_L', 'x_3': 'Frontal_Med_Orb_R',
    'x_4': 'Frontal_Sup_2_L', 'x_5': 'Frontal_Sup_2_R',
    'x_6': 'Hippocampus_L', 'x_7': 'Hippocampus_R',
    'x_8': 'Insula_L', 'x_9': 'Insula_R',
    'x_10': 'ParaHippocampal_L', 'x_11': 'ParaHippocampal_R',
    'x_12': 'Parietal_Sup_L', 'x_13': 'Parietal_Sup_R',
    'x_14': 'Precuneus_L', 'x_15': 'Precuneus_R',
    'x_16': 'Supp_Motor_Area_L', 'x_17': 'Supp_Motor_Area_R'
}


def load_and_structure_data(base_dir):
    """
    Loads all influence matrix CSVs and structures them into a single
    long-format pandas DataFrame.

    Args:
        base_dir (Path): The path to the 'Influence_Matrices' directory.

    Returns:
        pd.DataFrame: A DataFrame containing all data, or None if an error occurs.
    """
    if not base_dir.is_dir():
        print(f"Error: Input directory not found at '{base_dir}'")
        print("Please ensure the directory exists and the path is correct.")
        return None

    all_data = []
    groups = ["AD", "MCI", "Normal"]

    for group in groups:
        group_path = base_dir / group
        if not group_path.is_dir():
            print(f"Warning: Directory for group '{group}' not found. Skipping.")
            continue

        for file_path in group_path.glob("*.csv"):
            subject_id = file_path.stem.split('_')[0]
            try:
                # Load the matrix, setting the first column as the index
                df = pd.read_csv(file_path, index_col=0)
                # Unpivot the matrix from wide to long format
                df_long = df.stack().reset_index()
                df_long.columns = ['Target_ROI', 'Source_ROI', 'Influence_Value']
                df_long['Group'] = group
                df_long['Subject_ID'] = subject_id
                all_data.append(df_long)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    if not all_data:
        print("Error: No data was loaded. Please check the input directory.")
        return None

    # Concatenate all individual dataframes into one master dataframe
    master_df = pd.concat(all_data, ignore_index=True)
    # Filter out self-connections (diagonal elements)
    master_df = master_df[master_df['Source_ROI'] != master_df['Target_ROI']]

    print(f"Successfully loaded data for {master_df['Subject_ID'].nunique()} subjects across {len(groups)} groups.")
    return master_df


def run_group_analysis(df):
    """
    Performs the full statistical analysis pipeline on the aggregated data.

    Args:
        df (pd.DataFrame): The master dataframe from load_and_structure_data.

    Returns:
        pd.DataFrame: A DataFrame containing the final, significant results.
    """
    # Create a unique identifier for each connection
    df['Connection'] = df['Source_ROI'] + ' -> ' + df['Target_ROI']
    connections = df['Connection'].unique()

    primary_results = []
    print(f"\nPerforming primary analysis on {len(connections)} connections...")

    # Suppress warnings from scipy about empty slices, which can happen
    warnings.filterwarnings("ignore", category=UserWarning)

    for conn in connections:
        conn_data = df[df['Connection'] == conn]

        # Create a list of arrays, one for each group's data
        groups_data = [
            conn_data[conn_data['Group'] == 'AD']['Influence_Value'],
            conn_data[conn_data['Group'] == 'MCI']['Influence_Value'],
            conn_data[conn_data['Group'] == 'Normal']['Influence_Value']
        ]

        # --- Step 2: Check for Equal Variances ---
        levene_stat, levene_p = stats.levene(*groups_data)

        # --- Step 3: Perform the Appropriate ANOVA ---
        if levene_p > ALPHA:
            # Variances are equal, use standard ANOVA
            f_stat, p_val = stats.f_oneway(*groups_data)
            test_used = 'ANOVA'
        else:
            # Variances are unequal, use Welch's ANOVA from the pingouin library
            welch_df = conn_data[['Group', 'Influence_Value', 'Subject_ID']]
            try:
                aov = pg.welch_anova(data=welch_df, dv='Influence_Value', between='Group')
                p_val = aov['p-unc'].iloc[0]
                f_stat = aov['F'].iloc[0]
                test_used = "Welch's ANOVA"
            except Exception:  # Handle potential errors with single-value groups etc.
                p_val = np.nan
                f_stat = np.nan
                test_used = "Welch's ANOVA (Error)"

        primary_results.append({
            'Connection': conn,
            'Levene_p': levene_p,
            'ANOVA_p': p_val,
            'F_stat': f_stat,
            'Test_Used': test_used
        })

    warnings.resetwarnings()
    results_df = pd.DataFrame(primary_results).dropna(subset=['ANOVA_p'])

    # --- Step 4: Correction for Multiple Comparisons ---
    print(f"\nApplying FDR correction...")
    if results_df.empty:
        print("No valid p-values to correct. Exiting analysis.")
        return pd.DataFrame()

    is_significant, q_values = fdrcorrection(results_df['ANOVA_p'], alpha=ALPHA)
    results_df['FDR_q_value'] = q_values
    results_df['Significant'] = is_significant

    significant_connections = results_df[results_df['Significant']].copy()
    print(f"Found {len(significant_connections)} significant connections after FDR correction.")

    if len(significant_connections) == 0:
        return pd.DataFrame()  # Return empty dataframe if no significant results

    # --- Step 5: Post-Hoc Analysis ---
    print("\nPerforming post-hoc tests on significant connections...")
    post_hoc_results = []
    for _, row in significant_connections.iterrows():
        conn = row['Connection']
        conn_data = df[df['Connection'] == conn]

        # Use the appropriate post-hoc test based on the Levene's test result
        if row['Levene_p'] > ALPHA:
            # Tukey's HSD for equal variances
            tukey_res = pairwise_tukeyhsd(conn_data['Influence_Value'], conn_data['Group'], alpha=ALPHA)
            ph_df = pd.DataFrame(data=tukey_res._results_table.data[1:], columns=tukey_res._results_table.data[0])
            p_ad_mci = ph_df[(ph_df.group1.isin(['AD', 'MCI'])) & (ph_df.group2.isin(['AD', 'MCI']))]['p-adj'].iloc[0]
            p_ad_normal = \
            ph_df[(ph_df.group1.isin(['AD', 'Normal'])) & (ph_df.group2.isin(['AD', 'Normal']))]['p-adj'].iloc[0]
            p_mci_normal = \
            ph_df[(ph_df.group1.isin(['MCI', 'Normal'])) & (ph_df.group2.isin(['MCI', 'Normal']))]['p-adj'].iloc[0]

        else:
            # Games-Howell for unequal variances, using pingouin
            ph_df = pg.pairwise_gameshowell(data=conn_data, dv='Influence_Value', between='Group')
            p_ad_mci = ph_df[(ph_df.A.isin(['AD', 'MCI'])) & (ph_df.B.isin(['AD', 'MCI']))]['pval'].iloc[0]
            p_ad_normal = ph_df[(ph_df.A.isin(['AD', 'Normal'])) & (ph_df.B.isin(['AD', 'Normal']))]['pval'].iloc[0]
            p_mci_normal = ph_df[(ph_df.A.isin(['MCI', 'Normal'])) & (ph_df.B.isin(['MCI', 'Normal']))]['pval'].iloc[0]

        row['p_AD_vs_MCI'] = p_ad_mci
        row['p_AD_vs_Normal'] = p_ad_normal
        row['p_MCI_vs_Normal'] = p_mci_normal
        post_hoc_results.append(row)

    final_df = pd.DataFrame(post_hoc_results)

    # Map indices to anatomical names for final report
    def map_connection(conn_str):
        source, target = conn_str.split(' -> ')
        return f"{ROI_MAP.get(source, source)} -> {ROI_MAP.get(target, target)}"

    final_df['Connection'] = final_df['Connection'].apply(map_connection)

    # Reorder columns for final presentation
    cols_order = [
        'Connection', 'F_stat', 'ANOVA_p', 'FDR_q_value',
        'p_AD_vs_Normal', 'p_MCI_vs_Normal', 'p_AD_vs_MCI',
        'Test_Used', 'Levene_p'
    ]
    final_df = final_df[cols_order]

    return final_df.sort_values(by='FDR_q_value').reset_index(drop=True)


def main():
    """
    Main execution function.
    """
    print("--- Starting Group Connectivity Analysis ---")

    # Step 1: Load and structure data
    master_df = load_and_structure_data(BASE_INPUT_DIR)

    if master_df is None:
        return  # Stop execution if data loading failed

    # Step 2-5: Run the full analysis pipeline
    final_results_df = run_group_analysis(master_df)

    if not final_results_df.empty:
        # Save results to a CSV file
        output_filename = "group_analysis_significant_results.csv"
        final_results_df.to_csv(output_filename, index=False, float_format='%.4g')
        print(f"\n--- Analysis Complete ---")
        print(f"Significant results saved to '{output_filename}'")
        print("\nFinal Results:")
        print(final_results_df.to_string())
    else:
        print("\n--- Analysis Complete ---")
        print("No statistically significant differences were found between the groups after FDR correction.")


if __name__ == '__main__':
    # Before running, ensure you have the required libraries:
    # pip install pandas numpy scipy statsmodels pingouin
    main()
