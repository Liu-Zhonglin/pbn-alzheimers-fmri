# hybrid_pbn_engine.py (Final Production Version)
import argparse
import json
import os
import itertools
from operator import itemgetter

import pandas as pd
import numpy as np
from dynGENIE3 import dynGENIE3
from sklearn.preprocessing import minmax_scale
from joblib import Parallel, delayed

# Import our robust boolean function inference utility
from SAILoR_utils import infer_boolean_function

# --- 1. Define Controllable Parameters and Constants ---

REFERENCE_NETWORK_PATHS = [
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_10pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_15pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_20pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_25pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_30pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_35pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_40pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_45pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_50pct.csv',
    '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Reference_Networks/All_Subjects/reference_network_All_Subjects_top_5pct.csv'
]
HYBRID_ALPHA = 0.7
TOP_N_REGULATORS = 15
MAX_INPUTS = 6
NUM_PREDICTORS = 4


# --- 2. Helper Functions ---

def evaluate_boolean_function(func_str, parents, data_row):
    local_scope = {parent: data_row[parent] for parent in parents}
    return eval(func_str, {"__builtins__": {}}, local_scope)


def calculate_cod(func_str, parents, target, binarised_df):
    y_actual = binarised_df[target].values
    y_predicted = np.zeros(len(y_actual) - 1, dtype=int)
    for t in range(len(y_actual) - 1):
        data_row = binarised_df.iloc[t]
        y_predicted[t] = evaluate_boolean_function(func_str, parents, data_row)
    y_actual_shifted = y_actual[1:]
    ss_res = np.sum((y_actual_shifted - y_predicted) ** 2)
    ss_tot = np.sum((y_actual_shifted - np.mean(y_actual_shifted)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def process_single_node(i, roi_names, W_hybrid, binarised_df):
    target_name = roi_names[i]
    print(f"\n[Phase 3/5] Searching for predictors for node: {target_name} ({i + 1}/{len(roi_names)})...")

    influences_on_target = W_hybrid[:, i]
    top_regulator_indices = np.argsort(influences_on_target)[::-1][:TOP_N_REGULATORS]

    all_candidate_functions = []
    for k in range(1, MAX_INPUTS + 1):
        if k > len(top_regulator_indices):
            continue
        for parent_indices_tuple in itertools.combinations(top_regulator_indices, k):
            parent_indices = list(parent_indices_tuple)
            parent_names = [roi_names[p_idx] for p_idx in parent_indices]

            try:
                func_str = infer_boolean_function(parent_names, target_name, binarised_df)

                if not func_str or not isinstance(func_str, str):
                    continue

                cod = calculate_cod(func_str, parent_names, target_name, binarised_df)
                all_candidate_functions.append({
                    "function": func_str,
                    "inputs": parent_names,
                    "cod": cod
                })
            except Exception as e:
                print(
                    f"  [WARNING] Could not infer function for target {target_name} with parents {parent_names}. Error: {e}. Skipping.")
                continue

    print(
        f"[Phase 4/5] Ranking {len(all_candidate_functions)} candidates for {target_name} and selecting top {NUM_PREDICTORS}...")
    all_candidate_functions.sort(key=itemgetter("cod"), reverse=True)
    top_predictors = all_candidate_functions[:NUM_PREDICTORS]

    if top_predictors:
        cod_scores = np.array([p["cod"] for p in top_predictors])
        probabilities = softmax(cod_scores)
        for predictor, prob in zip(top_predictors, probabilities):
            predictor["probability"] = prob

    return target_name, {"functions": top_predictors}


# --- 3. Main PBN Inference Engine ---

def main():
    parser = argparse.ArgumentParser(description="Hybrid-COD PBN Inference Engine")
    parser.add_argument('--timeSeriesPath', required=True)
    parser.add_argument('--binarisedPath', required=True)
    parser.add_argument('--outputJsonPath', required=True)
    args = parser.parse_args()

    subject_id = os.path.basename(args.binarisedPath).replace("_binarised.csv", "")
    print(f"--- Starting Hybrid-COD PBN Inference for Subject: {subject_id} ---")

    # --- Phase 1: Load All Necessary Data ---
    print("\n[Phase 1/5] Loading time-series, binarized, and reference data...")
    timeseries_matrix = pd.read_csv(args.timeSeriesPath, header=0, index_col=0).values.T
    binarised_matrix = pd.read_csv(args.binarisedPath, header=None).values

    num_rois = timeseries_matrix.shape[1]
    roi_names = [f'x_{i}' for i in range(num_rois)]

    timeseries_df = pd.DataFrame(timeseries_matrix, columns=roi_names)
    binarised_df = pd.DataFrame(binarised_matrix, columns=roi_names)

    print(f"Data loaded and relabeled. Shape is (Time Points, ROIs): {timeseries_df.shape}")

    reference_networks = []
    for path in REFERENCE_NETWORK_PATHS:
        raw_matrix = pd.read_csv(path, header=None).values
        if raw_matrix.shape[0] > num_rois:
            sliced_matrix = raw_matrix[1:, 1:]
            reference_networks.append(sliced_matrix)
        else:
            reference_networks.append(raw_matrix)

    # --- Phase 2: Create the Hybrid Influence Matrix ---
    print("\n[Phase 2/5] Creating Hybrid Influence Matrix...")
    (W_dyn, _, _, _, _) = dynGENIE3([timeseries_df.values], [np.arange(len(timeseries_df))])
    W_dyn_normalized = minmax_scale(W_dyn)

    A_con = np.mean(reference_networks, axis=0)

    if W_dyn_normalized.shape != A_con.shape:
        print(
            f"FATAL ERROR: Shape mismatch between data matrix {W_dyn_normalized.shape} and reference matrix {A_con.shape}")
        exit()

    W_hybrid = (HYBRID_ALPHA * W_dyn_normalized) + ((1 - HYBRID_ALPHA) * A_con)
    print(f"Hybrid matrix created with alpha = {HYBRID_ALPHA}")

    # --- Phase 3 & 4 now run in parallel ---
    print("\n--- Starting parallel search for predictors for all nodes ---")

    # --- FINAL VERSION: Set n_jobs to -1 to use all available CPU cores ---
    results = Parallel(n_jobs=-1)(
        delayed(process_single_node)(i, roi_names, W_hybrid, binarised_df) for i in range(len(roi_names))
    )

    final_pbn_model = {"nodes": {node_name: funcs for node_name, funcs in results}}

    # --- Phase 5: Save the Final PBN Model ---
    print("\n[Phase 5/5] Saving the final PBN model...")
    os.makedirs(os.path.dirname(args.outputJsonPath), exist_ok=True)
    with open(args.outputJsonPath, 'w') as f:
        json.dump(final_pbn_model, f, indent=4)

    print(f"\n--- Hybrid-COD PBN Inference Complete! ---")
    print(f"Output saved to: {args.outputJsonPath}")


if __name__ == "__main__":
    main()