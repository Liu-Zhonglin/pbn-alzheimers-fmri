# pbn_engine.py
import argparse
import itertools
import json
import os
from operator import itemgetter

import pandas as pd
import numpy as np

# Import our newly created utility functions and classes
from SAILoR_utils import RegulatoryWeights, BinarisedData, infer_boolean_function, calculate_cod_for_rule


def find_top_predictors_for_node(target_name, target_idx,
                                 reg_weights, binarised_df, max_inputs=6, num_predictors=4):
    """
    Finds the top N predictor functions for a single target node.
    This is the core of the PBN inference.
    """
    candidate_predictors = []

    # Use dynGENIE3 results to get a ranked list of the most important regulators
    # We take a larger pool to generate combinations from (e.g., top 10)
    potential_regulator_indices = reg_weights.getRegulators(target_name)
    potential_regulator_names = [f'x{i}' for i in potential_regulator_indices]

    print(f"    - Analyzing {target_name} with {len(potential_regulator_names)} potential regulators...")

    # 1. Iterate through different numbers of inputs
    for k in range(1, min(max_inputs, len(potential_regulator_names)) + 1):
        # 2. Iterate through all combinations of these potential regulators
        for regulator_subset_names in itertools.combinations(potential_regulator_names, k):
            regulator_subset_names = list(regulator_subset_names)

            # 3. For each combination, infer the single best Boolean function
            # This uses the logic we extracted from the original SAILoR script
            function_str = infer_boolean_function(regulator_subset_names, target_name, binarised_df)

            # 4. Score that function using our new COD calculator
            cod = calculate_cod_for_rule(function_str, regulator_subset_names, target_name, binarised_df)

            candidate_predictors.append({
                "function": function_str,
                "cod": cod,
                "inputs": regulator_subset_names
            })

    # 5. Rank all the candidate functions by their COD score
    sorted_predictors = sorted(candidate_predictors, key=itemgetter('cod'), reverse=True)

    # 6. Select the top N overall predictors
    top_predictors = sorted_predictors[:num_predictors]

    # 7. Calculate probabilities based on their relative COD
    total_cod = sum(p['cod'] for p in top_predictors if p['cod'] > 0)
    if total_cod > 0:
        for p in top_predictors:
            # Ensure probability is non-negative
            p['probability'] = max(0, p['cod']) / total_cod
    else:
        # Handle cases where all CODs are zero or negative
        prob = 1.0 / len(top_predictors) if top_predictors else 0
        for p in top_predictors:
            p['probability'] = prob

    return top_predictors


def main():
    parser = argparse.ArgumentParser(description="PBN Inference Engine")
    parser.add_argument('--binarisedPath', required=True)
    parser.add_argument('--outputJsonPath', required=True)
    args = parser.parse_args()

    print(f"--- Starting PBN Inference for {os.path.basename(args.binarisedPath)} ---")

    # Load data and initialize regulatory weights using our utility classes
    bin_data = BinarisedData(args.binarisedPath)

    # This is the part that runs dynGENIE3 to get regulator importances
    from dynGENIE3 import dynGENIE3
    TS_data = [bin_data.df.values]
    time_points = [np.arange(bin_data.df.shape[0])]
    method_args = {'regulators': 'all', 'tree_method': 'RF', 'K': 'sqrt', 'ntrees': 1000}
    regulatory_probs, _, _, _, _ = dynGENIE3(TS_data, time_points, **method_args)
    reg_weights = RegulatoryWeights(regulatory_probs, bin_data.getGeneNames())

    # --- Main PBN Inference Loop ---
    pbn_model = {"nodes": {}}
    for i, gene_name in enumerate(bin_data.getGeneNames()):
        top_4_predictors = find_top_predictors_for_node(gene_name, i, reg_weights, bin_data.df)
        pbn_model["nodes"][gene_name] = {"functions": top_4_predictors}

    # Save the complete PBN structure to a JSON file
    os.makedirs(os.path.dirname(args.outputJsonPath), exist_ok=True)
    with open(args.outputJsonPath, 'w') as f:
        json.dump(pbn_model, f, indent=4)

    print(f"\n--- PBN Inference complete. Output saved to: {args.outputJsonPath} ---")


if __name__ == "__main__":
    main()