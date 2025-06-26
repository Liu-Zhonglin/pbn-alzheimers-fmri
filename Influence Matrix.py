import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def parse_boolean_expression(expression_str):
    """
    Parses a boolean expression string from the PBN model.
    Replaces logical operators with Python's evaluatable operators.

    Args:
        expression_str (str): The boolean expression (e.g., "x_1 & ~x_2").

    Returns:
        str: A Python-evaluable string.
    """
    # Replace logic operators with Python's bitwise operators
    return expression_str.replace(' & ', ' and ').replace(' | ', ' or ').replace('~', ' not ')


def calculate_predictor_influence(func_str, inputs):
    """
    Calculates the influence of each input node on a single predictor function.
    Implements Equation (5) from the paper.

    Args:
        func_str (str): The string representation of the boolean function.
        inputs (list): A list of input node names (e.g., ['x_1', 'x_2']).

    Returns:
        dict: A dictionary mapping each input node to its influence value.
    """
    num_inputs = len(inputs)
    influence = {inp: 0 for inp in inputs}
    total_states = 2 ** num_inputs

    # Iterate through all 2^k possible input states
    for i in range(total_states):
        state = {}
        binary_representation = bin(i)[2:].zfill(num_inputs)
        for j, inp in enumerate(inputs):
            state[inp] = int(binary_representation[j])

        try:
            base_result = eval(func_str, {}, state)
        except Exception as e:
            print(
                f"  [Warning] Error evaluating base expression: {func_str} with state {state}. Skipping state. Error: {e}")
            continue

        # Toggle each input one by one and check for output change
        for toggled_input in inputs:
            toggled_state = state.copy()
            toggled_state[toggled_input] = 1 - toggled_state[toggled_input]

            try:
                toggled_result = eval(func_str, {}, toggled_state)
            except Exception as e:
                print(
                    f"  [Warning] Error evaluating toggled expression: {func_str} with state {toggled_state}. Skipping state. Error: {e}")
                continue

            # If output flips, the toggled input has influence
            if base_result != toggled_result:
                influence[toggled_input] += 1

    # Normalize influence by the total number of states
    for inp in influence:
        influence[inp] /= total_states

    return influence


def calculate_and_save_influence_matrix(pbn_filepath, output_filepath, num_nodes=18):
    """
    Loads a PBN model, calculates the full influence matrix, and saves it.

    Args:
        pbn_filepath (Path): Path object for the input PBN model JSON file.
        output_filepath (Path): Path object for the output CSV file.
        num_nodes (int): The total number of nodes in the network.
    """
    try:
        with open(pbn_filepath, 'r') as f:
            pbn_model = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [Error] Could not read file {pbn_filepath}: {e}")
        return

    nodes_data = pbn_model.get("nodes", {})
    influence_matrix = np.zeros((num_nodes, num_nodes))
    node_names = [f"x_{i}" for i in range(num_nodes)]

    # Iterate through each target node (rows of the matrix)
    for i in range(num_nodes):
        target_node_name = f"x_{i}"

        if target_node_name not in nodes_data:
            continue  # This node has no predictors, so its row in the matrix remains zero.

        predictors = nodes_data[target_node_name].get("functions", [])
        total_influence_on_target = {}

        # Calculate influence for each predictor and sum them up, weighted by probability
        for predictor in predictors:
            func_str = predictor["function"]
            inputs = predictor["inputs"]
            prob = predictor["probability"]

            parsed_func = parse_boolean_expression(func_str)
            predictor_influence = calculate_predictor_influence(parsed_func, inputs)

            for input_node, influence_val in predictor_influence.items():
                total_influence_on_target.setdefault(input_node, 0.0)
                total_influence_on_target[input_node] += prob * influence_val

        # Populate the i-th row of the influence matrix
        for input_node_name, final_influence in total_influence_on_target.items():
            j = int(input_node_name.split('_')[1])
            influence_matrix[i, j] = final_influence

    # Save the result to CSV
    influence_df = pd.DataFrame(influence_matrix, index=node_names, columns=node_names)
    influence_df.to_csv(output_filepath)


def main():
    """
    Main function to batch process all PBN files in a structured directory.
    """
    # --- User Configuration ---
    # IMPORTANT: Change this to the path of your "SAILoR_Output" folder.
    base_input_dir = Path("./SAILoR_Output")

    # This will be the name of the folder where all results are saved.
    base_output_dir = Path("./Influence_Matrices")
    # --------------------------

    if not base_input_dir.is_dir():
        print(f"Error: The specified input directory does not exist: {base_input_dir}")
        print("Please make sure the script is in the correct location or update the 'base_input_dir' path.")
        return

    groups = ["AD", "MCI", "Normal"]

    print("Starting batch processing of PBN models...")

    for group in groups:
        input_group_path = base_input_dir / group
        output_group_path = base_output_dir / group

        # Create the output directory for the group if it doesn't exist
        output_group_path.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing Group: {group}")

        if not input_group_path.is_dir():
            print(f"  [Warning] Input directory not found for group '{group}'. Skipping.")
            continue

        # Find all PBN model files in the group directory
        pbn_files = list(input_group_path.glob("sub-*_hybrid_pbn_model.json"))

        if not pbn_files:
            print(f"  No PBN model files found in {input_group_path}")
            continue

        for pbn_file in pbn_files:
            # Construct the output filename
            subject_id = pbn_file.name.split('_')[0]
            output_filename = f"{subject_id}_influence_matrix.csv"
            output_filepath = output_group_path / output_filename

            print(f"  - Calculating matrix for: {pbn_file.name}")
            calculate_and_save_influence_matrix(pbn_file, output_filepath)
            print(f"    -> Saved to: {output_filepath}")

    print("\nBatch processing complete.")
    print(f"All influence matrices have been saved in the '{base_output_dir}' directory.")


if __name__ == '__main__':
    main()
