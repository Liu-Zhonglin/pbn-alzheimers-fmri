import pandas as pd
import numpy as np
import graphviz
from pathlib import Path
import argparse
import os

# --- ROI and Network Definitions ---
# This mapping provides the anatomical names for the node indices
ROI_MAP = {
    'x_0': 'Angular_L', 'x_1': 'Angular_R', 'x_2': 'Frontal_Med_Orb_L',
    'x_3': 'Frontal_Med_Orb_R', 'x_4': 'Frontal_Sup_2_L', 'x_5': 'Frontal_Sup_2_R',
    'x_6': 'Hippocampus_L', 'x_7': 'Hippocampus_R', 'x_8': 'Insula_L',
    'x_9': 'Insula_R', 'x_10': 'ParaHippocampal_L', 'x_11': 'ParaHippocampal_R',
    'x_12': 'Parietal_Sup_L', 'x_13': 'Parietal_Sup_R', 'x_14': 'Precuneus_L',
    'x_15': 'Precuneus_R', 'x_16': 'Supp_Motor_Area_L', 'x_17': 'Supp_Motor_Area_R'
}

# --- NEW: Abbreviation Mapping for compact visualization ---
ROI_ABBREVIATIONS = {
    'Angular_L': 'ANG_L', 'Angular_R': 'ANG_R',
    'Frontal_Med_Orb_L': 'FMO_L', 'Frontal_Med_Orb_R': 'FMO_R',
    'Frontal_Sup_2_L': 'FS2_L', 'Frontal_Sup_2_R': 'FS2_R',
    'Hippocampus_L': 'HIPP_L', 'Hippocampus_R': 'HIPP_R',
    'Insula_L': 'INS_L', 'Insula_R': 'INS_R',
    'ParaHippocampal_L': 'PHC_L', 'ParaHippocampal_R': 'PHC_R',
    'Parietal_Sup_L': 'PARS_L', 'Parietal_Sup_R': 'PARS_R',
    'Precuneus_L': 'PREC_L', 'Precuneus_R': 'PREC_R',
    'Supp_Motor_Area_L': 'SMA_L', 'Supp_Motor_Area_R': 'SMA_R'
}

# This dictionary maps each ROI to its large-scale brain network
NETWORK_MAP = {
    'Angular_L': 'DMN', 'Angular_R': 'DMN', 'Frontal_Med_Orb_L': 'DMN',
    'Frontal_Med_Orb_R': 'DMN', 'Precuneus_L': 'DMN', 'Precuneus_R': 'DMN',
    'Hippocampus_L': 'MTL', 'Hippocampus_R': 'MTL', 'ParaHippocampal_L': 'MTL',
    'ParaHippocampal_R': 'MTL', 'Parietal_Sup_L': 'ECN', 'Parietal_Sup_R': 'ECN',
    'Frontal_Sup_2_L': 'ECN', 'Frontal_Sup_2_R': 'ECN', 'Insula_L': 'SN',
    'Insula_R': 'SN', 'Supp_Motor_Area_L': 'SN', 'Supp_Motor_Area_R': 'SN'
}

# This dictionary defines the color for each brain network
NETWORK_COLORS = {
    'DMN': '#1f77b4', 'MTL': '#ff7f0e', 'ECN': '#2ca02c', 'SN': '#d62728'
}


def visualize_influence_matrix(csv_path, output_dir, threshold=0.1):
    """
    Loads an influence matrix from a CSV file and visualizes it as a
    weighted directed graph, applying a threshold and removing isolated nodes.

    Args:
        csv_path (Path): The path to the input influence matrix CSV file.
        output_dir (Path): The directory where the output graph will be saved.
        threshold (float): The minimum influence value to be displayed as a connection.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Identify nodes that have at least one connection above the threshold
    active_nodes = set()
    for source_id in df.columns:
        for target_id in df.index:
            if df.loc[target_id, source_id] > threshold:
                active_nodes.add(source_id)
                active_nodes.add(target_id)

    if not active_nodes:
        print(f"No connections found above the threshold of {threshold}. No graph will be generated.")
        return

    # Create a new directed graph with better layout engine for hierarchy
    dot = graphviz.Digraph(engine='dot',
                           graph_attr={'splines': 'true', 'overlap': 'false', 'rankdir': 'LR', 'nodesep': '0.8',
                                       'ranksep': '1.2'})

    # Add only the active nodes to the graph, using abbreviations
    for node_id in active_nodes:
        anatomical_name = ROI_MAP.get(node_id, node_id)
        abbreviation = ROI_ABBREVIATIONS.get(anatomical_name, anatomical_name)
        network = NETWORK_MAP.get(anatomical_name, 'Other')
        color = NETWORK_COLORS.get(network, 'grey')
        dot.node(abbreviation, style='filled', fillcolor=color,
                 fontcolor='white', shape='ellipse', fontsize='30')

    # Add edges to the graph for influences above the threshold
    for source_id in active_nodes:
        for target_id in active_nodes:
            influence = df.loc[target_id, source_id]  # Note: df.loc[row, col]

            if influence > threshold:
                source_name = ROI_ABBREVIATIONS.get(ROI_MAP.get(source_id, source_id))
                target_name = ROI_ABBREVIATIONS.get(ROI_MAP.get(target_id, target_id))

                # Scale pen width based on influence strength for visual emphasis
                penwidth = 1.0 + (influence - threshold) * 5.0

                dot.edge(source_name, target_name,
                         label=f"{influence:.2f}",
                         penwidth=str(penwidth),
                         color='black',
                         arrowsize='0.8',
                         fontsize='40')

    # Define the output file path
    output_filename = csv_path.stem.replace('_influence_matrix', '')
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_filename}_network_graph_abbrev_thresh_{threshold}"

    # Render and save the graph
    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"Successfully generated network graph: {output_path.with_suffix('.png')}")
    except Exception as e:
        print(f"Error rendering graph. Make sure Graphviz is installed and in your system's PATH. Error: {e}")


def main():
    """
    Main function to parse command-line arguments and run the visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize a PBN influence matrix as a directed graph.")
    parser.add_argument("file_path", type=str,
                        help="Path to the influence matrix CSV file.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save the output graph (default: current directory).")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Influence value threshold below which connections will not be drawn (default: 0.2).")

    args = parser.parse_args()

    file_path = Path(args.file_path)
    output_path = Path(args.output_dir)

    visualize_influence_matrix(file_path, output_path, args.threshold)


if __name__ == "__main__":
    # To run this script from the command line:
    # python <script_name>.py path/to/matrix.csv --output_dir /path/to/save --threshold 0.25
    #
    # Example for your request:
    # python visualize_influence.py /Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Influence_Matrices/AD/sub-036S6885_influence_matrix.csv --output_dir /Users/liuzhonglin/Downloads --threshold 0.25
    #
    # Before running, ensure you have the required libraries:
    # pip install pandas graphviz
    # You also need to install the Graphviz system package:
    # - on macOS: brew install graphviz
    # - on Debian/Ubuntu: sudo apt-get install graphviz
    main()
