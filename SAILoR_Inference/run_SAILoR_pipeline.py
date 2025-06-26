import os
import sys
import argparse
import traceback

# ==============================================================================
# --- SAILoR Pipeline Core Engine ---
# ==============================================================================
# This script is the core, general-purpose engine for the SAILoR analysis.
# It is designed to be called by a master script and accepts all necessary
# file paths and parameters via command-line arguments. It contains no
# hardcoded subject information.

def main():
    """
    Main function to parse arguments and run the SAILoR inference engine.
    """
    print("--- Starting SAILoR Python Pipeline (Core Engine) ---")

    # --- 1. Set up Paths and Argument Parser ---
    sailor_script_dir = os.path.dirname(os.path.abspath(__file__))
    dyngenie3_path = os.path.join(sailor_script_dir, 'dynGENIE3_python')
    sys.path.insert(0, sailor_script_dir)
    sys.path.insert(0, dyngenie3_path)

    # Use argparse to properly handle all command-line arguments.
    parser = argparse.ArgumentParser(description="SAILoR PBN Inference Pipeline Core Engine")
    parser.add_argument('--timeSeriesPath', required=True, help='Path to the continuous time series data file (.csv)')
    parser.add_argument('--binarisedPath', required=True, help='Path to the binarised data file (.csv)')
    parser.add_argument('--outputFilePath', required=True, help='Path to save the final Boolean model (.txt)')
    parser.add_argument('--referencePaths', required=True, help='String representation of a list of reference network paths')
    parser.add_argument('--exactNetworksIndices', required=True, help='String representation of a list of indices for the reference networks')

    args = parser.parse_args()

    # --- 2. Import SAILoR and Prepare Arguments ---
    try:
        import SAILoR
    except ImportError as e:
        print(f"\n--- FATAL ERROR: Could not import SAILoR module. ---\nDetails: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.outputFilePath), exist_ok=True)

    # Construct the sys.argv list that SAILoR.py's main function expects,
    # using the arguments parsed from the command line.
    sys.argv = [
        'SAILoR.py',
        '--timeSeriesPath', args.timeSeriesPath,
        '--binarisedPath', args.binarisedPath,
        '--referencePaths', args.referencePaths,
        '--outputFilePath', args.outputFilePath,
        '--exactNetworksIndices', args.exactNetworksIndices
    ]

    print("\n--- Configuration for this run ---")
    print(f"Time Series: {args.timeSeriesPath}")
    print(f"Output Model: {args.outputFilePath}")
    # The `eval()` function safely converts the string back into a Python list to count its length.
    print(f"Using {len(eval(args.referencePaths))} reference networks.")
    print("----------------------------------\n")

    # --- 3. Run the SAILoR Main Logic ---
    try:
        SAILoR.main()
        print("\n--- SAILoR Core Engine Finished Successfully ---")
        print(f"Output should be saved to: {args.outputFilePath}")
    except Exception as e:
        print(f"\n--- An exception occurred during SAILoR execution: ---")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()