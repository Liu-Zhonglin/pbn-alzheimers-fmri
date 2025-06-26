import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from collections import OrderedDict

print("--- Data Harmonization and 18-ROI Extraction Pipeline (Final Version) ---")

# ==============================================================================
# === 1. CONFIGURATION =========================================================
# ==============================================================================
# Path to the original BIDS-style data folder
base_data_path = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/Data'

# Path to a NEW folder where the harmonized .csv files will be saved
harmonized_output_path = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/pbn_data_harmonized/'

# Path to your full AAL3 atlas
atlas_filepath = '/Users/liuzhonglin/Desktop/URFP/Codes/Pipeline/atlas/AAL3/AAL3v1.nii.gz'

# The target TR we want all data to have
TARGET_TR = 0.61

# --- Define the 18 ROIs of interest using an Ordered Dictionary ---
ROIS_OF_INTEREST = OrderedDict([
    # DMN Network Regions
    ('Precuneus_L', 71), ('Precuneus_R', 72), ('Angular_L', 69), ('Angular_R', 70),
    ('Frontal_Med_Orb_L', 21), ('Frontal_Med_Orb_R', 22),
    # ECN Network Regions
    ('Frontal_Sup_2_L', 3), ('Frontal_Sup_2_R', 4), ('Parietal_Sup_L', 63), ('Parietal_Sup_R', 64),
    # SN Network Regions
    ('Insula_L', 33), ('Insula_R', 34), ('Supp_Motor_Area_L', 15), ('Supp_Motor_Area_R', 16),
    # MTL Network Regions
    ('Hippocampus_L', 41), ('Hippocampus_R', 42), ('ParaHippocampal_L', 43), ('ParaHippocampal_R', 44)
])

os.makedirs(harmonized_output_path, exist_ok=True)
print(f"Harmonized .csv files will be saved to: {harmonized_output_path}")

# ==============================================================================
# === 2. CREATE A CUSTOM 18-ROI ATLAS ==========================================
# ==============================================================================
print("\nCreating a custom atlas with only the 18 specified ROIs...")
full_atlas_img = nib.load(atlas_filepath)
full_atlas_data = full_atlas_img.get_fdata()
custom_atlas_data = np.zeros(full_atlas_data.shape, dtype=full_atlas_data.dtype)
aal_indices_to_keep = list(ROIS_OF_INTEREST.values())
for label_value in aal_indices_to_keep:
    custom_atlas_data[full_atlas_data == label_value] = label_value
custom_atlas_img = nib.Nifti1Image(custom_atlas_data, full_atlas_img.affine, full_atlas_img.header)
print("Custom 18-ROI atlas created successfully.")

# ==============================================================================
# === 3. FIND ALL SUBJECTS AND THEIR ORIGINAL TRs ==============================
# ==============================================================================
fmri_data_pattern = os.path.join(base_data_path, '**', '*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
fmri_filenames = sorted(glob.glob(fmri_data_pattern, recursive=True))
print(f"\nFound {len(fmri_filenames)} total subjects to process.")

# ==============================================================================
# === 4. HARMONIZE (IF NEEDED) AND EXTRACT TIME SERIES FOR EACH SUBJECT ========
# ==============================================================================
# This masker is now correctly configured to avoid the "18 vs 19" mismatch warning.
masker = NiftiLabelsMasker(labels_img=custom_atlas_img, labels=list(ROIS_OF_INTEREST.keys()),
                           standardize=True, memory='nilearn_cache', background_label=0)

for filepath in fmri_filenames:
    subject_id = os.path.basename(filepath).split('_')[0]
    print(f"\nProcessing {subject_id}...")

    try:
        original_img = nib.load(filepath)
        original_tr = original_img.header.get_zooms()[3]
        print(f"  - Original TR: {original_tr:.2f}s")
        img_to_process = original_img

        # This check now correctly uses np.isclose to avoid unnecessary resampling.
        if not np.isclose(original_tr, TARGET_TR):
            print(f"  - Resampling from {original_tr:.2f}s to {TARGET_TR:.2f}s...")
            target_affine = original_img.affine.copy()
            target_affine[3, 3] = TARGET_TR
            harmonized_img = image.resample_img(original_img, target_affine=target_affine,
                                                interpolation='continuous',
                                                copy_header=True, force_resample=True)
            img_to_process = harmonized_img
            print("  - Resampling complete.")
        else:
            print("  - TR already matches target. No resampling needed.")

        print("  - Extracting time series from the 18 specified ROIs...")
        time_series = masker.fit_transform(img_to_process)
        column_headers = list(ROIS_OF_INTEREST.keys())

        df_timeseries = pd.DataFrame(time_series, columns=column_headers)

        problematic_rois = []
        for roi_name in df_timeseries.columns:
            if df_timeseries[roi_name].std() == 0:
                problematic_rois.append(roi_name)

        if problematic_rois:
            print(f"  - !!! DATA QUALITY WARNING for {subject_id} !!!")
            print(f"  - The following ROIs had ZERO signal variance (all values are the same):")
            for roi in problematic_rois:
                print(f"    - {roi}")
            print("  - This likely means there was no overlap between the atlas ROI and this subject's brain data.")
            print("  - These ROIs will cause issues in downstream statistical analysis.")

        output_csv_path = os.path.join(harmonized_output_path, f"{subject_id}_time_series.csv")
        df_timeseries.to_csv(output_csv_path, index=False)
        print(f"  - Successfully saved 18-ROI time series to {output_csv_path}")

    except Exception as e:
        print(f"  - !!! ERROR processing {subject_id}: {e}")

print("\n\n--- Pipeline Complete ---")
print("All subjects have been processed and their 18-ROI time series have been extracted.")