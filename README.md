# pbn-alzheimers-fmri

### Modeling Progressive Brain Disconnection in Alzheimer's Disease

This repository provides a computational framework for modeling the progressive failure of brain connectivity in Alzheimer's Disease (AD). Using **Probabilistic Boolean Networks (PBNs)** and resting-state functional Magnetic Resonance Imaging (fMRI) data, this project quantifies the disruption of directed influence between key brain networks, offering insights into the "disconnection syndrome" hypothesis of AD.

The methodology is a direct implementation of the work described in the following paper:

> Liu, Zhonglin, & Zhang, Louxin. (2025). *Inferring Progressive Disconnection in Alzheimer’s Disease with Probabilistic Boolean Networks* (currently under review at APBC 2025).  
> [View on OpenReview](https://openreview.net/forum?id=qLbZfaluWq&noteId=qLbZfaluWq)

---

### Core Methodology 

The analysis pipeline transforms raw fMRI time-series data into a subject-specific influence network, allowing for group-level statistical analysis.

1. **Data Preprocessing & ROI Extraction**: Resting-state fMRI data for **Normal Control (NC)**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer's Disease (AD)** cohorts were sourced from the ADNI (Alzheimer’s Disease Neuroimaging Initiative) database. Time series for **18 Regions of Interest (ROIs)** across four major brain networks—the Default Mode Network (DMN), Executive Control Network (ECN), Salience Network (SN), and Medial Temporal Lobe (MTL)—were extracted using the AAL3 (Automated Anatomical Labeling 3) atlas.

2. **Signal Binarization**: The continuous Blood-Oxygen-Level-Dependent (BOLD) signals were denoised and binarized into 'low' and 'high' activity states. This was achieved using a novel, iterative **Hidden Markov Model (HMM)** adapted for resting-state data.

3. **PBN Inference**: A structure-aware pipeline was used to infer a PBN for each subject. This hybrid approach integrates data-driven dynamics with anatomical priors to identify predictive Boolean functions for each ROI. The predictive power of each function was scored using the **Coefficient of Determination (COD)**.

4. **Influence Analysis**: The final PBN model was used to compute an $18 \times 18$ directed **influence matrix** for each subject. This matrix quantifies the causal influence that each brain region exerts over others.

---

### Key Findings to Replicate 

The central finding of this research is the **progressive, linear decrease in directed influence** from the **Default Mode Network (DMN)** to the **Medial Temporal Lobe (MTL)** across the disease spectrum (NC → MCI → AD).

Specifically, the model identifies five significant connections showing this decline. The most significant pathway is from the **Right Precuneus (DMN) to the Left Hippocampus (MTL)**. This provides quantitative evidence for the failure of the brain's memory system, a key biomarker of Alzheimer's progression.

---

### Usage Guide 

Follow these steps to replicate the analysis pipeline.

#### 1. Data Acquisition and Preprocessing

First, acquire the necessary fMRI data from the [ADNI database](http://adni.loni.usc.edu) using the `participants.tsv` file in this repository.

The data must be preprocessed using **fMRIPrep**.
* Each fMRI scan requires its corresponding T1-weighted (a type of MRI scan that provides good contrast between different soft tissues) anatomical scan.
* Data must be organized in the BIDS (Brain Imaging Data Structure) format.
* **Note**: This is a computationally intensive step, and using a High-Performance Computing (HPC) cluster is highly recommended.

#### 2. Signal Extraction and Binarization

Once the data is preprocessed, run the following scripts in order:

1. **Harmonize and Extract ROIs**:
    ```bash
    python harmonize_and_extract.py
    ```
    This script harmonizes the Repetition Time (TR) to 0.61 seconds and extracts the mean BOLD time series for the 18 ROIs via [AAL3 atlas](https://www.oxcns.org/aal3.html).

2. **Denoise and Binarize**:
    ```matlab
    % Run in MATLAB
    Denoising_and_Binarization.m
    ```
    This MATLAB script performs denoising and HMM binarization, producing a binary time series for each ROI.

#### 3. PBN Inference and Analysis

This stage uses a structure-aware approach inspired by [SAILOR](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0304102).

1. **Generate Reference Networks**:
    ```bash
    python generate_custom_references.py
    ```
    This creates the consensus network that serves as an anatomical prior for the model.

2. **Run the Hybrid PBN Engine**:
    ```bash
    python hybrid_pbn_engine.py
    ```
    This is the core script for PBN inference, generating a subject-specific PBN in JSON (JavaScript Object Notation) format.

3. **Calculate Influence Matrix**:
    ```bash
    python Influence_Matrix.py
    ```
    This script processes the PBN output to derive the final $18 \times 18$ influence matrix for each subject.

4. **Perform Group Analysis**:
    ```bash
    python run_analysis.py
    ```
    This script conducts the statistical analysis, including Analysis of Variance (ANOVA) and Benjamini-Hochberg False Discovery Rate (FDR) correction, to identify significant group differences, as detailed in the paper.

---

### Acknowledgments

* This research was made possible by the generous support of the **Undergraduate Research Fellowship Programme (URFP)** at The University of Hong Kong (HKU), which funded a research visit to the National University of Singapore.
* Data were obtained from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**. We thank the ADNI investigators for their contribution to data collection and sharing. A full list of ADNI investigators can be found [here](http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf).
