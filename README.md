# pbn-alzheimers-fmri
Modeling the progressive failure of brain networks in Alzheimer's Disease using Probabilistic Boolean Networks and fMRI data.


# Alzheimer's Brain Disconnection Model

A computational model to analyze the progressive failure of brain connectivity in Alzheimer's Disease using fMRI data.

---

### About This Project

This repository contains the code and resources for implementing a framework to model Alzheimer's Disease (AD) as a "disconnection syndrome." The primary goal is to quantify how the directed influence between different brain regions weakens as the disease progresses from healthy aging (Normal Control) through Mild Cognitive Impairment (MCI) to a full AD diagnosis.

This work is an implementation of the methodology described in the paper:

> **"Inferring Progressive Disconnection in Alzheimer’s Disease with Probabilistic Boolean Networks"** by Zhonglin Liu and Louxin Zhang.

### Core Methodology

The model follows a pipeline that uses Probabilistic Boolean Networks (PBNs) to analyze resting-state fMRI data. The key steps are:

1.  **Data Preprocessing:** fMRI brain scans from the ADNI database are cleaned, and time-series data is extracted for 18 key Regions of Interest (ROIs).
2.  **Signal Binarization:** The continuous fMRI signal for each ROI is converted into a binary ('on'/'off') state sequence using a Hidden Markov Model (HMM).
3.  **PBN Inference:** A subject-specific Probabilistic Boolean Network is learned, identifying the most likely rules that govern how brain regions influence each other.
4.  **Influence Analysis:** The final PBN model is used to calculate a directed influence matrix, which shows the strength of the connection from one brain region to another.

### Key Findings to Replicate

The central finding of the original paper, which this code aims to model, is the **progressive, linear decrease in influence from the Default Mode Network (DMN) to the Medial Temporal Lobe (MTL)**. This represents a quantifiable failure of the brain's memory system and is a key biomarker for Alzheimer's progression.

### How to Use

To be updated

### Acknowledgments

*   This project is based on the novel framework proposed by **Zhonglin Liu and Louxin Zhang**.
*   The data used for this analysis was obtained from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** database.