# Radiomics CRT

This repo provides the implementation of paper entitled "Comparing deep learning and handcrafted radiomics to predict chemoradiotherapy response for locally advanced cervical cancer using pretreatment MRI".

## Prerequisites

Tested on Ubuntu 16.04.7 LTS

- Python 3.6.8 (Anaconda 4.8.3)
- CUDA 9.2

## Getting Started

Install dependencies conda environment using commands:

```bash
conda env create --name radiomics --file=env.yaml
conda activate radiomics
```

## List of Documents

All implementations of our model, optimization, evaluation, and experimental results can be found in the following Jupyter Notebooks:

- `DLR.ipynb` : experimental results for deep learning radiomics (DLR)
- `DLR_ActivationMap_visualization.ipynb` : feature map visualization of DLR
- `Featureselection_RFE_SVM_classifier.ipynb` : experimental results for handcrafted radiomics (HCR)

## Data Availability

The datasets generated and/or analyzed during the current study are not publicly available due to the privacy protection policy of personal medical information at our institution but are available from the corresponding author upon reasonable request.
