# Thesis Codebase

## Installation

1. Create a Python environment (Python 3.10+ recommended).  
2. Install the required dependencies:
   pip install -r requirements.txt

## Data Acquisition

The MIMIC-CXR-JPG dataset can be obtained from:
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

To comply with the data use agreement, we cannot share the experimental data directly.
Users must apply for access through PhysioNet and download the dataset themselves.

## Data Preprocessing

All preprocessing scripts are located in the data_setup folder.
The pipeline consists of the following steps:
1. Report mapping (reports_extraction.py)
2. Data cleaning (preprocessing.ipynb)
3. Metadata preprocessing (preprocessing.ipynb)
4. Embedding generation(preprocessing.ipynb)

## Experiments

1. Bootstrap experiments: scripts are in files starting with bootstrap*.
2. Single runs (no bootstrap): scripts are in files starting with domino_pipeline*.
3. LLM analysis: see the llm_analysis.py (Note: users must replace the placeholder with their own Hugging Face token.)

## Citation

This code uses the following models and frameworks.  
If you build upon this work, please cite them accordingly:

- **Domino**  
  https://github.com/HazyResearch/domino  

- **Gemma-2B-2-it**  
  https://huggingface.co/google/gemma-2-2b-it  

- **BioMedCLIP**  
  https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224  

- **CLIP**  
  https://huggingface.co/openai/clip-vit-large-patch14  

  ## Disclaimer

This repository is provided for research purposes only.  
For any enquiries, please contact: **yl28218@ic.ac.uk**









