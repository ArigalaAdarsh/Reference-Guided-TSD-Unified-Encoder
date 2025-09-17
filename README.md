# REFERENCE-GUIDED TARGETED SOUND DETECTION USING A UNIFIED ENCODER (ICASSP 2026 Submission)

This repository provides the **official implementation** of our unified encoder model for **Target Sound Detection (TSD)**, submitted to **ICASSP 2026**.  
It includes training and evaluation for the **URBAN-SED** and **URBAN*8K** datasets.  

## Datasets
- **UrbanSound8K**: [Download here](https://urbansounddataset.weebly.com/urbansound8k.html)  
- **URBAN-SED**: [Download here](https://zenodo.org/records/1324404)  
- Keep these in the datasets/URBAN-SED
- datasets/ UrbanSound8K

## Installation

```bash
git clone https://github.com/your-username/Reference-Guided-TSD-Unified-Encoder.git
cd Reference-Guided-TSD-Unified-Encoder
```


Install python >= 3.8

### 1. Install Requirements
Upgrade pip and install dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

2. Load the ConvNeXt checkpoint for the audio encoder (trained on the AudioSet2M). Download `convnext_tiny_471mAP.pth` from [Zenodo](https://zenodo.org/records/8020843) and place it in the `convnext` folder.  

## Preparing the Dataset
To make the dataset ready, run:

```bash
python data/extract_feature.py
```

Inside the script, change the parameters accordingly:

```python
what_type = 'train'  # options: 'train', 'val', 'test'
```
and update the respective dataset path locations.

## Training
To train the model, run:

```bash
bash bash/tsd.sh
```

## Evaluation
Check the `notebooks/` folder for evaluation scripts.  
For example, you can run:

```bash
Evaluating_ConvNeXt_multiplication.ipynb
```

---
