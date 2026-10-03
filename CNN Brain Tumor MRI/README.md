# Brain Tumor MRI Classification using PyTorch

---
#### DISCLAIMER: THIS IS NOT MEDICAL ADVICE, DO NOT SELF DIAGNOSE

<p align="center">
  <img src="BrainTumorCNNStepByStep.gif" alt="Brain Tumor MRI CNN step-by-step animation">
</p>

## Overview
A convolutional neural network (CNN) built with PyTorch that classifies brain MRI
scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, and
**no tumor**.

## Data
MRI images are expected under `data/`, split into `Training/` and `Testing/`
folders, each containing one subfolder per class:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

Images are loaded with `torchvision.datasets.ImageFolder` and resized to 128x128,
converted to tensors, and normalized.

## Model
A simple CNN defined with `nn.Sequential`:
- 3 convolutional blocks (Conv2d → ReLU → MaxPool2d), with channel depth
  32 → 64 → 128
- Flatten → Linear(128\*16\*16, 256) → ReLU → Dropout(0.5) → Linear(256, 4)

Trained with the Adam optimizer (`lr=0.001`) and cross-entropy loss, running on
GPU when available (falls back to CPU).

## Usage
Open and run `Brain_tumor_classification.ipynb`, which:
1. Loads and transforms the training/testing data
2. Defines the CNN model
3. Trains the model over 25 epochs, printing the running loss per epoch

## Files
- `Brain_tumor_classification.ipynb` — data loading, model definition, training loop
- `data/Training`, `data/Testing` — image folders, one subfolder per class
- `animation.py` — Manim animation that trains the real model (on a small, class-balanced
  subset for speed) and shows, from raw scan to prediction, how preprocessing, convolution,
  pooling, and the classifier work (`manim -pqh "animation.py" BrainTumorCNNStepByStep`)
- `BrainTumorCNNStepByStep.gif` — rendered output of `animation.py`