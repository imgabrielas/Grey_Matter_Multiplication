# Grey Matter Multiplication

A collection of DL projects.

| Project                                                             | Description                                                                                         |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [CNN Brain Tumor MRI](CNN%20Brain%20Tumor%20MRI/README.md)          | CNN that classifies brain MRI scans into glioma, meningioma, pituitary, or no tumor using PyTorch.  |
| [CNN Cloud classification](CNN%20Cloud%20classification/README.md) | CNN that classifies cloud images into 7 types using PyTorch.                                         |
| [RNN](RNN/README.md)                                                | RNN that forecasts next-day minimum temperature from a 30-day sliding window.                       |

## Repository structure

```
.
├── CNN Brain Tumor MRI/
│   ├── README.md
│   ├── Brain_tumor_classification.ipynb   # data loading, model, training, evaluation
│   └── data/
│       ├── Training/      # training images, one subfolder per class
│       └── Testing/       # test images, same 4 class subfolders
│
├── CNN Cloud classification/
│   ├── README.md
│   ├── clouds.ipynb          # data loading, model, training, evaluation
│   └── data/
│       ├── clouds_train/     # training images, one subfolder per class
│       └── clouds_test/      # test images, same 7 class subfolders
│
└── RNN/
    ├── README.md
    ├── rnn.ipynb              # preprocessing, model, training, evaluation
    ├── animation.py           # Manim animation of the model's forward pass
    ├── RNNStepByStep.gif      # rendered output of animation.py
    └── data.csv               # daily minimum temperatures, Melbourne (1981-1990)
```