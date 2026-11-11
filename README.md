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
│   ├── Brain_Tumor_Classification.ipynb
│   ├── animation.py
│   ├── BrainTumorCNNStepByStep.gif
│   └── data/
│       ├── Training/
│       └── Testing/
│
├── CNN Cloud classification/
│   ├── README.md
│   ├── CNN_clouds.ipynb
│   ├── CNN_animation.py
│   ├── CNN_animation.gif
│   └── data/
│       ├── clouds_train/
│       └── clouds_test/
│
├── CNN Mnist Fashion/
│   └── CNN_mnist_fashion.ipynb
│
├── Customer Purchase/
│   └── Customer_Purchase.ipynb
│
├── RNN/
│   ├── README.md
│   ├── RNN_Temperature.ipynb
│   ├── animation.py
│   ├── RNNStepByStep.gif
│   └── data.csv
│
└── Spoken Digit Recognition/
    ├── Spoken_Digit_Recognition.ipynb
    ├── data/
    │   └── FSDD/
    └── models/
        ├── cnn_digit.pt
        └── cnn_speaker.pt
```