# Grey Matter Multiplication

A collection of DL projects.

| Project                                                             | Description                                                                                         |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [CNN Brain Tumor MRI](CNN%20Brain%20Tumor%20MRI/README.md)          | CNN that classifies brain MRI scans into glioma, meningioma, pituitary, or no tumor using PyTorch.  |
| [CNN Cloud classification](CNN%20Cloud%20classification/README.md) | CNN that classifies cloud images into 7 types using PyTorch.                                         |
| [CNN Mnist Fashion](CNN%20Mnist%20Fashion/CNN_mnist_fashion.ipynb) | CNN that classifies Fashion-MNIST clothing images into 10 categories using TensorFlow/Keras.         |
| [Customer Purchase](Customer%20Purchase/Customer_Purchase.ipynb)   | Neural network that predicts which customers are likely to make a purchase from browsing behavior, using PyTorch. |
| [RNN](RNN/README.md)                                                | RNN that forecasts next-day minimum temperature from a 30-day sliding window.                       |
| [Spoken Digit Recognition](Spoken%20Digit%20Recognition/Spoken_Digit_Recognition.ipynb) | CNN on log-mel spectrograms that recognizes spoken digits (0–9) and speakers from the FSDD audio dataset, using PyTorch. |

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