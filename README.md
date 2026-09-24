# Grey Matter Multiplication

A collection of DL projects.

| Project                                                            | Description                                                                    |
|---------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [CNN Cloud classification](CNN%20Cloud%20classification/README.md) | CNN that classifies cloud images into 7 types using PyTorch.                  |
| [RNN](RNN/README.md)                                               | RNN that forecasts next-day minimum temperature from a 30-day sliding window. |

## Repository structure

```
.
├── CNN Cloud classification/
│   ├── README.md
│   ├── clouds.ipynb          # data loading, model, training, evaluation
│   └── data/
│       ├── clouds_train/     # training images, one subfolder per class
│       └── clouds_test/      # test images, same 7 class subfolders
└── RNN/
    ├── README.md
    ├── rnn.ipynb              # preprocessing, model, training, evaluation
    ├── animation.py           # Manim animation of the model's forward pass
    ├── RNNStepByStep.gif      # rendered output of animation.py
    └── data.csv               # daily minimum temperatures, Melbourne (1981-1990)
```