# Cloud Classification

<p align="center">
  <img src="CNN_animation.gif" alt="CNN step-by-step animation">
</p>

A PyTorch CNN that classifies cloud images into 7 types: cirriform, clear sky, cumulonimbus, cumulus, high cumuliform, stratiform, and stratocumulus clouds.

**Approach:** Images are loaded via `ImageFolder`, resized to 128x128, and augmented (random flip/rotation) for training. A small CNN (two conv+pool blocks) is trained with cross-entropy loss and Adam, then evaluated on a held-out test set using per-class precision and recall (`torchmetrics`).

**Files:**
- `clouds.ipynb` — data loading, model definition, training loop, and evaluation
- `data/clouds_train`, `data/clouds_test` — image folders, one subfolder per class
- `CNN_animation.py` — Manim animation that trains the real model and shows, from raw image to prediction, how data augmentation, convolution, pooling, and the classifier work (`manim -pqh "CNN_animation.py" CNNStepByStep`)
- `CNN_animation.gif` — rendered output of `CNN_animation.py`
