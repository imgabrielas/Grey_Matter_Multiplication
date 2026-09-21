# Cloud Classification

A PyTorch CNN that classifies cloud images into 7 types: cirriform, clear sky, cumulonimbus, cumulus, high cumuliform, stratiform, and stratocumulus clouds.

**Approach:** Images are loaded via `ImageFolder`, resized to 128x128, and augmented (random flip/rotation) for training. A small CNN (two conv+pool blocks) is trained with cross-entropy loss and Adam, then evaluated on a held-out test set using per-class precision and recall (`torchmetrics`).

**Files:**
- `clouds.ipynb` — data loading, model definition, training loop, and evaluation
- `data/clouds_train`, `data/clouds_test` — image folders, one subfolder per class
