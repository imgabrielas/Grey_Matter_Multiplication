"""
Manim animation that walks through the whole Cloud Classification CNN pipeline
(see clouds.ipynb) from raw image to predicted class: data augmentation, exactly
what a convolution kernel computes, the two conv+pool blocks, flatten, the linear
classifier, the training curve, and held-out test metrics.

The architecture, transforms and hyperparameters mirror the notebook exactly:
    Conv2d(3,32,k=3,pad=1) -> ELU -> MaxPool(2)
    Conv2d(32,64,k=3,pad=1) -> ELU -> MaxPool(2)
    Flatten -> Linear(64*32*32, 7)
    Adam(lr=0.001), CrossEntropyLoss, batch_size=16, 3 epochs

Unlike RNN/animation.py, this script actually trains the real model for real (3
epochs takes well under a minute on CPU on this dataset), so every number shown
-- the kernel's convolution math, the feature maps, the loss curve, the
precision/recall, the final prediction -- comes from a real forward/backward pass
on the real data, not a placeholder.

Render with, e.g.:
    manim -pql "CNN Cloud classification/CNN_animation.py" CNNStepByStep   # quick draft
    manim -pqh "CNN Cloud classification/CNN_animation.py" CNNStepByStep  # high quality
"""

import os
import random
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall
from torchvision import transforms
from torchvision.datasets import ImageFolder

from manim import *

# --------------------------------------------------------------------------------------
# 0. Paths / constants
# --------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(HERE, "data", "clouds_train")
TEST_DIR = os.path.join(HERE, "data", "clouds_test")
ASSETS_DIR = tempfile.mkdtemp(prefix="cnn_anim_assets_")

IMG_SIZE = 128
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SEED = 7

SHORT_NAMES = {
    "cirriform clouds": "Cirriform",
    "clear sky": "Clear sky",
    "cumulonimbus clouds": "Cumulonimbus",
    "cumulus clouds": "Cumulus",
    "high cumuliform clouds": "High cumuliform",
    "stratiform clouds": "Stratiform",
    "stratocumulus clouds": "Stratocumulus",
}


# --------------------------------------------------------------------------------------
# 1. Same model definition as the notebook
# --------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------------------
# 2. Train the real model, evaluate it, and precompute every visual asset up front so
#    the Scene itself only has to animate, not crunch numbers.
# --------------------------------------------------------------------------------------

def save_heatmap(channel_2d: torch.Tensor, path: str, cmap="inferno"):
    arr = channel_2d.detach().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    plt.imsave(path, arr, cmap=cmap)


def run_pipeline():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ])

    ds_train = ImageFolder(TRAIN_DIR, transform=train_transforms)
    ds_test = ImageFolder(TEST_DIR, transform=test_transforms)
    classes = ds_train.classes

    dl_train = DataLoader(ds_train, shuffle=True, batch_size=BATCH_SIZE)
    dl_test = DataLoader(ds_test, shuffle=False, batch_size=BATCH_SIZE)

    net = Net(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    epoch_losses = []
    net.train()
    for _ in range(NUM_EPOCHS):
        running = 0.0
        for images, labels in dl_train:
            optimizer.zero_grad()
            out = net(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        epoch_losses.append(running / len(dl_train))

    metric_precision = Precision(task="multiclass", num_classes=len(classes), average="macro")
    metric_recall = Recall(task="multiclass", num_classes=len(classes), average="macro")
    net.eval()
    with torch.no_grad():
        for images, labels in dl_test:
            out = net(images)
            _, preds = torch.max(out, 1)
            metric_precision(preds, labels)
            metric_recall(preds, labels)
    precision = metric_precision.compute().item()
    recall = metric_recall.compute().item()

    # ---- pick one real test image to trace through the whole network -----------------
    demo_path, demo_label_idx = random.choice(ds_test.samples)
    demo_raw = Image.open(demo_path).convert("RGB")
    demo_tensor = test_transforms(demo_raw)  # (3, 128, 128), no augmentation for a clean trace

    activations = {}
    hooks = []
    layer_names = ["conv1", "elu1", "pool1", "conv2", "elu2", "pool2"]
    for name, module in zip(layer_names, net.feature_extractor):
        hooks.append(module.register_forward_hook(
            lambda m, i, o, n=name: activations.__setitem__(n, o.detach()[0])
        ))

    net.eval()
    with torch.no_grad():
        logits = net(demo_tensor.unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=0)
    for h in hooks:
        h.remove()

    pred_idx = int(torch.argmax(probs))

    # ---- data-augmentation pipeline preview images ------------------------------------
    aug_path, _ = random.choice(ds_train.samples)
    aug_raw = Image.open(aug_path).convert("RGB")
    aug_flipped = transforms.functional.hflip(aug_raw)
    aug_rotated = transforms.functional.rotate(aug_flipped, angle=28)
    aug_resized = aug_rotated.resize((IMG_SIZE, IMG_SIZE))

    demo_raw.save(os.path.join(ASSETS_DIR, "raw.png"))
    aug_raw.save(os.path.join(ASSETS_DIR, "aug_raw.png"))
    aug_flipped.save(os.path.join(ASSETS_DIR, "aug_flipped.png"))
    aug_rotated.save(os.path.join(ASSETS_DIR, "aug_rotated.png"))
    aug_resized.save(os.path.join(ASSETS_DIR, "aug_resized.png"))

    # ---- feature-map thumbnails (8 representative channels per stage) ----------------
    channel_ids = list(range(8))
    feature_map_paths = {}
    for stage in ["conv1", "pool1", "conv2", "pool2"]:
        paths = []
        tensor = activations[stage]
        for c in channel_ids:
            p = os.path.join(ASSETS_DIR, f"{stage}_{c}.png")
            save_heatmap(tensor[c], p)
            paths.append(p)
        feature_map_paths[stage] = paths

    # ---- a tiny, real, hand-checkable convolution demo (single filter, 3 channels) ---
    kernel = net.feature_extractor[0].weight[0].detach()  # (3, 3, 3): out-channel 0
    bias = net.feature_extractor[0].bias[0].item()
    r0, c0 = 40, 40  # top-left corner of a 6x6 patch -> 4x4 valid-convolution output
    patch = demo_tensor[:, r0:r0 + 6, c0:c0 + 6]  # (3, 6, 6)
    patch_rgb = patch.permute(1, 2, 0).numpy()  # (6, 6, 3) in [0, 1]

    conv_grid = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            window = patch[:, i:i + 3, j:j + 3]
            conv_grid[i, j] = float((window * kernel).sum()) + bias

    return {
        "classes": classes,
        "epoch_losses": epoch_losses,
        "precision": precision,
        "recall": recall,
        "demo_label": classes[demo_label_idx],
        "pred_label": classes[pred_idx],
        "probs": probs.numpy(),
        "feature_map_paths": feature_map_paths,
        "patch_rgb": patch_rgb,
        "conv_grid": conv_grid,
        "kernel_r_channel": kernel[0].numpy(),  # 3x3, red channel only, for display
    }


DATA = run_pipeline()


# --------------------------------------------------------------------------------------
# 3. Scene
# --------------------------------------------------------------------------------------

class CNNStepByStep(Scene):
    def construct(self):
        self.show_title()
        self.show_data_pipeline()
        self.show_kernel_demo()
        self.show_feature_maps()
        self.show_flatten_and_classify()
        self.show_training_and_eval()

    # ---- 1. title ---------------------------------------------------------------------
    def show_title(self):
        title = Text("Cloud Classification CNN — Convolution by Convolution", weight=BOLD).scale(0.55)
        subtitle = Text(
            "3→32→64 channels · 128×128 input · 7 cloud classes · trained live, 3 epochs",
            font_size=22, color=GREY_B,
        ).next_to(title, DOWN)
        group = VGroup(title, subtitle)
        self.play(FadeIn(group, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(group))

    # ---- 2. augmentation pipeline -------------------------------------------------------
    def show_data_pipeline(self):
        heading = Text("1. Data augmentation", font_size=28, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        def img(path, label):
            im = ImageMobject(path).scale_to_fit_height(2.6)
            lbl = Text(label, font_size=18, color=GREY_B).next_to(im, DOWN, buff=0.15)
            return Group(im, lbl)

        raw = img(os.path.join(ASSETS_DIR, "aug_raw.png"), "raw image")
        flipped = img(os.path.join(ASSETS_DIR, "aug_flipped.png"), "random flip")
        rotated = img(os.path.join(ASSETS_DIR, "aug_rotated.png"), "random rotation")
        resized = img(os.path.join(ASSETS_DIR, "aug_resized.png"), "resize -> 128x128")

        row = Group(raw, flipped, rotated, resized).arrange(RIGHT, buff=0.9).shift(DOWN * 0.3)
        arrows = VGroup(*[
            Arrow(row[i][0].get_right(), row[i + 1][0].get_left(), buff=0.1, stroke_width=3)
            for i in range(3)
        ])

        self.play(FadeIn(raw))
        for prev, nxt, arrow in zip(row[:-1], row[1:], arrows):
            self.play(GrowArrow(arrow), FadeIn(nxt), run_time=0.6)
        self.wait(0.4)
        self.play(FadeOut(heading), FadeOut(row), FadeOut(arrows))

    # ---- 3. exact convolution math on a real 6x6 patch ---------------------------------
    def show_kernel_demo(self):
        heading = Text("2. What one convolution filter actually computes", font_size=26, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        patch_rgb = DATA["patch_rgb"]
        conv_grid = DATA["conv_grid"]

        cell = 0.5
        patch_squares = VGroup(*[
            Square(cell, stroke_width=1, stroke_color=WHITE, fill_opacity=1,
                   fill_color=rgb_to_color(patch_rgb[i, j]))
            for i in range(6) for j in range(6)
        ])
        patch_squares.arrange_in_grid(rows=6, cols=6, buff=0)
        patch_squares.move_to(LEFT * 3.3)
        patch_label = Text("6×6 crop of the real image (RGB)", font_size=18, color=GREY_B)
        patch_label.next_to(patch_squares, DOWN, buff=0.3)

        out_squares = VGroup(*[Square(cell, stroke_width=1, stroke_color=GREY_B) for _ in range(16)])
        out_squares.arrange_in_grid(rows=4, cols=4, buff=0)
        out_squares.move_to(RIGHT * 3.3)
        out_label = Text("output feature map (4×4)", font_size=18, color=GREY_B)
        out_label.next_to(out_squares, DOWN, buff=0.3)

        eq = Text("value = Σ (kernel · image patch) + bias", font_size=20).to_edge(DOWN, buff=0.5)

        self.play(FadeIn(patch_squares), FadeIn(patch_label), FadeIn(out_squares), FadeIn(out_label), Write(eq))

        vmin, vmax = conv_grid.min(), conv_grid.max()

        def grid_index(i, j):
            return i * 6 + j

        for i in range(4):
            for j in range(4):
                idxs = [grid_index(i + di, j + dj) for di in range(3) for dj in range(3)]
                window = VGroup(*[patch_squares[k] for k in idxs])
                highlight = SurroundingRectangle(window, color=YELLOW, buff=0.02, stroke_width=3)

                out_idx = i * 4 + j
                val = conv_grid[i, j]
                norm = (val - vmin) / (vmax - vmin + 1e-8)
                target_color = interpolate_color(BLUE, RED, norm)
                val_text = Text(f"{val:.1f}", font_size=14).move_to(out_squares[out_idx])

                self.play(
                    Create(highlight),
                    out_squares[out_idx].animate.set_fill(target_color, opacity=0.85),
                    FadeIn(val_text),
                    run_time=0.12,
                )
                self.remove(highlight)

        self.wait(0.5)
        note = Text("× 32 filters like this one, each scanning the full 128×128 image",
                     font_size=20, color=GREY_B).next_to(eq, UP, buff=0.2)
        self.play(FadeIn(note))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not None])

    # ---- 4. real feature maps through both conv blocks ----------------------------------
    def show_feature_maps(self):
        def stage_row(stage, label, n_channels):
            heading = Text(label, font_size=26, weight=BOLD).to_edge(UP)
            imgs = Group(*[ImageMobject(p).scale_to_fit_height(1.4) for p in DATA["feature_map_paths"][stage]])
            imgs.arrange_in_grid(rows=2, cols=4, buff=0.25).shift(DOWN * 0.2)
            caption = Text(f"8 of {n_channels} channels shown", font_size=18, color=GREY_B)
            caption.next_to(imgs, DOWN, buff=0.3)
            self.play(FadeIn(heading))
            self.play(LaggedStartMap(FadeIn, imgs, lag_ratio=0.08), FadeIn(caption))
            self.wait(0.6)
            self.play(FadeOut(heading), FadeOut(imgs), FadeOut(caption))

        stage_row("conv1", "3. Conv block 1 — 32 filters (post-conv, pre-activation)", 32)
        stage_row("pool1", "MaxPool 2×2 — 128×128 → 64×64", 32)
        stage_row("conv2", "4. Conv block 2 — 64 filters", 64)
        stage_row("pool2", "MaxPool 2×2 — 64×64 → 32×32", 64)

    # ---- 5. flatten -> linear -> class probabilities -------------------------------------
    def show_flatten_and_classify(self):
        heading = Text("5. Flatten → Linear classifier", font_size=26, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        thumbs = Group(*[ImageMobject(p).scale_to_fit_height(0.9) for p in DATA["feature_map_paths"]["pool2"]])
        thumbs.arrange_in_grid(rows=2, cols=4, buff=0.15).to_edge(LEFT, buff=1.0)
        self.play(FadeIn(thumbs))

        vector = Rectangle(width=0.4, height=3.2, color=GREEN, fill_opacity=0.3).next_to(thumbs, RIGHT, buff=1.2)
        vector_label = Text("64×32×32 = 65,536-d vector", font_size=16, color=GREY_B).next_to(vector, DOWN, buff=0.2)
        flatten_arrow = Arrow(thumbs.get_right(), vector.get_left(), buff=0.15, stroke_width=3)
        self.play(GrowArrow(flatten_arrow), FadeIn(vector), FadeIn(vector_label))

        classes = DATA["classes"]
        probs = DATA["probs"]
        pred_label = DATA["pred_label"]
        demo_label = DATA["demo_label"]

        bars = VGroup()
        labels = VGroup()
        max_width = 2.6
        for idx, cls in enumerate(classes):
            p = float(probs[idx])
            bar = Rectangle(width=max(p * max_width, 0.02), height=0.3,
                             color=GREEN if cls == pred_label else BLUE, fill_opacity=0.8)
            lbl = Text(f"{SHORT_NAMES[cls]} ({p:.0%})", font_size=16)
            bars.add(bar)
            labels.add(lbl)
        bars.arrange(DOWN, buff=0.18, aligned_edge=LEFT).next_to(vector, RIGHT, buff=1.3)
        for bar, lbl in zip(bars, labels):
            lbl.next_to(bar, RIGHT, buff=0.15)

        self.play(GrowArrow(Arrow(vector.get_right(), bars.get_left() + LEFT * 0.3, buff=0.1)))
        self.play(
            LaggedStart(*[GrowFromEdge(bar, LEFT) for bar in bars], lag_ratio=0.1),
            FadeIn(labels),
        )
        self.wait(0.5)

        verdict_color = GREEN if pred_label == demo_label else RED
        verdict = Text(
            f"Predicted: {SHORT_NAMES[pred_label]}   |   Actual: {SHORT_NAMES[demo_label]}",
            font_size=20, color=verdict_color,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(verdict))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not None])

    # ---- 6. training curve + held-out metrics -----------------------------------------
    def show_training_and_eval(self):
        heading = Text("6. Training & evaluation", font_size=26, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        losses = DATA["epoch_losses"]
        axes = Axes(
            x_range=[1, len(losses), 1], y_range=[0, max(losses) * 1.2, max(losses) / 4],
            x_length=6, y_length=3.5,
            axis_config={"include_numbers": False},
        ).shift(LEFT * 2.5)
        x_label = Text("epoch", font_size=18).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("loss", font_size=18).next_to(axes.y_axis, LEFT, buff=0.2)

        points = [axes.coords_to_point(i + 1, loss) for i, loss in enumerate(losses)]
        dots = VGroup(*[Dot(p, color=YELLOW) for p in points])
        line = VMobject(color=YELLOW).set_points_as_corners(points)
        tick_labels = VGroup(*[
            Text(str(i + 1), font_size=16).next_to(axes.coords_to_point(i + 1, 0), DOWN, buff=0.15)
            for i in range(len(losses))
        ])
        value_labels = VGroup(*[
            Text(f"{loss:.2f}", font_size=16, color=YELLOW).next_to(p, UP, buff=0.15)
            for p, loss in zip(points, losses)
        ])

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label), FadeIn(tick_labels))
        self.play(Create(line), FadeIn(dots), FadeIn(value_labels), run_time=1.2)

        metrics = VGroup(
            Text("Held-out test set (macro-averaged):", font_size=20, weight=BOLD),
            Text(f"Precision:  {DATA['precision']:.2f}", font_size=22, color=GREEN),
            Text(f"Recall:       {DATA['recall']:.2f}", font_size=22, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT, buff=1.0)

        self.play(FadeIn(metrics, shift=LEFT * 0.3))
        self.wait(2)