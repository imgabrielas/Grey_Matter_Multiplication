"""
Manim animation that walks through the whole Brain Tumor MRI CNN pipeline
(see Brain_tumor_classification.ipynb) from raw scan to predicted class:
the preprocessing pipeline, exactly what a convolution kernel computes, the
three conv+pool blocks, flatten, the linear classifier, the training curve,
and held-out test metrics.

The architecture and transforms mirror the notebook exactly:
    Conv2d(3,32,k=3,pad=1)   -> ReLU -> MaxPool(2)
    Conv2d(32,64,k=3,pad=1)  -> ReLU -> MaxPool(2)
    Conv2d(64,128,k=3,pad=1) -> ReLU -> MaxPool(2)
    Flatten -> Linear(128*16*16, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 4)
    Adam(lr=0.001), CrossEntropyLoss
    Resize((128,128)), ToTensor(), Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

The notebook itself trains on the full ~5,600-image training set for 50 epochs.
That is too slow to redo as a precompute step for an animation, so this script
trains the *real* model (no faked numbers) on a small, class-balanced subset
for a few epochs -- enough that every number shown (the kernel's convolution
math, the feature maps, the loss curve, the precision/recall, the final
prediction) comes from a genuine forward/backward pass, just over less data.

Render with, e.g.:
    manim -pql "CNN Brain Tumor MRI/animation.py" BrainTumorCNNStepByStep   # quick draft
    manim -pqh "CNN Brain Tumor MRI/animation.py" BrainTumorCNNStepByStep   # high quality
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
from torch.utils.data import DataLoader, Subset
from torchmetrics import Precision, Recall
from torchvision import transforms
from torchvision.datasets import ImageFolder

from manim import *

# --------------------------------------------------------------------------------------
# 0. Paths / constants
# --------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(HERE, "data", "Training")
TEST_DIR = os.path.join(HERE, "data", "Testing")
ASSETS_DIR = tempfile.mkdtemp(prefix="brain_cnn_anim_assets_")

IMG_SIZE = 128
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PER_CLASS_TRAIN = 60   # small, class-balanced subset -- keeps the precompute step fast
PER_CLASS_TEST = 20
SEED = 7

SHORT_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}


# --------------------------------------------------------------------------------------
# 1. Same model definition as the notebook
# --------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------------------
# 2. Train the real model on a small subset, evaluate it, and precompute every visual
#    asset up front so the Scene itself only has to animate, not crunch numbers.
# --------------------------------------------------------------------------------------

def save_heatmap(channel_2d: torch.Tensor, path: str, cmap="inferno"):
    arr = channel_2d.detach().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    plt.imsave(path, arr, cmap=cmap)


def class_balanced_indices(dataset: ImageFolder, per_class: int, rng: random.Random):
    idx_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        idx_by_class.setdefault(label, []).append(idx)
    chosen = []
    for idxs in idx_by_class.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_class])
    rng.shuffle(chosen)
    return chosen


def run_pipeline():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_train = ImageFolder(TRAIN_DIR, transform=transform)
    full_test = ImageFolder(TEST_DIR, transform=transform)
    classes = full_train.classes

    ds_train = Subset(full_train, class_balanced_indices(full_train, PER_CLASS_TRAIN, rng))
    ds_test = Subset(full_test, class_balanced_indices(full_test, PER_CLASS_TEST, rng))

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

    # ---- pick one real test scan to trace through the whole network ------------------
    demo_path, demo_label_idx = full_test.samples[class_balanced_indices(full_test, 1, rng)[0]]
    demo_raw = Image.open(demo_path).convert("RGB")
    demo_tensor = transform(demo_raw)  # (3, 128, 128)

    activations = {}
    hooks = []
    layer_names = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2",
                    "conv3", "relu3", "pool3", "flatten"]
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

    # ---- preprocessing pipeline preview images ----------------------------------------
    prep_path, _ = full_train.samples[class_balanced_indices(full_train, 1, rng)[0]]
    prep_raw = Image.open(prep_path).convert("RGB")
    prep_resized = prep_raw.resize((IMG_SIZE, IMG_SIZE))
    prep_normed = transforms.functional.normalize(
        transforms.functional.to_tensor(prep_resized), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    prep_normed_vis = ((prep_normed * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    demo_raw.save(os.path.join(ASSETS_DIR, "raw.png"))
    prep_raw.save(os.path.join(ASSETS_DIR, "prep_raw.png"))
    prep_resized.save(os.path.join(ASSETS_DIR, "prep_resized.png"))
    Image.fromarray(prep_normed_vis).save(os.path.join(ASSETS_DIR, "prep_normed.png"))

    # ---- feature-map thumbnails (8 representative channels per stage) ----------------
    channel_ids = list(range(8))
    feature_map_paths = {}
    for stage in ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]:
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
    r0, c0 = 48, 48  # top-left corner of a 6x6 patch -> 4x4 valid-convolution output
    patch = demo_tensor[:, r0:r0 + 6, c0:c0 + 6]  # (3, 6, 6), normalized values
    patch_vis = (patch * 0.5 + 0.5).clamp(0, 1)  # back to [0, 1] for display only
    patch_rgb = patch_vis.permute(1, 2, 0).numpy()  # (6, 6, 3)

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
    }


DATA = run_pipeline()


# --------------------------------------------------------------------------------------
# 3. Scene
# --------------------------------------------------------------------------------------

class BrainTumorCNNStepByStep(Scene):
    def construct(self):
        self.show_title()
        self.show_data_pipeline()
        self.show_kernel_demo()
        self.show_feature_maps()
        self.show_flatten_and_classify()
        self.show_training_and_eval()

    # ---- 1. title ---------------------------------------------------------------------
    def show_title(self):
        title = Text("Brain Tumor MRI CNN — Convolution by Convolution", weight=BOLD).scale(0.5)
        subtitle = Text(
            "3→32→64→128 channels · 128×128 input · 4 classes · trained live on a small subset",
            font_size=22, color=GREY_B,
        ).next_to(title, DOWN)
        disclaimer = Text(
            "Illustrative only — not medical advice, do not self diagnose",
            font_size=18, color=RED,
        ).next_to(subtitle, DOWN, buff=0.3)
        group = VGroup(title, subtitle, disclaimer)
        self.play(FadeIn(group, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(group))

    # ---- 2. preprocessing pipeline ------------------------------------------------------
    def show_data_pipeline(self):
        heading = Text("1. Preprocessing pipeline", font_size=28, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        def img(path, label):
            im = ImageMobject(path).scale_to_fit_height(2.8)
            lbl = Text(label, font_size=18, color=GREY_B).next_to(im, DOWN, buff=0.15)
            return Group(im, lbl)

        raw = img(os.path.join(ASSETS_DIR, "prep_raw.png"), "raw MRI scan")
        resized = img(os.path.join(ASSETS_DIR, "prep_resized.png"), "resize -> 128x128")
        normed = img(os.path.join(ASSETS_DIR, "prep_normed.png"), "normalize (mean=0.5, std=0.5)")

        row = Group(raw, resized, normed).arrange(RIGHT, buff=1.1).shift(DOWN * 0.3)
        arrows = VGroup(*[
            Arrow(row[i][0].get_right(), row[i + 1][0].get_left(), buff=0.1, stroke_width=3)
            for i in range(2)
        ])

        self.play(FadeIn(raw))
        for nxt, arrow in zip(row[1:], arrows):
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
        patch_label = Text("6×6 crop of the real scan (normalized)", font_size=18, color=GREY_B)
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

        def window_of(i, j):
            idxs = [grid_index(i + di, j + dj) for di in range(3) for dj in range(3)]
            return VGroup(*[patch_squares[k] for k in idxs])

        highlight = SurroundingRectangle(window_of(0, 0), color=YELLOW, buff=0.02, stroke_width=5)
        self.play(Create(highlight), run_time=0.3)

        for i in range(4):
            for j in range(4):
                out_idx = i * 4 + j
                val = conv_grid[i, j]
                norm = (val - vmin) / (vmax - vmin + 1e-8)
                target_color = interpolate_color(BLUE, RED, norm)
                val_text = Text(f"{val:.1f}", font_size=14).move_to(out_squares[out_idx])

                new_highlight = SurroundingRectangle(window_of(i, j), color=YELLOW, buff=0.02, stroke_width=5)
                self.play(
                    Transform(highlight, new_highlight),
                    out_squares[out_idx].animate.set_fill(target_color, opacity=0.85),
                    FadeIn(val_text),
                    run_time=0.35,
                )
                self.wait(0.15)
        self.play(FadeOut(highlight))

        self.wait(0.5)
        note = Text("× 32 filters like this one, each scanning the full 128×128 scan",
                     font_size=20, color=GREY_B).next_to(eq, UP, buff=0.2)
        self.play(FadeIn(note))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not None])

    # ---- 4. real feature maps through all three conv blocks -----------------------------
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

        stage_row("conv1", "3. Conv block 1 — 32 filters (post-conv, pre-pool)", 32)
        stage_row("pool1", "MaxPool 2×2 — 128×128 → 64×64", 32)
        stage_row("conv2", "4. Conv block 2 — 64 filters", 64)
        stage_row("pool2", "MaxPool 2×2 — 64×64 → 32×32", 64)
        stage_row("conv3", "5. Conv block 3 — 128 filters", 128)
        stage_row("pool3", "MaxPool 2×2 — 32×32 → 16×16", 128)

    # ---- 5. flatten -> linear -> class probabilities -------------------------------------
    def show_flatten_and_classify(self):
        heading = Text("6. Flatten → Linear classifier", font_size=26, weight=BOLD).to_edge(UP)
        self.play(FadeIn(heading))

        thumbs = Group(*[ImageMobject(p).scale_to_fit_height(0.9) for p in DATA["feature_map_paths"]["pool3"]])
        thumbs.arrange_in_grid(rows=2, cols=4, buff=0.15).to_edge(LEFT, buff=1.0)
        self.play(FadeIn(thumbs))

        vector = Rectangle(width=0.4, height=3.2, color=GREEN, fill_opacity=0.3).next_to(thumbs, RIGHT, buff=1.2)
        vector_label = Text("128×16×16 = 32,768-d vector", font_size=16, color=GREY_B).next_to(vector, DOWN, buff=0.2)
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
        heading = Text("7. Training & evaluation", font_size=26, weight=BOLD).to_edge(UP)
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
            Text("Held-out test subset (macro-averaged):", font_size=20, weight=BOLD),
            Text(f"Precision:  {DATA['precision']:.2f}", font_size=22, color=GREEN),
            Text(f"Recall:       {DATA['recall']:.2f}", font_size=22, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT, buff=1.0)

        self.play(FadeIn(metrics, shift=LEFT * 0.3))
        self.wait(2)