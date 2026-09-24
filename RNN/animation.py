"""
Manim animation that walks through exactly what TemperatureRNN (see rnn.ipynb) does,
one time step at a time: how the input at day t and the memory carried over from
day t-1 combine inside the 64 hidden neurons, and how the final hidden state is
turned into a predicted temperature.

The architecture mirrors the model defined in the notebook exactly:
    input_size=1, hidden_size=64, output_size=1, nn.RNN(batch_first=True), train_window=30

The notebook never saves trained weights to disk, so this script initialises the
same architecture with a fixed random seed (purely illustrative activations) and
runs one real forward pass on a real 30-day window pulled from data.csv. If you
later save trained weights (torch.save(model.state_dict(), 'RNN/model.pt')), this
script will automatically pick them up and animate the real, trained behaviour.

Render with, e.g.:
    manim -pql RNN/animation.py RNNStepByStep      # quick draft, 480p
    manim -pqh RNN/animation.py RNNStepByStep      # high quality, 1080p
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from manim import *

# --------------------------------------------------------------------------------------
# 1. Same model definition as the notebook
# --------------------------------------------------------------------------------------

INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
TRAIN_WINDOW = 30

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data.csv")
CHECKPOINT_PATH = os.path.join(HERE, "model.pt")


class TemperatureRNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)
        pred = self.linear(out[:, -1, :])
        return pred, out  # out: (batch, seq_len, hidden_size) -> hidden state at every step


# --------------------------------------------------------------------------------------
# 2. Load real data, build one real 30-day window, run one real forward pass
# --------------------------------------------------------------------------------------

def load_window_and_run_model(window_start_index=3000, seed=42):
    """Returns per-step hidden states, per-step inputs, the prediction and the target,
    all computed from a real forward pass through TemperatureRNN."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

    train = df.loc[:"1988"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train["Temp"].values.reshape(-1, 1))

    window_raw = df["Temp"].values[window_start_index: window_start_index + TRAIN_WINDOW]
    target_raw = df["Temp"].values[window_start_index + TRAIN_WINDOW]

    window_norm = scaler.transform(window_raw.reshape(-1, 1)).flatten()

    model = TemperatureRNN()
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        trained = True
    else:
        trained = False

    model.eval()
    x = torch.FloatTensor(window_norm).view(1, TRAIN_WINDOW, 1)

    with torch.no_grad():
        pred, hidden_seq = model(x)

    pred_c = scaler.inverse_transform(pred.numpy())[0, 0]

    return {
        "inputs_norm": window_norm,                      # (30,) values fed to the network, in [-1, 1]
        "inputs_c": window_raw,                           # (30,) same values in real Celsius
        "hidden_seq": hidden_seq.numpy()[0],               # (30, 64) hidden state after each day
        "prediction_c": pred_c,
        "target_c": target_raw,
        "trained": trained,
    }


DATA = load_window_and_run_model()

# --------------------------------------------------------------------------------------
# 3. Colour helper: map a hidden-neuron activation in [-1, 1] to a colour
# --------------------------------------------------------------------------------------

def activation_color(value: float) -> str:
    """tanh output in [-1, 1] -> blue (very negative) .. grey (0) .. red (very positive)."""
    value = float(np.clip(value, -1, 1))
    if value >= 0:
        return interpolate_color(WHITE, RED, value)
    return interpolate_color(WHITE, BLUE, -value)


GRID_ROWS, GRID_COLS = 8, 8  # 8x8 = 64, matches hidden_size exactly


class RNNStepByStep(Scene):
    def construct(self):
        self.show_title()
        input_node, hidden_grid, output_node, hh_loop, ih_arrow, ho_arrow = self.build_diagram()
        self.walk_through_time(input_node, hidden_grid, hh_loop, ih_arrow)
        self.show_prediction(hidden_grid, output_node, ho_arrow)

    # ---- scene 1: title -------------------------------------------------------------
    def show_title(self):
        title = Text("Temperature RNN — Neuron by Neuron", weight=BOLD).scale(0.8)
        subtitle = Text(
            f"input_size=1  ·  hidden_size=64  ·  window=30 days"
            f"  ·  weights: {'trained checkpoint' if DATA['trained'] else 'randomly initialised (illustrative)'}",
            font_size=22,
            color=GREY_B,
        )
        subtitle.next_to(title, DOWN)
        group = VGroup(title, subtitle)
        self.play(FadeIn(group, shift=UP * 0.3))
        self.wait(1)
        self.play(FadeOut(group))

    # ---- build the persistent diagram -----------------------------------------------
    def build_diagram(self):
        input_node = Circle(radius=0.35, color=YELLOW, fill_opacity=0.3)
        input_label = Text("x_t", font_size=24).move_to(input_node)
        input_group = VGroup(input_node, input_label).to_edge(LEFT).shift(UP * 0.5)

        dots = VGroup(*[
            Dot(radius=0.09, color=WHITE, fill_opacity=1)
            for _ in range(GRID_ROWS * GRID_COLS)
        ])
        dots.arrange_in_grid(rows=GRID_ROWS, cols=GRID_COLS, buff=0.22)
        dots.scale(1.0).move_to(ORIGIN).shift(UP * 0.3)
        grid_label = Text("hidden state h_t  (64 units)", font_size=22, color=GREY_B)
        grid_label.next_to(dots, DOWN, buff=0.35)
        hidden_group = VGroup(dots, grid_label)

        output_node = Circle(radius=0.35, color=GREEN, fill_opacity=0.3)
        output_label = Text("ŷ", font_size=26).move_to(output_node)
        output_group = VGroup(output_node, output_label).to_edge(RIGHT).shift(UP * 0.5)

        ih_arrow = Arrow(
            input_group.get_right(), dots.get_left(), buff=0.15, color=YELLOW, stroke_width=3
        )
        ih_label = Text("W_ih", font_size=20, color=YELLOW).next_to(ih_arrow, UP, buff=0.1)

        hh_loop = CurvedArrow(
            dots.get_top() + LEFT * 0.6, dots.get_top() + RIGHT * 0.6,
            angle=-TAU / 3, color=PURPLE, stroke_width=3,
        )
        hh_loop.next_to(dots, UP, buff=0.5)
        hh_label = Text("W_hh  (memory of h_{t-1})", font_size=20, color=PURPLE)
        hh_label.next_to(hh_loop, UP, buff=0.1)

        ho_arrow = Arrow(
            dots.get_right(), output_group.get_left(), buff=0.15, color=GREEN, stroke_width=3
        )
        ho_label = Text("W_out", font_size=20, color=GREEN).next_to(ho_arrow, UP, buff=0.1)
        ho_arrow.set_opacity(0)
        ho_label.set_opacity(0)

        eq = Text(
            "h_t = tanh( W_ih · x_t  +  W_hh · h_{t-1}  +  b )",
            font_size=24,
        ).to_edge(DOWN, buff=0.4)

        self.play(
            FadeIn(input_group),
            FadeIn(hidden_group),
            FadeIn(output_group),
        )
        self.play(
            GrowArrow(ih_arrow), FadeIn(ih_label),
            Create(hh_loop), FadeIn(hh_label),
        )
        self.play(FadeIn(ho_arrow.set_opacity(0)), FadeIn(ho_label.set_opacity(0)))
        self.play(Write(eq))
        self.wait(0.5)

        self.ih_label, self.hh_label, self.ho_label, self.eq = ih_label, hh_label, ho_label, eq
        self.input_label_obj, self.output_label_obj = input_label, output_label
        self.input_group, self.output_group = input_group, output_group

        return input_group, dots, output_group, hh_loop, ih_arrow, ho_arrow

    # ---- scene 2: step through the 30 days -------------------------------------------
    def walk_through_time(self, input_group, dots, hh_loop, ih_arrow):
        day_counter = Text("Day 1 / 30", font_size=24).to_corner(UR)
        self.play(FadeIn(day_counter))

        detailed_steps = set(range(1, 4)) | {TRAIN_WINDOW}  # detailed: first 3 + last day
        input_value_label = None

        for t in range(1, TRAIN_WINDOW + 1):
            x_val = DATA["inputs_norm"][t - 1]
            x_celsius = DATA["inputs_c"][t - 1]
            hidden_vals = DATA["hidden_seq"][t - 1]

            new_counter = Text(f"Day {t} / 30", font_size=24).to_corner(UR)

            if t in detailed_steps:
                new_input_label = Text(f"{x_celsius:.1f}°C", font_size=20, color=YELLOW)
                new_input_label.next_to(input_group, DOWN, buff=0.15)

                anims = [Transform(day_counter, new_counter)]
                if input_value_label is None:
                    anims.append(FadeIn(new_input_label))
                else:
                    anims.append(Transform(input_value_label, new_input_label))
                input_value_label = input_value_label or new_input_label

                self.play(*anims, run_time=0.4)
                self.play(Indicate(input_group, color=YELLOW, scale_factor=1.15), run_time=0.4)
                self.play(ShowPassingFlash(
                    ih_arrow.copy().set_color(YELLOW).set_stroke(width=6), time_width=0.6
                ), run_time=0.5)

                if t > 1:
                    self.play(ShowPassingFlash(
                        hh_loop.copy().set_color(PURPLE).set_stroke(width=6), time_width=0.6
                    ), run_time=0.5)
                else:
                    note = Text("h_0 = 0  (no memory yet)", font_size=20, color=GREY_B)
                    note.next_to(hh_loop, UP, buff=0.35)
                    self.play(FadeIn(note), run_time=0.3)
                    self.play(FadeOut(note), run_time=0.3)

                target_colors = [activation_color(v) for v in hidden_vals]
                self.play(
                    AnimationGroup(*[
                        dot.animate.set_color(c).set_fill(opacity=0.4 + 0.5 * abs(float(v)))
                        for dot, c, v in zip(dots, target_colors, hidden_vals)
                    ], lag_ratio=0.01),
                    run_time=0.6,
                )
                self.wait(0.15)
            else:
                if t == 4:
                    ff = Text("⏩ fast-forwarding through the remaining days …",
                              font_size=22, color=GREY_B)
                    ff.to_edge(DOWN, buff=1.0)
                    self.play(FadeIn(ff), run_time=0.3)
                    self.ff_label = ff

                new_input_label = Text(f"{x_celsius:.1f}°C", font_size=20, color=YELLOW)
                new_input_label.next_to(input_group, DOWN, buff=0.15)

                target_colors = [activation_color(v) for v in hidden_vals]
                self.play(
                    Transform(day_counter, new_counter),
                    Transform(input_value_label, new_input_label),
                    AnimationGroup(*[
                        dot.animate.set_color(c).set_fill(opacity=0.4 + 0.5 * abs(float(v)))
                        for dot, c, v in zip(dots, target_colors, hidden_vals)
                    ], lag_ratio=0.0),
                    run_time=0.08,
                )

            if t == TRAIN_WINDOW - 1 and hasattr(self, "ff_label"):
                self.play(FadeOut(self.ff_label), run_time=0.3)

        self.day_counter = day_counter
        if input_value_label is not None:
            self.input_value_label = input_value_label

    # ---- scene 3: final hidden state -> linear layer -> prediction -------------------
    def show_prediction(self, dots, output_group, ho_arrow):
        self.play(
            Circumscribe(dots, color=WHITE, buff=0.15),
            self.ho_label.animate.set_opacity(1),
        )
        self.play(GrowArrow(ho_arrow.set_opacity(1)))

        pred_c = DATA["prediction_c"]
        target_c = DATA["target_c"]

        pred_label = Text(f"{pred_c:.1f}°C", font_size=22, color=GREEN)
        pred_label.next_to(output_group, DOWN, buff=0.15)
        self.play(Indicate(output_group, color=GREEN, scale_factor=1.2), FadeIn(pred_label))

        result = VGroup(
            Text(f"Predicted:  {pred_c:.1f}°C", font_size=26, color=GREEN),
            Text(f"Actual:      {target_c:.1f}°C", font_size=26, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN, buff=0.5)

        self.play(FadeOut(self.eq), FadeIn(result))
        self.wait(2)