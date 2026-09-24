# Temperature RNN

<p align="center">
  <img src="RNNStepByStep.gif" alt="RNN step-by-step animation">
</p>

A PyTorch RNN that forecasts the next day's minimum temperature from a sliding window of the previous 30 days, trained on daily minimum temperatures in Melbourne, Australia (1981-1990).

**Approach:** The series is split by time (train: 1981-1988, test: 1989-1990) to avoid look-ahead bias, then scaled with a `MinMaxScaler` fit only on the training data. A sliding window turns the series into supervised pairs — 30 days of temperature in, the 31st day's temperature as the target. A single-layer `nn.RNN` (hidden_size=64) followed by a linear output layer is trained for 50 epochs with Adam (lr=0.001, batch_size=64), then evaluated on the held-out test set with RMSE (~2.4°C).

**Files:**
- `rnn.ipynb` — data loading, preprocessing, sliding-window construction, model definition, training loop, and evaluation/visualisation
- `data.csv` — daily minimum temperatures, Melbourne, Australia (1981-1990)
- `animation.py` — Manim animation that runs a real forward pass through the model and shows, day by day, how the input and the recurrent hidden state combine across the 64 hidden neurons to produce a prediction (`manim -pqh animation.py RNNStepByStep`)
- `RNNStepByStep.gif` — rendered output of `animation.py`