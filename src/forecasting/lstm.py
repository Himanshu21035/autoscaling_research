# src/forecasting/lstm.py
"""
LSTM Forecaster — Neural Baseline.

Improvements vs previous version:
  - seed parameter for full reproducibility
  - Guard: if sliding-window count < min_windows, falls back gracefully
  - training_summary() for experiment logging
  - _LSTMNet defined as a proper top-level class (importable without torch)
"""
import numpy as np
import torch
from src.forecasting.base import BaseForecaster
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("forecasting", {}).get("lstm", {})


class LSTMForecaster(BaseForecaster):

    def __init__(
        self,
        window_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        max_epochs: int | None = None,
        learning_rate: float | None = None,
        patience: int | None = None,
        min_series_length: int | None = None,
        min_windows: int = 20,
        seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__("LSTM")
        self.window_size       = window_size    or _CFG.get("window_size",    30)
        self.hidden_size       = hidden_size    or _CFG.get("hidden_size",    64)
        self.num_layers        = num_layers     or _CFG.get("num_layers",      2)
        self.max_epochs        = max_epochs     or _CFG.get("max_epochs",    100)
        self.learning_rate     = learning_rate  or _CFG.get("learning_rate", 1e-3)
        self.patience          = patience       or _CFG.get("patience",       10)
        self.min_series_length = (
            min_series_length or _CFG.get("min_series_length", 50)
        )
        self.min_windows = min_windows   # min sliding-window samples to train
        self.seed        = seed
        self.device      = device

        self._model = None
        self._scaler_mean: float = 0.0
        self._scaler_std:  float = 1.0
        self._series: list[float] = []
        self._training_summary: dict = {}

    # ── Fit ────────────────────────────────────────────────────────────

    def fit(self, series: np.ndarray) -> "LSTMForecaster":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        series = np.asarray(series, dtype=float)
        self._series = list(series)

        n_windows = len(series) - self.window_size
        if len(series) < self.min_series_length or n_windows < self.min_windows:
            logger.warning(
                f"LSTM: insufficient data "
                f"(series={len(series)}, windows={max(0, n_windows)}, "
                f"min_windows={self.min_windows}) — flat fallback"
            )
            self._fitted = True
            self._training_summary = {
                "status": "fallback", "reason": "insufficient_data"
            }
            return self

        # Normalise
        self._scaler_mean = float(np.mean(series))
        self._scaler_std  = float(np.std(series)) or 1.0
        norm = (series - self._scaler_mean) / self._scaler_std

        # Sliding windows
        X = np.array([norm[i: i + self.window_size]
                      for i in range(n_windows)])
        y = np.array([norm[i + self.window_size]
                      for i in range(n_windows)])

        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32)

        split = max(1, int(0.8 * len(X_t)))
        loader_train = DataLoader(
            TensorDataset(X_t[:split], y_t[:split]),
            batch_size=32, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        loader_val = DataLoader(
            TensorDataset(X_t[split:], y_t[split:]), batch_size=32
        )

        self._model = _build_lstm_net(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate
        )
        loss_fn = nn.MSELoss()
        best_val, best_epoch, patience_count = float("inf"), 0, 0
        train_losses, val_losses = [], []

        for epoch in range(self.max_epochs):
            self._model.train()
            epoch_loss = 0.0
            for xb, yb in loader_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self._model(xb).squeeze(-1)
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(loader_train))

            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self._model(xb).squeeze(-1)   # squeeze only last dim, not batch
                    # Guard: if pred is scalar (batch=1), unsqueeze
                    if pred.dim() == 0:
                        pred = pred.unsqueeze(0)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item()
            val_loss /= max(1, len(loader_val))
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(f"LSTM early stop: epoch={epoch}, best_val={best_val:.4f}")
                    break

        self._fitted = True
        self._training_summary = {
            "status":        "trained",
            "epochs_run":    epoch + 1,
            "best_epoch":    best_epoch,
            "best_val_loss": round(best_val, 6),
            "final_train_loss": round(train_losses[-1], 6),
            "series_len":    len(series),
            "n_windows":     n_windows,
            "seed":          self.seed,
        }
        logger.info(
            f"LSTM fitted: {len(series)} pts, {epoch+1} epochs, "
            f"best_val={best_val:.4f}"
        )
        return self

    # ── Predict ────────────────────────────────────────────────────────

    def predict(self, steps: int) -> np.ndarray:
        import torch

        self._require_fitted()
        if steps == 0:
            return np.array([])

        fallback = self._clip(
            np.full(steps, self._last_observed(self._series))
        )

        if self._model is None or len(self._series) < self.window_size:
            return fallback

        self._model.eval()
        norm = (
            np.array(self._series[-self.window_size:]) - self._scaler_mean
        ) / self._scaler_std

        window, preds = list(norm), []
        with torch.no_grad():
            for _ in range(steps):
                x = torch.tensor(
                    window[-self.window_size:], dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).to(self.device)
                out = self._model(x).item()
                preds.append(out)
                window.append(out)

        denorm = np.array(preds) * self._scaler_std + self._scaler_mean
        return self._clip(denorm)

    # ── Update / Reset ─────────────────────────────────────────────────

    def update(self, new_value: float) -> None:
        if self._fitted:
            self._series.append(new_value)

    def reset(self) -> None:
        self._model = None
        self._series = []
        self._scaler_mean = 0.0
        self._scaler_std  = 1.0
        self._fitted = False
        self._training_summary = {}

    # ── Diagnostics ────────────────────────────────────────────────────

    def training_summary(self) -> dict:
        return dict(self._training_summary)

    def metadata(self) -> dict:
        base = super().metadata()
        base.update(self._training_summary)
        return base
# Module-level builder — importable without torch at package load time
def _build_lstm_net(input_size: int, hidden_size: int, num_layers: int):
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return _Net()