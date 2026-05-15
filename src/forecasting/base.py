"""
Abstract base for all forecasters.

Research extensions beyond basic contract:
  - fit_latency_ms / predict_latency_ms: measured automatically on each call
  - metadata(): standardised dict for experiment logging
  - timed_fit() / timed_predict(): explicit latency-capturing wrappers
"""
import time
from abc import ABC, abstractmethod
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


class BaseForecaster(ABC):

    def __init__(self, name: str):
        self.name = name
        self._fitted            = False
        self.fit_latency_ms:     float | None = None   # set after fit()
        self.predict_latency_ms: float | None = None   # set after last predict()
        self._fit_series_len:    int   = 0

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def fit(self, series: np.ndarray) -> "BaseForecaster":
        """Train on historical RPS series (chronological order)."""

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Forecast `steps` values ahead. Returns np.ndarray >= 0."""

    @abstractmethod
    def update(self, new_value: float) -> None:
        """Incorporate one new observation (online update)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all fitted state. Called between experiments."""

    # ── Timed wrappers ─────────────────────────────────────────────────

    def timed_fit(self, series: np.ndarray) -> "BaseForecaster":
        """fit() with automatic latency capture in fit_latency_ms."""
        t0 = time.perf_counter()
        result = self.fit(series)
        self.fit_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._fit_series_len = len(series)
        logger.debug(
            f"{self.name}: fit {len(series)} pts in "
            f"{self.fit_latency_ms:.1f} ms"
        )
        return result

    def timed_predict(self, steps: int) -> np.ndarray:
        """predict() with automatic latency capture in predict_latency_ms."""
        t0 = time.perf_counter()
        result = self.predict(steps)
        self.predict_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ── Metadata ───────────────────────────────────────────────────────

    def metadata(self) -> dict:
        """
        Standardised metadata dict for experiment logging / CSV export.
        All forecasters emit the same keys so results are directly comparable.
        """
        return {
            "name":               self.name,
            "fitted":             self._fitted,
            "fit_series_len":     self._fit_series_len,
            "fit_latency_ms":     round(self.fit_latency_ms,     2)
                                  if self.fit_latency_ms     is not None else None,
            "predict_latency_ms": round(self.predict_latency_ms, 2)
                                  if self.predict_latency_ms is not None else None,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}: predict() called before fit(). "
                "Call fit(series) first."
            )

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        """RPS cannot be negative."""
        return np.maximum(np.asarray(arr, dtype=float), 0.0)

    def _last_observed(self, series_cache: list) -> float:
        """Safe last-value fallback — never returns 0 for non-empty series."""
        return float(series_cache[-1]) if series_cache else 0.0

    def __repr__(self):
        return f"{self.name}(fitted={self._fitted})"