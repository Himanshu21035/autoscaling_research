# src/metrics/window_buffer.py
import math
import threading
from collections import deque
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)

_FORECAST_CFG = CONFIG.get("forecasting", {})


class WindowBuffer:
    """
    Fixed-size sliding window of (timestamp, rps) observations.
    Used by forecasters to access recent history at each step.

    Thread-safe via a reentrant lock.
    """

    def __init__(
        self,
        maxlen: int = 60,
        min_ready_len: int | None = None,
    ):
        self.maxlen = maxlen
        # Read from config if not passed explicitly
        self.min_ready_len = (
            min_ready_len
            or _FORECAST_CFG.get("min_history_steps", 30)
        )
        self._timestamps: deque = deque(maxlen=maxlen)
        self._values: deque     = deque(maxlen=maxlen)
        self._lock = threading.RLock()

    # ── Public API ─────────────────────────────────────────────────────

    def push(self, timestamp, value: float):
        """
        Add one observation. Validates value before accepting.

        Raises:
            ValueError: if value is NaN, inf, or negative
        """
        if not math.isfinite(value):
            logger.warning(
                f"Rejected non-finite RPS value: {value} at {timestamp}"
            )
            return
        if value < 0:
            logger.warning(
                f"Rejected negative RPS value: {value} at {timestamp}"
            )
            return

        with self._lock:
            # Warn on duplicate timestamps — some forecasters break on them
            if self._timestamps and timestamp == self._timestamps[-1]:
                logger.warning(
                    f"Duplicate timestamp detected: {timestamp}. "
                    f"Overwriting previous value."
                )
                self._values[-1] = value
                return

            self._timestamps.append(timestamp)
            self._values.append(value)

    def to_series(self) -> pd.Series:
        """
        Returns buffer as pd.Series with sorted DatetimeIndex where possible.
        Falls back to RangeIndex if timestamps are not datetime-compatible.
        """
        with self._lock:
            timestamps = list(self._timestamps)
            values     = list(self._values)

        if not timestamps:
            return pd.Series([], dtype=float, name="rps")

        try:
            idx = pd.DatetimeIndex(timestamps)
            s = pd.Series(values, index=idx, name="rps", dtype=float)
            if not s.index.is_monotonic_increasing:
                s = s.sort_index()
            return s
        except (TypeError, ValueError):
            # Integer or mixed index — return as-is with RangeIndex
            return pd.Series(values, name="rps", dtype=float)

    def is_ready(self, min_len: int | None = None) -> bool:
        """True if enough history is available to forecast."""
        threshold = min_len if min_len is not None else self.min_ready_len
        with self._lock:
            return len(self._values) >= threshold

    def reset(self):
        """Clear all observations — use between simulation runs."""
        with self._lock:
            self._timestamps.clear()
            self._values.clear()
        logger.debug("WindowBuffer reset")

    def __len__(self):
        with self._lock:
            return len(self._values)