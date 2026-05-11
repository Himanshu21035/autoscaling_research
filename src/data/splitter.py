# src/data/splitter.py
"""
Train / validation / test splitter for time series.

Strict temporal split — no shuffling, no leakage.

  |────── train ──────|── val ──|── test ──|
  0                  70%       80%        100%

Returns numpy arrays (rps values only) for clean forecaster API.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_days: float
    val_days: float
    test_days: float


def split_by_days(
    df: pd.DataFrame,
    train_days: int = 60,
    val_days: int = 14,
    test_days: int = 14,
) -> DataSplit:
    """
    Chronological split — never shuffle time series data.
    Default: 60 train / 14 val / 14 test days (for BurstGPT's 121 days).
    Remaining days after test are discarded.
    """
    df = df.sort_index()
    start = df.index.min()

    train_end = start + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=val_days)
    test_end = val_end + pd.Timedelta(days=test_days)

    train = df[df.index < train_end]
    val = df[(df.index >= train_end) & (df.index < val_end)]
    test = df[(df.index >= val_end) & (df.index < test_end)]

    actual_train_days = (train.index.max() - train.index.min()).total_seconds() / 86400
    actual_val_days = (val.index.max() - val.index.min()).total_seconds() / 86400
    actual_test_days = (test.index.max() - test.index.min()).total_seconds() / 86400

    logger.info(
        f"Split complete:\n"
        f"  Train: {len(train):,} rows ({actual_train_days:.1f} days) "
        f"[{train.index.min()} → {train.index.max()}]\n"
        f"  Val:   {len(val):,} rows ({actual_val_days:.1f} days) "
        f"[{val.index.min()} → {val.index.max()}]\n"
        f"  Test:  {len(test):,} rows ({actual_test_days:.1f} days) "
        f"[{test.index.min()} → {test.index.max()}]"
    )

    return DataSplit(
        train=train, val=val, test=test,
        train_days=actual_train_days,
        val_days=actual_val_days,
        test_days=actual_test_days,
    )


def split_by_ratio(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DataSplit:
    """Alternative: split by percentage of total data (for shorter datasets like Azure 10-day)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end_idx = int(n * train_ratio)
    val_end_idx = train_end_idx + int(n * val_ratio)

    train = df.iloc[:train_end_idx]
    val = df.iloc[train_end_idx:val_end_idx]
    test = df.iloc[val_end_idx:]

    logger.info(f"Ratio split: train={len(train)}, val={len(val)}, test={len(test)}")
    return DataSplit(
        train=train, val=val, test=test,
        train_days=(train.index.max() - train.index.min()).total_seconds() / 86400,
        val_days=(val.index.max() - val.index.min()).total_seconds() / 86400,
        test_days=(test.index.max() - test.index.min()).total_seconds() / 86400,
    )
@dataclass
class TraceSplit:
    train:     np.ndarray
    val:       np.ndarray
    test:      np.ndarray
    train_pct: float
    val_pct:   float
    test_pct:  float

    def __post_init__(self):
        total = len(self.train) + len(self.val) + len(self.test)
        logger.info(
            f"TraceSplit: total={total} | "
            f"train={len(self.train)} ({self.train_pct:.0%}) | "
            f"val={len(self.val)} ({self.val_pct:.0%}) | "
            f"test={len(self.test)} ({self.test_pct:.0%})"
        )

    @property
    def train_val(self) -> np.ndarray:
        """train + val concatenated — for final model fit before test."""
        return np.concatenate([self.train, self.val])


def split(
    series: np.ndarray | pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.10,
) -> TraceSplit:
    if isinstance(series, pd.DataFrame):
        series = series["rps"].to_numpy(dtype=float)

    series = np.asarray(series, dtype=float)
    n = len(series)

    if n < 10:
        raise ValueError(f"Series too short to split: {n} steps")
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1, "
            f"got {train_frac + val_frac}"
        )

    # Compute boundaries with round() to avoid float accumulation error
    # e.g. int(1000 * (0.70 + 0.10)) = int(799.999...) = 799 — wrong
    i_train = round(n * train_frac)
    i_val   = i_train + round(n * val_frac)   # avoids cumulative float error

    test_frac = 1.0 - train_frac - val_frac
    return TraceSplit(
        train     = series[:i_train],
        val       = series[i_train:i_val],
        test      = series[i_val:],
        train_pct = train_frac,
        val_pct   = val_frac,
        test_pct  = test_frac,
    )
