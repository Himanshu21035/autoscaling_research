# src/data/splitter.py
import pandas as pd
from dataclasses import dataclass
from src.logger import get_logger

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
