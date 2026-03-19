# src/data/cleaner.py
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)


def forward_fill(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Fill gaps up to `limit` consecutive NaNs using forward fill."""
    n_missing_before = df["rps"].isna().sum()
    df = df.copy()
    df["rps"] = df["rps"].ffill(limit=limit)
    n_missing_after = df["rps"].isna().sum()
    logger.info(f"Forward fill: {n_missing_before} → {n_missing_after} NaNs "
                f"({n_missing_before - n_missing_after} filled)")
    return df


def remove_outliers(df: pd.DataFrame, n_std: float = 3.0) -> pd.DataFrame:
    """
    Replace values beyond n_std standard deviations from rolling mean with NaN,
    then forward fill. Uses rolling window to handle local distribution shifts.
    """
    df = df.copy()
    rolling_mean = df["rps"].rolling(window=60, center=True, min_periods=10).mean()
    rolling_std = df["rps"].rolling(window=60, center=True, min_periods=10).std()

    lower = rolling_mean - n_std * rolling_std
    upper = rolling_mean + n_std * rolling_std

    outlier_mask = (df["rps"] < lower) | (df["rps"] > upper)
    n_outliers = outlier_mask.sum()

    df.loc[outlier_mask, "rps"] = np.nan
    df["rps"] = df["rps"].ffill(limit=5).bfill(limit=5)

    logger.info(f"Outlier removal: {n_outliers} values replaced "
                f"({100 * n_outliers / len(df):.2f}% of data)")
    return df


def normalize_time_index(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Ensure the time index is complete and evenly spaced at `freq`.
    Missing timestamps get NaN rps (handled by forward_fill after this).
    """
    df = df.copy()
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "timestamp"
    if "source" in df.columns:
        df["source"] = df["source"].ffill()  # fill source column too
    logger.info(f"Time index normalized: {len(full_index)} timesteps at freq={freq}")
    return df


def clip_negative(df: pd.DataFrame) -> pd.DataFrame:
    """RPS can never be negative — clip any negatives to 0."""
    n_neg = (df["rps"] < 0).sum()
    if n_neg > 0:
        logger.warning(f"Clipping {n_neg} negative RPS values to 0")
        df = df.copy()
        df["rps"] = df["rps"].clip(lower=0)
    return df


def clean_pipeline(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Full cleaning pipeline in correct order:
    1. Normalize time index (fill gaps)
    2. Forward fill short gaps
    3. Remove statistical outliers
    4. Clip negatives
    """
    logger.info(f"Starting clean pipeline on {len(df)} rows from source={df['source'].iloc[0]}")
    df = normalize_time_index(df, freq=freq)
    df = forward_fill(df, limit=5)
    df = remove_outliers(df, n_std=3.0)
    df = clip_negative(df)

    # Final check: drop rows where rps still NaN after all attempts
    n_dropped = df["rps"].isna().sum()
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} rows with unfillable NaN rps")
        df = df.dropna(subset=["rps"])

    logger.info(f"Clean pipeline done: {len(df)} rows remaining, "
                f"mean_rps={df['rps'].mean():.2f}, "
                f"max_rps={df['rps'].max():.2f}")
    return df
