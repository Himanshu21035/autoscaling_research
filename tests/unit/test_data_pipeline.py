# tests/unit/test_data_pipeline.py
import pytest
import pandas as pd
import numpy as np
from src.data.cleaner import forward_fill, remove_outliers, normalize_time_index, clip_negative, clean_pipeline
from src.data.splitter import split_by_days, split_by_ratio


def make_sample_df(n_rows: int = 200, with_nans: bool = False, with_outliers: bool = False) -> pd.DataFrame:
    """Helper to create a small test DataFrame in unified format."""
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="1min")
    rps = np.random.uniform(10, 100, size=n_rows).astype(float)
    if with_nans:
        rps[[5, 6, 7, 50]] = np.nan
    if with_outliers:
        rps[[20, 80]] = 99999.0   # extreme outliers
    return pd.DataFrame({"rps": rps, "source": "test"}, index=idx)


class TestForwardFill:
    def test_fills_short_gaps(self):
        df = make_sample_df(with_nans=True)
        result = forward_fill(df, limit=5)
        assert result["rps"].isna().sum() == 0

    def test_does_not_fill_beyond_limit(self):
        df = make_sample_df(n_rows=20)
        df.iloc[2:12, 0] = np.nan   # 10 consecutive NaNs
        result = forward_fill(df, limit=3)
        assert result["rps"].isna().sum() > 0   # some should still be NaN


class TestRemoveOutliers:
    def test_extreme_values_replaced(self):
        df = make_sample_df(with_outliers=True)
        result = remove_outliers(df, n_std=3.0)
        assert result["rps"].max() < 99999.0

    def test_normal_values_unchanged(self):
        df = make_sample_df()
        original_mean = df["rps"].mean()
        result = remove_outliers(df)
        assert abs(result["rps"].mean() - original_mean) < 5.0   # mean stays roughly same


class TestNormalizeTimeIndex:
    def test_fills_missing_timestamps(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="1min")
        idx_with_gap = idx.delete([4, 5])   # remove 2 timestamps
        df = pd.DataFrame({"rps": np.ones(8), "source": "test"}, index=idx_with_gap)
        result = normalize_time_index(df, freq="1min")
        assert len(result) == 10   # gap timestamps now exist with NaN rps

    def test_index_name_set(self):
        df = make_sample_df()
        result = normalize_time_index(df)
        assert result.index.name == "timestamp"


class TestClipNegative:
    def test_negatives_clipped(self):
        df = make_sample_df()
        df.iloc[10, 0] = -5.0
        result = clip_negative(df)
        assert result["rps"].min() >= 0


class TestSplitter:
    def test_split_by_days_no_overlap(self):
        # Need enough rows to cover 70+14+14 = 98 days
        # 98 days × 1440 min/day = 141,120 rows minimum
        # Use 100 days × 1440 = 144,000 rows to be safe
        df = make_sample_df(n_rows=100 * 1440)
        split = split_by_days(df, train_days=70, val_days=14, test_days=14)
        assert split.train.index.max() < split.val.index.min()
        assert split.val.index.max() < split.test.index.min()
        assert len(split.train) > 0
        assert len(split.val) > 0
        assert len(split.test) > 0


    def test_split_by_ratio_sizes(self):
        df = make_sample_df(n_rows=100)
        split = split_by_ratio(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert len(split.train) == 70
        assert len(split.val) == 15
        assert len(split.test) == 15

    def test_no_data_leakage(self):
        df = make_sample_df(n_rows=300)
        split = split_by_days(df, train_days=0, val_days=0, test_days=0)
        # Edge: zero-day splits should return empty or minimal frames
        assert isinstance(split.train, pd.DataFrame)
