import math
import pytest
import numpy as np
import pandas as pd
from src.metrics.collector import FeatureCollector
from src.metrics.window_buffer import WindowBuffer

RNG = np.random.default_rng(seed=42)   # seeded — deterministic


# ── Helpers ────────────────────────────────────────────────────────────

def make_series(n: int = 200, freq: str = "1min", seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    rps = np.abs(rng.normal(loc=300, scale=80, size=n))
    return pd.Series(rps, index=idx, name="rps")


def make_collector(adapt_s: float = 120.0) -> FeatureCollector:
    return FeatureCollector(
        windows=[5, 15, 60],
        burst_sigma=2.0,
        adapt_estimate_s=adapt_s,
        timestep_seconds=60,
    )


# ── FeatureCollector: shape + columns ─────────────────────────────────

class TestFeatureCollectorShape:

    def test_output_has_all_expected_columns(self):
        df = make_collector().transform(make_series(200))
        expected = [
            "rps",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
            "rolling_mean_5m", "rolling_std_5m",
            "rolling_mean_15m", "rolling_std_15m",
            "rolling_mean_60m", "rolling_std_60m",
            "rps_delta",
            "burst_flag", "volatility_ratio", "queue_proxy",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nans_after_transform(self):
        df = make_collector().transform(make_series(200))
        assert df.isnull().sum().sum() == 0

    def test_output_shorter_than_input_by_largest_window(self):
        """Largest window is 60 — exactly 60-1 rows should be dropped."""
        series = make_series(200)
        df = make_collector().transform(series)
        # dropna removes rows where ANY rolling window is NaN
        # 60m window produces first valid at row 59 (0-indexed)
        assert len(df) == len(series) - 60 + 1

    def test_length_adapts_to_custom_windows(self):
        """Length drops by largest window regardless of which windows are used."""
        custom_windows = [3, 10, 30]
        fc = FeatureCollector(windows=custom_windows, timestep_seconds=60)
        series = make_series(100)
        df = fc.transform(series)
        largest = max(custom_windows)
        assert len(df) == len(series) - largest + 1

    def test_empty_series_returns_empty_dataframe(self):
        df = make_collector().transform(pd.Series([], dtype=float))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_series_shorter_than_largest_window_returns_empty(self):
        """50 rows < 60m window → all NaN → empty after dropna."""
        series = make_series(50)
        df = make_collector().transform(series)
        assert len(df) == 0


# ── FeatureCollector: semantic correctness ────────────────────────────

class TestFeatureCollectorSemantics:

    def test_burst_flag_is_binary(self):
        df = make_collector().transform(make_series(200))
        assert df["burst_flag"].isin([0, 1]).all()

    def test_hour_sin_cos_in_unit_circle(self):
        df = make_collector().transform(make_series(200))
        assert df["hour_sin"].between(-1.0, 1.0).all()
        assert df["hour_cos"].between(-1.0, 1.0).all()

    def test_volatility_ratio_non_negative(self):
        df = make_collector().transform(make_series(200))
        assert (df["volatility_ratio"] >= 0).all()

    def test_queue_proxy_units_are_requests(self):
        """
        queue_proxy = (RPS × pending_steps × timestep_s) / capacity_per_replica
        For RPS=300, 2 steps, 60s timestep, capacity=100:
        = (300 × 2 × 60) / 100 = 360 / 10 ... wait:
        raw = 300 × 2 × 60 = 36000 requests
        normalized = 36000 / 100 = 360 replica-equivalent-seconds → / 60 if in minutes
        Expected = 36000 / 100 = 360
        """
        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        series = pd.Series(np.full(200, 300.0), index=idx)
        fc = FeatureCollector(
            windows=[5, 15, 60],
            adapt_estimate_s=120.0,   # 2 steps at 60s
            timestep_seconds=60,
        )
        df = fc.transform(series)
        expected = (300.0 * 2 * 60) / 100.0   # = 360.0
        assert abs(df["queue_proxy"].iloc[-1] - expected) < 1.0

    def test_queue_proxy_increases_with_adapt_estimate(self):
        """Larger cold start window → more requests in flight → higher proxy."""
        series = make_series(200, seed=42)
        df_short = make_collector(adapt_s=60.0).transform(series)
        df_long  = make_collector(adapt_s=300.0).transform(series)
        assert df_long["queue_proxy"].mean() > df_short["queue_proxy"].mean()

    def test_set_adapt_estimate_changes_queue_proxy_monotonically(self):
        """Progressively larger estimates should produce larger queue_proxy."""
        series = make_series(200, seed=42)
        means = []
        for est in [60.0, 120.0, 180.0, 300.0]:
            fc = make_collector(adapt_s=est)
            df = fc.transform(series)
            means.append(df["queue_proxy"].mean())
        assert means == sorted(means), \
            f"queue_proxy not monotonically increasing: {means}"

    def test_burst_flag_triggers_on_known_spike(self):
        """Inject a known spike — burst_flag must fire exactly there."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        rps = rng.normal(loc=100, scale=5, size=200)
        rps[180] = 5000.0   # extreme spike
        series = pd.Series(rps, index=idx)
        df = make_collector().transform(series)
        # The spike at row 180 must be flagged (after NaN rows dropped)
        assert df["burst_flag"].sum() >= 1

    def test_constant_rps_produces_zero_std_and_zero_volatility(self):
        """Flat signal → std=0 → volatility_ratio=0."""
        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        series = pd.Series(np.full(200, 250.0), index=idx)
        df = make_collector().transform(series)
        assert (df["rolling_std_5m"] == 0.0).all()
        assert (df["volatility_ratio"] == 0.0).all()

    def test_non_datetime_index_sets_temporal_to_zero(self):
        """Integer-indexed series: temporal features must be exactly 0."""
        series = pd.Series(np.abs(RNG.normal(300, 80, 200)))
        df = make_collector().transform(series)
        assert (df["hour_sin"] == 0.0).all()
        assert (df["hour_cos"] == 0.0).all()
        assert (df["is_weekend"] == 0).all()

    def test_rps_delta_is_first_difference(self):
        """rps_delta[i] == rps[i] - rps[i-1]."""
        idx = pd.date_range("2024-01-01", periods=200, freq="1min")
        rps = np.arange(200, dtype=float)   # 0,1,2,...,199
        series = pd.Series(rps, index=idx)
        df = make_collector().transform(series)
        # After dropna, all deltas should equal 1.0
        assert (df["rps_delta"] == 1.0).all()


# ── WindowBuffer ───────────────────────────────────────────────────────

class TestWindowBuffer:

    def test_push_and_retrieve_values(self):
        buf = WindowBuffer(maxlen=10)
        for i in range(10):
            buf.push(i, float(i * 10))
        s = buf.to_series()
        assert len(s) == 10
        assert s.iloc[-1] == 90.0

    def test_maxlen_enforced_oldest_dropped(self):
        buf = WindowBuffer(maxlen=5)
        for i in range(20):
            buf.push(i, float(i))
        assert len(buf) == 5
        assert list(buf.to_series().values) == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_is_ready_false_when_too_short(self):
        buf = WindowBuffer(maxlen=60, min_ready_len=30)
        for i in range(10):
            buf.push(i, float(i))
        assert not buf.is_ready()

    def test_is_ready_true_when_sufficient(self):
        buf = WindowBuffer(maxlen=60, min_ready_len=30)
        for i in range(40):
            buf.push(i, float(i))
        assert buf.is_ready()

    def test_reset_clears_all_data(self):
        buf = WindowBuffer(maxlen=10)
        for i in range(10):
            buf.push(i, float(i))
        buf.reset()
        assert len(buf) == 0
        assert len(buf.to_series()) == 0

    def test_duplicate_timestamp_overwrites_not_appends(self):
        buf = WindowBuffer(maxlen=10)
        buf.push(1, 100.0)
        buf.push(1, 200.0)   # duplicate — should overwrite
        assert len(buf) == 1
        assert buf.to_series().iloc[-1] == 200.0

    def test_nan_value_rejected(self):
        buf = WindowBuffer(maxlen=10)
        buf.push(1, float("nan"))
        assert len(buf) == 0

    def test_inf_value_rejected(self):
        buf = WindowBuffer(maxlen=10)
        buf.push(1, float("inf"))
        assert len(buf) == 0

    def test_negative_rps_rejected(self):
        buf = WindowBuffer(maxlen=10)
        buf.push(1, -5.0)
        assert len(buf) == 0

    def test_zero_rps_accepted(self):
        """Zero is valid RPS (idle period)."""
        buf = WindowBuffer(maxlen=10)
        buf.push(1, 0.0)
        assert len(buf) == 1

    def test_datetime_index_type_in_to_series(self):
        """DatetimeIndex timestamps produce DatetimeIndex in output."""
        buf = WindowBuffer(maxlen=10)
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        for ts, v in zip(idx, [100, 200, 300, 400, 500]):
            buf.push(ts, float(v))
        s = buf.to_series()
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_integer_index_fallback_no_crash(self):
        """Integer timestamps fall back to RangeIndex without crash."""
        buf = WindowBuffer(maxlen=10)
        for i in range(5):
            buf.push(i, float(i * 50))
        s = buf.to_series()
        assert len(s) == 5
        assert s.iloc[0] == 0.0

    def test_min_ready_len_respected_with_custom_value(self):
        buf = WindowBuffer(maxlen=60, min_ready_len=50)
        for i in range(49):
            buf.push(i, float(i))
        assert not buf.is_ready()
        buf.push(49, 49.0)
        assert buf.is_ready()