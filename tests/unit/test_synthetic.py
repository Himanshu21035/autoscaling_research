# tests/unit/test_synthetic.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.synthetic import (
    generate_smooth,
    generate_bursty,
    generate_flash_crowd,
    generate_all,
    save_workload,
)

SEED = 42


# ── Shared Contract ────────────────────────────────────────────────────

def assert_workload_contract(series: pd.Series, name: str):
    assert isinstance(series, pd.Series),          f"{name}: must be pd.Series"
    assert isinstance(series.index, pd.DatetimeIndex), \
        f"{name}: must have DatetimeIndex"
    assert series.name == "rps",                   f"{name}: Series.name must be 'rps'"
    assert (series >= 0).all(),                    f"{name}: RPS must be non-negative"
    assert not series.isnull().any(),              f"{name}: must have no NaNs"
    assert len(series) > 0,                        f"{name}: must not be empty"
    # Strict: monotonically increasing AND unique timestamps
    assert series.index.is_monotonic_increasing,   f"{name}: index not sorted"
    assert series.index.is_unique,                 f"{name}: duplicate timestamps"
    # 1-minute frequency
    diffs = series.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta("1min")).all(),  f"{name}: not 1-minute frequency"


class TestWorkloadContract:

    def test_smooth_contract(self):
        assert_workload_contract(generate_smooth(n_days=2, seed=SEED), "smooth")

    def test_bursty_contract(self):
        assert_workload_contract(generate_bursty(n_days=2, seed=SEED), "bursty")

    def test_flash_crowd_contract(self):
        assert_workload_contract(generate_flash_crowd(n_days=7, seed=SEED), "flash_crowd")

    def test_all_workloads_deterministic(self):
        assert generate_smooth(seed=SEED).equals(generate_smooth(seed=SEED))
        assert generate_bursty(seed=SEED).equals(generate_bursty(seed=SEED))
        assert generate_flash_crowd(seed=SEED).equals(generate_flash_crowd(seed=SEED))

    def test_different_seeds_produce_different_output(self):
        assert not generate_smooth(seed=0).equals(generate_smooth(seed=99))
        assert not generate_bursty(seed=0).equals(generate_bursty(seed=99))

    def test_generate_all_uses_independent_seeds(self):
        """Smooth and bursty from generate_all must differ from each other."""
        workloads = generate_all(seed=SEED)
        assert not workloads["smooth"].equals(workloads["bursty"])
        assert not workloads["smooth"].equals(workloads["flash_crowd"])


# ── Smooth ─────────────────────────────────────────────────────────────

class TestSmoothWorkload:

    def test_correct_length(self):
        assert len(generate_smooth(n_days=3, seed=SEED)) == 3 * 1440

    def test_mean_near_base_rps(self):
        s = generate_smooth(n_days=7, base_rps=300.0, seed=SEED)
        assert abs(s.mean() - 300.0) < 20.0

    def test_peak_near_base_plus_amplitude(self):
        s = generate_smooth(n_days=7, base_rps=300.0, amplitude=150.0, seed=SEED)
        assert s.max() > 400.0

    def test_cv_below_threshold(self):
        """Smooth must be less volatile than bursty — CV < 0.6."""
        s = generate_smooth(n_days=7, seed=SEED)
        assert s.std() / s.mean() < 0.6


# ── Bursty ─────────────────────────────────────────────────────────────

class TestBurstyWorkload:

    def test_correct_length(self):
        assert len(generate_bursty(n_days=3, seed=SEED)) == 3 * 1440

    def test_std_higher_than_smooth(self):
        assert generate_bursty(n_days=7, seed=SEED).std() > \
               generate_smooth(n_days=7, seed=SEED).std()

    def test_has_spikes_above_3x_median(self):
        s = generate_bursty(n_days=7, seed=SEED)
        assert (s > 3 * s.median()).any()

    def test_max_substantially_above_mean(self):
        s = generate_bursty(n_days=7, seed=SEED)
        assert s.max() > 5 * s.mean()

    def test_cv_higher_than_smooth(self):
        smooth = generate_smooth(n_days=7, seed=SEED)
        bursty = generate_bursty(n_days=7, seed=SEED)
        assert bursty.std() / bursty.mean() > smooth.std() / smooth.mean()


# ── Flash Crowd ────────────────────────────────────────────────────────

class TestFlashCrowdWorkload:

    def test_correct_length(self):
        assert len(generate_flash_crowd(n_days=7, seed=SEED)) == 7 * 1440

    def test_spike_magnitude(self):
        s = generate_flash_crowd(
            n_days=7, base_rps=150.0, spike_multiplier=10.0, seed=SEED
        )
        assert s.max() >= 1400.0

    def test_spike_is_localized(self):
        """Less than 10% of steps should be above 2× baseline."""
        s = generate_flash_crowd(n_days=7, base_rps=150.0, seed=SEED)
        assert (s > 300.0).mean() < 0.10

    def test_pre_spike_is_flat(self):
        s = generate_flash_crowd(
            n_days=7, spike_start_day=3, spike_start_hour=14, seed=SEED
        )
        pre_spike = s.iloc[:3 * 1440]
        assert pre_spike.std() / pre_spike.mean() < 0.3

    def test_spike_at_correct_position(self):
        """Spike maximum must fall within the configured spike window."""
        base, mult = 150.0, 10.0
        s = generate_flash_crowd(
            n_days=7, base_rps=base, spike_multiplier=mult,
            spike_start_day=3, spike_start_hour=14,
            spike_duration_minutes=90, seed=SEED,
        )
        spike_start = 3 * 1440 + 14 * 60
        spike_end   = spike_start + 90
        assert s.iloc[spike_start:spike_end].max() >= base * mult * 0.95


# ── Parameter Validation ───────────────────────────────────────────────

class TestParameterValidation:

    def test_smooth_negative_base_rps_raises(self):
        with pytest.raises(ValueError, match="base_rps"):
            generate_smooth(base_rps=-1.0)

    def test_smooth_zero_base_rps_raises(self):
        with pytest.raises(ValueError, match="base_rps"):
            generate_smooth(base_rps=0.0)

    def test_smooth_n_days_zero_raises(self):
        with pytest.raises(ValueError, match="n_days"):
            generate_smooth(n_days=0)

    def test_bursty_spike_probability_above_1_raises(self):
        with pytest.raises(ValueError, match="spike_probability"):
            generate_bursty(spike_probability=1.5)

    def test_bursty_spike_probability_zero_raises(self):
        with pytest.raises(ValueError, match="spike_probability"):
            generate_bursty(spike_probability=0.0)

    def test_bursty_inverted_multiplier_range_raises(self):
        with pytest.raises(ValueError, match="spike_multiplier_range"):
            generate_bursty(spike_multiplier_range=(8.0, 3.0))

    def test_flash_crowd_spike_day_exceeds_n_days_raises(self):
        with pytest.raises(ValueError, match="spike_start_day"):
            generate_flash_crowd(n_days=3, spike_start_day=5)

    def test_flash_crowd_invalid_hour_raises(self):
        with pytest.raises(ValueError, match="spike_start_hour"):
            generate_flash_crowd(spike_start_hour=25)

    def test_flash_crowd_negative_base_rps_raises(self):
        with pytest.raises(ValueError, match="base_rps"):
            generate_flash_crowd(base_rps=-50.0)


# ── Save / Load Roundtrip ──────────────────────────────────────────────

class TestSaveAndLoad:

    def test_csv_roundtrip_values(self, tmp_path):
        original = generate_smooth(n_days=1, seed=SEED)
        save_workload(original, "smooth", out_dir=tmp_path)
        loaded = pd.read_csv(
            tmp_path / "smooth.csv", index_col="timestamp", parse_dates=True
        )["rps"]

        # CSV reload drops freq metadata — compare values and index only
        np.testing.assert_allclose(original.values, loaded.values, rtol=1e-5)
        assert list(original.index) == list(loaded.index)
        assert len(original) == len(loaded)
    
    def test_generate_all_creates_three_files(self, tmp_path):
        generate_all(seed=SEED, out_dir=tmp_path)
        names = {f.stem for f in tmp_path.glob("*.csv")}
        assert names == {"smooth", "bursty", "flash_crowd"}

    def test_generate_all_saved_content_passes_contract(self, tmp_path):
        """Files must reload as valid workloads — not just exist."""
        generate_all(seed=SEED, out_dir=tmp_path)
        for name in ["smooth", "bursty", "flash_crowd"]:
            loaded = pd.read_csv(
                tmp_path / f"{name}.csv",
                index_col="timestamp", parse_dates=True,
            )["rps"]
            assert (loaded >= 0).all(), f"{name}: negative RPS after reload"
            assert not loaded.isnull().any(), f"{name}: NaN after reload"
            assert len(loaded) == 7 * 1440, f"{name}: wrong length after reload"