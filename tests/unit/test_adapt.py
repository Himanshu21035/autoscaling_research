# tests/unit/test_adapt.py
import math
import pytest
from src.simulator.adapt import ADAPTTracker


def make_adapt(**kwargs) -> ADAPTTracker:
    defaults = dict(
        alpha=0.3, cold_start_s=120.0,
        cold_start_min_s=30.0, cold_start_max_s=600.0,
        epsilon_steps=1, timestep_seconds=60,
    )
    defaults.update(kwargs)
    return ADAPTTracker(**defaults)


# ── EWMA Correctness ───────────────────────────────────────────────────

class TestEWMA:

    def test_single_observation_partial_update(self):
        a = make_adapt(alpha=0.3, cold_start_s=120.0)
        est = a.observe(180.0)
        assert abs(est - (0.3 * 180.0 + 0.7 * 120.0)) < 1e-6

    def test_repeated_same_observation_converges(self):
        a = make_adapt(alpha=0.3, cold_start_s=120.0)
        for _ in range(200):
            a.observe(240.0)
        assert abs(a.estimate_s - 240.0) < 1.0

    def test_estimate_moves_toward_observations(self):
        a = make_adapt(alpha=0.5, cold_start_s=60.0)
        for _ in range(10):
            a.observe(300.0)
        assert a.estimate_s > 60.0

    def test_estimate_decreases_with_lower_observations(self):
        a = make_adapt(alpha=0.5, cold_start_s=300.0)
        for _ in range(10):
            a.observe(60.0)
        assert a.estimate_s < 300.0

    def test_alpha_1_always_takes_latest_observation(self):
        a = make_adapt(alpha=1.0, cold_start_s=120.0)
        a.observe(250.0)
        assert a.estimate_s == 250.0
        a.observe(80.0)
        assert a.estimate_s == 80.0

    def test_higher_alpha_reacts_faster(self):
        fast = make_adapt(alpha=0.8, cold_start_s=120.0)
        slow = make_adapt(alpha=0.1, cold_start_s=120.0)
        for _ in range(5):
            fast.observe(300.0)
            slow.observe(300.0)
        assert fast.estimate_s > slow.estimate_s

    def test_observation_count_increments(self):
        a = make_adapt()
        assert a.n_observations == 0
        a.observe(120.0)
        a.observe(150.0)
        assert a.n_observations == 2


# ── Timestamp-Pair API ─────────────────────────────────────────────────

class TestObserveEvent:

    def test_observe_event_computes_duration(self):
        """observe_event(100, 220) == observe(120)"""
        a1 = make_adapt(alpha=0.5, cold_start_s=120.0)
        a2 = make_adapt(alpha=0.5, cold_start_s=120.0)
        a1.observe_event(t_requested=100.0, t_ready=220.0)
        a2.observe(120.0)
        assert abs(a1.estimate_s - a2.estimate_s) < 1e-9

    def test_observe_event_invalid_order_ignored(self):
        a = make_adapt()
        est_before = a.estimate_s
        a.observe_event(t_requested=200.0, t_ready=100.0)   # ready < requested
        assert a.estimate_s == est_before
        assert a.n_observations == 0

    def test_observe_event_equal_timestamps_ignored(self):
        a = make_adapt()
        est_before = a.estimate_s
        a.observe_event(100.0, 100.0)
        assert a.estimate_s == est_before


# ── Bounds + Validation ────────────────────────────────────────────────

class TestBoundsAndValidation:

    def test_observation_clipped_to_max(self):
        a = make_adapt(cold_start_max_s=600.0, alpha=1.0)
        a.observe(9999.0)
        assert a.estimate_s == 600.0

    def test_observation_clipped_to_min(self):
        a = make_adapt(cold_start_min_s=30.0, alpha=1.0)
        a.observe(1.0)
        assert a.estimate_s == 30.0

    def test_clipped_observation_counted_but_bounded(self):
        """Clipped observation still increments n_observations."""
        a = make_adapt(cold_start_max_s=600.0, alpha=1.0)
        a.observe(9999.0)
        assert a.n_observations == 1
        assert a.estimate_s == 600.0   # bounded, not 9999

    def test_estimate_near_max_clipped_correctly(self):
        """Even when estimate is already at max, observation is clipped."""
        a = make_adapt(cold_start_s=590.0, cold_start_max_s=600.0, alpha=0.5)
        a.observe(99999.0)
        assert a.estimate_s <= 600.0

    def test_non_positive_observation_ignored(self):
        a = make_adapt(cold_start_s=120.0)
        est_before = a.estimate_s
        a.observe(0.0)
        a.observe(-50.0)
        assert a.estimate_s == est_before
        assert a.n_observations == 0

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            make_adapt(alpha=0.0)

    def test_alpha_above_1_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            make_adapt(alpha=1.1)

    def test_min_above_max_raises(self):
        with pytest.raises(ValueError, match="cold_start_max_s"):
            make_adapt(cold_start_min_s=300.0, cold_start_max_s=100.0)

    def test_prior_below_min_raises(self):
        with pytest.raises(ValueError, match="cold_start_s"):
            make_adapt(cold_start_s=10.0, cold_start_min_s=30.0)

    def test_prior_above_max_raises(self):
        with pytest.raises(ValueError, match="cold_start_s"):
            make_adapt(cold_start_s=700.0, cold_start_max_s=600.0)


# ── Diagnostics / Summary ─────────────────────────────────────────────

class TestSummary:

    def test_summary_keys_present(self):
        a = make_adapt()
        s = a.summary()
        for key in ["estimate_s", "n_observations", "optimal_horizon",
                    "history_mean", "history_std", "alpha", "prior_s"]:
            assert key in s, f"Missing key: {key}"

    def test_summary_empty_history(self):
        a = make_adapt(cold_start_s=120.0)
        s = a.summary()
        assert s["n_observations"] == 0
        assert s["history_mean"] is None
        assert s["history_std"]  is None
        assert s["estimate_s"]   == 120.0

    def test_summary_single_observation(self):
        a = make_adapt(alpha=0.3, cold_start_s=120.0)
        a.observe(180.0)
        s = a.summary()
        assert s["n_observations"] == 1
        assert s["history_mean"]   == 180.0
        assert s["history_std"]    == 0.0

    def test_summary_mean_is_welford_consistent(self):
        """Welford mean must match simple arithmetic mean."""
        import statistics
        a = make_adapt()
        observations = [60.0, 90.0, 120.0, 150.0, 180.0]
        for v in observations:
            a.observe(v)
        s = a.summary()
        expected_mean = statistics.mean(observations)
        assert abs(s["history_mean"] - expected_mean) < 0.01

    def test_summary_std_is_welford_consistent(self):
        """Welford std must match population std."""
        import statistics
        a = make_adapt()
        observations = [60.0, 90.0, 120.0, 150.0, 180.0]
        for v in observations:
            a.observe(v)
        s = a.summary()
        expected_std = statistics.pstdev(observations)
        assert abs(s["history_std"] - expected_std) < 0.01

    def test_summary_optimal_horizon_consistent(self):
        a = make_adapt(cold_start_s=120.0, timestep_seconds=60, epsilon_steps=1)
        s = a.summary()
        assert s["optimal_horizon"] == a.optimal_horizon()


# ── FH OPT ────────────────────────────────────────────────────────────

class TestFHOPT:

    def test_optimal_horizon_formula(self):
        a = make_adapt(cold_start_s=120.0, timestep_seconds=60, epsilon_steps=1)
        assert a.optimal_horizon() == 3   # ceil(120/60) + 1

    def test_horizon_increases_with_longer_estimate(self):
        a = make_adapt(cold_start_s=60.0, timestep_seconds=60, epsilon_steps=0)
        h_before = a.optimal_horizon()
        for _ in range(30):
            a.observe(300.0)
        assert a.optimal_horizon() > h_before

    def test_horizon_at_least_1(self):
        a = make_adapt(cold_start_s=30.0, cold_start_min_s=30.0,
                       timestep_seconds=60, epsilon_steps=0)
        assert a.optimal_horizon() >= 1

    def test_fh_opt_adapts_after_observations(self):
        a = make_adapt(cold_start_s=60.0, timestep_seconds=60, epsilon_steps=1)
        h_initial = a.optimal_horizon()
        for _ in range(20):
            a.observe(300.0)
        assert a.optimal_horizon() > h_initial


# ── Reset ─────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_restores_prior(self):
        a = make_adapt(cold_start_s=120.0)
        for _ in range(20):
            a.observe(300.0)
        a.reset()
        assert a.estimate_s == 120.0

    def test_reset_clears_observation_count(self):
        a = make_adapt()
        a.observe(150.0)
        a.reset()
        assert a.n_observations == 0

    def test_reset_clears_welford_stats(self):
        a = make_adapt()
        for _ in range(10):
            a.observe(200.0)
        a.reset()
        assert a._welford_mean == 0.0
        assert a._welford_M2   == 0.0

    def test_post_reset_ewma_uses_prior(self):
        a = make_adapt(alpha=0.3, cold_start_s=120.0)
        for _ in range(50):
            a.observe(500.0)
        a.reset()
        est = a.observe(180.0)
        assert abs(est - (0.3 * 180.0 + 0.7 * 120.0)) < 1e-6

    def test_summary_after_reset_is_clean(self):
        a = make_adapt()
        for _ in range(10):
            a.observe(300.0)
        a.reset()
        s = a.summary()
        assert s["n_observations"] == 0
        assert s["history_mean"]   is None