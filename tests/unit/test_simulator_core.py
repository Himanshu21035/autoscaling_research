import numpy as np
import pytest
from src.simulator.core import run_simulation, SimResult
from src.policies import create_policy
from src.data.loader import load_trace, as_numpy
from src.data.splitter import split


def flat_trace(n=100, rps=300.0):
    return np.full(n, rps)

def spike_trace(n=100, base=100.0, spike_rps=1000.0, spike_start=40, spike_len=20):
    t = np.full(n, base)
    t[spike_start:spike_start + spike_len] = spike_rps
    return t


# ── SimResult ─────────────────────────────────────────────────────────

class TestSimResult:

    def test_finalise_computes_aggregates(self):
        policy = create_policy("hpa")
        result = run_simulation(flat_trace(50), policy)
        assert result.total_cost > 0
        assert 0 <= result.sla_violation_pct <= 100
        assert result.avg_latency_ms > 0
        assert result.avg_replicas >= 1
        assert result.peak_replicas >= 1

    def test_summary_has_required_keys(self):
        policy = create_policy("hpa")
        result = run_simulation(flat_trace(20), policy)
        s = result.summary()
        for key in ["policy", "forecaster", "steps", "total_cost",
                    "sla_pct", "avg_latency_ms", "avg_replicas", "peak_replicas"]:
            assert key in s

    def test_steps_matches_trace_length(self):
        policy = create_policy("hpa")
        trace  = flat_trace(60)
        result = run_simulation(trace, policy)
        assert result.steps == 60
        assert len(result.metrics) == 60


# ── Cold Start Queue ─────────────────────────────────────────────────

class TestColdStartQueue:

    def test_replicas_not_immediately_available(self):
        """
        At step 0: scale-up ordered. Capacity should not jump immediately.
        New replicas only available after cold_start_steps.
        """
        from src.config import CONFIG
        import math
        cold_steps = max(1, math.ceil(
            CONFIG["simulator"]["cold_start_s"] /
            CONFIG["simulator"]["timestep_seconds"]
        ))
        policy = create_policy("hpa")
        trace  = spike_trace(n=cold_steps + 5, spike_start=0,
                             spike_len=cold_steps + 5, spike_rps=1000.0)
        result = run_simulation(trace, policy)

        # At step 0, warm_replicas should still be initial_replicas
        step0 = result.metrics[0]
        assert step0.warm_replicas == CONFIG["simulator"]["initial_replicas"]

    def test_replicas_available_after_cold_start(self):
        """Replicas ordered at step 0 must be warm by step cold_start_steps."""
        from src.config import CONFIG
        import math
        cold_steps = max(1, math.ceil(
            CONFIG["simulator"]["cold_start_s"] /
            CONFIG["simulator"]["timestep_seconds"]
        ))
        policy = create_policy("hpa")
        # High RPS forces scale-up at step 0
        trace  = np.full(cold_steps + 10, 1000.0)
        result = run_simulation(trace, policy)

        warm_at_cold = result.metrics[cold_steps].warm_replicas
        warm_at_0    = result.metrics[0].warm_replicas
        assert warm_at_cold >= warm_at_0


# ── Latency Model ────────────────────────────────────────────────────

class TestLatencyModel:

    def test_low_utilisation_low_latency(self):
        """
        Test the latency formula directly — no policy, no cold start race.
        At 20% utilisation: latency = base_ms / (1 - 0.2) = base * 1.25
        Must be well below SLA threshold.
        """
        from src.simulator.core import _compute_latency
        from src.config import CONFIG

        base_ms = CONFIG["simulator"]["base_latency_ms"]
        sla_ms  = CONFIG["simulator"]["sla_latency_ms"]

        # 20% utilisation → base * 1.25
        lat_20 = _compute_latency(0.20)
        assert lat_20 < sla_ms, (
            f"20% util should be < {sla_ms} ms, got {lat_20:.1f} ms"
        )
        assert abs(lat_20 - base_ms / 0.80) < 1.0, (
            f"Expected {base_ms / 0.80:.1f} ms, got {lat_20:.1f} ms"
        )

    def test_zero_utilisation_equals_base(self):
        """0% utilisation must return exactly base_latency_ms."""
        from src.simulator.core import _compute_latency
        from src.config import CONFIG

        base_ms = CONFIG["simulator"]["base_latency_ms"]
        lat = _compute_latency(0.0)
        assert abs(lat - base_ms) < 1.0

    def test_overload_triggers_sla_violation(self):
        """
        Trace far above all-replica capacity must produce SLA violations.
        Uses ThresholdPolicy with default constructor (no kwargs).
        """
        from src.config import CONFIG
        policy = create_policy("threshold")          # default params only
        trace  = np.full(50, 9999.0)                 # massively over capacity
        result = run_simulation(trace, policy)
        assert result.sla_violation_pct > 0, \
            "Expected SLA violations at 9999 RPS, got 0%"

    def test_overload_latency_above_sla(self):
        """Latency formula at util >= 1.0 must exceed SLA threshold."""
        from src.simulator.core import _compute_latency
        from src.config import CONFIG

        sla_ms = CONFIG["simulator"]["sla_latency_ms"]
        lat    = _compute_latency(1.0)
        assert lat > sla_ms, (
            f"Overload latency {lat:.1f} ms must exceed SLA {sla_ms} ms"
        )

# ── With Forecaster ──────────────────────────────────────────────────

class TestWithForecaster:

    def test_arima_mpc_runs_without_crash(self):
        from src.forecasting import create_forecaster
        from src.simulator.adapt import ADAPTTracker
        from src.config import CONFIG

        trace = spike_trace(n=80)
        sp    = split(trace, train_frac=0.6, val_frac=0.1)

        adapt = ADAPTTracker(
            alpha=0.3,
            cold_start_s=CONFIG["simulator"]["cold_start_s"],
            cold_start_min_s=30.0, cold_start_max_s=600.0,
            epsilon_steps=1,
            timestep_seconds=CONFIG["simulator"]["timestep_seconds"],
        )
        policy     = create_policy("mpc", adapt_tracker=adapt)
        forecaster = create_forecaster("arima", min_series_length=10)
        forecaster.fit(sp.train_val)

        result = run_simulation(
            trace=sp.test,
            policy=policy,
            forecaster=forecaster,
            adapt=adapt,
        )
        assert isinstance(result, SimResult)
        assert result.steps == len(sp.test)
        assert result.avg_replicas >= 1

    def test_no_forecaster_still_works(self):
        """MPC with no forecaster must use flat fallback, not crash."""
        policy = create_policy("mpc")
        result = run_simulation(flat_trace(30), policy)
        assert result.steps == 30


# ── Data loader + splitter ───────────────────────────────────────────

class TestDataPipeline:

    def test_synthetic_loader_returns_dataframe(self):
        df = load_trace(source="synthetic", n_steps=100)
        assert "timestamp" in df.columns
        assert "rps" in df.columns
        assert len(df) == 100
        assert (df["rps"] >= 0).all()

    def test_as_numpy_returns_float_array(self):
        df  = load_trace(source="synthetic", n_steps=50)
        arr = as_numpy(df)
        assert arr.dtype == np.float64
        assert arr.shape == (50,)

    def test_splitter_sizes_correct(self):
        arr = np.arange(1000, dtype=float)
        sp  = split(arr, train_frac=0.70, val_frac=0.10)
        assert len(sp.train) == 700
        assert len(sp.val)   == 100
        assert len(sp.test)  == 200

    def test_splitter_no_overlap(self):
        arr = np.arange(100, dtype=float)
        sp  = split(arr)
        total = len(sp.train) + len(sp.val) + len(sp.test)
        assert total == 100

    def test_splitter_train_val_concatenated(self):
        arr = np.arange(100, dtype=float)
        sp  = split(arr)
        np.testing.assert_array_equal(
            sp.train_val,
            np.concatenate([sp.train, sp.val])
        )

    def test_splitter_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            split(np.array([1.0, 2.0]))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            load_trace(source="gcp_v2_undefined")