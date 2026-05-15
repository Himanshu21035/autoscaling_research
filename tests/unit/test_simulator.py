import pytest
import pandas as pd
import numpy as np
from src.simulator.core import AutoscalerSimulator
from src.simulator.cold_start import ColdStartTracker
from src.simulator.latency_model import mm1_latency, MAX_LATENCY_MS


# ── ColdStartTracker Tests ─────────────────────────────────────────────────

class TestColdStartTracker:

    def test_replicas_not_ready_before_delay(self):
        tracker = ColdStartTracker(cold_start_seconds=120, timestep_seconds=60)
        tracker.request_scale_up(3, current_step=0)
        # Step 1 — should NOT be ready yet (needs 2 steps)
        ready = tracker.update(current_step=1)
        assert ready == 0

    def test_replicas_ready_after_delay(self):
        tracker = ColdStartTracker(cold_start_seconds=120, timestep_seconds=60)
        tracker.request_scale_up(3, current_step=0)
        # Step 2 — should be ready (120s / 60s = 2 steps)
        ready = tracker.update(current_step=2)
        assert ready == 3

    def test_warming_count_decreases_after_ready(self):
        tracker = ColdStartTracker(cold_start_seconds=60, timestep_seconds=60)
        tracker.request_scale_up(2, current_step=0)
        assert tracker.warming_count() == 2
        tracker.update(current_step=1)
        assert tracker.warming_count() == 0

    def test_multiple_batches_tracked_independently(self):
        tracker = ColdStartTracker(cold_start_seconds=60, timestep_seconds=60)
        tracker.request_scale_up(2, current_step=0)
        tracker.request_scale_up(3, current_step=1)
        # At step 1: first batch (ordered at 0) is ready
        ready = tracker.update(current_step=1)
        assert ready == 2
        assert tracker.warming_count() == 3

    def test_reset_clears_all_warming(self):
        tracker = ColdStartTracker(cold_start_seconds=120, timestep_seconds=60)
        tracker.request_scale_up(5, current_step=0)
        tracker.reset()
        assert tracker.warming_count() == 0


# ── Latency Model Tests ────────────────────────────────────────────────────

class TestLatencyModel:

    def test_zero_load_returns_base_latency(self):
        assert mm1_latency(rps=0, capacity=1000, base_latency_ms=50) == 50.0

    def test_half_load_doubles_latency(self):
        # utilization=0.5 → latency = base / (1-0.5) = 2x base
        result = mm1_latency(rps=500, capacity=1000, base_latency_ms=50)
        assert abs(result - 100.0) < 0.001

    def test_overloaded_returns_max_latency(self):
        result = mm1_latency(rps=1100, capacity=1000, base_latency_ms=50)
        assert result == MAX_LATENCY_MS

    def test_zero_capacity_returns_max_latency(self):
        result = mm1_latency(rps=100, capacity=0, base_latency_ms=50)
        assert result == MAX_LATENCY_MS


# ── AutoscalerSimulator Tests ──────────────────────────────────────────────

class TestAutoscalerSimulator:

    def make_sim(self, cold_start_seconds=120) -> AutoscalerSimulator:
        return AutoscalerSimulator(
            cold_start_seconds=cold_start_seconds,
            timestep_seconds=60,
            capacity_per_replica=100,
            base_latency_ms=50,
            initial_replicas=2,
            min_replicas=1,
            max_replicas=50,
        )

    def test_no_violation_when_rps_below_capacity(self):
        sim = self.make_sim()
        # 2 replicas × 100 RPS = 200 capacity, send 150 RPS
        result = sim.step(rps=150, decision=2)
        assert result["violation"] == 0.0

    def test_violation_when_rps_exceeds_capacity(self):
        sim = self.make_sim()
        # 2 replicas × 100 = 200 capacity, send 300 RPS
        result = sim.step(rps=300, decision=2)
        assert result["violation"] > 0.0

    def test_scale_down_is_instant(self):
        sim = self.make_sim()
        sim.active_replicas = 10
        sim.step(rps=100, decision=3)   # scale down from 10 to 3
        assert sim.active_replicas == 3

    def test_scale_up_delayed_by_cold_start(self):
        sim = self.make_sim(cold_start_seconds=120)  # 2 steps delay
        sim.active_replicas = 2
        sim.step(rps=100, decision=5)   # order 3 more replicas
        # Step 0 done — replicas should still be 2 (warming not ready)
        assert sim.active_replicas == 2
        assert sim.state["warming_replicas"] == 3

    def test_scale_up_ready_after_cold_start_steps(self):
        sim = self.make_sim(cold_start_seconds=60)   # 1 step delay
        sim.active_replicas = 2
        sim.step(rps=100, decision=5)   # order 3 more at step 0
        sim.step(rps=100, decision=5)   # step 1 — replicas should arrive
        assert sim.active_replicas == 5

    def test_cost_accumulates_correctly(self):
        sim = self.make_sim()
        sim.active_replicas = 4
        result = sim.step(rps=200, decision=4)
        # 4 replicas × $0.01/min × 1 min = $0.04
        assert abs(result["cost"] - 0.04) < 1e-9

    def test_reset_clears_all_state(self):
        sim = self.make_sim()
        sim.step(rps=200, decision=5)
        sim.step(rps=300, decision=5)
        sim.reset()
        assert sim.current_step == 0
        assert sim.total_cost == 0.0
        assert len(sim.get_metrics()) == 0

    def test_metrics_dataframe_has_correct_columns(self):
        sim = self.make_sim()
        for _ in range(5):
            sim.step(rps=200, decision=3)
        df = sim.get_metrics()
        expected_cols = ["rps", "active_replicas", "capacity",
                         "violation", "latency_ms", "cost"]
        for col in expected_cols:
            assert col in df.columns

    def test_replicas_clamped_to_min_max(self):
        sim = self.make_sim()
        result = sim.step(rps=10, decision=0)    # below min_replicas=1
        assert sim.active_replicas >= 1
        result = sim.step(rps=10, decision=999)  # above max_replicas=50
        # warming will be capped
        total = sim.active_replicas + sim.state["warming_replicas"]
        assert total <= 50

    def test_full_simulation_run(self):
        """End-to-end: run 100 steps with realistic RPS, check no crashes."""
        sim = self.make_sim()
        rps_trace = np.random.uniform(50, 400, size=100)
        for rps in rps_trace:
            result = sim.step(rps=rps, decision=max(1, int(rps / 100) + 1))
        df = sim.get_metrics()
        assert len(df) == 100
        assert df["cost"].sum() > 0
        assert df["violation"].between(0, 1).all()

class TestADAPTTracker:

    def test_initial_estimate_equals_init_value(self):
        from src.simulator.adapt import ADAPTTracker
        adapt = ADAPTTracker(alpha=0.3, init_cold_start_s=120.0)
        assert adapt.estimate_s == 120.0

    def test_single_observation_updates_correctly(self):
        from src.simulator.adapt import ADAPTTracker
        adapt = ADAPTTracker(alpha=0.3, init_cold_start_s=120.0)
        adapt.observe(180.0)
        # 0.3 × 180 + 0.7 × 120 = 54 + 84 = 138
        assert abs(adapt.estimate_s - 138.0) < 0.001

    def test_estimate_converges_with_repeated_observations(self):
        from src.simulator.adapt import ADAPTTracker
        adapt = ADAPTTracker(alpha=0.5, init_cold_start_s=120.0)
        for _ in range(20):
            adapt.observe(60.0)
        # Should converge close to 60
        assert adapt.estimate_s < 70.0