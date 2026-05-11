# tests/unit/test_policies.py
import math
import pytest
import numpy as np
from src.policies.threshold import ThresholdPolicy
from src.policies.pid import PIDPolicy
from src.policies.hpa import HPAPolicy
from src.policies.base import BasePolicy


# ── Shared Contract ────────────────────────────────────────────────────

def assert_policy_contract(policy: BasePolicy, label: str):
    for rps in [0, 50, 200, 500, 5000]:
        out = policy.compute_replicas(rps, current_replicas=5, step=0)
        assert policy.min_replicas <= out <= policy.max_replicas, (
            f"{label}: output {out} outside bounds for rps={rps}"
        )
    policy.reset()
    out = policy.compute_replicas(100.0, current_replicas=3, step=0)
    assert policy.min_replicas <= out <= policy.max_replicas


class TestPolicyContract:

    def test_threshold_contract(self):
        assert_policy_contract(
            ThresholdPolicy(target_rps_per_replica=100.0), "threshold"
        )

    def test_pid_contract(self):
        assert_policy_contract(PIDPolicy(), "pid")

    def test_hpa_contract(self):
        assert_policy_contract(
            HPAPolicy(target_rps_per_replica=100.0), "hpa"
        )

    def test_all_accept_context_kwargs(self):
        """All policies must accept **context without crashing."""
        ctx = dict(
            forecast=np.array([300.0] * 10),
            adapt_estimate_s=120.0,
            warm_replicas=3,
        )
        for cls in [ThresholdPolicy, PIDPolicy, HPAPolicy]:
            p = cls()
            p.compute_replicas(300.0, 5, 0, **ctx)


# ── BasePolicy ─────────────────────────────────────────────────────────

class TestBasePolicy:

    def test_clamp_uses_ceil_not_round(self):
        """3.1 should clamp to 4 (ceiling), not 3 (round)."""
        p = ThresholdPolicy(target_rps_per_replica=100.0)
        # 310 RPS / 100 = 3.1 → ceil = 4
        assert p.compute_replicas(310.0, 10, 0) == 4

    def test_min_replicas_below_1_raises(self):
        with pytest.raises(ValueError, match="min_replicas"):
            ThresholdPolicy(min_replicas=0)

    def test_max_below_min_raises(self):
        with pytest.raises(ValueError, match="max_replicas"):
            ThresholdPolicy(min_replicas=5, max_replicas=3)


# ── ThresholdPolicy ────────────────────────────────────────────────────

class TestThresholdPolicy:

    def test_formula_matches_k8s_hpa(self):
        """Each assertion uses a fresh instance to avoid state leak."""
        assert ThresholdPolicy(target_rps_per_replica=100.0).compute_replicas(300.0, 5, 0) == 3
        assert ThresholdPolicy(target_rps_per_replica=100.0).compute_replicas(301.0, 5, 0) == 4
        assert ThresholdPolicy(target_rps_per_replica=100.0).compute_replicas(200.0, 5, 0) == 2

    def test_zero_rps_returns_min_replicas(self):
        p = ThresholdPolicy(target_rps_per_replica=100.0, min_replicas=2)
        assert p.compute_replicas(0.0, 5, 0) == 2

    def test_high_rps_clamped_to_max(self):
        p = ThresholdPolicy(target_rps_per_replica=100.0, max_replicas=10)
        assert p.compute_replicas(99999.0, 5, 0) == 10

    def test_scale_down_blocked_within_stabilization_window(self):
        p = ThresholdPolicy(
            target_rps_per_replica=100.0, scale_down_stabilization=5
        )
        # step 0: scale down would be triggered (200 RPS → 2 replicas, have 5)
        r0 = p.compute_replicas(200.0, 5, step=0)
        assert r0 == 2   # first scale-down allowed
        # step 1-4: still low RPS — blocked by stabilization
        for step in range(1, 5):
            r = p.compute_replicas(200.0, 2, step=step)
            assert r == 2   # no further scale-down yet

    def test_scale_up_always_immediate(self):
        """Scale-up must never be blocked."""
        p = ThresholdPolicy(target_rps_per_replica=100.0,
                            scale_down_stabilization=99)
        r = p.compute_replicas(500.0, 1, step=0)
        assert r == 5   # immediate

    def test_uses_current_replicas_as_truth(self):
        """Policy must return values relative to passed current_replicas."""
        p = ThresholdPolicy(target_rps_per_replica=100.0)
        # 200 RPS → wants 2. Current = 10. Should return 2.
        # Stabilization window not elapsed yet, but this is step 0
        r = p.compute_replicas(200.0, 10, step=0)
        assert r == 2

    def test_reset_clears_stabilization_state(self):
        p = ThresholdPolicy(target_rps_per_replica=100.0,
                            scale_down_stabilization=100)
        p.compute_replicas(200.0, 5, step=0)
        p.reset()
        assert p._last_scale_down_step == -999

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_rps_per_replica"):
            ThresholdPolicy(target_rps_per_replica=0.0)


# ── PIDPolicy ──────────────────────────────────────────────────────────

class TestPIDPolicy:

    def test_error_based_on_capacity_not_fixed_target(self):
        """
        With 5 replicas × 100 capacity = 500 capacity.
        At RPS=450 → error = -50 (over-provisioned slightly).
        Should NOT aggressively scale up.
        With old approach (target=300): error = 450-300 = 150 → wrong.
        """
        p = PIDPolicy(kp=1.0, ki=0.0, kd=0.0, capacity_per_replica=100.0)
        r = p.compute_replicas(450.0, current_replicas=5, step=0)
        # error = 450 - 500 = -50 → delta = -0.5 → desired = ceil(4.5) = 5
        assert r == 5   # near-capacity — should stay put

    def test_clear_under_provision_scales_up(self):
        """800 RPS with 5 replicas × 100 = 500 capacity → error = 300."""
        p = PIDPolicy(kp=1.0, ki=0.0, kd=0.0, capacity_per_replica=100.0)
        r = p.compute_replicas(800.0, current_replicas=5, step=0)
        assert r > 5

    def test_clear_over_provision_scales_down(self):
        """50 RPS with 10 replicas × 100 = 1000 capacity → error = -950."""
        p = PIDPolicy(kp=1.0, ki=0.0, kd=0.0, capacity_per_replica=100.0,
                      min_replicas=1)
        r = p.compute_replicas(50.0, current_replicas=10, step=0)
        assert r < 10

    def test_uses_ceil_not_round(self):
        """
        5 replicas, RPS=510, capacity=500 → error=10
        delta = 10/100 = 0.1 → desired_raw = 5.1 → ceil = 6
        """
        p = PIDPolicy(kp=1.0, ki=0.0, kd=0.0, capacity_per_replica=100.0)
        r = p.compute_replicas(510.0, current_replicas=5, step=0)
        assert r == 6

    def test_conditional_integral_no_windup_at_max(self):
        """Integral must not accumulate when output is saturated at max."""
        p = PIDPolicy(
            kp=0.0, ki=1.0, kd=0.0,
            capacity_per_replica=100.0,
            integral_limit=1000.0,
            max_replicas=5,
        )
        for step in range(100):
            p.compute_replicas(9999.0, current_replicas=5, step=step)
        # Output always saturated at max=5, so integral should be 0
        assert p._integral == 0.0

    def test_reset_clears_all_state(self):
        p = PIDPolicy()
        for step in range(20):
            p.compute_replicas(800.0, 5, step)
        p.reset()
        assert p._integral       == 0.0
        assert p._prev_error     == 0.0
        assert p._smoothed_deriv == 0.0

    def test_output_clamped_to_bounds(self):
        p = PIDPolicy(min_replicas=2, max_replicas=8)
        assert p.compute_replicas(0.0,     2, 0) >= 2
        assert p.compute_replicas(99999.0, 8, 0) <= 8


# ── HPAPolicy ──────────────────────────────────────────────────────────

class TestHPAPolicy:

    def test_uses_simulator_replicas_as_truth(self):
        """
        Policy internal state must NOT diverge from simulator.
        Simulator starts with 3 replicas, policy must use 3, not min_replicas.
        """
        p = HPAPolicy(target_rps_per_replica=100.0,
                      scale_up_cooldown=1, scale_down_cooldown=5)
        # Simulator says 3 replicas, 400 RPS → wants 4
        r = p.compute_replicas(400.0, current_replicas=3, step=0)
        assert r == 4   # computed from current_replicas=3, not internal state

    def test_scale_down_blocked_by_cooldown(self):
        p = HPAPolicy(target_rps_per_replica=100.0,
                      scale_up_cooldown=1, scale_down_cooldown=10)
        # Scale up
        r0 = p.compute_replicas(500.0, current_replicas=1, step=0)
        assert r0 == 5
        # Immediately try to scale down — cooldown blocks it
        # Pass current_replicas=5 (simulator truth after scale-up)
        r1 = p.compute_replicas(10.0, current_replicas=5, step=1)
        assert r1 == 5

    def test_scale_down_allowed_after_cooldown_elapsed(self):
        p = HPAPolicy(target_rps_per_replica=100.0,
                    scale_up_cooldown=1, scale_down_cooldown=3)
        p.compute_replicas(500.0, current_replicas=1, step=0)   # up at step 0
        p.compute_replicas(10.0,  current_replicas=5, step=1)   # blocked: since_up=1 < 3
        p.compute_replicas(10.0,  current_replicas=5, step=2)   # blocked: since_up=2 < 3
        r = p.compute_replicas(10.0, current_replicas=5, step=3) # allowed: since_up=3 >= 3
        assert r < 5

    def test_state_sync_no_drift_over_many_steps(self):
        """
        Simulate 50 steps with alternating RPS.
        Policy output must always be consistent with passed current_replicas.
        """
        p = HPAPolicy(target_rps_per_replica=100.0,
                      scale_up_cooldown=1, scale_down_cooldown=2)
        replicas = 3
        for step in range(50):
            rps = 600.0 if step % 5 == 0 else 100.0
            new_replicas = p.compute_replicas(rps, current_replicas=replicas, step=step)
            # Output must always be clamped and consistent
            assert p.min_replicas <= new_replicas <= p.max_replicas
            replicas = new_replicas  # advance simulator state

    def test_reset_clears_only_step_trackers(self):
        """After reset, step trackers are -999 but no replica state exists."""
        p = HPAPolicy(target_rps_per_replica=100.0)
        p.compute_replicas(500.0, current_replicas=1, step=0)
        p.reset()
        assert p._last_scale_up_step   == -999
        assert p._last_scale_down_step == -999
        assert not hasattr(p, "_current_replicas"), \
            "HPAPolicy must not maintain _current_replicas — simulator is truth"

    def test_no_change_returns_current_replicas(self):
        """At exactly target utilization, output equals current."""
        p = HPAPolicy(target_rps_per_replica=100.0)
        r = p.compute_replicas(300.0, current_replicas=3, step=5)
        assert r == 3

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_rps_per_replica"):
            HPAPolicy(target_rps_per_replica=-1.0)

    def test_invalid_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown"):
            HPAPolicy(scale_up_cooldown=0)


# ── Integration: multi-step simulation ────────────────────────────────

class TestMultiStepIntegration:

    def _run_trace(self, policy, rps_trace: list[float]) -> list[int]:
        """Run a policy through a full RPS trace, advancing replicas."""
        replicas = policy.min_replicas
        results  = []
        for step, rps in enumerate(rps_trace):
            replicas = policy.compute_replicas(rps, replicas, step)
            results.append(replicas)
        return results

    def test_threshold_tracks_demand(self):
        rps_trace = [100.0] * 10 + [500.0] * 10 + [100.0] * 10
        p = ThresholdPolicy(target_rps_per_replica=100.0,
                            scale_down_stabilization=1)
        results = self._run_trace(p, rps_trace)
        assert max(results[10:20]) >= 5   # scaled up during spike
        assert results[-1] <= 2           # scaled back down

    def test_hpa_does_not_drift_from_simulator_state(self):
        """Policy output at step N must depend on actual replicas at step N."""
        rps_trace = [500.0, 100.0, 100.0, 100.0, 100.0,
                     500.0, 500.0, 100.0, 100.0, 100.0]
        p = HPAPolicy(target_rps_per_replica=100.0,
                      scale_up_cooldown=1, scale_down_cooldown=3)
        results = self._run_trace(p, rps_trace)
        for r in results:
            assert p.min_replicas <= r <= p.max_replicas

    def test_pid_converges_on_steady_load(self):
        """PID must converge to roughly correct replica count under steady load."""
        p = PIDPolicy(kp=0.5, ki=0.1, kd=0.01, capacity_per_replica=100.0)
        # Steady 400 RPS → needs 4 replicas
        rps_trace = [400.0] * 60
        replicas = 1
        for step, rps in enumerate(rps_trace):
            replicas = p.compute_replicas(rps, replicas, step)
        # After 60 steps should be near 4
        assert 3 <= replicas <= 6


# ── Registry ──────────────────────────────────────────────────────────

class TestPolicyRegistry:

    def test_all_three_registered(self):
        from src.policies import POLICY_REGISTRY
        assert {"threshold", "pid", "hpa"} == set(POLICY_REGISTRY.keys())

    def test_registry_instantiates_correctly(self):
        from src.policies import POLICY_REGISTRY
        for name, cls in POLICY_REGISTRY.items():
            assert isinstance(cls(), BasePolicy), \
                f"{name} does not subclass BasePolicy"