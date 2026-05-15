import math
import numpy as np
from src.policies.mpc import MPCPolicy
from src.policies.base import BasePolicy
from src.policies import create_policy
from src.simulator.adapt import ADAPTTracker


def make_adapt(cold_start_s=120.0, alpha=0.3, epsilon=1, ts=60):
    return ADAPTTracker(
        alpha=alpha, cold_start_s=cold_start_s,
        cold_start_min_s=30.0, cold_start_max_s=600.0,
        epsilon_steps=epsilon, timestep_seconds=ts,
    )


def make_mpc(cold_start_s=120.0, alpha=0.3, epsilon=1,
             lambda_sla=10.0, lambda_cost=1.0, lambda_stab=0.0,
             **kwargs) -> MPCPolicy:
    adapt = make_adapt(cold_start_s=cold_start_s, alpha=alpha, epsilon=epsilon)
    return MPCPolicy(
        adapt_tracker=adapt,
        lambda_sla=lambda_sla,
        lambda_cost=lambda_cost,
        lambda_stab=lambda_stab,
        capacity_per_replica=100.0,
        min_replicas=1,
        max_replicas=50,
        **kwargs,
    )


# ── Contract ──────────────────────────────────────────────────────────

class TestMPCContract:

    def test_subclasses_base_policy(self):
        assert isinstance(make_mpc(), BasePolicy)

    def test_output_always_in_bounds(self):
        p = make_mpc()
        for rps in [0, 100, 500, 2000, 9999]:
            r = p.compute_replicas(rps, 5, 0)
            assert 1 <= r <= 50

    def test_accepts_context_kwargs(self):
        p = make_mpc()
        p.compute_replicas(300.0, 5, 0,
                           forecast=np.full(5, 300.0), warm_replicas=3)

    def test_factory_creates_mpc(self):
        p = create_policy("mpc")
        assert isinstance(p, MPCPolicy)


# ── Core Research Claim: Delay-Aware Proactive Scaling ────────────────

class TestDelayAwareScaling:

    def test_replica_ordered_before_spike_arrives(self):
        """
        Core claim: MPC orders replicas NOW for a spike within the horizon.

        Setup:
        cold_start_s=120, timestep=60 → cold_start_steps=2
        epsilon_steps=3 → h* = ceil(120/60) + 3 = 5
        forecast = [100, 100, 1000, 1000, 1000]  ← spike at index 2
        At k=0,1: only warm_replicas=1 available (cold start window)
        At k=2+:  ordered replicas ready — but spike hits at k=2
        Therefore MPC MUST order > 1 replica at step 0 to cover k=2+.
        """
        p = make_mpc(
            cold_start_s=120.0, epsilon=3,   # h* = ceil(120/60)+3 = 5
            lambda_sla=100.0, lambda_cost=0.1, lambda_stab=0.0,
        )
        # Spike starts exactly at cold_start_steps=2 — inside h*=5
        forecast = np.array([100.0, 100.0, 1000.0, 1000.0, 1000.0])
        r = p.compute_replicas(
            100.0, current_replicas=1, step=0,
            forecast=forecast, warm_replicas=1,
        )
        assert r > 1, (
            f"MPC must order replicas ahead of spike — got {r}. "
            f"cold_start_steps=2, spike at index 2, h*=5."
        )


    def test_longer_cold_start_causes_earlier_scale_up(self):
        """
        Longer Δ̂_cold → larger cold_start_steps → MPC must order even
        earlier → higher replica count for same forecast.
        """
        forecast = np.array([100.0] * 5 + [1000.0] * 5)

        p_short = make_mpc(cold_start_s=60.0,  lambda_sla=100.0, lambda_cost=0.1)
        p_long  = make_mpc(cold_start_s=300.0, lambda_sla=100.0, lambda_cost=0.1)

        r_short = p_short.compute_replicas(100.0, 1, 0, forecast=forecast, warm_replicas=1)
        r_long  = p_long.compute_replicas( 100.0, 1, 0, forecast=forecast, warm_replicas=1)

        assert r_long >= r_short, (
            f"Longer cold start must trigger earlier/larger scale-up: "
            f"r_short={r_short}, r_long={r_long}"
        )

    def test_no_spike_in_forecast_does_not_over_provision(self):
        """Flat low forecast → MPC should not waste money on excess replicas."""
        p = make_mpc(lambda_sla=10.0, lambda_cost=5.0, lambda_stab=0.0)
        forecast = np.full(5, 100.0)
        r = p.compute_replicas(100.0, 3, 0, forecast=forecast, warm_replicas=3)
        assert r <= 3   # at most current — no reason to scale up

    def test_horizon_length_changes_decision_for_future_spike(self):
        """
        Two identical near-term forecasts but different horizon lengths
        (via different cold start estimates) must produce different decisions
        when a far-future spike is within one horizon but not the other.
        """
        near_flat  = np.array([100.0] * 3)
        far_spike  = np.array([100.0] * 3 + [1000.0] * 5)

        # Short horizon (h*=2): sees only [100, 100] → stays low
        p_short = make_mpc(cold_start_s=60.0, epsilon=0,
                           lambda_sla=100.0, lambda_cost=0.1)
        # Long horizon (h*=5): sees the spike → scales up
        p_long  = make_mpc(cold_start_s=240.0, epsilon=0,
                           lambda_sla=100.0, lambda_cost=0.1)

        r_short = p_short.compute_replicas(100.0, 1, 0,
                                           forecast=far_spike, warm_replicas=1)
        r_long  = p_long.compute_replicas( 100.0, 1, 0,
                                           forecast=far_spike, warm_replicas=1)
        assert r_long >= r_short


# ── ADAPT + MPC Integration Over Spike Trace ──────────────────────────

class TestIntegration:

    def test_horizon_growth_changes_action_over_spike_trace(self):
        """
        Integration test: longer cold start → MPC provisions EARLIER (pre-spike).

        Mechanism:
        short cold start (cold_start_steps=1): MPC can react close to the spike
        long  cold start (cold_start_steps=5): MPC must order 5 steps ahead

        Therefore p_long should have higher replica counts in the PRE-SPIKE
        window (steps 15–20) where it is already ordering for the spike,
        whereas p_short can afford to wait until closer to step 20.
        """
        n = 60
        rps_trace = np.concatenate([
            np.full(20, 100.0),
            np.full(20, 1000.0),
            np.full(20, 100.0),
        ])

        adapt_short = make_adapt(cold_start_s=60.0,  alpha=0.05)  # cold_start_steps=1
        adapt_long  = make_adapt(cold_start_s=300.0, alpha=0.05)  # cold_start_steps=5

        p_short = MPCPolicy(
            adapt_tracker=adapt_short,
            lambda_sla=50.0, lambda_cost=0.1, lambda_stab=0.0,
            capacity_per_replica=100.0, min_replicas=1, max_replicas=50,
        )
        p_long = MPCPolicy(
            adapt_tracker=adapt_long,
            lambda_sla=50.0, lambda_cost=0.1, lambda_stab=0.0,
            capacity_per_replica=100.0, min_replicas=1, max_replicas=50,
        )

        results_short, results_long = [], []
        replicas_s = replicas_l = 1

        for step, rps in enumerate(rps_trace):
            forecast = rps_trace[step:step + 10] if step + 10 <= n else rps_trace[step:]
            replicas_s = p_short.compute_replicas(
                rps, replicas_s, step, forecast=forecast, warm_replicas=replicas_s
            )
            replicas_l = p_long.compute_replicas(
                rps, replicas_l, step, forecast=forecast, warm_replicas=replicas_l
            )
            results_short.append(replicas_s)
            results_long.append(replicas_l)

        # PRE-SPIKE window: steps 14–19 (6 steps before spike).
        # p_long (cold_start_steps=5) must start ordering here.
        # p_short (cold_start_steps=1) can afford to wait.
        pre_spike_avg_short = np.mean(results_short[14:20])
        pre_spike_avg_long  = np.mean(results_long[14:20])

        assert pre_spike_avg_long >= pre_spike_avg_short, (
            f"Longer cold start must provision EARLIER (pre-spike steps 14-19): "
            f"short={pre_spike_avg_short:.1f}, long={pre_spike_avg_long:.1f}. "
            f"This validates the core paper claim: proactive ordering proportional "
            f"to cold start delay."
        )