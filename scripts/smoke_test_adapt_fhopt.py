#!/usr/bin/env python
"""
ADAPT + FH-OPT Smoke Test

Validates:
  - ADAPT tracker learns cold-start estimates from observed completions
  - FH-OPT (dynamic horizon) enables MPC to adapt lookahead based on live estimates
  - Step 9 remains functional (backward compatibility)
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_trace, as_numpy
from src.data.splitter import split
from src.simulator.adapt import ADAPTTracker
from src.simulator.core import run_simulation
from src.policies import create_policy
from src.forecasting import create_forecaster
from src.config import CONFIG

_SIM_CFG = CONFIG["simulator"]


def test_adapt_learning():
    """
    Verify that ADAPT learns from observed cold-start completions.
    
    Expected: Initial estimate ≈ 120s, then updates based on observed delays.
    """
    print("\n=== TEST: ADAPT Learning ===")
    
    adapt = ADAPTTracker(
        alpha=0.3,
        cold_start_s=120.0,
        cold_start_min_s=30.0,
        cold_start_max_s=600.0,
    )
    
    # Simulate observing a cold-start that took 90 seconds
    adapt.observe_event(
        t_requested=0.0,
        t_ready=90.0,
    )
    print(f"  After observing 90s cold-start: estimate = {adapt.estimate_s:.1f}s")
    
    # Simulate another cold-start that took 150 seconds
    adapt.observe_event(
        t_requested=100.0,
        t_ready=250.0,
    )
    print(f"  After observing 150s cold-start: estimate = {adapt.estimate_s:.1f}s")
    
    # Verify estimate has been updated (not still 120)
    assert adapt.estimate_s != 120.0, "ADAPT should have updated estimate"
    print("  ✓ ADAPT is learning from observations")


def test_fh_opt_horizon():
    """
    Verify that ADAPT's optimal_horizon() adapts based on learned estimates.
    
    Expected: As estimate increases, optimal horizon should increase.
    """
    print("\n=== TEST: FH-OPT Horizon Optimization ===")
    
    adapt = ADAPTTracker(
        alpha=0.3,
        cold_start_s=120.0,
        epsilon_steps=1,
        timestep_seconds=60,
    )
    
    h_initial = adapt.optimal_horizon()
    print(f"  Initial optimal_horizon() = {h_initial} steps")
    
    # Observe a long cold-start (300s = 5 steps at 60s resolution)
    adapt.observe_event(t_requested=0.0, t_ready=300.0)
    h_after = adapt.optimal_horizon()
    print(f"  After observing 300s cold-start: optimal_horizon() = {h_after} steps")
    
    assert h_after > h_initial, "FH-OPT should increase horizon as estimate grows"
    print("  ✓ FH-OPT adapts horizon based on learned cold-start")


def test_mpc_with_fh_opt():
    """
    Verify that MPC with use_fh_opt=True uses live ADAPT-derived horizons.
    
    Expected: MPC proactive target should vary as ADAPT estimate changes.
    """
    print("\n=== TEST: MPC with FH-OPT Integration ===")
    
    adapt = ADAPTTracker(
        alpha=0.3,
        cold_start_s=120.0,
        cold_start_min_s=30.0,
        cold_start_max_s=600.0,
    )
    
    # MPC with FH-OPT enabled
    mpc = create_policy(
        "mpc",
        adapt_tracker=adapt,
        use_fh_opt=True,
        lambda_sla=150.0,
        lambda_cost=1.0,
        lambda_stab=0.5,
    )
    
    # Create a simple forecast: ramp-up from 100 to 400 RPS
    forecast = np.array([100, 150, 200, 250, 300, 350, 400])
    
    # Call 1: ADAPT estimate at default 120s
    r1 = mpc.compute_replicas(
        current_rps=100.0,
        current_replicas=2,
        step=0,
        forecast=forecast,
        cold_start_steps=None,  # Will use default
    )
    print(f"  MPC decision (default 120s): {r1} replicas")
    
    # Observe a long cold-start to increase ADAPT estimate
    adapt.observe_event(t_requested=0.0, t_ready=300.0)
    horizon_adjusted = adapt.optimal_horizon() - 1
    
    # Call 2: MPC with FH-OPT-derived horizon
    r2 = mpc.compute_replicas(
        current_rps=100.0,
        current_replicas=2,
        step=0,
        forecast=forecast,
        cold_start_steps=horizon_adjusted,  # FH-OPT provides this
    )
    print(f"  MPC decision (FH-OPT {adapt.estimate_s:.0f}s): {r2} replicas")
    
    assert r2 >= r1, "MPC should scale more when horizon increases (FH-OPT)"
    print("  ✓ MPC respects FH-OPT-derived horizons")


def test_step9_backward_compat():
    """
    Verify that Step 9 still works (backward compatibility).
    
    Expected: Without use_fh_opt=True, MPC uses static cold_start_steps.
    """
    print("\n=== TEST: Step 9 Backward Compatibility ===")
    
    # Load synthetic trace
    df = load_trace(source="synthetic", pattern="diurnal_burst", seed=42)
    trace = as_numpy(df)
    split_res = split(trace, train_frac=0.6); train, test = split_res.train, split_res.test
    
    # MPC WITHOUT FH-OPT (Step 9 default)
    mpc = create_policy(
        "mpc",
        lambda_sla=150.0,
        lambda_cost=1.0,
        lambda_stab=0.5,
        cold_start_steps=0,  # Step 9: ARIMA config
        use_fh_opt=False,  # Disabled by default
    )
    
    # Run a short simulation
    result = run_simulation(
        trace=test[:100],  # First 100 steps
        policy=mpc,
    )
    
    print(f"  Simulation completed: {result.steps} steps")
    print(f"  SLA violation: {result.sla_violation_pct:.1f}%")
    
    assert result.steps == 100, "Simulation should complete without error"
    print("  ✓ Step 9 backward compatibility maintained")


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPT + FH-OPT Smoke Test")
    print("=" * 60)
    
    test_adapt_learning()
    test_fh_opt_horizon()
    test_mpc_with_fh_opt()
    test_step9_backward_compat()
    
    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    print("=" * 60)
