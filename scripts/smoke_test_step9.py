#!/usr/bin/env python
"""
Step 9 Smoke Test — Forecaster → Simulator → Metrics Table

Runs four configurations on a synthetic diurnal+burst trace:
  1. HPA (no forecaster)          — reactive baseline
  2. ARIMA + MPC                  — statistical forecaster + novel policy
  3. Prophet + MPC                — seasonal forecaster + novel policy
  4. LSTM + MPC                   — neural forecaster + novel policy

Expected output:
  Policy          Forecaster  SLA%    AvgReplicas  AvgLatency  TotalCost
  ─────────────────────────────────────────────────────────────────────
  HPAPolicy       none        xx.x%   x.x          xxxx ms     xxxx
  MPCPolicy       ARIMA       xx.x%   x.x          xxxx ms     xxxx
  MPCPolicy       Prophet     xx.x%   x.x          xxxx ms     xxxx
  MPCPolicy       LSTM        xx.x%   x.x          xxxx ms     xxxx
"""
import sys
import numpy as np
from pathlib import Path

# Ensure src is on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))


from simulator import adapt
from src.data.loader import load_trace, as_numpy
from src.data.splitter import split
from src.simulator.adapt import ADAPTTracker
from src.simulator.core import run_simulation
from src.policies import create_policy
from src.forecasting import create_forecaster
from src.config import CONFIG
from src.api.store import metrics_store, results_store
_SIM_CFG = CONFIG["simulator"]

FORECASTER_OVERRIDES = {
    "arima":   {"lambda_sla": 150.0, "lambda_cost": 1.0, "lambda_stab": 0.5},
    "prophet": {"lambda_sla": 150.0, "lambda_cost": 1.0, "lambda_stab": 0.5},
    "lstm":    {"lambda_sla": 50.0,  "lambda_cost": 1.0, "lambda_stab": 0.5},
}
FORECASTER_CONFIG = {
    "arima":   {"lambda_sla": 150.0, "lambda_cost": 1.0, "lambda_stab": 0.5, "forecast_margin": 1.0, "cold_start_steps": 0},
    "prophet": {"lambda_sla": 150.0, "lambda_cost": 1.0, "lambda_stab": 0.5, "forecast_margin": 1.06, "cold_start_steps": 2},
    "lstm":    {"lambda_sla": 50.0,  "lambda_cost": 1.0, "lambda_stab": 0.5, "forecast_margin": 1.15, "cold_start_steps": 0},
}
def make_adapt() -> ADAPTTracker:
    return ADAPTTracker(
        alpha=0.3,
        cold_start_s=_SIM_CFG["cold_start_s"],
        cold_start_min_s=30.0,
        cold_start_max_s=600.0,
        epsilon_steps=1,
        timestep_seconds=_SIM_CFG["timestep_seconds"],
    )


def run_hpa(test: np.ndarray) -> dict:
    policy = create_policy("hpa")
    result = run_simulation(trace=test, policy=policy)
    return result.summary()


def run_mpc_with_forecaster(
    train: np.ndarray,
    test:  np.ndarray,
    forecaster_name: str,
    forecaster_kwargs: dict,
) -> dict:
    adapt      = make_adapt()
    cfg = FORECASTER_CONFIG[forecaster_name]
    
    policy = create_policy(
        "mpc",
        adapt_tracker=adapt,
        lambda_sla=cfg["lambda_sla"],
        lambda_cost=cfg["lambda_cost"],
        lambda_stab=cfg["lambda_stab"],
        forecast_margin=cfg["forecast_margin"],   # pass through
        cold_start_steps=cfg["cold_start_steps"],
    )
    forecaster = create_forecaster(forecaster_name, **forecaster_kwargs)

    print(f"  Fitting {forecaster_name} on {len(train)} steps...", flush=True)
    forecaster.timed_fit(train)
    print(
        f"  {forecaster_name} fit in "
        f"{forecaster.fit_latency_ms:.0f} ms", flush=True
    )

    result = run_simulation(
        trace=test,
        policy=policy,
        forecaster=forecaster,
        adapt=adapt,
        forecast_every=1,
    )
    return result.summary()


def print_table(results: list[dict]):
    header = f"{'Policy':<14} {'Forecaster':<12} {'SLA%':>6} {'AvgRep':>8} {'AvgLat ms':>10} {'TotalCost':>10}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r['policy']:<14} {r['forecaster']:<12} "
            f"{r['sla_pct']:>5.1f}% {r['avg_replicas']:>8.1f} "
            f"{r['avg_latency_ms']:>9.0f}  "
            f"{r['total_cost']:>10.1f}"
        )
    print("─" * len(header))


def main():
    print("Loading trace...", flush=True)
    df    = load_trace(source="synthetic", pattern="diurnal_burst", seed=42)
    trace = as_numpy(df)
    sp    = split(trace, train_frac=0.70, val_frac=0.10)

    print(
        f"Trace: {len(trace)} steps | "
        f"train={len(sp.train)} val={len(sp.val)} test={len(sp.test)}"
    )

    results = []

    # 1 — HPA baseline
    print("\n[1/4] HPA baseline...", flush=True)
    results.append(run_hpa(sp.test))

    # 2 — ARIMA + MPC
    print("\n[2/4] ARIMA + MPC...", flush=True)
    results.append(run_mpc_with_forecaster(
        sp.train_val, sp.test,
        "arima", {"min_series_length": 10},
    ))

    # 3 — Prophet + MPC
    print("\n[3/4] Prophet + MPC...", flush=True)
    results.append(run_mpc_with_forecaster(
        sp.train_val, sp.test,
        "prophet", {"min_series_length": 20},
    ))

    # 4 — LSTM + MPC
    print("\n[4/4] LSTM + MPC...", flush=True)
    results.append(run_mpc_with_forecaster(
        sp.train_val, sp.test,
        "lstm", {
            "window_size": 30, "hidden_size": 64,
            "num_layers": 2, "max_epochs": 50,
            "min_series_length": 50, "seed": 42,
        },
    ))

    print_table(results)

    # ── Push to API stores ─────────────────────────────────────────────────
    results_store.set(results)                    # summary table → /v1/policy/results/latest
    print(f"  pushed {len(results)} run summaries to results_store")
    # ──────────────────────────────────────────────────────────────────────

    # Key assertion
    hpa_sla = results[0]["sla_pct"]
    for r in results[1:]:
        if r["sla_pct"] > hpa_sla * 1.5:
            print(
                f"\n⚠ WARNING: {r['forecaster']}+MPC SLA {r['sla_pct']:.1f}% "
                f"is worse than HPA {hpa_sla:.1f}% — check lambda weights"
            )

    print("\n✅ Step 9 smoke test complete")

if __name__ == "__main__":
    main()