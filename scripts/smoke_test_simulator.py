# scripts/smoke_test_simulator.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.simulator.core import AutoscalerSimulator

# Load 1 day of BurstGPT
df = pd.read_csv("data/processed/burstgpt_test.csv",
                 index_col="timestamp", parse_dates=True)
rps_trace = df["rps"].iloc[:1440]   # first day (1440 min)

sim = AutoscalerSimulator(cold_start_seconds=120)

for ts, rps in rps_trace.items():
    # Naive policy: ceil(rps / 100) replicas
    import math
    decision = max(1, math.ceil(rps / 100))
    sim.step(rps=float(rps), decision=decision, timestamp=ts)

metrics = sim.get_metrics()
print(f"Steps simulated  : {len(metrics)}")
print(f"Total cost       : ${metrics['cost'].sum():.4f}")
print(f"SLA violations   : {100 * metrics['violation'].mean():.2f}%")
print(f"Avg latency      : {metrics['latency_ms'].mean():.1f} ms")
print(f"Avg replicas     : {metrics['active_replicas'].mean():.1f}")
print(f"Peak replicas    : {metrics['active_replicas'].max()}")
print(f"ADAPT final est  : {sim.adapt.estimate_s:.1f}s")
print(f"ADAPT observations: {sim.adapt.n_observations}")