# scripts/generate_synthetic.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import generate_all
import pandas as pd

print("Generating synthetic workloads...")
workloads = generate_all(seed=42)

print("\n── Workload Statistics ──────────────────────────────────────────")
print(f"{'Workload':<14} {'Duration':>10} {'Mean RPS':>10} {'Std RPS':>10} "
      f"{'Max RPS':>10} {'CV':>8}")
print("─" * 66)

for name, s in workloads.items():
    duration_days = len(s) / 1440
    cv = s.std() / s.mean()
    print(
        f"{name:<14} {duration_days:>9.0f}d {s.mean():>10.1f} "
        f"{s.std():>10.1f} {s.max():>10.1f} {cv:>8.3f}"
    )

print("\n── Contrast with Real Traces (from your EDA) ───────────────────")
print("BurstGPT:  mean=294.6  std=646.4  max=10110  CV=2.19  (target: bursty)")
print("Azure:     mean=329.6  std=75.1   max=502     CV=0.23  (target: smooth)")

print("\n✅ Saved to data/synthetic/")