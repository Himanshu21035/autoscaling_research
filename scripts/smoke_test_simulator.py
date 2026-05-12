# scripts/smoke_test_simulator.py
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.policies import create_policy
from src.simulator.core import run_simulation


def main() -> None:
    # Load 1 day of BurstGPT from the repository root, not the scripts folder.
    data_path = project_root / "data" / "processed" / "burstgpt_test.csv"
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    rps_trace = df["rps"].iloc[:1440].to_numpy(dtype=float)

    # Preserve the old smoke-test behavior: naive ceil(rps / 100) scaling.
    policy = create_policy(
        "threshold",
        target_rps_per_replica=100.0,
        scale_down_stabilization=3,
    )

    result = run_simulation(trace=rps_trace, policy=policy)
    summary = result.summary()

    print(f"Steps simulated  : {summary['steps']}")
    print(f"Total cost       : ${summary['total_cost']:.4f}")
    print(f"SLA violations   : {summary['sla_pct']:.2f}%")
    print(f"Avg latency      : {summary['avg_latency_ms']:.1f} ms")
    print(f"Avg replicas     : {summary['avg_replicas']:.1f}")
    print(f"Peak replicas    : {summary['peak_replicas']}")


if __name__ == "__main__":
    main()