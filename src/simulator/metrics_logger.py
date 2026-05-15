from dataclasses import dataclass, field, asdict
import pandas as pd


@dataclass
class StepMetrics:
    step: int
    timestamp: object         # pd.Timestamp
    rps: float
    active_replicas: int
    warming_replicas: int
    capacity: float
    utilization: float
    violation: float          # fraction of demand unserved (0-1)
    latency_ms: float
    cost: float               # replica-minutes used this step
    scaling_action: int       # +N or -N replicas ordered this step
    cold_start_seconds: int
    adapt_estimate_s: float


class MetricsLogger:
    """Accumulates per-step metrics and exports to DataFrame/CSV."""

    def __init__(self):
        self._rows: list[StepMetrics] = []

    def log(self, metrics: StepMetrics):
        self._rows.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self._rows]).set_index("step")

    def save(self, path: str):
        self.to_dataframe().to_csv(path)

    def reset(self):
        self._rows = []

    def __len__(self):
        return len(self._rows)