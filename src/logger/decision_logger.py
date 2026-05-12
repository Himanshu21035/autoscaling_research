"""
Decision Logger — writes one CSV row per scaling decision.

Output : outputs/decisions_{policy}_{forecaster}.csv
Purpose: paper figures + Grafana file-based datasource.

Usage:
    with DecisionLogger("mpc", "lstm") as dl:
        dl.log(DecisionRecord(...))
"""
from __future__ import annotations
import csv
import time
from dataclasses import dataclass, fields, astuple
from pathlib import Path

OUTPUT_DIR = Path("outputs")


@dataclass
class DecisionRecord:
    step:             int
    timestamp:        float
    policy:           str
    forecaster:       str
    current_rps:      float
    current_replicas: int
    desired_replicas: int
    forecast_peak:    float
    reactive_floor:   int
    proactive_target: int
    sla_violated:     bool
    latency_ms:       float
    cost:             float


class DecisionLogger:

    def __init__(self, policy: str, forecaster: str):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._path   = OUTPUT_DIR / f"decisions_{policy}_{forecaster}.csv"
        self._file   = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([f.name for f in fields(DecisionRecord)])
        self._file.flush()

    def log(self, record: DecisionRecord) -> None:
        self._writer.writerow(astuple(record))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
