"""
Experiment job model — the core data structure for a simulation run.

RunStatus lifecycle:  PENDING → RUNNING → COMPLETED | FAILED
"""
from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class ExperimentRun:
    # Identity
    run_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)

    # Config
    policy:           str   = "mpc"
    forecaster:       str   = "lstm"
    batch:            str   = ""
    workload:         str   = "diurnal_burst"   # synthetic pattern name
    cold_start_s:     float = 120.0
    train_frac:       float = 0.70
    val_frac:         float = 0.10
    forecast_margin:  float = 1.15
    lambda_sla:       float = 50.0
    lambda_cost:      float = 1.0
    lambda_stab:      float = 0.5
    cold_start_steps: int   = 0
    use_fh_opt:       bool  = False
    seed:             int   = 42

    # State
    status:     RunStatus = RunStatus.PENDING
    started_at: float | None = None
    ended_at:   float | None = None
    error:      str | None   = None

    # Results (populated after completion)
    summary:    dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        if self.started_at and self.ended_at:
            return round(self.ended_at - self.started_at, 2)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":           self.run_id,
            "created_at":       self.created_at,
            "policy":           self.policy,
            "forecaster":       self.forecaster,
            "batch":            self.batch,
            "workload":         self.workload,
            "cold_start_s":     self.cold_start_s,
            "train_frac":       self.train_frac,
            "val_frac":         self.val_frac,
            "forecast_margin":  self.forecast_margin,
            "lambda_sla":       self.lambda_sla,
            "lambda_cost":      self.lambda_cost,
            "lambda_stab":      self.lambda_stab,
            "cold_start_steps": self.cold_start_steps,
            "use_fh_opt":       self.use_fh_opt,
            "seed":             self.seed,
            "status":           self.status.value,
            "started_at":       self.started_at,
            "ended_at":         self.ended_at,
            "duration_s":       self.duration_s,
            "error":            self.error,
            "summary":          self.summary,
        }
