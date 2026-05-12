"""
Pydantic V2 schemas — all Field(..., example=...) replaced with
json_schema_extra for Swagger/OpenAPI examples.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    version: str

    model_config = ConfigDict(
        json_schema_extra={"example": {"status": "ok", "version": "1.0.0"}}
    )


class ReadyResponse(BaseModel):
    ready:                 bool
    detail:                str       = ""
    available_policies:    list[str] = []
    available_forecasters: list[str] = []


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricPoint(BaseModel):
    step:             int
    rps:              float
    replicas:         int
    latency_ms:       float
    sla_violated:     bool
    cost:             float
    utilisation:      float = 0.0
    adapt_estimate_s: float = 0.0


class MetricsCurrentResponse(MetricPoint):
    policy:     str = ""
    forecaster: str = ""


class MetricsHistoryResponse(BaseModel):
    total:      int
    policy:     str | None = None
    forecaster: str | None = None
    records:    list[MetricPoint]


# ── Policy decide ─────────────────────────────────────────────────────────────

class ForecastMeta(BaseModel):
    source: str
    steps:  int
    peak:   float
    mean:   float
    values: list[float] = []


class PolicyDecideRequest(BaseModel):
    current_rps:      float              = Field(...,   ge=0)
    current_replicas: int                = Field(...,   ge=1)
    step:             int                = Field(0,     ge=0)
    forecast:         list[float] | None = None
    policy:           str                = "mpc"
    forecaster:       str                = "none"
    lambda_sla:       float | None       = None
    lambda_cost:      float | None       = None
    lambda_stab:      float | None       = None
    forecast_margin:  float | None       = None

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "current_rps": 250.0, "current_replicas": 3, "step": 42,
            "policy": "mpc", "forecaster": "none",
            "forecast": [260.0, 310.0, 280.0],
        }
    })


class PolicyDecideResponse(BaseModel):
    desired_replicas: int
    policy:           str
    forecaster:       str
    reactive_floor:   int
    proactive_target: int
    decision_ms:      float
    forecast_meta:    ForecastMeta


class PolicyStatusResponse(BaseModel):
    active_policy:         str
    active_forecaster:     str
    available_policies:    list[str]
    available_forecasters: list[str]
    simulator_config:      dict[str, Any]
    policy_config:         dict[str, Any]


# ── Experiment runs ───────────────────────────────────────────────────────────

class RunSubmitRequest(BaseModel):
    policy:           str   = Field("mpc")
    forecaster:       str   = Field("lstm")
    workload:         str   = Field(
        "diurnal_burst",
        description="Synthetic pattern: diurnal_burst | smooth | bursty | bimodal | flash_crowd",
    )
    cold_start_s:     float = Field(120.0, ge=30,   le=600)
    train_frac:       float = Field(0.70,  ge=0.5,  le=0.9)
    val_frac:         float = Field(0.10,  ge=0.05, le=0.2)
    forecast_margin:  float = Field(1.15,  ge=0.5,  le=3.0)
    lambda_sla:       float = Field(50.0,  ge=1.0,  le=1000.0)
    lambda_cost:      float = Field(1.0,   ge=0.1,  le=100.0)
    lambda_stab:      float = Field(0.5,   ge=0.0,  le=10.0)
    cold_start_steps: int   = Field(0,     ge=0,    le=10)
    use_fh_opt:       bool = Field(False, description="Enable ADAPT dynamic horizon")
    seed:             int   = Field(42)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "policy": "mpc", "forecaster": "lstm",
            "workload": "diurnal_burst", "cold_start_s": 120.0,
            "forecast_margin": 1.15, "lambda_sla": 50.0,
            "lambda_cost": 1.0, "lambda_stab": 0.5,
            "cold_start_steps": 0, "use_fh_opt": False, "seed": 42,
        }
    })


class RunSubmitResponse(BaseModel):
    run_id:  str
    status:  str
    message: str


class RunDetailResponse(BaseModel):
    run_id:           str
    status:           str
    policy:           str
    forecaster:       str
    workload:         str
    cold_start_s:     float
    forecast_margin:  float
    lambda_sla:       float
    lambda_cost:      float
    lambda_stab:      float
    cold_start_steps: int
    use_fh_opt:       bool
    seed:             int
    created_at:       float
    started_at:       float | None
    ended_at:         float | None
    duration_s:       float | None
    error:            str | None
    summary:          dict[str, Any]


class RunListResponse(BaseModel):
    total:  int
    offset: int
    limit:  int
    runs:   list[RunDetailResponse]


# ── Results ───────────────────────────────────────────────────────────────────

class RunResult(BaseModel):
    policy:        str
    forecaster:    str
    sla_pct:       float
    avg_rep:       float
    avg_lat_ms:    float
    total_cost:    float
    peak_replicas: int = 0
    steps:         int = 0


class ResultsLatestResponse(BaseModel):
    runs:   list[RunResult]
    source: str
    count:  int
