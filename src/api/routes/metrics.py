"""
GET /metrics/current          — latest step for a policy+forecaster combo
GET /metrics/history?n=100    — last N steps
Both support ?policy= and ?forecaster= query filters.
"""
from fastapi import APIRouter, Query, HTTPException
from src.api.schemas import (
    MetricsCurrentResponse, MetricsHistoryResponse, MetricPoint,
)
from src.api.store import metrics_store

router = APIRouter()

_POLICY_HINT     = "Filter by policy name e.g. mpc, hpa, pid, threshold"
_FORECASTER_HINT = "Filter by forecaster name e.g. lstm, arima, prophet"


@router.get("/current", response_model=MetricsCurrentResponse, tags=["Metrics"])
def metrics_current(
    policy:     str | None = Query(None, description=_POLICY_HINT),
    forecaster: str | None = Query(None, description=_FORECASTER_HINT),
) -> MetricsCurrentResponse:
    """Return the most recent step metrics, optionally filtered."""
    record = metrics_store.latest(policy=policy, forecaster=forecaster)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No metrics yet. Run smoke_test_step9.py first, "
                   "or check your policy/forecaster filter.",
        )
    return MetricsCurrentResponse(**_to_metric_point(record))


@router.get("/history", response_model=MetricsHistoryResponse, tags=["Metrics"])
def metrics_history(
    n:          int        = Query(100, ge=1, le=50_000, description="Records to return"),
    policy:     str | None = Query(None, description=_POLICY_HINT),
    forecaster: str | None = Query(None, description=_FORECASTER_HINT),
) -> MetricsHistoryResponse:
    """Return the last N step metrics, optionally filtered."""
    records = metrics_store.last_n(n, policy=policy, forecaster=forecaster)
    return MetricsHistoryResponse(
        total=len(records),
        policy=policy,
        forecaster=forecaster,
        records=[MetricPoint(**_to_metric_point(r)) for r in records],
    )


def _to_metric_point(r: dict) -> dict:
    """Normalise store record keys to MetricPoint field names."""
    return {
        "step":             r["step"],
        "rps":              r["rps"],
        "replicas":         r["replicas"],
        "latency_ms":       r["latency_ms"],
        "sla_violated":     r["sla_violated"],
        "cost":             r["cost"],
        "utilisation":      r.get("utilisation", 0.0),
        "adapt_estimate_s": r.get("adapt_estimate_s", 0.0),
    }
