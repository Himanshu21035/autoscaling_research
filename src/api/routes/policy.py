"""
POST /policy/decide          — stateless single-step decision (with real forecaster)
GET  /policy/status          — registry + active config
GET  /policy/results/latest  — last smoke-test run summaries
"""
from __future__ import annotations
import math
import time
import numpy as np
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    PolicyDecideRequest, PolicyDecideResponse, ForecastMeta,
    PolicyStatusResponse, ResultsLatestResponse, RunResult,
)
from src.api.store import results_store
from src.config import CONFIG
from src.policies import POLICY_REGISTRY
from src.forecasting import available_forecasters

router = APIRouter()

_SIM = CONFIG["simulator"]
_CAP = float(_SIM["capacity_per_replica"])
_MIN = int(_SIM.get("min_replicas", 1))
_MAX = int(_SIM.get("max_replicas", 50))

_VALID_POLICIES    = sorted(POLICY_REGISTRY.keys())
_VALID_FORECASTERS = ["arima", "prophet", "lstm", "none"]


@router.post("/decide", response_model=PolicyDecideResponse, tags=["Policy"])
def policy_decide(req: PolicyDecideRequest) -> PolicyDecideResponse:
    """
    Run one stateless scaling decision.

    If forecaster != 'none' and no forecast list is provided, the forecaster
    is instantiated and a short horizon is predicted from the current RPS
    (flat seed — useful for demo/API testing without a pre-fitted model).

    For production accuracy, pass a pre-computed forecast list.
    """
    t0 = time.perf_counter()

    # ── Validate policy ───────────────────────────────────────────────
    policy_key = req.policy.strip().lower()
    if policy_key not in POLICY_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown policy '{req.policy}'. Valid: {_VALID_POLICIES}",
        )

    # ── Build policy kwargs from request overrides ────────────────────
    policy_kwargs: dict = {}
    if policy_key == "mpc":
        if req.lambda_sla      is not None: policy_kwargs["lambda_sla"]      = req.lambda_sla
        if req.lambda_cost     is not None: policy_kwargs["lambda_cost"]     = req.lambda_cost
        if req.lambda_stab     is not None: policy_kwargs["lambda_stab"]     = req.lambda_stab
        if req.forecast_margin is not None: policy_kwargs["forecast_margin"] = req.forecast_margin

    from src.policies import create_policy
    try:
        policy = create_policy(policy_key, **policy_kwargs)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Resolve forecast ──────────────────────────────────────────────
    forecast_key = req.forecaster.strip().lower()
    forecast_arr: np.ndarray | None = None
    forecast_source = "none"

    if req.forecast:
        # Caller provided explicit forecast values — use them directly
        forecast_arr    = np.array(req.forecast, dtype=float)
        forecast_source = "provided"

    elif forecast_key != "none":
        # Caller wants a forecaster but didn't supply values:
        # generate a short flat-seed forecast (demo mode)
        if forecast_key not in ["arima", "prophet", "lstm"]:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown forecaster '{req.forecaster}'. "
                       f"Valid: {_VALID_FORECASTERS}",
            )
        try:
            from src.forecasting import create_forecaster
            fc   = create_forecaster(forecast_key)
            seed = np.full(20, req.current_rps)   # flat seed from current RPS
            fc.fit(seed)
            horizon      = max(1, math.ceil(
                float(_SIM.get("cold_start_s", 120))
                / float(_SIM.get("timestep_seconds", 60))
            ) + 1)
            forecast_arr    = fc.predict(horizon)
            forecast_source = "generated"
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Forecaster '{forecast_key}' dependency not installed: {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Forecaster '{forecast_key}' failed: {exc}",
            )

    # ── Compute decision ──────────────────────────────────────────────
    try:
        desired = policy.compute_replicas(
            current_rps=req.current_rps,
            current_replicas=req.current_replicas,
            step=req.step,
            forecast=forecast_arr,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Policy compute failed: {exc}")

    # ── Diagnostics ───────────────────────────────────────────────────
    reactive_floor   = max(_MIN, math.ceil(req.current_rps / max(_CAP, 1.0)))
    peak             = float(np.max(forecast_arr)) \
                       if forecast_arr is not None and len(forecast_arr) > 0 \
                       else req.current_rps
    proactive_target = max(reactive_floor, math.ceil(peak / max(_CAP, 1.0)))

    forecast_meta = ForecastMeta(
        source  = forecast_source,
        steps   = len(forecast_arr) if forecast_arr is not None else 0,
        peak    = round(peak, 2),
        mean    = round(float(np.mean(forecast_arr)), 2)
                  if forecast_arr is not None and len(forecast_arr) > 0 else 0.0,
        values  = forecast_arr.tolist() if forecast_arr is not None else [],
    )

    return PolicyDecideResponse(
        desired_replicas = desired,
        policy           = policy_key,
        forecaster       = forecast_key,
        reactive_floor   = reactive_floor,
        proactive_target = proactive_target,
        decision_ms      = round((time.perf_counter() - t0) * 1000, 3),
        forecast_meta    = forecast_meta,
    )


@router.get("/status", response_model=PolicyStatusResponse, tags=["Policy"])
def policy_status() -> PolicyStatusResponse:
    """Return available registries and active config."""
    return PolicyStatusResponse(
        active_policy         = CONFIG.get("policies",    {}).get("default", "mpc"),
        active_forecaster     = CONFIG.get("forecasting", {}).get("default", "lstm"),
        available_policies    = _VALID_POLICIES,
        available_forecasters = available_forecasters(),
        simulator_config      = dict(_SIM),
        policy_config         = CONFIG.get("policies", {}),
    )


@router.get("/results/latest", response_model=ResultsLatestResponse, tags=["Results"])
def results_latest() -> ResultsLatestResponse:
    """Return smoke-test run summaries (populated by smoke_test_step9.py)."""
    runs = results_store.get_all()
    if not runs:
        return ResultsLatestResponse(runs=[], source="none", count=0)

    # Map SimResult.summary() keys → RunResult fields
    parsed = []
    for r in runs:
        parsed.append(RunResult(
            policy     = r.get("policy",         ""),
            forecaster = r.get("forecaster",      ""),
            sla_pct    = r.get("sla_pct",         0.0),
            avg_rep    = r.get("avg_replicas",    0.0),
            avg_lat_ms = r.get("avg_latency_ms",  0.0),
            total_cost = r.get("total_cost",      0.0),
            peak_replicas = r.get("peak_replicas", 0),
            steps      = r.get("steps",            0),
        ))

    return ResultsLatestResponse(runs=parsed, source="cache", count=len(parsed))
