"""
/v1/runs — experiment job management

POST /v1/runs               — submit a new experiment run
GET  /v1/runs               — list all runs (filterable by status)
GET  /v1/runs/{run_id}      — get a specific run
DELETE /v1/runs/{run_id}    — cancel a pending run (best-effort)
"""
from fastapi import APIRouter, HTTPException, Query
from src.api.schemas import (
    RunSubmitRequest, RunSubmitResponse,
    RunDetailResponse, RunListResponse,
)
from src.api.models import ExperimentRun, RunStatus
from src.api.database import run_db
from src.api.runner import submit_run
from src.policies import POLICY_REGISTRY
from src.forecasting import available_forecasters

router = APIRouter()

_VALID_POLICIES    = set(POLICY_REGISTRY.keys())
_VALID_WORKLOADS   = {"diurnal_burst", "smooth", "bursty", "bimodal", "flash_crowd", "slow_ramp_up", "periodic_spikes"}


@router.post("", response_model=RunSubmitResponse, status_code=202, tags=["Runs"])
def submit(req: RunSubmitRequest) -> RunSubmitResponse:
    """
    Submit an experiment run.

    Returns immediately with a run_id. Poll GET /v1/runs/{run_id}
    for status. The simulation runs in a background thread.
    """
    if req.policy not in _VALID_POLICIES:
        raise HTTPException(422, f"Unknown policy '{req.policy}'. Valid: {sorted(_VALID_POLICIES)}")
    if req.forecaster not in {*available_forecasters(), "none"}:
        raise HTTPException(422, f"Unknown/unavailable forecaster '{req.forecaster}'.")
    if req.workload not in _VALID_WORKLOADS:
        raise HTTPException(422, f"Unknown workload '{req.workload}'. Valid: {sorted(_VALID_WORKLOADS)}")

    run = ExperimentRun(
        policy=req.policy, forecaster=req.forecaster,
        batch=req.batch,
        workload=req.workload, cold_start_s=req.cold_start_s,
        train_frac=req.train_frac, val_frac=req.val_frac,
        forecast_margin=req.forecast_margin,
        lambda_sla=req.lambda_sla, lambda_cost=req.lambda_cost,
        lambda_stab=req.lambda_stab, cold_start_steps=req.cold_start_steps,
        use_fh_opt=req.use_fh_opt,
        seed=req.seed,
    )

    submit_run(run)

    return RunSubmitResponse(
        run_id=run.run_id,
        status=RunStatus.PENDING.value,
        message=f"Run {run.run_id} queued. "
                f"Poll GET /v1/runs/{run.run_id} for status.",
    )


@router.get("", response_model=RunListResponse, tags=["Runs"])
def list_runs(
    status: str | None = Query(None, description="Filter: pending|running|completed|failed"),
    limit:  int        = Query(20,   ge=1, le=200),
    offset: int        = Query(0,    ge=0),
) -> RunListResponse:
    """List experiment runs, newest first."""
    if status and status not in RunStatus._value2member_map_:
        raise HTTPException(422, f"Invalid status '{status}'.")
    rows  = run_db.list_all(status=status, limit=limit, offset=offset)
    total = run_db.count(status=status)
    return RunListResponse(
        total=total, offset=offset, limit=limit,
        runs=[_row_to_detail(r) for r in rows],
    )


@router.get("/{run_id}", response_model=RunDetailResponse, tags=["Runs"])
def get_run(run_id: str) -> RunDetailResponse:
    """Get a specific experiment run by ID."""
    row = run_db.get(run_id)
    if not row:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    return _row_to_detail(row)


@router.delete("/{run_id}", status_code=204, tags=["Runs"])
def cancel_run(run_id: str) -> None:
    """
    Cancel a PENDING run.
    RUNNING runs cannot be cancelled (no preemption in ThreadPoolExecutor).
    """
    row = run_db.get(run_id)
    if not row:
        raise HTTPException(404, f"Run '{run_id}' not found.")
    if row["status"] == RunStatus.RUNNING.value:
        raise HTTPException(409, "Cannot cancel a running job.")
    if row["status"] in (RunStatus.COMPLETED.value, RunStatus.FAILED.value):
        raise HTTPException(409, f"Run already {row['status']}.")
    run_db.update_status(run_id, RunStatus.FAILED, error="Cancelled by user.")


def _row_to_detail(row: dict) -> RunDetailResponse:
    return RunDetailResponse(
        run_id=row["run_id"], status=row["status"],
        policy=row.get("policy", ""),
        forecaster=row.get("forecaster", ""),
        batch=row.get("batch", ""),
        workload=row.get("workload", ""),
        cold_start_s=row.get("cold_start_s", 120.0),
        forecast_margin=row.get("forecast_margin", 1.15),
        lambda_sla=row.get("lambda_sla", 50.0),
        lambda_cost=row.get("lambda_cost", 1.0),
        lambda_stab=row.get("lambda_stab", 0.5),
        cold_start_steps=row.get("cold_start_steps", 0),
        use_fh_opt=row.get("use_fh_opt", False),
        seed=row.get("seed", 42),
        created_at=row.get("created_at", 0.0),
        started_at=row.get("started_at"),
        ended_at=row.get("ended_at"),
        duration_s=row.get("duration_s"),
        error=row.get("error"),
        summary=row.get("summary", {}),
    )
