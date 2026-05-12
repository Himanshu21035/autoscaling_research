"""
GET /health  — liveness probe  (always 200 if process is alive)
GET /ready   — readiness probe (validates registries + config)
"""
from fastapi import APIRouter
from src.api.schemas import HealthResponse, ReadyResponse
from src.policies import POLICY_REGISTRY
from src.forecasting import available_forecasters

router  = APIRouter()
VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


@router.get("/ready", response_model=ReadyResponse, tags=["Health"])
def ready() -> ReadyResponse:
    """
    Validates:
      - config.yaml loaded with required simulator keys
      - policy registry has all 4 expected policies
      - at least one forecaster dependency is installed
      - AutoscalerSimulator can be instantiated
    """
    errors: list[str] = []

    # 1. Config
    try:
        from src.config import CONFIG
        required = ["capacity_per_replica", "timestep_seconds", "cold_start_s",
                    "sla_latency_ms", "min_replicas", "max_replicas"]
        missing = [k for k in required if k not in CONFIG["simulator"]]
        if missing:
            errors.append(f"Missing simulator config keys: {missing}")
    except Exception as e:
        errors.append(f"Config load failed: {e}")

    # 2. Policy registry
    try:
        expected = {"hpa", "mpc", "pid", "threshold"}
        found    = set(POLICY_REGISTRY.keys())
        if not expected.issubset(found):
            errors.append(f"Policy registry missing: {expected - found}")
    except Exception as e:
        errors.append(f"Policy registry error: {e}")

    # 3. Forecaster registry
    avail: list[str] = []
    try:
        avail = available_forecasters()
        if not avail:
            errors.append("No forecaster dependencies installed (pmdarima/prophet/torch)")
    except Exception as e:
        errors.append(f"Forecaster registry error: {e}")

    # 4. Simulator instantiation
    try:
        from src.simulator.core import AutoscalerSimulator
        sim = AutoscalerSimulator()
        assert sim.capacity_per_replica > 0
    except Exception as e:
        errors.append(f"Simulator instantiation failed: {e}")

    if errors:
        return ReadyResponse(
            ready=False,
            detail="; ".join(errors),
            available_policies=list(POLICY_REGISTRY.keys()),
            available_forecasters=avail,
        )

    return ReadyResponse(
        ready=True,
        detail="all systems nominal",
        available_policies=list(POLICY_REGISTRY.keys()),
        available_forecasters=avail,
    )
