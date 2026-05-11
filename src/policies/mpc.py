# src/policies/mpc.py
"""
MPC Policy — Model Predictive Control with cold start co-optimization.

Core innovation:
  - Reactive floor: never goes below what current RPS demands
  - Proactive target: scales ahead of forecast peak by cold_start_steps
  - ADAPT integration: uses live cold start estimate for horizon
  - Multi-objective: lambda_sla >> lambda_cost so SLA dominates
"""
import math
import numpy as np
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("policies", {}).get("mpc", {})
_SIM = CONFIG["simulator"]


class MPCPolicy(BasePolicy):

    def __init__(
        self,
        adapt_tracker=None,
        lambda_sla:          float | None = None,
        lambda_cost:         float | None = None,
        lambda_stab:         float | None = None,
        forecast_margin:     float | None = None,
        capacity_per_replica: float | None = None,
        min_replicas:        int   | None = None,
        max_replicas:        int   | None = None,
        max_scale_rate:      int = 5,
        cold_start_steps: int | None = None,
    ):
        super().__init__("MPC")

        self.adapt_tracker = adapt_tracker

        # ── Weights ────────────────────────────────────────────────────
        self.lambda_sla  = lambda_sla  if lambda_sla  is not None else float(_CFG.get("lambda_sla",  50.0))
        self.lambda_cost = lambda_cost if lambda_cost is not None else float(_CFG.get("lambda_cost",  1.0))
        self.lambda_stab = lambda_stab if lambda_stab is not None else float(_CFG.get("lambda_stab",  0.5))

        # ── Capacity / replica bounds ──────────────────────────────────
        self.capacity_per_replica = (
            capacity_per_replica if capacity_per_replica is not None
            else float(_SIM["capacity_per_replica"])
        )
        self.min_replicas = (
            int(min_replicas) if min_replicas is not None
            else int(_SIM.get("min_replicas", 1))
        )
        self.max_replicas = (
            int(max_replicas) if max_replicas is not None
            else int(_SIM.get("max_replicas", 50))
        )
        self.max_scale_rate = max_scale_rate

        # ── Forecast margin ────────────────────────────────────────────
        self.forecast_margin = (
            forecast_margin if forecast_margin is not None
            else float(_CFG.get("forecast_margin", 1.15))
        )

        self.cold_start_steps = (
            int(cold_start_steps) if cold_start_steps is not None
            else int(_CFG.get("cold_start_steps",
                    math.ceil(float(_SIM.get("cold_start_s", 120))
                            / float(_SIM.get("timestep_seconds", 60)))))
        )

        logger.info(
            f"MPCPolicy init | lam_sla={self.lambda_sla} "
            f"lam_cost={self.lambda_cost} lam_stab={self.lambda_stab} "
            f"cold_start_steps={self.cold_start_steps} "
            f"forecast_margin={self.forecast_margin} "
            f"adapt={'yes' if adapt_tracker else 'no'}"
        )

    def compute_replicas(
        self,
        current_rps:      float,
        current_replicas: int,
        step:             int,
        forecast:         np.ndarray | None = None,
        warm_replicas:    int | None = None,
        **kwargs,
    ) -> int:
        warm = warm_replicas if warm_replicas is not None else current_replicas

        # ── 1. Reactive floor — hard lower bound from current load ─────
        reactive_floor = max(
            self.min_replicas,
            math.ceil(current_rps / max(self.capacity_per_replica, 1.0)),
        )

        # ── 2. Proactive target — cold-start-aware peak lookahead ──────
        if forecast is not None and len(forecast) > 0:
            # Skip the first cold_start_steps (already committed/warming),
            # then look at the next half-horizon for the upcoming peak.
            offset  = self.cold_start_steps
            usable  = forecast[offset:] if len(forecast) > offset else forecast
            cutoff  = max(1, len(usable) // 2)
            raw     = float(np.max(usable))
            peak_forecast = raw * self.forecast_margin
        else:
            peak_forecast = current_rps * self.forecast_margin

        proactive_target = max(
            reactive_floor,
            math.ceil(peak_forecast / max(self.capacity_per_replica, 1.0)),
        )

        # ── 3. Search over candidate replica counts ────────────────────
        lo = max(self.min_replicas, warm - self.max_scale_rate)
        hi = min(self.max_replicas, warm + self.max_scale_rate)

        best_replicas = warm
        best_cost     = float("inf")

        for r in range(lo, hi + 1):
            capacity = r * self.capacity_per_replica
            util     = current_rps / max(capacity, 1.0)

            sla_pen       = self.lambda_sla  * (max(0.0, util - 1.0) ** 2) * 1000.0
            cost_pen      = self.lambda_cost * (r / self.max_replicas)
            stab_pen      = self.lambda_stab * (abs(r - warm) / self.max_replicas)
            proactive_gap = max(0, proactive_target - r)
            proactive_pen = self.lambda_sla  * 1.0 * (proactive_gap / self.max_replicas)

            total = sla_pen + cost_pen + stab_pen + proactive_pen

            if total < best_cost:
                best_cost     = total
                best_replicas = r

        # ── 4. Reactive floor — hard constraint, always enforced ───────
        final = max(best_replicas, reactive_floor)
        return max(self.min_replicas, min(self.max_replicas, final))

    def reset(self) -> None:
        if self.adapt_tracker:
            self.adapt_tracker.reset()