# src/policies/mpc.py
"""
MPC Policy with ADAPT + FH OPT — Core Novel Contribution.

TRUE DELAY-AWARE MPC:
  A replica ordered at step k becomes ready at step k + cold_start_steps,
  where cold_start_steps = ceil(Δ̂_cold / timestep).

  This means the optimizer must place orders h* steps BEFORE they are
  needed — not when demand arrives. This is the core paper claim:
  proactive scaling with cold-start-aware delay modelling.

Horizon state evolution (per candidate decision r_now):
  capacity[k] = (warm_replicas + ordered_replicas_ready_by_k) × cap_per_replica
  where ordered_replicas_ready_by_k = r_now if k >= cold_start_steps else 0

Objective per horizon step k:
  cost += λ_sla  × max(0, demand[k] - capacity[k])²   SLA penalty
        + λ_cost × r_now                               running cost
        + λ_stab × (r_now - r_prev)²                   stability

Solver: exhaustive integer search over [min_replicas, max_replicas].
  Exact for integer problems with ≤ 50 replicas. O(N×h*) per step.
  No cvxpy required — avoids a heavy dependency for a 1D integer problem.

Context keys:
  forecast             : np.ndarray  — RPS, length >= h*
  warm_replicas        : int         — replicas already past cold start
  observed_cold_start_s: float       — triggers ADAPT.observe() if present
"""
import math
import numpy as np
from src.policies.base import BasePolicy
from src.simulator.adapt import ADAPTTracker
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_MPC_CFG = CONFIG.get("policies", {}).get("mpc", {})
_SIM_CFG = CONFIG["simulator"]


class MPCPolicy(BasePolicy):

    def __init__(
        self,
        adapt_tracker: ADAPTTracker | None = None,
        lambda_sla: float | None = None,
        lambda_cost: float | None = None,
        lambda_stab: float | None = None,
        capacity_per_replica: float | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
    ):
        super().__init__(min_replicas, max_replicas)

        self.adapt = adapt_tracker or ADAPTTracker()
        self.lambda_sla  = (
            lambda_sla  if lambda_sla  is not None
            else _MPC_CFG.get("lambda_sla",  10.0)
        )
        self.lambda_cost = (
            lambda_cost if lambda_cost is not None
            else _MPC_CFG.get("lambda_cost", 1.0)
        )
        self.lambda_stab = (
            lambda_stab if lambda_stab is not None
            else _MPC_CFG.get("lambda_stab", 0.5)
        )
        self.capacity_per_replica = (
            capacity_per_replica if capacity_per_replica is not None
            else _SIM_CFG["capacity_per_replica"]
        )

        if any(v < 0 for v in [self.lambda_sla, self.lambda_cost, self.lambda_stab]):
            raise ValueError("All lambda weights must be >= 0")

        self._prev_replicas: int = self.min_replicas

    # ── Core Solver ────────────────────────────────────────────────────

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **context,
    ) -> int:
        forecast: np.ndarray = context.get("forecast", None)
        warm_replicas: int   = context.get("warm_replicas", current_replicas)

        # Update ADAPT if a new observation is available
        if "observed_cold_start_s" in context:
            self.adapt.observe(context["observed_cold_start_s"])

        # FH OPT: get dynamic horizon
        h_star = self.adapt.optimal_horizon()

        # Cold start delay in steps — how long before ordered replicas are ready
        cold_start_steps = max(
            1, math.ceil(self.adapt.estimate_s / _SIM_CFG["timestep_seconds"])
        )

        # Build forecast of length h*
        if forecast is None or len(forecast) == 0:
            forecast = np.full(h_star, current_rps)
        if len(forecast) < h_star:
            forecast = np.concatenate([
                forecast,
                np.full(h_star - len(forecast), forecast[-1])
            ])
        forecast = forecast[:h_star]

        # Exhaustive integer search: find replica count minimising objective
        best_action = self._clamp(current_rps / self.capacity_per_replica)
        best_cost   = float("inf")

        for candidate in range(self.min_replicas, self.max_replicas + 1):
            cost = self._objective(
                candidate, forecast, warm_replicas,
                h_star, cold_start_steps
            )
            if cost < best_cost:
                best_cost   = cost
                best_action = candidate

        self._prev_replicas = best_action
        logger.debug(
            f"MPC step={step} h*={h_star} "
            f"cold_start_steps={cold_start_steps} "
            f"Δ̂={self.adapt.estimate_s:.0f}s "
            f"-> {best_action} replicas (cost={best_cost:.2f})"
        )
        return best_action

    def _objective(
        self,
        replicas_now: int,
        forecast: np.ndarray,
        warm_replicas: int,
        h_star: int,
        cold_start_steps: int,
    ) -> float:
        """
        Delay-aware MPC objective over h* steps.

        Capacity at each step k:
          k < cold_start_steps: only warm_replicas available
                                (ordered replicas still booting)
          k >= cold_start_steps: warm_replicas + replicas_now fully ready
        """
        total_cost = 0.0
        prev = self._prev_replicas

        for k, demand in enumerate(forecast):
            # Delay model: ordered replicas only available after cold start
            if k < cold_start_steps:
                effective_replicas = warm_replicas
            else:
                effective_replicas = replicas_now

            effective_capacity = effective_replicas * self.capacity_per_replica

            shortfall  = max(0.0, demand - effective_capacity)
            sla_cost   = self.lambda_sla  * shortfall ** 2
            run_cost   = self.lambda_cost * replicas_now
            stab_cost  = self.lambda_stab * (replicas_now - prev) ** 2

            total_cost += sla_cost + run_cost + stab_cost
            prev = replicas_now

        return total_cost

    def reset(self):
        self.adapt.reset()
        self._prev_replicas = self.min_replicas

    def __repr__(self):
        return (
            f"MPCPolicy(lambda_sla={self.lambda_sla}, "
            f"lambda_cost={self.lambda_cost}, "
            f"lambda_stab={self.lambda_stab}, "
            f"h*={self.adapt.optimal_horizon()})"
        )