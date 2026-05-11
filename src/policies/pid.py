# src/policies/pid.py
"""
PID Controller Policy.

Error signal: utilization error
  capacity  = current_replicas × capacity_per_replica
  error     = current_rps - capacity   (negative = over-provisioned)

This correctly reflects actual provisioning state:
  5 replicas × 100 RPS = 500 capacity
  450 RPS arriving → error = -50 (slight over-provision, fine)
  550 RPS arriving → error = +50 (under-provisioned, scale up)

A fixed global target_rps (the previous approach) would have given:
  error = 450 - 300 = +150 → PID thinks it's massively under-provisioned
  when actually it has spare capacity. This was wrong.

Anti-windup:
  Integral only accumulates when output is NOT saturated (not at min/max).
  This prevents integral runaway at the limits.

Paper role: Baseline 2 — adaptive reactive controller.
"""
import math
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_POL_CFG = CONFIG.get("policies", {}).get("pid", {})


class PIDPolicy(BasePolicy):

    def __init__(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        capacity_per_replica: float | None = None,
        integral_limit: float | None = None,
        derivative_smoothing: float | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
    ):
        super().__init__(min_replicas, max_replicas)
        self.capacity_per_replica = (
            capacity_per_replica
            if capacity_per_replica is not None
            else CONFIG["simulator"]["capacity_per_replica"]
        )
        self.kp = kp if kp is not None else _POL_CFG.get("kp", 0.5)
        self.ki = ki if ki is not None else _POL_CFG.get("ki", 0.1)
        self.kd = kd if kd is not None else _POL_CFG.get("kd", 0.05)
        self.integral_limit = (
            integral_limit
            if integral_limit is not None
            else _POL_CFG.get("integral_limit", 500.0)
        )
        # EMA smoothing on derivative (alpha=1.0 = no smoothing)
        self.derivative_smoothing = (
            derivative_smoothing
            if derivative_smoothing is not None
            else _POL_CFG.get("derivative_smoothing", 0.3)
        )

        # State
        self._integral        = 0.0
        self._prev_error      = 0.0
        self._smoothed_deriv  = 0.0

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **context,
    ) -> int:
        # Error = RPS - current capacity (not vs fixed target)
        capacity = current_replicas * self.capacity_per_replica
        error    = current_rps - capacity

        # Smoothed derivative via EMA (reduces noise amplification)
        raw_deriv            = error - self._prev_error
        self._smoothed_deriv = (
            self.derivative_smoothing * raw_deriv
            + (1 - self.derivative_smoothing) * self._smoothed_deriv
        )
        self._prev_error = error

        pid_output    = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * self._smoothed_deriv
        )

        # Convert to replica delta and compute desired
        replica_delta = pid_output / self.capacity_per_replica
        desired_raw   = current_replicas + replica_delta
        desired       = self._clamp(desired_raw)   # ceil + clamp

        # Conditional integral: only accumulate when NOT saturated
        # This prevents integral windup at min/max boundaries
        is_saturated = (
            desired == self.min_replicas or desired == self.max_replicas
        )
        if not is_saturated:
            self._integral = max(
                -self.integral_limit,
                min(self.integral_limit, self._integral + error)
            )

        return desired

    def reset(self):
        self._integral       = 0.0
        self._prev_error     = 0.0
        self._smoothed_deriv = 0.0

    def __repr__(self):
        return (
            f"PIDPolicy(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"capacity={self.capacity_per_replica} RPS/replica)"
        )