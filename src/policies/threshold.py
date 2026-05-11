# src/policies/threshold.py
"""
Threshold (HPA-pure) Policy — K8s HPA formula with stabilization window.

Formula:
  desired = ceil(current_rps / target_rps_per_replica)

Stabilization:
  scale_down_stabilization: steps before allowing a scale-down
  (K8s default = 5 min = 5 steps at 1-min timestep)
  scale_up is always immediate (K8s default).

Paper role: Baseline 1 — raw K8s HPA behavior.
"""
import math
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_POL_CFG = CONFIG.get("policies", {}).get("threshold", {})


class ThresholdPolicy(BasePolicy):

    def __init__(
        self,
        target_rps_per_replica: float | None = None,
        scale_down_stabilization: int | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
    ):
        super().__init__(min_replicas, max_replicas)
        cap = CONFIG["simulator"]["capacity_per_replica"]
        self.target_rps_per_replica = (
            target_rps_per_replica
            if target_rps_per_replica is not None
            else _POL_CFG.get("target_rps_per_replica", cap)
        )
        self.scale_down_stabilization = (
            scale_down_stabilization
            if scale_down_stabilization is not None
            else _POL_CFG.get("scale_down_stabilization", 5)
        )
        if self.target_rps_per_replica <= 0:
            raise ValueError(
                f"target_rps_per_replica must be > 0, "
                f"got {self.target_rps_per_replica}"
            )
        if self.scale_down_stabilization < 1:
            raise ValueError(
                f"scale_down_stabilization must be >= 1, "
                f"got {self.scale_down_stabilization}"
            )

        self._last_scale_down_step = -999

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,   # source of truth — used directly
        step: int,
        **context,
    ) -> int:
        desired = self._clamp(current_rps / self.target_rps_per_replica)

        if desired < current_replicas:
            # Scale down: enforce stabilization window
            if step - self._last_scale_down_step < self.scale_down_stabilization:
                return current_replicas   # blocked — return actual current
            self._last_scale_down_step = step

        return desired

    def reset(self):
        self._last_scale_down_step = -999

    def __repr__(self):
        return (
            f"ThresholdPolicy(target={self.target_rps_per_replica} RPS/replica, "
            f"scale_down_stab={self.scale_down_stabilization})"
        )