# src/policies/hpa.py
import math
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_POL_CFG = CONFIG.get("policies", {}).get("hpa", {})


class HPAPolicy(BasePolicy):

    def __init__(
        self,
        target_rps_per_replica: float | None = None,
        scale_up_cooldown: int | None = None,
        scale_down_cooldown: int | None = None,
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
        self.scale_up_cooldown = (
            scale_up_cooldown
            if scale_up_cooldown is not None
            else _POL_CFG.get("scale_up_cooldown", 1)
        )
        self.scale_down_cooldown = (
            scale_down_cooldown
            if scale_down_cooldown is not None
            else _POL_CFG.get("scale_down_cooldown", 5)
        )

        if self.target_rps_per_replica <= 0:
            raise ValueError(
                f"target_rps_per_replica must be > 0, "
                f"got {self.target_rps_per_replica}"
            )
        if self.scale_up_cooldown < 1 or self.scale_down_cooldown < 1:
            raise ValueError(
                "scale_up_cooldown and scale_down_cooldown must be >= 1"
            )

        # Track last step ANY scaling event occurred
        # Scale-down is blocked for scale_down_cooldown steps
        # after either a scale-up OR a scale-down event
        self._last_scale_up_step   = -999
        self._last_scale_down_step = -999

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **context,
    ) -> int:
        desired = self._clamp(current_rps / self.target_rps_per_replica)

        if desired > current_replicas:
            if step - self._last_scale_up_step >= self.scale_up_cooldown:
                self._last_scale_up_step = step
                return desired
            return current_replicas

        elif desired < current_replicas:
            # Scale-down blocked if within cooldown of EITHER last scale-up
            # OR last scale-down — prevents thrashing after a spike
            steps_since_up   = step - self._last_scale_up_step
            steps_since_down = step - self._last_scale_down_step
            if (steps_since_up  >= self.scale_down_cooldown and
                    steps_since_down >= self.scale_down_cooldown):
                self._last_scale_down_step = step
                return desired
            return current_replicas

        return current_replicas

    def reset(self):
        self._last_scale_up_step   = -999
        self._last_scale_down_step = -999

    def __repr__(self):
        return (
            f"HPAPolicy(target={self.target_rps_per_replica} RPS/replica, "
            f"up_cd={self.scale_up_cooldown}, down_cd={self.scale_down_cooldown})"
        )