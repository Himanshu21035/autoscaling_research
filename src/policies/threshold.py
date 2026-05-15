import math
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("policies", {}).get("threshold", {})
_SIM = CONFIG.get("simulator", {})


class ThresholdPolicy(BasePolicy):

    def __init__(
        self,
        target_rps_per_replica: float | None = None,
        scale_down_stabilization: int | None = None,
    ):
        super().__init__("Threshold")

        self.target_rps_per_replica = (
            target_rps_per_replica if target_rps_per_replica is not None
            else float(_CFG.get("target_rps_per_replica",
                       _SIM.get("capacity_per_replica", 100.0)))
        )
        self.scale_down_stabilization = (
            scale_down_stabilization if scale_down_stabilization is not None
            else int(_CFG.get("scale_down_stabilization", 3))
        )
        self._scale_down_counter = 0

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **kwargs,
    ) -> int:
        desired = math.ceil(current_rps / max(self.target_rps_per_replica, 1.0))
        desired = self._clamp(desired)

        if desired > current_replicas:
            self._scale_down_counter = 0
            return desired

        if desired < current_replicas:
            self._scale_down_counter += 1
            if self._scale_down_counter >= self.scale_down_stabilization:
                self._scale_down_counter = 0
                return desired
            return current_replicas

        self._scale_down_counter = 0
        return current_replicas

    def reset(self) -> None:
        self._scale_down_counter = 0