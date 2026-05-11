import math
from src.policies.base import BasePolicy
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("policies", {}).get("pid", {})


class PIDPolicy(BasePolicy):

    def __init__(
        self,
        kp: float | None = None,
        ki: float | None = None,
        kd: float | None = None,
        target_utilisation: float | None = None,
    ):
        super().__init__("PID")

        self.kp = kp if kp is not None else float(_CFG.get("kp", 1.0))
        self.ki = ki if ki is not None else float(_CFG.get("ki", 0.1))
        self.kd = kd if kd is not None else float(_CFG.get("kd", 0.05))
        self.target_utilisation = (
            target_utilisation if target_utilisation is not None
            else float(_CFG.get("target_utilisation", 0.7))
        )

        self._integral  = 0.0
        self._prev_error = 0.0

    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **kwargs,
    ) -> int:
        capacity    = current_replicas * self.capacity_per_replica
        utilisation = current_rps / max(capacity, 1.0)
        error       = utilisation - self.target_utilisation

        self._integral  += error
        derivative       = error - self._prev_error
        self._prev_error = error

        adjustment = self.kp * error + self.ki * self._integral + self.kd * derivative
        desired    = current_replicas + math.ceil(adjustment * current_replicas)

        return self._clamp(desired)

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0