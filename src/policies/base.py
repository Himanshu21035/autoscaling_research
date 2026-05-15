"""
Abstract base class for all autoscaling policies.

The simulator calls policy.compute_replicas() once per timestep.

Context dict keys (passed as **context):
  Reactive baselines use: nothing beyond positional args
  MPC (Step 7) uses:
    forecast        : np.ndarray  — RPS forecast h steps ahead
    adapt_estimate_s: float       — ADAPT cold start estimate (Δ̂_cold)
    warm_replicas   : int         — replicas already past cold start
    feature_row     : pd.Series   — current feature vector (queue_proxy etc.)
    forecast_errors : list[float] — rolling residuals for GRACE
  FH OPT (Step 7) uses:
    adapt_estimate_s: float       — sets h* = Δ̂_cold + epsilon_margin
  GRACE (Step 11) uses:
    forecast_errors : list[float] — confidence weight computation
"""
import math
from abc import ABC, abstractmethod
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_SIM = CONFIG.get("simulator", {})


class BasePolicy(ABC):

    def __init__(self, name: str):
        self.name             = name
        self.min_replicas     = int(_SIM.get("min_replicas",          1))
        self.max_replicas     = int(_SIM.get("max_replicas",         50))
        self.capacity_per_replica = float(_SIM.get("capacity_per_replica", 100.0))

        if self.min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {self.min_replicas}")
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"max_replicas ({self.max_replicas}) < "
                f"min_replicas ({self.min_replicas})"
            )

    @abstractmethod
    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **kwargs,
    ) -> int:
        """Return desired replica count for this timestep."""

    def reset(self) -> None:
        """Reset any stateful policy variables between experiments."""

    def _clamp(self, replicas: int) -> int:
        return max(self.min_replicas, min(self.max_replicas, replicas))

    def __repr__(self):
        return f"{self.name}Policy(min={self.min_replicas}, max={self.max_replicas})"