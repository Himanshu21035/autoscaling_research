# src/policies/base.py
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

_SIM_CFG = CONFIG["simulator"]


class BasePolicy(ABC):

    def __init__(
        self,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
    ):
        self.min_replicas = (
            min_replicas if min_replicas is not None
            else _SIM_CFG.get("min_replicas", 1)
        )
        self.max_replicas = (
            max_replicas if max_replicas is not None
            else _SIM_CFG.get("max_replicas", 50)
        )
        if self.min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {self.min_replicas}")
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"max_replicas ({self.max_replicas}) must be >= "
                f"min_replicas ({self.min_replicas})"
            )

    @abstractmethod
    def compute_replicas(
        self,
        current_rps: float,
        current_replicas: int,
        step: int,
        **context,
    ) -> int:
        """
        Compute desired replica count for this timestep.

        Args:
            current_rps:      observed RPS at this step
            current_replicas: replicas currently active (post-cold-start)
                              SOURCE OF TRUTH — policies must use this,
                              not internal shadow state
            step:             simulation step index (0-indexed)
            **context:        optional signals for advanced policies
                              (see module docstring for keys)

        Returns:
            desired_replicas: int in [min_replicas, max_replicas]
        """

    @abstractmethod
    def reset(self):
        """Restore to initial state. Called between experiments."""

    def _clamp(self, replicas: float) -> int:
        """Ceil then clamp — always provision up, never under."""
        return max(self.min_replicas, min(self.max_replicas, math.ceil(replicas)))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(min={self.min_replicas}, max={self.max_replicas})"
        )