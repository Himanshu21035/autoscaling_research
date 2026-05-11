# src/simulator/adapt.py
from src.logger import get_logger

logger = get_logger(__name__)


class ADAPTTracker:
    """
    ADAPT — Adaptive cold start delay tracker.
    Uses EWMA to estimate cold start latency as a live variable
    instead of treating it as a fixed constant.

    Novel contribution: feeds live Δ_cold estimate into MPC/FH-OPT
    so the optimizer self-calibrates as cluster conditions change.
    """

    def __init__(self, alpha: float = 0.3, init_cold_start_s: float = 120.0):
        """
        Args:
            alpha: EWMA smoothing factor (0=never update, 1=always use latest)
            init_cold_start_s: initial estimate before any observations
        """
        self.alpha = alpha
        self._estimate_s: float = init_cold_start_s
        self._observations: list[float] = []
        self._n_updates: int = 0
        logger.info(
            f"ADAPTTracker init | alpha={alpha} | "
            f"init_estimate={init_cold_start_s}s"
        )

    def observe(self, actual_cold_start_s: float) -> float:
        """
        Record a measured cold start duration and update EWMA estimate.

        Call this when a replica transitions pending → running.

        Args:
            actual_cold_start_s: measured duration in seconds

        Returns:
            Updated estimate
        """
        self._observations.append(actual_cold_start_s)
        self._estimate_s = (
            self.alpha * actual_cold_start_s
            + (1.0 - self.alpha) * self._estimate_s
        )
        self._n_updates += 1
        logger.debug(
            f"ADAPT update #{self._n_updates}: "
            f"observed={actual_cold_start_s:.1f}s → "
            f"estimate={self._estimate_s:.1f}s"
        )
        return self._estimate_s

    @property
    def estimate_s(self) -> float:
        """Current cold start estimate in seconds."""
        return self._estimate_s

    @property
    def estimate_steps(self) -> int:
        """Current estimate converted to simulator steps (rounded up)."""
        from math import ceil
        # Imported here to avoid circular — timestep injected at use site
        return max(1, ceil(self._estimate_s / 60.0))

    def get_estimate_steps(self, timestep_seconds: int) -> int:
        """Convert estimate to steps given a specific timestep size."""
        from math import ceil
        return max(1, ceil(self._estimate_s / timestep_seconds))

    @property
    def n_observations(self) -> int:
        return self._n_updates

    def summary(self) -> dict:
        return {
            "estimate_s": round(self._estimate_s, 2),
            "n_observations": self._n_updates,
            "alpha": self.alpha,
        }

    def reset(self, init_cold_start_s: float | None = None):
        if init_cold_start_s is not None:
            self._estimate_s = init_cold_start_s
        self._observations.clear()
        self._n_updates = 0