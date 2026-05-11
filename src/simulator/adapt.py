# src/simulator/adapt.py
"""
ADAPT — Adaptive Cold Start Tracker

Novel contribution: treats Δ_cold as a live dynamic variable updated
via EWMA, rather than a fixed constant (as in all 47 surveyed papers).

Higher-level API:
  observe_event(t_requested, t_ready) — computes duration from timestamps
  observe(duration_s)                 — direct duration input

Rolling stats are cached on each update (O(1) amortized via Welford).
"""
import math
from collections import deque
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)

_SIM_CFG   = CONFIG["simulator"]
_ADAPT_CFG = CONFIG.get("adapt", {})


class ADAPTTracker:

    def __init__(
        self,
        alpha: float | None = None,
        cold_start_s: float | None = None,
        cold_start_min_s: float | None = None,
        cold_start_max_s: float | None = None,
        epsilon_steps: int | None = None,
        timestep_seconds: int | None = None,
        history_maxlen: int = 100,
    ):
        self.alpha = (
            alpha if alpha is not None
            else _ADAPT_CFG.get("alpha", 0.3)
        )
        self.cold_start_s = (
            cold_start_s if cold_start_s is not None
            else _SIM_CFG.get("cold_start_s", 120.0)
        )
        self.cold_start_min_s = (
            cold_start_min_s if cold_start_min_s is not None
            else _ADAPT_CFG.get("cold_start_min_s", 30.0)
        )
        self.cold_start_max_s = (
            cold_start_max_s if cold_start_max_s is not None
            else _ADAPT_CFG.get("cold_start_max_s", 600.0)
        )
        self.epsilon_steps = (
            epsilon_steps if epsilon_steps is not None
            else _ADAPT_CFG.get("epsilon_steps", 1)
        )
        self.timestep_seconds = (
            timestep_seconds if timestep_seconds is not None
            else _SIM_CFG.get("timestep_seconds", 60)
        )

        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.cold_start_min_s <= 0:
            raise ValueError(f"cold_start_min_s must be > 0, got {self.cold_start_min_s}")
        if self.cold_start_max_s <= self.cold_start_min_s:
            raise ValueError(
                f"cold_start_max_s ({self.cold_start_max_s}) must be > "
                f"cold_start_min_s ({self.cold_start_min_s})"
            )
        if not (self.cold_start_min_s <= self.cold_start_s <= self.cold_start_max_s):
            raise ValueError(
                f"cold_start_s={self.cold_start_s} must be within "
                f"[{self.cold_start_min_s}, {self.cold_start_max_s}]"
            )

        # State
        self._estimate_s: float     = self.cold_start_s
        self._n_observations: int   = 0
        self._history: deque[float] = deque(maxlen=history_maxlen)

        # Welford online stats cache (O(1) per update)
        self._welford_mean: float = 0.0
        self._welford_M2: float   = 0.0   # sum of squared deviations

        logger.info(
            f"ADAPTTracker init | prior={self.cold_start_s}s | "
            f"alpha={self.alpha} | epsilon={self.epsilon_steps} steps"
        )

    # ── Public API ─────────────────────────────────────────────────────

    def observe_event(self, t_requested: float, t_ready: float) -> float:
        """
        Higher-level API: compute duration from request/ready timestamps.

        Args:
            t_requested: simulation time (s) when scale-up was issued
            t_ready:     simulation time (s) when replica passed readiness probe

        Returns:
            updated Δ̂_cold estimate
        """
        if t_ready <= t_requested:
            logger.warning(
                f"ADAPT: t_ready ({t_ready}) <= t_requested ({t_requested}) "
                f"— observation ignored"
            )
            return self._estimate_s
        return self.observe(t_ready - t_requested)

    def observe(self, observed_cold_start_s: float) -> float:
        """
        Update EWMA estimate with a new cold start duration.

        Args:
            observed_cold_start_s: measured boot time in seconds

        Returns:
            updated Δ̂_cold estimate
        """
        if observed_cold_start_s <= 0:
            logger.warning(
                f"ADAPT: non-positive observation {observed_cold_start_s}s — ignored"
            )
            return self._estimate_s

        clipped = max(
            self.cold_start_min_s,
            min(self.cold_start_max_s, observed_cold_start_s)
        )
        if clipped != observed_cold_start_s:
            logger.debug(
                f"ADAPT observation clipped: "
                f"{observed_cold_start_s:.1f}s -> {clipped:.1f}s"
            )

        # EWMA update
        self._estimate_s = (
            self.alpha * clipped + (1.0 - self.alpha) * self._estimate_s
        )

        # Welford online mean/variance update (O(1), numerically stable)
        self._n_observations += 1
        delta = clipped - self._welford_mean
        self._welford_mean += delta / self._n_observations
        delta2 = clipped - self._welford_mean
        self._welford_M2 += delta * delta2

        self._history.append(clipped)

        logger.debug(
            f"ADAPT obs #{self._n_observations}: "
            f"measured={clipped:.1f}s | estimate={self._estimate_s:.1f}s"
        )
        return self._estimate_s

    # ── FH OPT ─────────────────────────────────────────────────────────

    def optimal_horizon(self) -> int:
        """
        FH OPT: h*(t) = ceil(Δ̂_cold(t) / timestep_seconds) + epsilon_steps

        Guarantees MPC always looks at least one full cold start window
        ahead. epsilon_steps adds a safety margin.
        """
        return max(
            1,
            math.ceil(self._estimate_s / self.timestep_seconds) + self.epsilon_steps
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def estimate_s(self) -> float:
        return self._estimate_s

    @property
    def n_observations(self) -> int:
        return self._n_observations

    # ── Diagnostics ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """
        Research-grade diagnostics using cached Welford stats.
        O(1) — no recomputation over history.
        """
        if self._n_observations == 0:
            history_mean = None
            history_std  = None
        elif self._n_observations == 1:
            history_mean = round(self._welford_mean, 2)
            history_std  = 0.0
        else:
            history_mean = round(self._welford_mean, 2)
            history_std  = round(
                math.sqrt(self._welford_M2 / self._n_observations), 2
            )

        return {
            "estimate_s":      round(self._estimate_s, 2),
            "n_observations":  self._n_observations,
            "optimal_horizon": self.optimal_horizon(),
            "history_mean":    history_mean,
            "history_std":     history_std,
            "alpha":           self.alpha,
            "prior_s":         self.cold_start_s,
        }

    def reset(self):
        """Restore to prior. Called between experiments."""
        self._estimate_s     = self.cold_start_s
        self._n_observations = 0
        self._history.clear()
        self._welford_mean   = 0.0
        self._welford_M2     = 0.0
        logger.debug("ADAPTTracker reset to prior")