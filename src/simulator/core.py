"""
Discrete-time simulation engine.

One step = one timestep (default 60s).

Per step:
  1. Get current RPS from trace
  2. Policy computes desired replicas
     (with optional forecast from forecaster)
  3. Cold start: new replicas enter warming queue
  4. Capacity = warm_replicas * capacity_per_replica
  5. Compute latency, SLA violation, cost
  6. ADAPT observes any completed cold start events
  7. Log metrics

Returns SimResult dataclass with full per-step metrics.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from polars.selectors import duration
from src.simulator.cold_start import ColdStartTracker
from src.simulator.latency_model import mm1_latency
from src.simulator.metrics_logger import MetricsLogger, StepMetrics
from src.config import CONFIG
from src.logger import get_logger
from src.simulator.adapt import ADAPTTracker
from src.config import CONFIG

import math
from dataclasses import dataclass, field
from src.policies.base import BasePolicy
from src.simulator.adapt import ADAPTTracker
logger = get_logger(__name__)

SIM_CFG = CONFIG["simulator"]
_SIM_CFG = CONFIG["simulator"]
adapt_cfg = CONFIG.get("adapt", {})

class AutoscalerSimulator:
    """
    Discrete-time autoscaling simulator.

    Each step:
      1. Receive RPS for this timestep
      2. Policy decides target replica count
      3. Scale-down is instant
      4. Scale-up goes through cold start tracker
      5. Compute capacity, violation, latency, cost
      6. Log metrics
    """

    def __init__(
        self,
        cold_start_seconds: int = 120,
        timestep_seconds: int = None,
        capacity_per_replica: float = None,
        base_latency_ms: float = None,
        initial_replicas: int = 2,
        min_replicas: int = 1,
        max_replicas: int = 50,
        price_per_replica_per_minute: float = 0.01,
    ):
        self.timestep_seconds = timestep_seconds or SIM_CFG["timestep_seconds"]
        self.capacity_per_replica = capacity_per_replica or SIM_CFG["capacity_per_replica"]
        self.base_latency_ms = base_latency_ms or SIM_CFG["base_latency_ms"]
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.price_per_replica_per_minute = price_per_replica_per_minute
        self.cold_start_seconds = cold_start_seconds

        self._cold_start_tracker = ColdStartTracker(
            cold_start_seconds=cold_start_seconds,
            timestep_seconds=self.timestep_seconds,
        )
        self._metrics_logger = MetricsLogger()

        # State
        self.active_replicas: int = initial_replicas
        self.current_step: int = 0
        self.current_timestamp: pd.Timestamp | None = None
        self.total_cost: float = 0.0
        self.adapt = ADAPTTracker(
            alpha=adapt_cfg.get("alpha", 0.3),
            init_cold_start_s=float(cold_start_seconds),
        )

        logger.info(
            f"AutoscalerSimulator init | cold_start={cold_start_seconds}s | "
            f"capacity_per_replica={self.capacity_per_replica} RPS | "
            f"timestep={self.timestep_seconds}s"
        )

    # ------------------------------------------------------------------
    # Core step method
    # ------------------------------------------------------------------

    def step(
        self,
        rps: float,
        decision: int,
        timestamp: pd.Timestamp | None = None,
    ) -> dict:
        """
        Advance simulation by one timestep.

        Args:
            rps: observed request rate this timestep (RPS)
            decision: target replica count from policy
            timestamp: wall-clock time (optional, for logging)

        Returns:
            dict of metrics for this step
        """
        self.current_timestamp = timestamp
        decision = int(np.clip(decision, self.min_replicas, self.max_replicas))

        # --- 1. Process newly ready warming replicas ---
        newly_ready, actual_durations = self._cold_start_tracker.update(self.current_step)
        self.active_replicas += newly_ready

        # --- 2. Apply scaling decision ---
        scaling_action = decision - (self.active_replicas +
                                     self._cold_start_tracker.warming_count())

        if scaling_action > 0:
            # Scale UP: goes through cold start
            self._cold_start_tracker.request_scale_up(scaling_action, self.current_step)

        elif scaling_action < 0:
            # Scale DOWN: instant — remove from active
            scale_down = abs(scaling_action)
            self.active_replicas = max(
                self.min_replicas,
                self.active_replicas - scale_down
            )
        for duration in actual_durations:
            self.adapt.observe(duration)
        # --- 3. Compute capacity and performance metrics ---
        capacity = self.active_replicas * self.capacity_per_replica
        utilization = rps / capacity if capacity > 0 else 1.0
        violation = max(0.0, (rps - capacity) / rps) if rps > 0 else 0.0
        latency_ms = mm1_latency(rps, capacity, self.base_latency_ms)

        # --- 4. Compute cost (replica-minutes) ---
        minutes_per_step = self.timestep_seconds / 60.0
        step_cost = self.active_replicas * self.price_per_replica_per_minute * minutes_per_step
        self.total_cost += step_cost

        # --- 5. Log metrics ---
        metrics = StepMetrics(
            step=self.current_step,
            timestamp=timestamp or self.current_step,
            rps=rps,
            active_replicas=self.active_replicas,
            warming_replicas=self._cold_start_tracker.warming_count(),
            capacity=capacity,
            utilization=round(utilization, 4),
            violation=round(violation, 4),
            latency_ms=round(latency_ms, 2),
            cost=round(step_cost, 6),
            scaling_action=scaling_action,
            cold_start_seconds=self.cold_start_seconds,
            adapt_estimate_s=round(self.adapt.estimate_s, 1)
        )
        self._metrics_logger.log(metrics)

        self.current_step += 1
        return {
            "step": metrics.step,
            "rps": rps,
            "active_replicas": self.active_replicas,
            "warming_replicas": self._cold_start_tracker.warming_count(),
            "capacity": capacity,
            "utilization": utilization,
            "violation": violation,
            "latency_ms": latency_ms,
            "cost": step_cost,
            "scaling_action": scaling_action,
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def reset(self, initial_replicas: int = 2):
        """Reset simulator state for a new experiment run."""
        self.active_replicas = initial_replicas
        self.current_step = 0
        self.current_timestamp = None
        self.total_cost = 0.0
        self._cold_start_tracker.reset()
        self._metrics_logger.reset()
        logger.debug("Simulator reset")

    def get_metrics(self) -> pd.DataFrame:
        """Return all logged metrics as a DataFrame."""
        return self._metrics_logger.to_dataframe()

    def save_metrics(self, path: str):
        """Save metrics to CSV."""
        self._metrics_logger.save(path)
        logger.info(f"Metrics saved to {path}")

    @property
    def state(self) -> dict:
        """Current simulator state snapshot."""
        return {
            "step": self.current_step,
            "active_replicas": self.active_replicas,
            "warming_replicas": self._cold_start_tracker.warming_count(),
            "total_cost": round(self.total_cost, 4),
        }
@dataclass
class StepMetrics:
    step:             int
    rps:              float
    desired_replicas: int
    warm_replicas:    int
    capacity:         float
    latency_ms:       float
    sla_violated:     bool
    cost:             float
    adapt_estimate_s: float


@dataclass
class SimResult:
    policy_name:      str
    forecaster_name:  str
    steps:            int
    metrics:          list[StepMetrics] = field(default_factory=list)

    # Aggregate (computed by finalise())
    total_cost:       float = 0.0
    sla_violation_pct: float = 0.0
    avg_latency_ms:   float = 0.0
    avg_replicas:     float = 0.0
    peak_replicas:    int   = 0

    def finalise(self) -> "SimResult":
        if not self.metrics:
            return self
        self.total_cost        = sum(m.cost         for m in self.metrics)
        self.sla_violation_pct = (
            sum(1 for m in self.metrics if m.sla_violated) / len(self.metrics) * 100
        )
        self.avg_latency_ms    = np.mean([m.latency_ms    for m in self.metrics])
        self.avg_replicas      = np.mean([m.warm_replicas for m in self.metrics])
        self.peak_replicas     = max(m.warm_replicas for m in self.metrics)
        return self

    def summary(self) -> dict:
        return {
            "policy":         self.policy_name,
            "forecaster":     self.forecaster_name,
            "steps":          self.steps,
            "total_cost":     round(self.total_cost, 4),
            "sla_pct":        round(self.sla_violation_pct, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_replicas":   round(self.avg_replicas, 1),
            "peak_replicas":  self.peak_replicas,
        }


def run_simulation(
    trace:          np.ndarray,
    policy:         BasePolicy,
    forecaster=None,            # BaseForecaster | None
    adapt:          ADAPTTracker | None = None,
    forecast_every: int = 1,    # reforecast every N steps (1 = every step)
    refine_fit_every: int = 0,  # refit forecaster every N steps (0 = never)
) -> SimResult:
    """
    Run one simulation episode.

    Args:
        trace:           1D RPS array (test split)
        policy:          instantiated BasePolicy
        forecaster:      optional BaseForecaster (already fitted on train split)
        adapt:           optional ADAPTTracker (shared with MPC policy if given)
        forecast_every:  how often to call forecaster.predict() [default: every step]
        refine_fit_every: refit forecaster on expanding window every N steps (0=off)

    Returns:
        SimResult with per-step metrics and aggregates
    """
    # Config
    cap_per_replica = _SIM_CFG["capacity_per_replica"]
    cold_start_s    = _SIM_CFG["cold_start_s"]
    timestep_s      = _SIM_CFG["timestep_seconds"]
    sla_latency_ms  = _SIM_CFG["sla_latency_ms"]
    cost_per_replica_step = _SIM_CFG.get("cost_per_replica_step", 1.0)

    cold_start_steps = max(1, math.ceil(cold_start_s / timestep_s))

    policy_name     = type(policy).__name__
    forecaster_name = forecaster.name if forecaster else "none"

    result = SimResult(
        policy_name=policy_name,
        forecaster_name=forecaster_name,
        steps=len(trace),
    )

    # State
    warm_replicas   = _SIM_CFG.get("initial_replicas", 2)
    warming_queue: list[tuple[int, int]] = []  # (ready_at_step, n_replicas)
    current_forecast: np.ndarray | None = None
    train_buffer: list[float] = []

    for step, rps in enumerate(trace):
        # ── 1. Reforecast ──────────────────────────────────────────────
        if forecaster is not None and step % forecast_every == 0:
            h = adapt.optimal_horizon() if adapt else cold_start_steps + 1
            try:
                current_forecast = forecaster.timed_predict(h)
            except Exception as e:
                logger.warning(f"Step {step}: forecast failed ({e})")
                current_forecast = np.full(h, rps)

        # ── 2. Policy decision ─────────────────────────────────────────
        adapt_est = adapt.estimate_s if adapt else cold_start_s
        context = {
            "warm_replicas":      warm_replicas,
            "adapt_estimate_s":   adapt_est,
        }
        if current_forecast is not None:
            context["forecast"] = current_forecast
        
        # ── FH-OPT: Pass live ADAPT-derived horizon to MPC if available
        if adapt is not None and hasattr(policy, "use_fh_opt") and policy.use_fh_opt:
            context["cold_start_steps"] = max(0, adapt.optimal_horizon() - 1)

        desired = policy.compute_replicas(
            current_rps=float(rps),
            current_replicas=warm_replicas,
            step=step,
            **context,
        )

        # ── 3. Cold start queue ────────────────────────────────────────
        delta = desired - warm_replicas
        if delta > 0:
            ready_at = step + cold_start_steps
            warming_queue.append((ready_at, delta))
            # Notify ADAPT that a scale-up event was requested
            if adapt:
                adapt.observe_event(
                    t_requested=step * timestep_s,
                    t_ready=ready_at * timestep_s,
                )
        elif delta < 0:
            warm_replicas = max(
                _SIM_CFG.get("min_replicas", 1),
                warm_replicas + delta,
            )

        # Promote replicas that have finished warming
        newly_ready = [n for (r, n) in warming_queue if r <= step]
        warming_queue = [(r, n) for (r, n) in warming_queue if r > step]
        warm_replicas += sum(newly_ready)
        warm_replicas = min(warm_replicas, _SIM_CFG.get("max_replicas", 50))

        # ── 4. Capacity & metrics ──────────────────────────────────────
        capacity   = warm_replicas * cap_per_replica
        utilisation = min(1.0, float(rps) / max(capacity, 1.0))
        latency_ms  = _compute_latency(utilisation)
        sla_violated = latency_ms > sla_latency_ms
        cost         = warm_replicas * cost_per_replica_step

        # ── 5. Online forecaster update ────────────────────────────────
        if forecaster is not None:
            forecaster.update(float(rps))
            train_buffer.append(float(rps))

            if (refine_fit_every > 0
                    and step > 0
                    and step % refine_fit_every == 0):
                try:
                    forecaster.timed_fit(np.array(train_buffer))
                    logger.debug(f"Step {step}: refitted forecaster")
                except Exception as e:
                    logger.warning(f"Step {step}: refit failed ({e})")

        result.metrics.append(StepMetrics(
            step=step,
            rps=float(rps),
            desired_replicas=desired,
            warm_replicas=warm_replicas,
            capacity=capacity,
            latency_ms=latency_ms,
            sla_violated=sla_violated,
            cost=cost,
            adapt_estimate_s=adapt_est,
        ))

    return result.finalise()


def _compute_latency(utilisation: float) -> float:
    """
    M/M/1 queue-inspired latency model.

    utilisation = rps / capacity
    latency_ms  = base_ms / (1 - utilisation) when util < 1
                = sla_latency_ms * 3           when overloaded (violation)
    """
    base_ms      = _SIM_CFG.get("base_latency_ms", 50.0)
    sla_ms       = _SIM_CFG["sla_latency_ms"]
    overload_mul = _SIM_CFG.get("overload_latency_multiplier", 3.0)

    if utilisation >= 1.0:
        return sla_ms * overload_mul
    return base_ms / max(1.0 - utilisation, 0.001)