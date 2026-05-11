# src/simulator/core.py
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

logger = get_logger(__name__)

SIM_CFG = CONFIG["simulator"]
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