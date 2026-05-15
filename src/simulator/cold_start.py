from dataclasses import dataclass, field
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WarmingReplica:
    count: int
    ready_at_step: int   # simulator step number when this batch becomes ready
    ordered_at_step: int 


class ColdStartTracker:
    """
    Tracks replicas currently warming up (ordered but not yet ready).
    Scale-up is delayed by cold_start_steps.
    Scale-down is instant (just terminate ready replicas).
    """

    def __init__(self, cold_start_seconds: int = 120, timestep_seconds: int = 60):
        self.cold_start_seconds = cold_start_seconds
        self.timestep_seconds = timestep_seconds
        # How many simulator steps until a new replica is ready
        self.cold_start_steps = max(1, cold_start_seconds // timestep_seconds)
        self._warming: list[WarmingReplica] = []
        logger.info(
            f"ColdStartTracker init: cold_start={cold_start_seconds}s, "
            f"timestep={timestep_seconds}s, "
            f"cold_start_steps={self.cold_start_steps}"
        )

    def request_scale_up(self, count: int, current_step: int) -> int:
        if count <= 0:
            return current_step
        ready_at = current_step + self.cold_start_steps
        self._warming.append(WarmingReplica(
            count=count,
            ready_at_step=ready_at,
            ordered_at_step=current_step   # ← ADD THIS
        ))
        return ready_at
    # In ColdStartTracker.update() — replace existing method

    def update(self, current_step: int) -> tuple[int, list[float]]:
        """
        Check if any warming replicas are now ready.

        Returns:
            (newly_ready_count, list_of_actual_durations_in_seconds)
        """
        newly_ready = 0
        actual_durations: list[float] = []
        still_warming = []

        for batch in self._warming:
            if batch.ready_at_step <= current_step:
                newly_ready += batch.count
                # Actual duration = steps waited × timestep
                actual_steps = batch.ready_at_step - batch.ordered_at_step
                actual_durations.append(actual_steps * self.timestep_seconds)
                logger.debug(
                    f"Step {current_step}: {batch.count} replicas ready, "
                    f"actual duration={actual_steps * self.timestep_seconds}s"
                )
            else:
                still_warming.append(batch)

        self._warming = still_warming
        return newly_ready, actual_durations

    def warming_count(self) -> int:
        """Total replicas currently warming up (ordered but not ready)."""
        return sum(b.count for b in self._warming)

    def reset(self):
        self._warming = []