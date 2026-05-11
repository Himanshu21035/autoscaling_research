# src/policies/__init__.py
from src.policies.threshold import ThresholdPolicy
from src.policies.pid import PIDPolicy
from src.policies.hpa import HPAPolicy
from src.policies.mpc import MPCPolicy
from src.policies.base import BasePolicy

POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    "threshold": ThresholdPolicy,
    "pid":       PIDPolicy,
    "hpa":       HPAPolicy,
    "mpc":       MPCPolicy,
}


def create_policy(name: str, **kwargs) -> BasePolicy:
    """
    Factory for experiment orchestration.

    Args:
        name:    policy name — one of {threshold, pid, hpa, mpc}
        **kwargs: passed directly to the policy constructor

    Returns:
        instantiated BasePolicy subclass

    Raises:
        ValueError: if name is not in POLICY_REGISTRY
    """
    key = name.strip().lower()
    if key not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. "
            f"Valid options: {sorted(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[key](**kwargs)


__all__ = [
    "ThresholdPolicy", "PIDPolicy", "HPAPolicy", "MPCPolicy",
    "POLICY_REGISTRY", "create_policy",
]