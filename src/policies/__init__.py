# src/policies/__init__.py
from src.policies.threshold import ThresholdPolicy
from src.policies.pid import PIDPolicy
from src.policies.hpa import HPAPolicy

POLICY_REGISTRY = {
    "threshold": ThresholdPolicy,
    "pid":       PIDPolicy,
    "hpa":       HPAPolicy,
}

__all__ = ["ThresholdPolicy", "PIDPolicy", "HPAPolicy", "POLICY_REGISTRY"]