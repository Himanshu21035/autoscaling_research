# src/forecasting/__init__.py
"""
Forecaster registry with lazy optional-dependency imports.

Package import never fails even if prophet or torch are absent.
ImportError surfaces only when the specific forecaster is instantiated.
"""
from __future__ import annotations
from src.forecasting.base import BaseForecaster


def _import_arima():
    from src.forecasting.arima import ARIMAForecaster
    return ARIMAForecaster


def _import_prophet():
    from src.forecasting.prophet import ProphetForecaster
    return ProphetForecaster


def _import_lstm():
    from src.forecasting.lstm import LSTMForecaster
    return LSTMForecaster


# Lazy registry: values are callables that return the class
_LAZY_REGISTRY: dict[str, callable] = {
    "arima":   _import_arima,
    "prophet": _import_prophet,
    "lstm":    _import_lstm,
}


def create_forecaster(name: str, **kwargs) -> BaseForecaster:
    """
    Factory with lazy imports. Safe to call even if some deps are missing.

    Raises:
        ValueError:  unknown forecaster name
        ImportError: required library not installed (pmdarima / prophet / torch)
    """
    key = name.strip().lower()
    if key not in _LAZY_REGISTRY:
        raise ValueError(
            f"Unknown forecaster '{name}'. "
            f"Valid options: {sorted(_LAZY_REGISTRY.keys())}"
        )
    cls = _LAZY_REGISTRY[key]()
    return cls(**kwargs)


def available_forecasters() -> list[str]:
    """Returns names of forecasters whose dependencies are installed."""
    available = []
    checks = {
        "arima":   ("pmdarima",),
        "prophet": ("prophet",),
        "lstm":    ("torch",),
    }
    for name, deps in checks.items():
        try:
            for dep in deps:
                __import__(dep)
            available.append(name)
        except ImportError:
            pass
    return available


# Expose classes directly for type-checking without triggering imports
def __getattr__(name: str):
    mapping = {
        "ARIMAForecaster":   _import_arima,
        "ProphetForecaster": _import_prophet,
        "LSTMForecaster":    _import_lstm,
    }
    if name in mapping:
        return mapping[name]()
    raise AttributeError(f"module 'src.forecasting' has no attribute '{name}'")


__all__ = [
    "BaseForecaster",
    "ARIMAForecaster", "ProphetForecaster", "LSTMForecaster",
    "create_forecaster", "available_forecasters",
]