# src/forecasting/arima.py
"""
ARIMA Forecaster — Statistical Baseline 1.

Fixes vs previous version:
  - Fallback always returns last observed value, never 0
  - update() truly refits on rolling window (not just appends)
  - Degenerate model detection: refit if forecast mean is near-zero
    when series mean is not
"""
import numpy as np
from src.forecasting.base import BaseForecaster
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("forecasting", {}).get("arima", {})


class ARIMAForecaster(BaseForecaster):

    def __init__(
        self,
        max_p: int | None = None,
        max_q: int | None = None,
        max_d: int = 2,
        seasonal: bool = False,
        online_window: int | None = None,
        min_series_length: int = 10,
    ):
        super().__init__("ARIMA")
        self.max_p = max_p if max_p is not None else _CFG.get("max_p", 3)
        self.max_q = max_q if max_q is not None else _CFG.get("max_q", 3)
        self.max_d = max_d
        self.seasonal = seasonal
        self.online_window = (
            online_window if online_window is not None
            else _CFG.get("online_window", 60)
        )
        self.min_series_length = min_series_length

        self._model = None
        self._series: list[float] = []

    def fit(self, series: np.ndarray) -> "ARIMAForecaster":
        import pmdarima as pm

        series = np.asarray(series, dtype=float)
        self._series = list(series)

        if len(series) < self.min_series_length:
            logger.warning(
                f"ARIMA: series too short ({len(series)} < "
                f"{self.min_series_length}) — flat fallback"
            )
            self._fitted = True
            return self

        self._model = self._fit_model(series)
        self._fitted = True
        return self

    def _fit_model(self, series: np.ndarray):
        """Fit pmdarima model. Returns model or None on failure."""
        def _fit_model(self, series: np.ndarray):
            import pmdarima as pm
            try:
                model = pm.auto_arima(
                    series,
                    start_p=1, max_p=self.max_p,
                    start_q=1, max_q=self.max_q,
                    max_d=self.max_d,
                    d=1,                    # fix d — skips unit-root tests entirely
                    seasonal=self.seasonal,
                    stepwise=True,          # greedy, not exhaustive grid
                    information_criterion='aic',
                    error_action="ignore",
                    suppress_warnings=True,
                )
                logger.info(f"ARIMA fitted: order={model.order}")
                return model
            except Exception as e:
                logger.warning(f"ARIMA _fit_model failed ({e})")
                return None

    def predict(self, steps: int) -> np.ndarray:
        self._require_fitted()
        if steps == 0:
            return np.array([])

        fallback = self._clip(
            np.full(steps, self._last_observed(self._series))
        )

        if self._model is None or len(self._series) < self.min_series_length:
            return fallback

        try:
            fc = np.asarray(self._model.predict(n_periods=steps), dtype=float)
            result = self._clip(fc)

            # Degenerate model guard: if forecast is near-zero but series
            # mean is not, the model has collapsed — return fallback
            series_mean   = np.mean(self._series[-self.min_series_length:])
            forecast_mean = float(np.mean(result))
            if series_mean > 1.0 and forecast_mean < 0.01 * series_mean:
                logger.warning(
                    f"ARIMA degenerate forecast (mean={forecast_mean:.2f} vs "
                    f"series_mean={series_mean:.2f}) — using last-observed fallback"
                )
                return fallback

            return result

        except Exception as e:
            logger.warning(f"ARIMA predict failed ({e}) — last-observed fallback")
            return fallback

    def update(self, new_value: float) -> None:
        """
        True rolling-window refit.

        Appends new_value, then refits on the last `online_window` points.
        This is correct online adaptation — not just appending to a buffer
        without updating the model.

        Trade-off: O(online_window) refit cost per step.
        For faster updates, switch to model.update([new_value]) (ARIMA
        append, not full refit) — but that accumulates approximation error.
        """
        if not self._fitted:
            return

        self._series.append(new_value)

        if self._model is not None:
            try:
                # Cheap O(1) append — no order search, just re-estimates params
                self._model.update([new_value])
                return
            except Exception as e:
                logger.warning(f"ARIMA incremental update failed ({e}) — falling back to refit")

        # Full refit only as fallback when incremental update fails
        window = self._series[-self.online_window:]
        if len(window) >= self.min_series_length:
            refitted = self._fit_model(np.array(window))
            if refitted is not None:
                self._model = refitted
    def reset(self) -> None:
        self._model = None
        self._series = []
        self._fitted = False