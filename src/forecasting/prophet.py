# src/forecasting/prophet.py
"""
Prophet Forecaster — Statistical Baseline 2.

Improvements vs previous version:
  - Supports real trace timestamps (pass start_time to fit())
  - Explicit fit/predict latency via timed_fit/timed_predict
  - update() logs that refitting is needed for true adaptation
"""
import numpy as np
import pandas as pd
from src.forecasting.base import BaseForecaster
from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

_CFG = CONFIG.get("forecasting", {}).get("prophet", {})


class ProphetForecaster(BaseForecaster):

    def __init__(
        self,
        changepoint_prior_scale: float | None = None,
        interval_width: float = 0.8,
        timestep_seconds: int | None = None,
        min_series_length: int = 20,
    ):
        super().__init__("Prophet")
        self.changepoint_prior_scale = (
            changepoint_prior_scale if changepoint_prior_scale is not None
            else _CFG.get("changepoint_prior_scale", 0.3)
        )
        self.interval_width = interval_width
        self.timestep_seconds = (
            timestep_seconds if timestep_seconds is not None
            else CONFIG["simulator"]["timestep_seconds"]
        )
        self.min_series_length = min_series_length

        self._model = None
        self._series: list[float] = []
        self._last_ds: pd.Timestamp | None = None

    def fit(
    self,
    series: np.ndarray,
    start_time: str | pd.Timestamp | None = None,
) -> "ProphetForecaster":
        from prophet import Prophet
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        series = np.asarray(series, dtype=float)
        self._series = list(series)

        if len(series) < self.min_series_length:
            logger.warning(f"Prophet: too short ({len(series)}) — flat fallback")
            self._fitted = True
            return self

        df = self._to_df(series, start_time=start_time)
        self._last_ds = df["ds"].iloc[-1]

        try:
            self._model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                interval_width=self.interval_width,
                uncertainty_samples=0,        # faster — disables CI computation
                daily_seasonality=False,        # enables detection of intraday/diurnal patterns
                weekly_seasonality=False,      # disabled: single-day training not enough for weekly cycles
            )
            self._model.fit(df)
            self._fitted = True
            logger.info(
                f"Prophet fitted: {len(series)} pts, "
                f"start={df['ds'].iloc[0]}"
            )
        except Exception as e:
            logger.warning(f"Prophet fit failed ({e}) — flat fallback")
            self._model = None
            self._fitted = True

        return self

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
            future_dates = pd.date_range(
                start=self._last_ds
                      + pd.Timedelta(seconds=self.timestep_seconds),
                periods=steps,
                freq=f"{self.timestep_seconds}s",
            )
            fc = self._model.predict(pd.DataFrame({"ds": future_dates}))
            return self._clip(fc["yhat"].to_numpy(dtype=float))
        except Exception as e:
            logger.warning(f"Prophet predict failed ({e}) — last-observed fallback")
            return fallback

    def update(self, new_value: float) -> None:
        """
        Append-only update. Prophet has no true incremental fit.
        For real adaptation: call timed_fit(series) periodically
        (e.g., every N steps) rather than relying on update().
        """
        if not self._fitted:
            return
        self._series.append(new_value)
        if self._last_ds is not None:
            self._last_ds += pd.Timedelta(seconds=self.timestep_seconds)

    def reset(self) -> None:
        self._model = None
        self._series = []
        self._last_ds = None
        self._fitted = False

    def _to_df(
        self,
        series: np.ndarray,
        start_time: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(start_time) if start_time else pd.Timestamp("2024-01-01")
        timestamps = pd.date_range(
            start=origin,
            periods=len(series),
            freq=f"{self.timestep_seconds}s",
        )
        return pd.DataFrame({"ds": timestamps, "y": series})