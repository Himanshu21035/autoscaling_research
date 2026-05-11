# src/metrics/collector.py
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)

_METRICS_CFG = CONFIG.get("metrics", {})
_FORECAST_CFG = CONFIG.get("forecasting", {})


class FeatureCollector:
    """
    Transforms a raw RPS time series into a feature matrix
    consumed by all forecasters and the GRACE confidence gate.

    Feature groups:
      1. Temporal    — cyclical time encodings
      2. Statistical — rolling window statistics
      3. Domain      — burst/volatility signals (novel inputs)

    NOTE on GRACE forecast residuals:
      This collector does NOT compute forecast errors.
      GRACE (Step 11) maintains its own rolling residual buffer,
      fed by the forecasting module after each predict() call.
      This is intentional — residuals require ground truth which
      is only available post-hoc during simulation.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
        burst_sigma: float | None = None,
        adapt_estimate_s: float = 120.0,
        timestep_seconds: int | None = None,
    ):
        # Load from config with constructor override
        self.windows = windows or _METRICS_CFG.get("rolling_windows", [5, 15, 60])
        self.burst_sigma = burst_sigma or _METRICS_CFG.get("burst_sigma", 2.0)
        self.timestep_seconds = (
            timestep_seconds or CONFIG["simulator"]["timestep_seconds"]
        )
        self.adapt_estimate_s = adapt_estimate_s

    # ── Public API ────────────────────────────────────────────────────

    def transform(self, series: pd.Series) -> pd.DataFrame:
        """
        Args:
            series: pd.Series with DatetimeIndex (preferred), values = RPS

        Returns:
            pd.DataFrame with all features, NaN rows from rolling removed
        """
        if len(series) == 0:
            logger.warning("Empty series passed to FeatureCollector.transform")
            return pd.DataFrame()

        df = pd.DataFrame({"rps": series.astype(float)})
        logger.info(f"FeatureCollector.transform: {len(df)} rows")

        # Order matters: rolling stats must exist before domain features
        df = self._add_temporal(df)
        df = self._add_rolling_stats(df)    # produces rolling_std_5m etc.
        df = self._add_domain_features(df)  # consumes rolling_std_5m etc.

        n_before = len(df)
        df = df.dropna()
        logger.info(
            f"Dropped {n_before - len(df)} NaN rows "
            f"(largest rolling window={max(self.windows)})"
        )
        return df

    def set_adapt_estimate(self, adapt_estimate_s: float):
        """Update ADAPT cold start estimate — call this each simulation step."""
        self.adapt_estimate_s = adapt_estimate_s

    # ── 1. Temporal Features ──────────────────────────────────────────

    def _add_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                "Index is not DatetimeIndex — temporal features set to 0. "
                "Pass a DatetimeIndex series for full feature coverage."
            )
            for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
                df[col] = 0.0
            return df

        hour = df.index.hour
        dow  = df.index.dayofweek  # 0=Monday, 6=Sunday

        df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"]    = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]    = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = (dow >= 5).astype(int)
        return df

    # ── 2. Statistical Features ───────────────────────────────────────

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.windows:
            roll = df["rps"].rolling(window=w, min_periods=w)
            df[f"rolling_mean_{w}m"] = roll.mean()
            df[f"rolling_std_{w}m"]  = roll.std()

        df["rps_delta"] = df["rps"].diff()
        return df

    # ── 3. Domain-Specific Features (Novel) ──────────────────────────

    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Derive column names dynamically from self.windows
        smallest_window = min(self.windows)
        largest_window  = max(self.windows)

        col_mean_large = f"rolling_mean_{largest_window}m"
        col_std_large  = f"rolling_std_{largest_window}m"
        col_std_small  = f"rolling_std_{smallest_window}m"

        # Guard: rolling stats must exist before this is called
        assert col_mean_large in df.columns, (
            f"_add_rolling_stats() must be called before _add_domain_features(). "
            f"Column '{col_mean_large}' is required for burst_flag and volatility_ratio."
        )
        assert col_std_small in df.columns, (
            f"Column '{col_std_small}' missing — ensure windows includes "
            f"at least one short window for volatility_ratio."
        )

        mean_large = df[col_mean_large]
        std_large  = df[col_std_large]

        # ── burst_flag ────────────────────────────────────────────────────
        df["burst_flag"] = (
            df["rps"] > (mean_large + self.burst_sigma * std_large)
        ).astype(int)

        # ── volatility_ratio ──────────────────────────────────────────────
        # short-window std / long-window mean — instability signal
        df["volatility_ratio"] = (
            df[col_std_small] / mean_large.clip(lower=1.0)
        )

        # ── queue_proxy ───────────────────────────────────────────────────
        # Total requests committed during cold start window
        # Units: [req/s] × [steps] × [s/step] = [requests]
        pending_steps = max(1, int(self.adapt_estimate_s / self.timestep_seconds))
        raw_requests  = (
            df["rps"]
            .rolling(window=pending_steps, min_periods=pending_steps)
            .sum()
            * self.timestep_seconds
        )
        capacity_per_replica = CONFIG["simulator"]["capacity_per_replica"]   # 100 RPS
        df["queue_proxy"] = raw_requests / capacity_per_replica

        return df