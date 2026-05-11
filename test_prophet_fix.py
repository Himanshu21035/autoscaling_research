#!/usr/bin/env python
"""Quick diagnostic: compare Prophet vs ARIMA forecasts on diurnal+burst data."""
import numpy as np
from src.data.loader import load_trace, as_numpy
from src.data.splitter import split
from src.forecasting import create_forecaster

# Load the same trace used in smoke_test_step9
df = load_trace(source="synthetic", pattern="diurnal_burst", seed=42)
trace = as_numpy(df)
sp = split(trace, train_frac=0.70, val_frac=0.10)

print(f"Trace shape: {trace.shape}")
print(f"Train: {len(sp.train)} | Val: {len(sp.val)} | Test: {len(sp.test)}")
print(f"Train min/mean/max: {sp.train.min():.1f} / {sp.train.mean():.1f} / {sp.train.max():.1f}")

# Fit both forecasters
arima = create_forecaster("arima", min_series_length=10)
prophet = create_forecaster("prophet", min_series_length=20)

print("\nFitting forecasters...")
arima.timed_fit(sp.train_val)
prophet.timed_fit(sp.train_val)

print(f"ARIMA fit in {arima.fit_latency_ms:.1f} ms")
print(f"Prophet fit in {prophet.fit_latency_ms:.1f} ms")

# Predict next 10 steps
h = 10
arima_fc = arima.timed_predict(h)
prophet_fc = prophet.timed_predict(h)

print(f"\n--- Forecasts for next {h} steps ---")
print(f"ARIMA:   {arima_fc}")
print(f"Prophet: {prophet_fc}")

# Compare with actual test data
actual = sp.test[:h]
print(f"Actual:  {actual}")

# Metrics
arima_mae = np.mean(np.abs(arima_fc - actual))
prophet_mae = np.mean(np.abs(prophet_fc - actual))

print(f"\n--- MAE (first {h} steps) ---")
print(f"ARIMA MAE:   {arima_mae:.1f}")
print(f"Prophet MAE: {prophet_mae:.1f}")
print(f"Actual mean: {actual.mean():.1f}")

if prophet_mae > arima_mae * 1.5:
    print(f"\n⚠ WARNING: Prophet MAE is {prophet_mae/arima_mae:.1f}x worse than ARIMA")
else:
    print(f"\n✅ Prophet MAE is {arima_mae/prophet_mae:.2f}x ARIMA (acceptable)")
