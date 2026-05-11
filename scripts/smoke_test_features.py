# scripts/smoke_test_features.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.metrics.collector import FeatureCollector

df_raw = pd.read_csv("data/processed/burstgpt_test.csv",
                     index_col="timestamp", parse_dates=True)
series = df_raw["rps"].iloc[:1440]

fc = FeatureCollector()
features = fc.transform(series)

print(f"Input rows      : {len(series)}")
print(f"Output rows     : {len(features)}")
print(f"Feature columns : {len(features.columns)}")
print(f"Burst events    : {features['burst_flag'].sum()} "
      f"({100*features['burst_flag'].mean():.1f}%)")
print(f"Max volatility  : {features['volatility_ratio'].max():.3f}")
print(f"Avg queue proxy : {features['queue_proxy'].mean():.1f}")
print("\nFeature matrix sample:")
print(features[["rps","rolling_mean_5m","burst_flag",
                "volatility_ratio","queue_proxy"]].tail(5).to_string())