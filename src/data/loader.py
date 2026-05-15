from pathlib import Path
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import CONFIG
from src.data.synthetic_patterns import _generate_synthetic
logger = get_logger(__name__)

RAW_DIR = Path(CONFIG["data"]["raw_dir"])
PROCESSED_DIR = Path(CONFIG["data"]["processed_dir"])
SCALE_FACTORS = CONFIG.get("data", {}).get("scale_factors", {})

_DATA_CFG  = CONFIG.get("data", {})
_DATA_DIR  = Path(_DATA_CFG.get("data_dir", "data/raw"))
_TIMESTEP  = CONFIG["simulator"]["timestep_seconds"]

def load_burstgpt(
    files: list[str] | None = None,
    resample_freq: str = "1min",
    scale_factor:float=200.0,
) -> pd.DataFrame:
    """
    Load BurstGPT CSVs and aggregate to RPS (requests per minute = requests/60).

    BurstGPT schema:
      Timestamp  : seconds since 00:00:00 on day 0 (float)
      Model      : 'ChatGPT' or 'GPT-4'
      Request tokens, Response tokens, Total tokens, Log Type

    Returns unified DataFrame: timestamp (datetime index), rps (float), source (str)
    """
    if files is None:
        burstgpt_dir = RAW_DIR / "burstgpt"
        files = list(burstgpt_dir.glob("BurstGPT_without_fails_*.csv"))

    if not files:
        raise FileNotFoundError(f"No BurstGPT files found in {RAW_DIR / 'burstgpt'}")

    chunks = []
    for fpath in sorted(files):
        logger.info(f"Loading BurstGPT file: {fpath.name}")
        df = pd.read_csv(fpath)

        base_date = pd.Timestamp("2023-01-01")
        df["timestamp"] = base_date + pd.to_timedelta(df["Timestamp"], unit="s")
        chunks.append(df[["timestamp"]])

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values("timestamp").set_index("timestamp")

    # Count requests per window → convert to RPS
    window_seconds = pd.Timedelta(resample_freq).total_seconds()
    rps_series = combined.resample(resample_freq).size() / window_seconds
    scale_factor = SCALE_FACTORS.get("burstgpt", 600.0)
    rps_scaled=rps_series*scale_factor
    result = pd.DataFrame({
        "rps": rps_scaled.values,
        "source": "burstgpt"
    }, index=rps_scaled.index)

    result.index.name = "timestamp"
    logger.info(f"BurstGPT loaded: {len(result)} rows, "
                f"duration={result.index[-1] - result.index[0]}, "
                f"mean_rps={result['rps'].mean():.2f}")
    return result


def load_azure(
    files: list[str] | None = None,
    resample_freq: str = "1min",
    scale_factor:float=200.0,
) -> pd.DataFrame:
    """
    Load Azure LLM 2024 CSVs and aggregate to RPS.

    Azure schema:
      TIMESTAMP      : ISO timestamp string
      ContextTokens  : int
      GeneratedTokens: int

    Returns unified DataFrame: timestamp (datetime index), rps (float), source (str)
    """
    if files is None:
        azure_dir = RAW_DIR / "azure"
        files = list(azure_dir.glob("AzureLLMInferenceTrace2024_*.csv"))

    if not files:
        raise FileNotFoundError(f"No Azure files found in {RAW_DIR / 'azure'}")

    chunks = []
    for fpath in sorted(files):
        logger.info(f"Loading Azure file: {fpath.name}")
        df = pd.read_csv(fpath)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"])
        chunks.append(df[["timestamp"]])

    combined = pd.concat(chunks, ignore_index=True)
    combined = combined.sort_values("timestamp").set_index("timestamp")

    window_seconds = pd.Timedelta(resample_freq).total_seconds()
    rps_series = combined.resample(resample_freq).size() / window_seconds
    scale_factor = SCALE_FACTORS.get("azure", 60.0)
    rps_scaled=rps_series*scale_factor
    result = pd.DataFrame({
        "rps": rps_scaled.values,
        "source": "azure"
    }, index=rps_scaled.index)

    result.index.name = "timestamp"
    logger.info(f"Azure loaded: {len(result)} rows, "
                f"duration={result.index[-1] - result.index[0]}, "
                f"mean_rps={result['rps'].mean():.2f}")
    return result


def load_alibaba(
    file_path: str | None = None,
    microservice_id: str | None = None,
    resample_freq: str = "1min",
    scale_factor:float=200.0,
) -> pd.DataFrame:
    """
    Load Alibaba Microservices Trace from Zenodo (record 14245634).

    Zenodo schema (no header, 7 columns):
      col0: node_id or call_id  (int)   — NOT a timestamp, skip
      col1: timestamp_ms        (int)   — ms from trace start (use this)
      col2: microservice_id     (str)   — hex hash of service name
      col3: container_id        (str)   — hex hash of instance
      col4: replica_count       (int)   — number of running replicas
      col5: call_rate           (float) — calls per 30s window
      col6: avg_response_time_ms(float) — average latency

    RPS = call_rate / 30.0
    """
    if file_path is None:
        alibaba_dir = RAW_DIR / "alibaba"
        candidates = (
            list(alibaba_dir.rglob("*.csv")) +
            list(alibaba_dir.rglob("*.csv.gz"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No Alibaba CSV files found in {alibaba_dir}\n"
                f"Download from: https://zenodo.org/records/14245634"
            )
        # Use the largest file — most data
        file_path = max(candidates, key=lambda p: p.stat().st_size)
        logger.info(f"Auto-selected file: {Path(file_path).name} "
                    f"({Path(file_path).stat().st_size / 1024 / 1024:.1f} MB)")

    col_names = [
        "node_id",          # col0 — skip
        "timestamp_ms",     # col1 — USE THIS
        "ms_id",            # col2 — microservice hash
        "container_id",     # col3 — instance hash
        "replica_count",    # col4
        "call_rate",        # col5 — calls per 30s
        "avg_rt_ms",        # col6 — response time
    ]

    logger.info(f"Loading Alibaba file: {Path(file_path).name}")
    df = pd.read_csv(file_path, header=None, names=col_names)
    logger.info(f"Raw rows loaded: {len(df):,}")

    # Pick microservice with most records for richest time-series
    if microservice_id is None:
        microservice_id = df["ms_id"].value_counts().index[0]
        top5 = df["ms_id"].value_counts().head(5)
        logger.info(f"Top 5 microservices by record count:\n{top5.to_string()}")
        logger.info(f"Auto-selected ms_id: {microservice_id[:16]}...")

    df = df[df["ms_id"] == microservice_id].copy()
    logger.info(f"Rows after ms filter: {len(df):,}")

    # Convert relative ms timestamp to absolute datetime
    # Trace starts at 2021-07-01 (Alibaba 2021 trace period)
    base = pd.Timestamp("2021-07-01")
    df["timestamp"] = base + pd.to_timedelta(df["timestamp_ms"], unit="ms")
    df = df.sort_values("timestamp").set_index("timestamp")
    scale_factor = SCALE_FACTORS.get("alibaba", 50.0)
    # call_rate is calls per 30s → divide by 30 to get RPS
    df["rps"] = (df["call_rate"] * scale_factor) / 30.0

    # Resample to uniform frequency (mean within each window)
    rps_series = df["rps"].resample(resample_freq).mean()

    result = pd.DataFrame({
        "rps": rps_series.values,
        "source": "alibaba"
    }, index=rps_series.index)

    result.index.name = "timestamp"

    duration = result.index[-1] - result.index[0]
    logger.info(
        f"Alibaba loaded ✓ | rows={len(result):,} | "
        f"duration={duration} | "
        f"mean_rps={result['rps'].mean():.3f} | "
        f"max_rps={result['rps'].max():.3f}"
    )
    return result
def load_trace(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Load a workload trace.

    Args:
        source: "azure" | "synthetic"
        **kwargs: passed to the specific loader

    Returns:
        pd.DataFrame with columns [timestamp, rps], sorted by timestamp,
        resampled to simulator timestep, no NaNs.
    """
    loaders = {
        "azure":     _load_azure,
        "synthetic": _load_synthetic,
    }
    key = source.strip().lower()
    if key not in loaders:
        raise ValueError(
            f"Unknown source '{source}'. "
            f"Valid options: {sorted(loaders.keys())}"
        )
    df = loaders[key](**kwargs)
    df = _normalise(df)
    logger.info(
        f"Loaded '{source}' trace: {len(df)} steps, "
        f"mean_rps={df['rps'].mean():.1f}, "
        f"max_rps={df['rps'].max():.1f}"
    )
    return df


def as_numpy(df: pd.DataFrame) -> np.ndarray:
    """Extract rps column as float64 numpy array."""
    return df["rps"].to_numpy(dtype=float)
def _load_azure(
    path: str | Path | None = None,
    rps_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """
    Load Azure LLM trace CSV.

    Expected CSV columns (configurable):
      timestamp: ISO8601 or Unix epoch seconds
      rps:       requests per second (float)

    If the file does not exist, falls back to synthetic data with a warning.
    """
    cfg       = _DATA_CFG.get("azure", {})
    path      = Path(path or cfg.get("path", _DATA_DIR / "azure_trace.csv"))
    rps_col   = rps_col  or cfg.get("rps_col",  "rps")
    time_col  = time_col or cfg.get("time_col", "timestamp")

    if not path.exists():
        logger.warning(
            f"Azure trace not found at {path} — "
            f"falling back to synthetic data"
        )
        return _load_synthetic()

    df = pd.read_csv(path)

    # Flexible timestamp parsing: epoch seconds or ISO string
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df["timestamp"] = pd.to_datetime(df[time_col], unit="s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df[time_col], utc=True)

    df = df.rename(columns={rps_col: "rps"})[["timestamp", "rps"]]
    df["rps"] = pd.to_numeric(df["rps"], errors="coerce").fillna(0.0)
    return df


# ── Synthetic Loader ──────────────────────────────────────────────────

def _load_synthetic(
    pattern: str = "diurnal_burst",
    steps:   int = 288,
    seed:    int = 42,
    **kwargs,
) -> "pd.DataFrame":
    """
    Load a synthetic workload trace.

    Supported patterns:
        diurnal_burst   daily sine + random bursts  (original)
        smooth          slow sine, low noise         (new)
        bursty          low base + sharp spikes      (new)
        bimodal         alternating low/high blocks  (new)
        flash_crowd     quiet then one big spike     (new)

    Returns:
        DataFrame with columns [timestamp, rps]
    """
    return _generate_synthetic(pattern=pattern, steps=steps, seed=seed)



# ── Normalise ─────────────────────────────────────────────────────────
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to simulator timestep, forward-fill gaps, clip negatives.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    df = (
        df["rps"]
        .resample(f"{_TIMESTEP}s")
        .mean()
        .interpolate("linear")
        .ffill()          # pandas 2.x — replaces fillna(method="ffill")
        .fillna(0.0)
        .clip(lower=0.0)
        .reset_index()
    )
    df.columns = ["timestamp", "rps"]
    return df