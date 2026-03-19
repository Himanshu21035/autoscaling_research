
from pathlib import Path
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)

RAW_DIR = Path(CONFIG["data"]["raw_dir"])
PROCESSED_DIR = Path(CONFIG["data"]["processed_dir"])
SCALE_FACTORS = CONFIG.get("data", {}).get("scale_factors", {})

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
