# src/data/synthetic.py
"""
Synthetic workload generator for controlled experiments.

Three workload types used in the 240-experiment grid:
  4 forecasters × 4 optimizers × 5 cold starts × 3 workloads = 240

  1. smooth      — sinusoidal daily pattern + Gaussian noise
  2. bursty      — smooth baseline + random Poisson spikes
  3. flash_crowd — flat baseline + one fixed catastrophic step spike

Design note — flash_crowd spike position:
  The spike onset is fixed at (spike_start_day, spike_start_hour) from config,
  NOT randomized. This is intentional: a fixed onset makes the flash_crowd
  workload a deterministic worst-case probe, not a random stress test.
  Randomizing onset would conflate "did the system handle a spike" with
  "did the spike happen to land at a favorable forecasting phase".
  All 240 experiments see the same flash_crowd trace for fair comparison.

RNG independence:
  Each generator accepts a seed offset. generate_all() applies
  seed + offset per workload so traces are independent but reproducible.
    smooth:      seed + 0
    bursty:      seed + 1
    flash_crowd: seed + 2
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.logger import get_logger
from src.config import CONFIG

logger = get_logger(__name__)

_SYN_CFG   = CONFIG.get("synthetic", {})
_GLOBAL_SEED = _SYN_CFG.get("seed", 42)
_N_DAYS    = _SYN_CFG.get("n_days", 7)
SYNTHETIC_DIR = Path(CONFIG["data"]["synthetic_dir"])


# ── Parameter Validation ───────────────────────────────────────────────

def _validate_positive(value: float, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")

def _validate_probability(value: float, name: str):
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")

def _validate_range(lo: float, hi: float, name: str):
    if lo >= hi:
        raise ValueError(f"{name}: low={lo} must be < high={hi}")

def _validate_n_days(n_days: int):
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")


# ── Generators ─────────────────────────────────────────────────────────

def generate_smooth(
    n_days: int | None = None,
    base_rps: float | None = None,
    amplitude: float | None = None,
    noise_std: float | None = None,
    seed: int = _GLOBAL_SEED,
) -> pd.Series:
    cfg       = _SYN_CFG.get("smooth", {})
    n_days    = n_days    if n_days    is not None else _N_DAYS
    base_rps  = base_rps  if base_rps  is not None else cfg.get("base_rps",  300.0)
    amplitude = amplitude if amplitude is not None else cfg.get("amplitude", 150.0)
    noise_std = noise_std if noise_std is not None else cfg.get("noise_std",  20.0)

    _validate_n_days(n_days)
    _validate_positive(base_rps,  "base_rps")
    _validate_positive(amplitude, "amplitude")
    _validate_positive(noise_std, "noise_std")

    rng = np.random.default_rng(seed)
    n   = n_days * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    t   = np.arange(n)

    signal = base_rps + amplitude * np.sin(2 * np.pi * t / 1440)
    rps    = np.clip(signal + rng.normal(0, noise_std, size=n), 0, None)

    logger.info(
        f"smooth | n_days={n_days} seed={seed} | "
        f"mean={rps.mean():.1f} std={rps.std():.1f} "
        f"max={rps.max():.1f} CV={rps.std()/rps.mean():.3f}"
    )
    return pd.Series(rps, index=idx, name="rps")


def generate_bursty(
    n_days: int | None = None,
    base_rps: float | None = None,
    noise_std: float | None = None,
    spike_probability: float | None = None,
    spike_multiplier_range: tuple | None = None,
    spike_duration_range: tuple | None = None,
    seed: int = _GLOBAL_SEED,
) -> pd.Series:
    cfg                    = _SYN_CFG.get("bursty", {})
    n_days                 = n_days            if n_days            is not None else _N_DAYS
    base_rps               = base_rps          if base_rps          is not None else cfg.get("base_rps",          200.0)
    noise_std              = noise_std         if noise_std         is not None else cfg.get("noise_std",           30.0)
    spike_probability      = spike_probability if spike_probability is not None else cfg.get("spike_probability",   0.01)
    spike_multiplier_range = spike_multiplier_range if spike_multiplier_range is not None \
                             else tuple(cfg.get("spike_multiplier_range", [3.0, 8.0]))
    spike_duration_range   = spike_duration_range if spike_duration_range is not None \
                             else tuple(cfg.get("spike_duration_range", [5, 20]))

    _validate_n_days(n_days)
    _validate_positive(base_rps, "base_rps")
    _validate_positive(noise_std, "noise_std")
    _validate_probability(spike_probability, "spike_probability")
    _validate_range(*spike_multiplier_range, "spike_multiplier_range")
    _validate_range(*spike_duration_range,   "spike_duration_range")
    rng = np.random.default_rng(seed)
    n   = n_days * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    t   = np.arange(n)

    baseline = base_rps + 50 * np.sin(2 * np.pi * t / 1440)
    rps      = baseline + rng.normal(0, noise_std, size=n)

    n_spikes = 0
    i = 0
    while i < n:
        if rng.random() < spike_probability:
            duration   = int(rng.uniform(*spike_duration_range))
            multiplier = rng.uniform(*spike_multiplier_range)
            end = min(i + duration, n)
            rps[i:end] *= multiplier
            n_spikes += 1
            i = end
        else:
            i += 1

    rps = np.clip(rps, 0, None)
    cv  = rps.std() / rps.mean()

    logger.info(
        f"bursty | n_days={n_days} seed={seed} | {n_spikes} spikes | "
        f"mean={rps.mean():.1f} std={rps.std():.1f} "
        f"max={rps.max():.1f} CV={cv:.3f}"
    )
    # Sanity check: CV should be substantially higher than smooth
    if cv < 0.5:
        logger.warning(
            f"bursty CV={cv:.3f} is lower than expected (>0.5). "
            f"Consider increasing spike_probability or spike_multiplier_range."
        )
    return pd.Series(rps, index=idx, name="rps")


def generate_flash_crowd(
    n_days: int | None = None,
    base_rps: float | None = None,
    noise_std: float | None = None,
    spike_multiplier: float | None = None,
    spike_start_day: int | None = None,
    spike_start_hour: int | None = None,
    spike_duration_minutes: int | None = None,
    seed: int = _GLOBAL_SEED,
) -> pd.Series:
    cfg                    = _SYN_CFG.get("flash_crowd", {})
    n_days                 = n_days                 if n_days                 is not None else _N_DAYS
    base_rps               = base_rps               if base_rps               is not None else cfg.get("base_rps",               150.0)
    noise_std              = noise_std              if noise_std              is not None else cfg.get("noise_std",                15.0)
    spike_multiplier       = spike_multiplier       if spike_multiplier       is not None else cfg.get("spike_multiplier",         10.0)
    spike_start_day        = spike_start_day        if spike_start_day        is not None else cfg.get("spike_start_day",            3)
    spike_start_hour       = spike_start_hour       if spike_start_hour       is not None else cfg.get("spike_start_hour",          14)
    spike_duration_minutes = spike_duration_minutes if spike_duration_minutes is not None else cfg.get("spike_duration_minutes",    90)

    _validate_n_days(n_days)
    _validate_positive(base_rps, "base_rps")
    _validate_positive(noise_std, "noise_std")
    _validate_positive(spike_multiplier, "spike_multiplier")
    if spike_start_day >= n_days:
        raise ValueError(f"spike_start_day={spike_start_day} must be < n_days={n_days}")
    if not (0 <= spike_start_hour < 24):
        raise ValueError(f"spike_start_hour={spike_start_hour} must be in [0, 23]")

    rng = np.random.default_rng(seed)
    n   = n_days * 1440
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")

    rps         = base_rps + rng.normal(0, noise_std, size=n)
    spike_start = spike_start_day * 1440 + spike_start_hour * 60
    spike_end   = min(spike_start + spike_duration_minutes, n)
    rps[spike_start:spike_end] = base_rps * spike_multiplier
    rps = np.clip(rps, 0, None)

    logger.info(
        f"flash_crowd | n_days={n_days} seed={seed} | "
        f"spike steps {spike_start}–{spike_end} "
        f"({spike_duration_minutes}min @ {base_rps * spike_multiplier:.0f} RPS) | "
        f"base={base_rps:.0f} mean={rps.mean():.1f}"
    )
    return pd.Series(rps, index=idx, name="rps")


# ── Save / Generate All ────────────────────────────────────────────────

def save_workload(series: pd.Series, name: str, out_dir: Path | None = None) -> Path:
    """Save a workload series to {out_dir}/{name}.csv"""
    target = (out_dir or SYNTHETIC_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.csv"
    df = series.reset_index()
    df.columns = ["timestamp", "rps"]
    df.to_csv(path, index=False)
    logger.info(f"Saved {name} → {path} ({len(df)} rows)")
    return path


def generate_all(
    seed: int = _GLOBAL_SEED,
    out_dir: Path | None = None,
) -> dict[str, pd.Series]:
    """
    Generate and save all three workload types.

    Seeds are offset per workload for independence:
      smooth:      seed + 0
      bursty:      seed + 1
      flash_crowd: seed + 2

    Returns dict of {name: pd.Series}.
    Integration point for run_phase1.py:
      from src.data.synthetic import generate_all
      workloads = generate_all()
    """
    workloads = {
        "smooth":      generate_smooth(seed=seed + 0),
        "bursty":      generate_bursty(seed=seed + 1),
        "flash_crowd": generate_flash_crowd(seed=seed + 2),
    }
    for name, series in workloads.items():
        save_workload(series, name, out_dir=out_dir)
    return workloads