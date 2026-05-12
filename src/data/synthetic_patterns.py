"""
Synthetic workload generators — add to src/data/loader.py
inside _load_synthetic(), replacing the existing pattern dispatch block.

Patterns:
  diurnal_burst   — daily sine wave with random burst spikes (existing)
  smooth          — slow sine wave, low noise, no spikes
  bursty          — low baseline with sharp random burst events
  bimodal         — alternates between two distinct load levels
  flash_crowd     — long quiet period then one massive sustained spike
  slow_ramp_up    — gradual increase from low to high over 24h
  periodic_spikes — regular, predictable spikes every ~2-3 hours
"""
import math
import numpy as np
import pandas as pd


def _generate_synthetic(pattern: str, steps: int = 288, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic RPS trace.

    All patterns produce `steps` rows at 5-min resolution (default 288 = 24h).
    Returns DataFrame with columns: [timestamp, rps]
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(steps)

    if pattern == "diurnal_burst":
        # ── Original pattern (keep unchanged) ────────────────────────
        base   = 200 + 150 * np.sin(2 * np.pi * t / (24 * 60 / 60))
        noise  = rng.normal(0, 20, steps)
        bursts = np.zeros(steps)
        for _ in range(4):
            idx = rng.integers(48, 240)
            width = rng.integers(6, 18)
            height = rng.uniform(80, 200)
            for j in range(max(0, idx - width // 2),
                           min(steps, idx + width // 2)):
                bursts[j] += height * math.exp(
                    -0.5 * ((j - idx) / (width / 4)) ** 2
                )
        rps = np.clip(base + noise + bursts, 5, 500)

    elif pattern == "smooth":
        # ── Gentle sine, low noise, no spikes ────────────────────────
        # Models a predictable workload — ARIMA should excel here
        base  = 200 + 150 * np.sin(2 * np.pi * t / 288 - np.pi / 2)
        noise = rng.normal(0, 20, steps)
        rps   = np.clip(base + noise, 5, 300)

    elif pattern == "bursty":
        # ── Low baseline + multiple sharp short bursts ────────────────
        # Models social-media/gaming traffic — LSTM should outperform ARIMA
        base   = np.full(steps, 30.0) + rng.normal(0, 4, steps)
        bursts = np.zeros(steps)
        # 8–12 random bursts, each 3–8 steps wide, height 100–400 RPS
        n_bursts = rng.integers(8, 13)
        for _ in range(n_bursts):
            idx    = rng.integers(10, steps - 10)
            width  = rng.integers(3, 9)
            height = rng.uniform(100, 400)
            for j in range(max(0, idx - width // 2),
                           min(steps, idx + width // 2)):
                bursts[j] += height * math.exp(
                    -0.5 * ((j - idx) / max(width / 4, 1)) ** 2
                )
        rps = np.clip(base + bursts, 5, 600)

    elif pattern == "bimodal":
        # ── Alternates between LOW (~40 RPS) and HIGH (~200 RPS) ─────
        # Models batch-job or shift-change traffic patterns
        rps = np.zeros(steps)
        block = 36   # ~3 hours per block at 5-min resolution
        for i in range(steps):
            block_idx = (i // block) % 2
            if block_idx == 0:
                rps[i] = rng.normal(40,  8)
            else:
                rps[i] = rng.normal(200, 20)
        rps = np.clip(rps, 5, 400)

    elif pattern == "flash_crowd":
        # ── Long quiet, then one massive sustained spike ───────────────
        # Models product launches, viral events — worst case for all policies
        rps = np.full(steps, 30.0) + rng.normal(0, 3, steps)
        # Spike starts at 60% of trace, lasts 25% of trace
        spike_start = int(steps * 0.60)
        spike_end   = int(steps * 0.85)
        ramp_len    = 12   # 12 steps = 1hr ramp-up
        for i in range(spike_start, spike_end):
            if i < spike_start + ramp_len:
                # linear ramp up
                frac = (i - spike_start) / ramp_len
                rps[i] += frac * 450
            else:
                rps[i] += 450 + rng.normal(0, 15)
        rps = np.clip(rps, 5, 600)

    elif pattern == "slow_ramp_up":
        # ── Gradual increase from low to high over 24h ────────────────
        # Models day-of-week growth, seasonal trending — tests proactive scaling
        start_rps = 50.0
        end_rps   = 350.0
        base = np.linspace(start_rps, end_rps, steps)
        noise = rng.normal(0, 5, steps)
        # Add one small burst mid-day to test responsiveness
        burst_idx = steps // 2
        burst_width = 8
        for j in range(max(0, burst_idx - burst_width // 2),
                       min(steps, burst_idx + burst_width // 2)):
            base[j] += 80 * math.exp(-0.5 * ((j - burst_idx) / (burst_width / 4)) ** 2)
        rps = np.clip(base + noise, 5, 500)

    elif pattern == "periodic_spikes":
        # ── Regular, predictable spikes every ~2-3 hours ───────────────
        # Models batch jobs, cron tasks, scheduled reports — ARIMA/Prophet advantage
        base = np.full(steps, 60.0) + rng.normal(0, 3, steps)
        spikes = np.zeros(steps)
        spike_period = 24  # spikes every 24 steps = 2 hours
        spike_height = 200
        spike_width = 6
        for spike_start_idx in range(12, steps, spike_period):
            for j in range(max(0, spike_start_idx - spike_width // 2),
                           min(steps, spike_start_idx + spike_width // 2)):
                spikes[j] += spike_height * math.exp(
                    -0.5 * ((j - spike_start_idx) / (spike_width / 4)) ** 2
                )
        rps = np.clip(base + spikes, 5, 500)

    else:
        raise ValueError(f"Unknown pattern '{pattern}'")

    timestamps = pd.date_range("2024-01-01", periods=steps, freq="5min")
    return pd.DataFrame({"timestamp": timestamps, "rps": rps.astype(float)})
