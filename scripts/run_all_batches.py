"""
scripts/run_all_batches.py
──────────────────────────
Python equivalent of the PowerShell batch submission scripts.
Submits all 7 experiment batches to the local API and polls until complete.

Usage:
    # Run everything (all 7 batches, ~200 runs)
    python scripts/run_all_batches.py

    # Run specific batches only
    python scripts/run_all_batches.py --batches b1-hpa b2-lstm

    # Dry run — print configs without submitting
    python scripts/run_all_batches.py --dry-run

    # Skip batches that already have completed runs in DB
    python scripts/run_all_batches.py --skip-existing
"""

import argparse
import time
import sys
from itertools import product

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE    = "http://localhost:8000"
POLL_EVERY  = 10   # seconds between status checks
RUN_TIMEOUT = 900  # seconds before marking a run as timed out (15 min)

SEEDS_MAIN        = [42, 123, 456, 789, 1337]
SEEDS_COLDSTART   = [42]                          # single seed for sensitivity
WORKLOADS_MAIN    = ["smooth", "bursty", "bimodal", "diurnal_burst", "flash_crowd", "slow_ramp_up"]
WORKLOADS_FHOPT   = ["diurnal_burst", "flash_crowd"]
COLD_START_LEVELS = [30, 60, 120, 180, 300]

# ── Batch definitions (mirrors PowerShell paste.txt exactly) ──────────────────

def build_b1_hpa():
    """Batch 1 — HPA baseline: 5 seeds × 6 workloads = 30 runs"""
    runs = []
    for seed, wl in product(SEEDS_MAIN, WORKLOADS_MAIN):
        runs.append({
            "batch":      "b1-hpa",
            "policy":     "hpa",
            "forecaster": "none",
            "workload":   wl,
            "seed":       seed,
        })
    return runs


def build_b2_lstm():
    """Batch 2 — MPC+LSTM: 5 seeds × 6 workloads = 30 runs"""
    runs = []
    for seed, wl in product(SEEDS_MAIN, WORKLOADS_MAIN):
        runs.append({
            "batch":           "b2-lstm",
            "policy":          "mpc",
            "forecaster":      "lstm",
            "workload":        wl,
            "seed":            seed,
            "lambda_sla":      50,
            "forecast_margin": 1.15,
        })
    return runs


def build_b3_prophet():
    """Batch 3 — MPC+Prophet: 5 seeds × 6 workloads = 30 runs"""
    runs = []
    for seed, wl in product(SEEDS_MAIN, WORKLOADS_MAIN):
        runs.append({
            "batch":           "b3-prophet",
            "policy":          "mpc",
            "forecaster":      "prophet",
            "workload":        wl,
            "seed":            seed,
            "lambda_sla":      150,
            "forecast_margin": 1.06,
        })
    return runs


def build_b4_prophet_coldstart():
    """Batch 4 — Prophet cold-start sensitivity: 1 seed × 5 levels × 2 workloads = 10 runs"""
    runs = []
    for cs, wl in product(COLD_START_LEVELS, WORKLOADS_FHOPT):
        runs.append({
            "batch":           "b4-prophet-coldstart",
            "policy":          "mpc",
            "forecaster":      "prophet",
            "workload":        wl,
            "seed":            42,
            "cold_start_s":    cs,
            "lambda_sla":      150,
            "forecast_margin": 1.06,
        })
    return runs


def build_b5_lstm_coldstart():
    """Batch 5 — LSTM cold-start sensitivity: 1 seed × 5 levels × 2 workloads = 10 runs"""
    runs = []
    for cs, wl in product(COLD_START_LEVELS, WORKLOADS_FHOPT):
        runs.append({
            "batch":           "b5-lstm-coldstart",
            "policy":          "mpc",
            "forecaster":      "lstm",
            "workload":        wl,
            "seed":            42,
            "cold_start_s":    cs,
            "lambda_sla":      50,
            "forecast_margin": 1.15,
        })
    return runs


def build_b6_hpa_coldstart():
    """Batch 6 — HPA cold-start sensitivity: 1 seed × 5 levels × 2 workloads = 10 runs"""
    runs = []
    for cs, wl in product(COLD_START_LEVELS, WORKLOADS_FHOPT):
        runs.append({
            "batch":        "b6-hpa-coldstart",
            "policy":       "hpa",
            "forecaster":   "none",
            "workload":     wl,
            "seed":         42,
            "cold_start_s": cs,
        })
    return runs


def build_b7_fhopt():
    """Batch 7 — FH-OPT A/B: 5 seeds × 2 workloads × 2 forecasters × 2 FH modes = 40 runs"""
    runs = []
    forecaster_cfg = {
        "lstm":    {"lambda_sla": 50,  "forecast_margin": 1.15},
        "prophet": {"lambda_sla": 150, "forecast_margin": 1.06},
    }
    fh_key_map = {
        ("lstm",    False): "lstm_fh_off",
        ("lstm",    True):  "lstm_fh_on",
        ("prophet", False): "prophet_fh_off",
        ("prophet", True):  "prophet_fh_on",
    }
    for seed, wl, forecaster, use_fh in product(
        SEEDS_MAIN, WORKLOADS_FHOPT, ["lstm", "prophet"], [False, True]
    ):
        cfg = forecaster_cfg[forecaster]
        runs.append({
            "batch":            "b7-fhopt",
            "policy":           "mpc",
            "forecaster":       forecaster,
            "workload":         wl,
            "seed":             seed,
            "lambda_sla":       cfg["lambda_sla"],
            "forecast_margin":  cfg["forecast_margin"],
            "cold_start_steps": 2,
            "use_fh_opt":       use_fh,
            "fh_key":           fh_key_map[(forecaster, use_fh)],
        })
    return runs


ALL_BATCHES = {
    "b1-hpa":               build_b1_hpa,
    "b2-lstm":              build_b2_lstm,
    "b3-prophet":           build_b3_prophet,
    "b4-prophet-coldstart": build_b4_prophet_coldstart,
    "b5-lstm-coldstart":    build_b5_lstm_coldstart,
    "b6-hpa-coldstart":     build_b6_hpa_coldstart,
    "b7-fhopt":             build_b7_fhopt,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_existing_run_keys(batch_name):
    """Return set of (forecaster, workload, seed, use_fh_opt) for completed runs in batch."""
    try:
        r = requests.get(f"{API_BASE}/v1/runs", params={"limit": 500}, timeout=10)
        runs = r.json().get("runs", [])
        keys = set()
        for run in runs:
            cfg = run.get("config", {})
            if isinstance(cfg, str):
                import json
                cfg = json.loads(cfg)
            if cfg.get("batch") == batch_name and run.get("status") == "completed":
                keys.add((
                    cfg.get("forecaster"),
                    cfg.get("workload"),
                    cfg.get("seed"),
                    cfg.get("use_fh_opt", False),
                    cfg.get("cold_start_s"),
                ))
        return keys
    except Exception:
        return set()


def submit_run(cfg):
    r = requests.post(f"{API_BASE}/v1/runs", json=cfg, timeout=15)
    r.raise_for_status()
    return r.json()["run_id"]


def poll_batch(run_ids, batch_name):
    """Poll until all run_ids in batch reach terminal state."""
    pending = set(run_ids)
    completed = failed = timed_out = 0
    start = time.time()

    print(f"\n  Polling {len(pending)} runs for {batch_name}...")

    while pending:
        time.sleep(POLL_EVERY)
        elapsed = time.time() - start

        still_pending = set()
        for run_id in pending:
            try:
                r = requests.get(f"{API_BASE}/v1/runs/{run_id}", timeout=10)
                status = r.json().get("status", "unknown")
            except Exception:
                still_pending.add(run_id)
                continue

            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
                print(f"    ⚠️  {run_id} FAILED: {r.json().get('error','?')[:80]}")
            elif elapsed > RUN_TIMEOUT:
                timed_out += 1
                print(f"    ⏱️  {run_id} timed out after {RUN_TIMEOUT}s")
            else:
                still_pending.add(run_id)

        pending = still_pending
        done = completed + failed + timed_out
        total = len(run_ids)
        pct = 100 * done / total if total else 0
        print(f"    [{batch_name}] {done}/{total} done ({pct:.0f}%) "
              f"| ✅ {completed} ❌ {failed} ⏱️ {timed_out} "
              f"| elapsed {elapsed:.0f}s", end="")

    print()  # newline after carriage-return line
    return completed, failed, timed_out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Submit all experiment batches to the API")
    parser.add_argument("--batches",       nargs="+", default=list(ALL_BATCHES.keys()),
                        choices=list(ALL_BATCHES.keys()),
                        help="Which batches to run (default: all 7)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print configs without submitting")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that already have completed results in DB")
    parser.add_argument("--api",           default=API_BASE,
                        help=f"API base URL (default: {API_BASE})")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.api

    # ── Health check ──────────────────────────────────────────────────────────
    if not args.dry_run:
        if not check_api():
            print(f"❌ API not reachable at {API_BASE}")
            print("   Start it with: uvicorn src.api.app:app --reload --port 8000")
            sys.exit(1)
        print(f"✅ API reachable at {API_BASE}")

    # ── Run each batch ────────────────────────────────────────────────────────
    grand_total = grand_ok = grand_fail = 0

    for batch_name in args.batches:
        runs = ALL_BATCHES[batch_name]()

        # Skip existing
        if args.skip_existing and not args.dry_run:
            existing = get_existing_run_keys(batch_name)
            before = len(runs)
            runs = [
                r for r in runs
                if (r.get("forecaster"), r.get("workload"), r.get("seed"),
                    r.get("use_fh_opt", False), r.get("cold_start_s")) not in existing
            ]
            skipped = before - len(runs)
            if skipped:
                print(f"  ↩️  Skipped {skipped} already-completed runs in {batch_name}")

        print(f"\n{'═'*60}")
        print(f"  {batch_name}  ({len(runs)} runs)")
        print(f"{'═'*60}")

        if args.dry_run:
            for r in runs[:3]:
                print(f"  DRY RUN: {r}")
            if len(runs) > 3:
                print(f"  ... and {len(runs)-3} more")
            continue

        if not runs:
            print(f"  ✅ All runs already completed — skipping")
            continue

        # Submit all
        run_ids = []
        for i, cfg in enumerate(runs):
            try:
                run_id = submit_run(cfg)
                run_ids.append(run_id)
                label = f"{cfg.get('forecaster','?')} {cfg.get('workload','?')} seed={cfg.get('seed','?')}"
                print(f"  [{i+1:3d}/{len(runs)}] ✅ {run_id}  {label}")
            except Exception as e:
                print(f"  [{i+1:3d}/{len(runs)}] ❌ Submit failed: {e}  cfg={cfg}")
            time.sleep(0.05)  # avoid hammering API

        # Poll
        if run_ids:
            ok, fail, timeout = poll_batch(run_ids, batch_name)
            grand_total += len(run_ids)
            grand_ok    += ok
            grand_fail  += fail + timeout
            print(f"  {batch_name} done: ✅ {ok}  ❌ {fail}  ⏱️ {timeout}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not args.dry_run and grand_total > 0:
        print(f"\n{'═'*60}")
        print(f"  ALL BATCHES COMPLETE")
        print(f"  Total runs: {grand_total}")
        print(f"  Completed:  {grand_ok}")
        print(f"  Failed:     {grand_fail}")
        print(f"{'═'*60}")
        print("\nNext step: open notebooks/analysis.ipynb and run all cells.")


if __name__ == "__main__":
    main()
