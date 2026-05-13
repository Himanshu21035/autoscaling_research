"""
Background experiment runner.

Executes a full run_simulation() call in a ThreadPoolExecutor,
pushes per-step metrics into metrics_store, and persists to SQLite.
"""
from __future__ import annotations
from random import seed
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from src.api.models import ExperimentRun, RunStatus
from src.api.database import run_db
from src.api.store import metrics_store, results_store

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="exp-runner")


def submit_run(run: ExperimentRun) -> None:
    run_db.insert(run)
    _executor.submit(_execute_run, run)


def _execute_run(run: ExperimentRun) -> None:
    t_start = time.time()
    run_db.update_status(run.run_id, RunStatus.RUNNING, started_at=t_start)

    try:
        from src.data.loader import load_trace, as_numpy
        from src.data.splitter import split
        from src.policies import create_policy
        from src.simulator.adapt import ADAPTTracker
        from src.simulator.core import run_simulation
        from src.config import CONFIG

        _SIM = CONFIG["simulator"]
        cap  = float(_SIM["capacity_per_replica"])

        # ── 1. Load trace ──────────────────────────────────────────────
        df    = load_trace(source="synthetic", pattern=run.workload, seed=run.seed)
        trace = as_numpy(df)
        sp    = split(trace, train_frac=run.train_frac, val_frac=run.val_frac)

        # ── 2. Build policy ────────────────────────────────────────────
        adapt = ADAPTTracker(
            alpha=0.3,
            cold_start_s=run.cold_start_s,
            cold_start_min_s=30.0,
            cold_start_max_s=600.0,
            epsilon_steps=1,
            timestep_seconds=_SIM["timestep_seconds"],
        )

        policy_kwargs: dict = {}
        if run.policy == "mpc":
            policy_kwargs = dict(
                adapt_tracker=adapt,
                lambda_sla=run.lambda_sla,
                lambda_cost=run.lambda_cost,
                lambda_stab=run.lambda_stab,
                forecast_margin=run.forecast_margin,
                cold_start_steps=run.cold_start_steps,
                use_fh_opt=run.use_fh_opt,
            )
        policy = create_policy(run.policy, **policy_kwargs)

        # ── 3. Fit forecaster ──────────────────────────────────────────
        forecaster = None
        if run.forecaster != "none":
            from src.forecasting import create_forecaster
            forecaster = create_forecaster(run.forecaster)
            forecaster.timed_fit(sp.train_val)

        # ── 4. Run simulation ──────────────────────────────────────────
        sim_result = run_simulation(
            trace=sp.test,
            policy=policy,
            forecaster=forecaster,
            adapt=adapt if run.policy == "mpc" else None,
            forecast_every=1,
            cold_start_s=run.cold_start_s,
            seed=run.seed,
            cold_start_noise=run.policy != "hpa",
        )

        # ── 5. Push ALL per-step metrics to metrics_store ──────────────
        for m in sim_result.metrics:
            capacity = m.warm_replicas * cap
            metrics_store.push(
                policy           = run.policy,
                forecaster       = run.forecaster,
                step             = m.step,
                rps              = m.rps,
                replicas         = m.warm_replicas,
                latency_ms       = m.latency_ms,
                sla_violated     = m.sla_violated,
                cost             = m.cost,
                utilisation      = m.rps / max(capacity, 1.0),
                adapt_estimate_s = m.adapt_estimate_s,
            )

        # ── 6. Push summary to results_store ──────────────────────────
        summary = sim_result.summary()
        existing = results_store.get_all()
        updated = [
            r for r in existing
            if not (r.get("policy") == run.policy
                    and r.get("forecaster") == run.forecaster)
        ]
        updated.append(summary)
        results_store.set(updated)

        # ── 7. Persist to SQLite ───────────────────────────────────────
        t_end = time.time()
        run_db.update_status(
            run.run_id,
            RunStatus.COMPLETED,
            summary=summary,
            started_at=t_start,
            ended_at=t_end,
        )

    except Exception:
        run_db.update_status(
            run.run_id,
            RunStatus.FAILED,
            error=traceback.format_exc(),
            started_at=t_start,
            ended_at=time.time(),
        )
