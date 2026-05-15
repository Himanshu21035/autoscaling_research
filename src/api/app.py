from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, metrics, policy, runs, prometheus_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.config import CONFIG
    from src.api.database import run_db
    from src.api.store import metrics_store, results_store

    metrics_store.clear()

    # ── Seed results_store from SQLite on startup ──────────────────────────
    # This means /v1/metrics/prometheus returns summary data immediately
    # after a container restart, without needing to re-run experiments.
    completed = run_db.list_all(status="completed", limit=200)
    if completed:
        summaries = [r["summary"] for r in completed if r.get("summary")]
        results_store.set(summaries)
        print(f"[startup] Seeded results_store with {len(summaries)} completed runs from DB")
    else:
        results_store.clear()

    sim = CONFIG.get("simulator", {})
    print(
        f"[startup] DB={run_db._path} | "
        f"capacity_per_replica={sim.get('capacity_per_replica')} | "
        f"cold_start_s={sim.get('cold_start_s')} | "
        f"sla_latency_ms={sim.get('sla_latency_ms')}"
    )
    yield


app = FastAPI(
    title       = "Autoscaler Research API",
    description = (
        "MPC-based autoscaling with cold-start co-optimization.\n\n"
        "**Workflow:**\n"
        "1. `POST /v1/runs` — submit an experiment\n"
        "2. `GET /v1/runs/{run_id}` — poll until `status=completed`\n"
        "3. `GET /v1/metrics/prometheus` — Prometheus scrape endpoint\n"
        "4. `GET /v1/policy/results/latest` — summary comparison table\n"
    ),
    version   = "1.0.0",
    docs_url  = "/docs",
    redoc_url = "/redoc",
    lifespan  = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PREFIX = "/v1"

app.include_router(health.router)
app.include_router(metrics.router,            prefix=f"{PREFIX}/metrics")
app.include_router(prometheus_metrics.router, prefix=f"{PREFIX}/metrics")
app.include_router(policy.router,             prefix=f"{PREFIX}/policy")
app.include_router(runs.router,               prefix=f"{PREFIX}/runs")
