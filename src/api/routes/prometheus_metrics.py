"""
GET /v1/metrics/prometheus — Prometheus scrape endpoint.

Data priority:
  1. metrics_store  — per-step live data (populated by runner.py)
  2. results_store  — run summaries (always populated after completion)

This dual-source approach means Grafana always has something to show
even when per-step metrics are missing (e.g. first run after restart).
"""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from src.api.store import metrics_store, results_store

router = APIRouter()


@router.get(
    "/prometheus",
    response_class=PlainTextResponse,
    tags=["Metrics"],
    summary="Prometheus scrape endpoint",
)
def prometheus_scrape() -> str:
    lines: list[str] = []

    # ── Source 1: live per-step metrics (best) ────────────────────────
    records = metrics_store.last_n(1000)
    latest: dict[tuple, dict] = {}
    for r in records:
        key = (r["policy"], r["forecaster"])
        latest[key] = r

    if latest:
        _write_step_metrics(lines, latest)
        return "\n".join(lines) + "\n"

    # ── Source 2: fall back to results_store summaries ─────────────────
    runs = results_store.get_all()
    if not runs:
        return "# No metrics yet — submit a run via POST /v1/runs\n"

    _write_summary_metrics(lines, runs)
    return "\n".join(lines) + "\n"


def _write_step_metrics(lines: list[str], latest: dict) -> None:
    """Emit per-step gauge metrics from metrics_store."""
    defs = [
        ("autoscaler_replicas",         "replicas",         "gauge", "Active warm replicas"),
        ("autoscaler_rps",              "rps",              "gauge", "Requests per second"),
        ("autoscaler_latency_ms",       "latency_ms",       "gauge", "Step latency ms"),
        ("autoscaler_cost",             "cost",             "gauge", "Step cost"),
        ("autoscaler_utilisation",      "utilisation",      "gauge", "Capacity utilisation 0-1"),
        ("autoscaler_adapt_estimate_s", "adapt_estimate_s", "gauge", "ADAPT cold start estimate s"),
    ]
    for metric_name, field, mtype, help_text in defs:
        lines += [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} {mtype}"]
        for (policy, forecaster), r in latest.items():
            lines.append(
                f'{metric_name}{{policy="{policy}",forecaster="{forecaster}"}} {r.get(field, 0)}'
            )

    lines += ["# HELP autoscaler_sla_violated SLA violation flag (1=violated)",
              "# TYPE autoscaler_sla_violated gauge"]
    for (policy, forecaster), r in latest.items():
        val = 1 if r.get("sla_violated") else 0
        lines.append(
            f'autoscaler_sla_violated{{policy="{policy}",forecaster="{forecaster}"}} {val}'
        )


def _write_summary_metrics(lines: list[str], runs: list[dict]) -> None:
    """
    Emit summary-level metrics from results_store.
    These are static values (not time-series) but they populate the
    KPI stat panels and the comparison table in Grafana immediately.
    """
    defs = [
        ("autoscaler_sla_pct",       "sla_pct",        "gauge", "SLA violation percent"),
        ("autoscaler_avg_replicas",  "avg_replicas",   "gauge", "Average replica count"),
        ("autoscaler_avg_latency_ms","avg_latency_ms", "gauge", "Average latency ms"),
        ("autoscaler_total_cost",    "total_cost",     "gauge", "Total cost for run"),
        ("autoscaler_peak_replicas", "peak_replicas",  "gauge", "Peak replica count"),
    ]
    for metric_name, field, mtype, help_text in defs:
        lines += [f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} {mtype}"]
        for r in runs:
            policy     = r.get("policy",     "unknown")
            forecaster = r.get("forecaster", "none")
            val        = r.get(field, 0)
            lines.append(
                f'{metric_name}{{policy="{policy}",forecaster="{forecaster}"}} {val}'
            )
