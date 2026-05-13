"""
API integration tests — all endpoints, edge cases, and invariants.

Run:
    pytest tests/test_api.py -v
"""
import math
import time
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.store import metrics_store, results_store
from src.api.database import RunDatabase
from src.api.models import RunStatus
import tempfile
from pathlib import Path

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_stores():
    metrics_store.clear()
    results_store.clear()
    yield
    metrics_store.clear()
    results_store.clear()


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_has_version():
    r = client.get("/health")
    assert "version" in r.json()


# ── /ready ────────────────────────────────────────────────────────────────────

def test_ready_shape():
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["ready"], bool)
    assert "available_policies"    in body
    assert "available_forecasters" in body


def test_ready_has_all_policies():
    r = client.get("/ready")
    for p in ("hpa", "mpc", "pid", "threshold"):
        assert p in r.json()["available_policies"]


# ── /v1/metrics/current ───────────────────────────────────────────────────────

def test_metrics_current_empty():
    r = client.get("/v1/metrics/current")
    assert r.status_code == 404


def test_metrics_current_returns_latest():
    metrics_store.push(policy="mpc", forecaster="lstm", step=0,
                       rps=200.0, replicas=3, latency_ms=120.0,
                       sla_violated=False, cost=3.0)
    r = client.get("/v1/metrics/current")
    assert r.status_code == 200
    assert r.json()["rps"] == 200.0


def test_metrics_filter_by_policy():
    metrics_store.push(policy="hpa",  forecaster="none", step=0, rps=100.0,
                       replicas=2, latency_ms=80.0, sla_violated=False, cost=2.0)
    metrics_store.push(policy="mpc",  forecaster="lstm", step=1, rps=250.0,
                       replicas=4, latency_ms=140.0, sla_violated=False, cost=4.0)
    r = client.get("/v1/metrics/current?policy=hpa")
    assert r.status_code == 200
    assert r.json()["replicas"] == 2


# ── /v1/metrics/history ───────────────────────────────────────────────────────

def test_metrics_history_empty():
    r = client.get("/v1/metrics/history")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_metrics_history_n_limit():
    for i in range(10):
        metrics_store.push(policy="mpc", forecaster="lstm", step=i,
                           rps=float(i * 10), replicas=2, latency_ms=50.0,
                           sla_violated=False, cost=2.0)
    r = client.get("/v1/metrics/history?n=5")
    assert r.json()["total"] == 5


def test_metrics_history_forecaster_filter():
    metrics_store.push(policy="mpc", forecaster="arima", step=0, rps=100.0,
                       replicas=2, latency_ms=80.0, sla_violated=False, cost=2.0)
    metrics_store.push(policy="mpc", forecaster="lstm",  step=1, rps=200.0,
                       replicas=3, latency_ms=100.0, sla_violated=False, cost=3.0)
    r = client.get("/v1/metrics/history?forecaster=arima")
    records = r.json()["records"]
    assert all(rec["rps"] == 100.0 for rec in records)


# ── /v1/policy/decide ────────────────────────────────────────────────────────

def test_decide_hpa_no_forecast():
    r = client.post("/v1/policy/decide", json={
        "current_rps": 150.0, "current_replicas": 2,
        "policy": "hpa", "forecaster": "none",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["desired_replicas"] >= 1
    assert body["forecast_meta"]["source"] == "none"


def test_decide_mpc_with_provided_forecast():
    r = client.post("/v1/policy/decide", json={
        "current_rps": 200.0, "current_replicas": 3,
        "policy": "mpc", "forecaster": "none",
        "forecast": [210.0, 250.0, 300.0, 280.0],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_meta"]["source"] == "provided"
    assert body["forecast_meta"]["peak"]   == 300.0


def test_decide_invalid_policy_422():
    r = client.post("/v1/policy/decide", json={
        "current_rps": 100.0, "current_replicas": 2,
        "policy": "banana", "forecaster": "none",
    })
    assert r.status_code == 422


def test_decide_reactive_floor_invariant():
    """desired_replicas >= ceil(rps / capacity) always."""
    from src.config import CONFIG
    cap   = float(CONFIG["simulator"]["capacity_per_replica"])
    rps   = 450.0
    floor = math.ceil(rps / cap)
    r = client.post("/v1/policy/decide", json={
        "current_rps": rps, "current_replicas": 1,
        "policy": "mpc", "forecaster": "none",
    })
    assert r.status_code == 200
    assert r.json()["desired_replicas"] >= floor


def test_decide_mpc_lambda_override():
    r = client.post("/v1/policy/decide", json={
        "current_rps": 200.0, "current_replicas": 2,
        "policy": "mpc", "forecaster": "none",
        "lambda_sla": 200.0, "forecast_margin": 1.3,
    })
    assert r.status_code == 200


def test_decide_decision_ms_is_positive():
    r = client.post("/v1/policy/decide", json={
        "current_rps": 100.0, "current_replicas": 2,
        "policy": "hpa", "forecaster": "none",
    })
    assert r.json()["decision_ms"] >= 0


# ── /v1/policy/status ────────────────────────────────────────────────────────

def test_policy_status_shape():
    r = client.get("/v1/policy/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("available_policies", "available_forecasters", "simulator_config"):
        assert key in body


# ── /v1/policy/results/latest ────────────────────────────────────────────────

def test_results_latest_empty():
    r = client.get("/v1/policy/results/latest")
    assert r.json()["source"] == "none"
    assert r.json()["count"]  == 0


def test_results_latest_after_set():
    results_store.set([
        {"policy": "mpc",  "forecaster": "lstm", "sla_pct": 3.1,
         "avg_replicas": 3.4, "avg_latency_ms": 221.0, "total_cost": 986.0,
         "peak_replicas": 5, "steps": 288},
        {"policy": "hpa",  "forecaster": "none", "sla_pct": 14.9,
         "avg_replicas": 3.0, "avg_latency_ms": 374.0, "total_cost": 858.0,
         "peak_replicas": 4, "steps": 288},
    ])
    r = client.get("/v1/policy/results/latest")
    body = r.json()
    assert body["count"]  == 2
    assert body["source"] == "cache"


# ── /v1/runs — submit + get ───────────────────────────────────────────────────

def test_run_submit_returns_202():
    r = client.post("/v1/runs", json={
        "policy": "hpa", "forecaster": "none", "workload": "diurnal_burst",
    })
    assert r.status_code == 202
    body = r.json()
    assert "run_id" in body
    assert body["status"] == "pending"


def test_run_get_after_submit():
    r  = client.post("/v1/runs", json={"policy": "hpa", "forecaster": "none",
                                       "workload": "diurnal_burst"})
    rid = r.json()["run_id"]
    r2  = client.get(f"/v1/runs/{rid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["run_id"]    == rid
    assert body["policy"]    == "hpa"
    assert body["forecaster"] == "none"
    assert body["batch"] == ""
    assert body["use_fh_opt"] is False
    assert body["status"]    in ("pending", "running", "completed", "failed")


def test_run_submit_with_fh_opt_roundtrip():
    r = client.post("/v1/runs", json={
        "policy": "mpc",
        "forecaster": "none",
        "workload": "diurnal_burst",
        "use_fh_opt": True,
        "cold_start_steps": 2,
    })
    assert r.status_code == 202
    rid = r.json()["run_id"]

    r2 = client.get(f"/v1/runs/{rid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["policy"] == "mpc"
    assert body["cold_start_steps"] == 2
    assert body["use_fh_opt"] is True


def test_run_submit_with_batch_roundtrip():
    r = client.post("/v1/runs", json={
        "policy": "mpc",
        "forecaster": "none",
        "workload": "diurnal_burst",
        "batch": "batch-7-fhopt",
    })
    assert r.status_code == 202
    rid = r.json()["run_id"]

    r2 = client.get(f"/v1/runs/{rid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["batch"] == "batch-7-fhopt"


def test_run_get_unknown_404():
    r = client.get("/v1/runs/doesnotexist")
    assert r.status_code == 404


def test_run_invalid_policy_422():
    r = client.post("/v1/runs", json={"policy": "banana", "forecaster": "none",
                                      "workload": "diurnal_burst"})
    assert r.status_code == 422


def test_run_invalid_workload_422():
    r = client.post("/v1/runs", json={"policy": "hpa", "forecaster": "none",
                                      "workload": "nonexistent_workload"})
    assert r.status_code == 422


def test_run_list_returns_submitted():
    r  = client.post("/v1/runs", json={"policy": "hpa", "forecaster": "none",
                                       "workload": "diurnal_burst"})
    rid = r.json()["run_id"]
    r2  = client.get("/v1/runs")
    assert r2.status_code == 200
    ids = [run["run_id"] for run in r2.json()["runs"]]
    assert rid in ids


def test_run_list_status_filter():
    client.post("/v1/runs", json={"policy": "hpa", "forecaster": "none",
                                  "workload": "diurnal_burst"})
    r = client.get("/v1/runs?status=pending")
    assert r.status_code == 200
    # All returned runs should be pending (or already moved to running)
    for run in r.json()["runs"]:
        assert run["status"] in ("pending", "running")


# ── RunDatabase unit tests ────────────────────────────────────────────────────

def test_database_insert_and_get():
    with tempfile.TemporaryDirectory() as d:
        with RunDatabase(Path(d) / "test.db") as db:
            from src.api.models import ExperimentRun
            run = ExperimentRun(policy="mpc", forecaster="lstm")
            db.insert(run)
            row = db.get(run.run_id)
            assert row is not None
            assert row["run_id"]    == run.run_id
            assert row["policy"]    == "mpc"
            assert row["forecaster"] == "lstm"


def test_database_persists_use_fh_opt():
    with tempfile.TemporaryDirectory() as d:
        with RunDatabase(Path(d) / "test.db") as db:
            from src.api.models import ExperimentRun
            run = ExperimentRun(policy="mpc", forecaster="none", use_fh_opt=True)
            db.insert(run)
            row = db.get(run.run_id)
            assert row is not None
            assert row["use_fh_opt"] is True


def test_database_persists_batch():
    with tempfile.TemporaryDirectory() as d:
        with RunDatabase(Path(d) / "test.db") as db:
            from src.api.models import ExperimentRun
            run = ExperimentRun(policy="mpc", forecaster="none", batch="batch-7-fhopt")
            db.insert(run)
            row = db.get(run.run_id)
            assert row is not None
            assert row["batch"] == "batch-7-fhopt"


def test_database_update_status():
    with tempfile.TemporaryDirectory() as d:
        with RunDatabase(Path(d) / "test.db") as db:
            from src.api.models import ExperimentRun
            run = ExperimentRun(policy="hpa", forecaster="none")
            db.insert(run)
            db.update_status(run.run_id, RunStatus.COMPLETED,
                             summary={"sla_pct": 3.1}, ended_at=time.time())
            row = db.get(run.run_id)
            assert row["status"]            == "completed"
            assert row["summary"]["sla_pct"] == 3.1


def test_database_list_and_count():
    with tempfile.TemporaryDirectory() as d:
        with RunDatabase(Path(d) / "test.db") as db:
            from src.api.models import ExperimentRun
            for i in range(5):
                db.insert(ExperimentRun(policy="hpa", forecaster="none"))
            assert db.count() == 5
            rows = db.list_all(limit=3)
            assert len(rows) == 3
