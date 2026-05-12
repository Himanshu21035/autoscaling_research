"""
RunDatabase unit tests — Windows-safe.

Key fix: call db.close() explicitly before the TemporaryDirectory context
exits. On Windows, SQLite holds the file open until the connection is closed;
TemporaryDirectory.__exit__ tries to delete the file and gets WinError 32.

Pattern for every test:
    with tempfile.TemporaryDirectory() as d:
        db = RunDatabase(Path(d) / "test.db")
        try:
            ... test body ...
        finally:
            db.close()   # ← releases Windows file lock BEFORE rmtree
"""
import tempfile
import time
from pathlib import Path

import pytest

from src.api.database import RunDatabase
from src.api.models import ExperimentRun, RunStatus


def _make_db(d: str) -> RunDatabase:
    return RunDatabase(Path(d) / "test.db")


def test_database_insert_and_get():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            run = ExperimentRun(policy="mpc", forecaster="lstm")
            db.insert(run)
            row = db.get(run.run_id)
            assert row is not None
            assert row["run_id"]     == run.run_id
            assert row["policy"]     == "mpc"
            assert row["forecaster"] == "lstm"
            assert row["status"]     == "pending"
        finally:
            db.close()


def test_database_update_status():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            run = ExperimentRun(policy="hpa", forecaster="none")
            db.insert(run)
            db.update_status(
                run.run_id, RunStatus.COMPLETED,
                summary={"sla_pct": 3.1},
                ended_at=time.time(),
            )
            row = db.get(run.run_id)
            assert row["status"]             == "completed"
            assert row["summary"]["sla_pct"] == 3.1
            assert row["ended_at"] is not None
        finally:
            db.close()


def test_database_list_and_count():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            for _ in range(5):
                db.insert(ExperimentRun(policy="hpa", forecaster="none"))
            assert db.count() == 5
            rows = db.list_all(limit=3)
            assert len(rows) == 3
        finally:
            db.close()


def test_database_list_status_filter():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            r1 = ExperimentRun(policy="mpc", forecaster="lstm")
            r2 = ExperimentRun(policy="hpa", forecaster="none")
            db.insert(r1)
            db.insert(r2)
            db.update_status(r1.run_id, RunStatus.COMPLETED,
                             summary={}, ended_at=time.time())
            assert db.count(status="completed") == 1
            assert db.count(status="pending")   == 1
            rows = db.list_all(status="completed")
            assert len(rows) == 1
            assert rows[0]["run_id"] == r1.run_id
        finally:
            db.close()


def test_database_context_manager():
    """Verify `with RunDatabase(...) as db:` closes on exit — no manual close needed."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.db"
        with RunDatabase(path) as db:
            run = ExperimentRun(policy="pid", forecaster="none")
            db.insert(run)
            assert db.get(run.run_id) is not None
        # After __exit__ the connection is None — file lock released
        assert db._conn is None


def test_database_get_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            assert db.get("does-not-exist") is None
        finally:
            db.close()


def test_database_duration_computed():
    with tempfile.TemporaryDirectory() as d:
        db = _make_db(d)
        try:
            run = ExperimentRun(policy="mpc", forecaster="arima")
            db.insert(run)
            t0 = time.time()
            db.update_status(run.run_id, RunStatus.COMPLETED,
                             summary={}, started_at=t0,
                             ended_at=t0 + 5.0)
            row = db.get(run.run_id)
            assert row["duration_s"] == pytest.approx(5.0, abs=0.01)
        finally:
            db.close()
