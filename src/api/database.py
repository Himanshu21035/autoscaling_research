"""
SQLite persistence layer for experiment runs.

Windows fix: sqlite3 keeps the file open as long as the connection object
lives. Tests must call db.close() explicitly (or use `with RunDatabase(...):`).
Use check_same_thread=False + WAL mode for safe multi-threaded access.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
from pathlib import Path
from src.api.models import ExperimentRun, RunStatus

DB_PATH = Path("outputs/runs.db")


class RunDatabase:

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a persistent connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
        return self._conn

    def close(self) -> None:
        """Explicitly close the connection — required on Windows before tmpdir cleanup."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Schema ────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id     TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'pending',
                    config     TEXT NOT NULL,
                    summary    TEXT NOT NULL DEFAULT '{}',
                    error      TEXT,
                    started_at REAL,
                    ended_at   REAL
                )
            """)
            conn.commit()

    # ── Write ops ─────────────────────────────────────────────────────

    def insert(self, run: ExperimentRun) -> None:
        config = {
            "policy": run.policy, "forecaster": run.forecaster,
            "batch": run.batch,
            "workload": run.workload, "cold_start_s": run.cold_start_s,
            "train_frac": run.train_frac, "val_frac": run.val_frac,
            "forecast_margin": run.forecast_margin,
            "lambda_sla": run.lambda_sla, "lambda_cost": run.lambda_cost,
            "lambda_stab": run.lambda_stab,
            "cold_start_steps": run.cold_start_steps,
            "use_fh_opt": run.use_fh_opt,
            "seed": run.seed,
        }
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO runs (run_id, created_at, status, config) VALUES (?,?,?,?)",
                (run.run_id, run.created_at, run.status.value, json.dumps(config)),
            )
            conn.commit()

    def update_status(
        self,
        run_id:     str,
        status:     RunStatus,
        summary:    dict | None  = None,
        error:      str | None   = None,
        started_at: float | None = None,
        ended_at:   float | None = None,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE runs
                   SET status=?, summary=?, error=?, started_at=?, ended_at=?
                   WHERE run_id=?""",
                (
                    status.value,
                    json.dumps(summary or {}),
                    error,
                    started_at,
                    ended_at,
                    run_id,
                ),
            )
            conn.commit()

    # ── Read ops ──────────────────────────────────────────────────────

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_all(
        self,
        status: str | None = None,
        limit:  int = 50,
        offset: int = 0,
    ) -> list[dict]:
        query  = "SELECT * FROM runs"
        params: list = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._get_conn().execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self, status: str | None = None) -> int:
        query  = "SELECT COUNT(*) FROM runs"
        params: list = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        with self._lock:
            return self._get_conn().execute(query, params).fetchone()[0]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d       = dict(row)
        config  = json.loads(d.pop("config",  "{}"))
        summary = json.loads(d.pop("summary", "{}"))
        d.update(config)
        d["summary"]    = summary
        d["duration_s"] = (
            round(d["ended_at"] - d["started_at"], 2)
            if d.get("ended_at") and d.get("started_at") else None
        )
        return d


# Module-level singleton
run_db = RunDatabase()
