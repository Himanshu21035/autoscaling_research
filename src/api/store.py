"""
In-memory stores — written by the simulator/smoke-tests, read by the API.

MetricsStore  : ring buffer of per-step metrics from run_simulation()
ResultsStore  : list of SimResult summaries from smoke_test_step9.py

Wire-up (add to smoke_test_step9.py after each run_simulation call):

    from src.api.store import metrics_store, results_store

    # push per-step metrics
    for m in sim_result.metrics:
        metrics_store.push(
            policy     = sim_result.policy_name,
            forecaster = sim_result.forecaster_name,
            step       = m.step,
            rps        = m.rps,
            replicas   = m.warm_replicas,
            latency_ms = m.latency_ms,
            sla_violated = m.sla_violated,
            cost       = m.cost,
            utilisation      = m.capacity and m.rps / m.capacity or 0.0,
            adapt_estimate_s = m.adapt_estimate_s,
        )

    # push run summary at the end
    results_store.append(sim_result.summary())
"""
from __future__ import annotations
import threading
from collections import deque
from typing import Any


class MetricsStore:
    """Thread-safe ring buffer. Each record carries policy+forecaster labels."""

    def __init__(self, maxlen: int = 50_000):
        self._buf:  deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(
        self,
        policy:           str,
        forecaster:       str,
        step:             int,
        rps:              float,
        replicas:         int,
        latency_ms:       float,
        sla_violated:     bool,
        cost:             float,
        utilisation:      float = 0.0,
        adapt_estimate_s: float = 0.0,
    ) -> None:
        record = dict(
            policy=policy, forecaster=forecaster,
            step=step, rps=rps, replicas=replicas,
            latency_ms=latency_ms, sla_violated=sla_violated,
            cost=cost, utilisation=utilisation,
            adapt_estimate_s=adapt_estimate_s,
        )
        with self._lock:
            self._buf.append(record)

    def latest(
        self,
        policy:     str | None = None,
        forecaster: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            items = list(self._buf)
        filtered = _filter(items, policy, forecaster)
        return filtered[-1] if filtered else None

    def last_n(
        self,
        n:          int,
        policy:     str | None = None,
        forecaster: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buf)
        filtered = _filter(items, policy, forecaster)
        return filtered[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


class ResultsStore:
    """Holds smoke-test run summaries (one dict per SimResult.summary())."""

    def __init__(self):
        self._runs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, run: dict[str, Any]) -> None:
        with self._lock:
            self._runs.append(run)

    def set(self, runs: list[dict[str, Any]]) -> None:
        """Replace entire results set (called at end of smoke test)."""
        with self._lock:
            self._runs = list(runs)

    def get_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._runs)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


def _filter(
    items:      list[dict[str, Any]],
    policy:     str | None,
    forecaster: str | None,
) -> list[dict[str, Any]]:
    if policy:
        items = [r for r in items if r.get("policy") == policy]
    if forecaster:
        items = [r for r in items if r.get("forecaster") == forecaster]
    return items


# Module-level singletons — import these everywhere
metrics_store = MetricsStore(maxlen=50_000)
results_store = ResultsStore()
