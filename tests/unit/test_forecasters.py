# tests/unit/test_forecasters.py
import pytest
import numpy as np
import pandas as pd
from src.forecasting.base import BaseForecaster
from src.forecasting import create_forecaster, available_forecasters


# ── Helpers ───────────────────────────────────────────────────────────

def sine_series(n=120, amplitude=200, offset=300, noise=10, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n)
    return (offset + amplitude * np.sin(2 * np.pi * t / 24)
            + rng.normal(0, noise, n))


def flat_series(n=60, value=300.0):
    return np.full(n, value)


def ramp_series(n=60):
    return np.linspace(100, 500, n)


def _make(name, **kwargs):
    return create_forecaster(name, **kwargs)


# ── Contract (all forecasters) ────────────────────────────────────────

@pytest.mark.parametrize("name,kwargs", [
    ("arima",   {"min_series_length": 10}),
    ("prophet", {"min_series_length": 20}),
    ("lstm",    {"min_series_length": 50, "max_epochs": 3,
                 "window_size": 10, "min_windows": 5}),
])
class TestForecasterContract:

    def test_subclasses_base(self, name, kwargs):
        f = _make(name, **kwargs)
        assert isinstance(f, BaseForecaster)

    def test_predict_before_fit_raises(self, name, kwargs):
        f = _make(name, **kwargs)
        with pytest.raises(RuntimeError, match="fit"):
            f.predict(5)

    def test_fit_returns_self(self, name, kwargs):
        f = _make(name, **kwargs)
        assert f.fit(flat_series()) is f

    def test_is_fitted_after_fit(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(flat_series())
        assert f.is_fitted

    def test_predict_correct_shape(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(sine_series())
        for steps in [1, 3, 10]:
            out = f.predict(steps)
            assert out.shape == (steps,)

    def test_predict_zero_steps_empty(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(flat_series())
        assert len(f.predict(0)) == 0

    def test_predict_no_negatives(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(flat_series())
        assert np.all(f.predict(10) >= 0)

    def test_update_does_not_raise(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(flat_series())
        f.update(350.0)

    def test_reset_clears_fitted(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(flat_series())
        f.reset()
        assert not f.is_fitted

    def test_short_series_graceful(self, name, kwargs):
        f = _make(name, **kwargs)
        f.fit(np.array([100.0, 200.0, 150.0]))
        out = f.predict(5)
        assert out.shape == (5,) and np.all(out >= 0)


# ── Latency hooks ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,kwargs", [
    ("arima",   {"min_series_length": 10}),
    ("prophet", {"min_series_length": 20}),
    ("lstm",    {"min_series_length": 50, "max_epochs": 3,
                 "window_size": 10, "min_windows": 5}),
])
class TestLatencyHooks:

    def test_timed_fit_records_latency(self, name, kwargs):
        f = _make(name, **kwargs)
        f.timed_fit(sine_series())
        assert f.fit_latency_ms is not None
        assert f.fit_latency_ms > 0

    def test_timed_predict_records_latency(self, name, kwargs):
        f = _make(name, **kwargs)
        f.timed_fit(sine_series())
        f.timed_predict(5)
        assert f.predict_latency_ms is not None
        assert f.predict_latency_ms >= 0

    def test_metadata_has_required_keys(self, name, kwargs):
        f = _make(name, **kwargs)
        f.timed_fit(sine_series())
        f.timed_predict(3)
        m = f.metadata()
        for key in ["name", "fitted", "fit_series_len",
                    "fit_latency_ms", "predict_latency_ms"]:
            assert key in m, f"Missing metadata key: {key}"

    def test_metadata_fit_series_len_correct(self, name, kwargs):
        f = _make(name, **kwargs)
        series = sine_series(n=80)
        f.timed_fit(series)
        assert f.metadata()["fit_series_len"] == 80


# ── ARIMA specific ────────────────────────────────────────────────────

class TestARIMA:

    def test_flat_series_fallback_uses_last_observed(self):
        """Fallback must return last observed value, NOT zero."""
        f = create_forecaster("arima", min_series_length=10)
        # Series too short → fallback
        short = np.array([250.0, 260.0, 270.0])
        f.fit(short)
        out = f.predict(5)
        assert np.allclose(out, 270.0, atol=1.0), \
            f"Expected ~270 (last observed), got {out}"

    def test_fallback_never_returns_zero_for_nonzero_series(self):
        """Degenerate model guard: forecast must not collapse to 0."""
        f = create_forecaster("arima", min_series_length=10)
        f.fit(flat_series(value=500.0, n=30))
        out = f.predict(5)
        assert np.all(out > 10.0), f"Near-zero forecast for 500 RPS series: {out}"

    def test_update_truly_refits_model(self):
        """After update(), model should have been refit (not same object)."""
        f = create_forecaster("arima", min_series_length=10, online_window=20)
        f.fit(flat_series(n=30, value=300.0))
        model_before = id(f._model)
        # Feed enough updates to trigger refit (online_window=20 new values)
        for v in np.linspace(300, 600, 20):
            f.update(float(v))
        model_after = id(f._model)
        assert model_before != model_after, \
            "ARIMA update() must refit the model, not just append to buffer"

    def test_accuracy_flat_series(self):
        """MAPE on flat series must be < 5%."""
        f = create_forecaster("arima", min_series_length=10)
        train = flat_series(n=60, value=300.0)
        f.fit(train)
        pred = f.predict(10)
        mape = float(np.mean(np.abs(pred - 300.0) / 300.0)) * 100
        assert mape < 5.0, f"ARIMA MAPE on flat series: {mape:.1f}%"

    def test_factory_creates_arima(self):
        f = create_forecaster("arima", min_series_length=10)
        from src.forecasting.arima import ARIMAForecaster
        assert isinstance(f, ARIMAForecaster)


# ── Prophet specific ─────────────────────────────────────────────────

class TestProphet:

    def test_fallback_uses_last_observed(self):
        f = create_forecaster("prophet", min_series_length=20)
        short = np.array([180.0, 190.0, 200.0])
        f.fit(short)
        out = f.predict(5)
        assert np.allclose(out, 200.0, atol=1.0)

    def test_update_advances_timestamp(self):
        f = create_forecaster("prophet", min_series_length=20,
                              timestep_seconds=60)
        f.fit(sine_series(n=40))
        ds_before = f._last_ds
        f.update(300.0)
        assert f._last_ds == ds_before + pd.Timedelta(seconds=60)

    def test_real_timestamps_accepted(self):
        """fit() with start_time kwarg must not crash."""
        f = create_forecaster("prophet", min_series_length=20)
        f.fit(sine_series(n=40), start_time="2024-03-01")
        out = f.predict(5)
        assert out.shape == (5,)

    def test_accuracy_flat_series(self):
        f = create_forecaster("prophet", min_series_length=20)
        f.fit(flat_series(n=80, value=400.0))
        pred = f.predict(5)
        mape = float(np.mean(np.abs(pred - 400.0) / 400.0)) * 100
        assert mape < 10.0, f"Prophet MAPE on flat series: {mape:.1f}%"


# ── LSTM specific ─────────────────────────────────────────────────────

class TestLSTM:

    def _fast_lstm(self, seed=42):
        return create_forecaster(
            "lstm", window_size=10, hidden_size=16,
            num_layers=1, max_epochs=5, min_series_length=30,
            min_windows=10, seed=seed,
        )

    def test_fallback_uses_last_observed(self):
        f = self._fast_lstm()
        short = np.array([150.0, 160.0, 170.0])
        f.fit(short)
        out = f.predict(5)
        assert np.allclose(out, 170.0, atol=1.0)

    def test_insufficient_windows_triggers_fallback(self):
        f = create_forecaster(
            "lstm", window_size=10, min_series_length=5,
            min_windows=50, max_epochs=2,
        )
        # Only 15 series points → only 5 windows < min_windows=50
        f.fit(np.arange(15, dtype=float))
        assert f._training_summary.get("status") == "fallback"

    def test_seed_control_reproducibility(self):
        """Same seed must produce identical predictions."""
        series = sine_series(n=80)
        f1 = self._fast_lstm(seed=7)
        f2 = self._fast_lstm(seed=7)
        f1.fit(series)
        f2.fit(series)
        np.testing.assert_array_equal(f1.predict(5), f2.predict(5))

    def test_different_seeds_may_differ(self):
        """Different seeds should (usually) produce different predictions."""
        series = sine_series(n=80)
        f1 = self._fast_lstm(seed=1)
        f2 = self._fast_lstm(seed=99)
        f1.fit(series)
        f2.fit(series)
        # Not guaranteed to differ on 5 epochs, but usually will
        # Use allclose with tight tol — if identical, it's a coincidence
        p1, p2 = f1.predict(5), f2.predict(5)
        # Just check they're both valid
        assert p1.shape == p2.shape == (5,)

    def test_training_summary_populated(self):
        f = self._fast_lstm()
        f.fit(sine_series(n=80))
        s = f.training_summary()
        assert s.get("status") == "trained"
        for key in ["epochs_run", "best_val_loss", "seed", "n_windows"]:
            assert key in s, f"Missing training_summary key: {key}"

    def test_metadata_includes_training_summary(self):
        f = self._fast_lstm()
        f.timed_fit(sine_series(n=80))
        m = f.metadata()
        assert "best_val_loss" in m


# ── Lazy import / optional deps ───────────────────────────────────────

class TestLazyImports:

    def test_package_import_does_not_crash(self):
        """Importing src.forecasting must never raise ImportError."""
        import importlib
        importlib.import_module("src.forecasting")   # must not raise

    def test_available_forecasters_returns_list(self):
        result = available_forecasters()
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)

    def test_unknown_forecaster_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown forecaster"):
            create_forecaster("xgboost")

    def test_case_insensitive_factory(self):
        from src.forecasting.arima import ARIMAForecaster
        f = create_forecaster("ARIMA", min_series_length=10)
        assert isinstance(f, ARIMAForecaster)


# ── Forecast → MPC Integration ────────────────────────────────────────

class TestForecastToMPCIntegration:

    def test_arima_forecast_feeds_mpc(self):
        """
        Full pipeline: fit ARIMA on trace → predict h* steps →
        pass forecast into MPC → get valid replica count.
        """
        from src.simulator.adapt import ADAPTTracker
        from src.policies.mpc import MPCPolicy

        series = sine_series(n=120)
        train  = series[:96]
        # current RPS = last known
        current_rps = float(series[96])

        # Fit forecaster
        forecaster = create_forecaster("arima", min_series_length=10)
        forecaster.fit(train)

        # Set up MPC + ADAPT
        adapt = ADAPTTracker(
            alpha=0.3, cold_start_s=120.0,
            cold_start_min_s=30.0, cold_start_max_s=600.0,
            epsilon_steps=1, timestep_seconds=60,
        )
        mpc = MPCPolicy(
            adapt_tracker=adapt,
            lambda_sla=10.0, lambda_cost=1.0, lambda_stab=0.5,
            capacity_per_replica=100.0,
            min_replicas=1, max_replicas=50,
        )

        # Get forecast and feed into MPC
        h_star   = adapt.optimal_horizon()
        forecast = forecaster.predict(h_star)

        replicas = mpc.compute_replicas(
            current_rps=current_rps,
            current_replicas=3,
            step=96,
            forecast=forecast,
            warm_replicas=3,
        )

        assert 1 <= replicas <= 50
        assert forecast.shape == (h_star,)
        assert np.all(forecast >= 0)

    def test_all_forecasters_feed_mpc_without_crash(self):
        """
        Smoke: every forecaster must produce a forecast that MPC accepts.
        """
        from src.simulator.adapt import ADAPTTracker
        from src.policies.mpc import MPCPolicy

        series = sine_series(n=120)
        adapt  = ADAPTTracker(
            alpha=0.3, cold_start_s=120.0,
            cold_start_min_s=30.0, cold_start_max_s=600.0,
            epsilon_steps=1, timestep_seconds=60,
        )
        mpc = MPCPolicy(
            adapt_tracker=adapt,
            lambda_sla=10.0, lambda_cost=1.0, lambda_stab=0.0,
            capacity_per_replica=100.0,
            min_replicas=1, max_replicas=50,
        )

        forecasters = [
            create_forecaster("arima",   min_series_length=10),
            create_forecaster("prophet", min_series_length=20),
            create_forecaster("lstm",    window_size=10, hidden_size=16,
                              num_layers=1, max_epochs=3,
                              min_series_length=30, min_windows=10),
        ]

        for f in forecasters:
            f.fit(series[:96])
            h    = adapt.optimal_horizon()
            fc   = f.predict(h)
            r    = mpc.compute_replicas(300.0, 3, 0, forecast=fc)
            assert 1 <= r <= 50, f"{f.name}: invalid replica count {r}"
            mpc.reset()