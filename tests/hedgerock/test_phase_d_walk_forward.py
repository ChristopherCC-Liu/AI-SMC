"""Phase D walk-forward harness tests.

Smoke + invariant coverage. The full 2024 run lives in
``scripts/run_phase_d_walk_forward.py`` (one-shot CLI).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.hedgerock.phase_d_walk_forward import (
    DEFAULT_INIT_EQUITY,
    DEFAULT_SPREAD_PTS,
    WalkForwardConfig,
    run_walk_forward,
)


# ---------------------------------------------------------------------------
# Synthetic lake stub
# ---------------------------------------------------------------------------


def _ohlcv(
    *,
    start: datetime, n_bars: int, bar_minutes: int,
    base: float = 2000.0, slope: float = 0.0, noise: float = 5.0,
) -> pl.DataFrame:
    """Synthetic OHLCV — deterministic sinusoidal-ish series."""
    rows = []
    price = base
    for i in range(n_bars):
        ts = start + timedelta(minutes=bar_minutes * i)
        # Triangle wave gives both up and down legs with predictable amplitude.
        delta = ((i % 50) - 25) * 0.2 + slope
        price = price + delta
        rows.append({
            "ts": ts,
            "open": price - delta / 2,
            "high": price + noise,
            "low": price - noise,
            "close": price,
            "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _FakeLake:
    def __init__(self, *, h1: pl.DataFrame, h4: pl.DataFrame, d1: pl.DataFrame) -> None:
        self._h1 = h1
        self._h4 = h4
        self._d1 = d1

    def query(self, instrument, timeframe, start, end):
        if str(timeframe) == "H1":
            df = self._h1
        elif str(timeframe) == "H4":
            df = self._h4
        elif str(timeframe) == "D1":
            df = self._d1
        else:
            return pl.DataFrame()
        if df.is_empty():
            return df
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


def _build_lake(*, start: datetime, days: int, slope: float = 0.0) -> _FakeLake:
    """2-week test window by default."""
    n_h1 = 24 * days
    n_h4 = 6 * days
    n_d1 = days
    h1 = _ohlcv(start=start, n_bars=n_h1, bar_minutes=60, slope=slope)
    h4 = _ohlcv(start=start, n_bars=n_h4, bar_minutes=240, slope=slope * 4)
    d1 = _ohlcv(start=start, n_bars=n_d1, bar_minutes=1440, slope=slope * 24)
    return _FakeLake(h1=h1, h4=h4, d1=d1)


@pytest.fixture
def small_lake() -> _FakeLake:
    return _build_lake(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        days=21,  # 3 weeks — enough for H1 lookback (240 = 10 days)
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_smoke_short_window_runs_to_completion(small_lake) -> None:
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        # Lower h1_lookback so the small-window run actually exercises
        # rule_engine for many bars.
        h1_lookback=120,
        h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)

    # Both runs completed.
    assert result.static_metrics.final_equity > 0
    assert result.dynamic_metrics.final_equity > 0
    # Envelope log captured some bars (after warm-up).
    assert len(result.envelope_log) > 0


@pytest.mark.unit
def test_envelope_log_has_complete_fields(small_lake) -> None:
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    assert len(result.envelope_log) > 10
    expected_keys = {
        "ts", "equity", "regime_v2", "confidence", "mode", "risk_tier",
        "lot_factor", "max_next_lot", "takeprofit_points",
        "recovery_multiplier", "max_orders_buy", "max_orders_sell",
        "cooldown_until", "reason",
    }
    for entry in result.envelope_log:
        missing = expected_keys - set(entry.keys())
        assert not missing, f"envelope log missing fields: {missing}"


# ---------------------------------------------------------------------------
# Phase C invariants must hold inside the replay loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_never_aggressive_in_first_few_bars(small_lake) -> None:
    """Cold-start: simulator has 0 closed trades → recent_sample_count<5
    → insufficient_sample → never aggressive in the early phase."""
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    early = result.envelope_log[:30]
    aggressive_early = [e for e in early if e["risk_tier"] == "aggressive"]
    assert not aggressive_early, (
        f"unexpected early aggressive bars: "
        f"{[(e['ts'], e['reason']) for e in aggressive_early]}"
    )


@pytest.mark.unit
def test_replay_envelope_ts_monotonic(small_lake) -> None:
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    timestamps = [e["ts"] for e in result.envelope_log]
    assert timestamps == sorted(timestamps), "envelope log timestamps not monotonic"


@pytest.mark.unit
def test_replay_dynamic_trades_no_more_than_static_in_observe_phase(
    small_lake,
) -> None:
    """Observe-mode bars produce no NEW entries (existing positions still TP).
    With cold-start always in observe (history insufficient), dynamic
    should have ≤ static trade count when the regime keeps it observing."""
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    # Sanity: dynamic total trades ≤ static (observe mode prunes new opens).
    assert result.dynamic_metrics.n_trades <= result.static_metrics.n_trades


@pytest.mark.unit
def test_replay_metrics_record_mode_distribution(small_lake) -> None:
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    m = result.dynamic_metrics
    total_mode_bars = (
        m.bars_in_hedgerock + m.bars_in_observe + m.bars_in_halt + m.bars_in_momentum
    )
    assert total_mode_bars == len(result.envelope_log)


@pytest.mark.unit
def test_static_baseline_does_not_blow_up_on_calm_window(small_lake) -> None:
    """Triangle-wave price with small amplitude shouldn't trigger
    the 80% DD blowup. (Sanity check on the simulator math.)"""
    cfg = WalkForwardConfig(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 22, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    # If this fires, our default params or simulator math has drifted —
    # tests need attention before trusting the metrics.
    assert not result.static_metrics.blowup
    assert not result.dynamic_metrics.blowup
