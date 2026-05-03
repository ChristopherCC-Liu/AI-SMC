"""Ticket 4 Step 3 — XAUUSD-only regime classifier tests.

The sidecar regime classifier maps a window's H1 bars to one or more
of the seven REGIME_BUCKETS. Pure function: input is bar arrays,
output is a tuple of bucket labels. No production import; no clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.hedgerock.evolution.regime_classifier import (
    classify_window,
    compute_window_features,
)


def _bars(*, n: int, start_close: float = 2000.0,
          drift: float = 0.0, vol_pct: float = 0.001,
          start_ts: datetime | None = None,
          gap_at: int | None = None,
          gap_hours: int = 0):
    """Synthetic H1 bar series. Each bar steps close by `drift`,
    high = close * (1+vol), low = close * (1-vol)."""
    ts = start_ts or datetime(2024, 1, 1, tzinfo=timezone.utc)
    close = start_close
    rows = []
    for i in range(n):
        if gap_at is not None and i == gap_at and gap_hours > 0:
            ts += timedelta(hours=gap_hours)
        rows.append({
            "ts": ts, "high": close * (1 + vol_pct),
            "low": close * (1 - vol_pct), "close": close,
        })
        close += drift
        ts += timedelta(hours=1)
    return rows


# ---------------------------------------------------------------------------
# 1. compute_window_features — basic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_window_features_returns_known_shape() -> None:
    bars = _bars(n=100, start_close=2000.0, drift=0.0, vol_pct=0.001)
    f = compute_window_features(bars)
    assert "n_bars" in f
    assert "median_vol_rank" in f
    assert "trend_bars_h4_proxy" in f
    assert "weekend_gap_count" in f
    assert "max_single_bar_move_pct" in f


@pytest.mark.unit
def test_compute_window_features_handles_empty() -> None:
    f = compute_window_features([])
    assert f["n_bars"] == 0


# ---------------------------------------------------------------------------
# 2. classify_window — bucket assignment
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_window_low_vol_drift_zero_yields_range_low_vol() -> None:
    bars = _bars(n=500, start_close=2000.0, drift=0.0, vol_pct=0.0005)
    buckets = classify_window(bars)
    assert "range_low_vol" in buckets


@pytest.mark.unit
def test_classify_window_high_vol_drift_zero_yields_range_high_vol() -> None:
    bars = _bars(n=500, start_close=2000.0, drift=0.0, vol_pct=0.005)
    buckets = classify_window(bars)
    assert "range_high_vol" in buckets


@pytest.mark.unit
def test_classify_window_steady_uptrend_yields_trend_up() -> None:
    bars = _bars(n=500, start_close=2000.0, drift=0.5, vol_pct=0.0005)
    buckets = classify_window(bars)
    assert "trend_up" in buckets


@pytest.mark.unit
def test_classify_window_steady_downtrend_yields_trend_down() -> None:
    bars = _bars(n=500, start_close=2000.0, drift=-0.5, vol_pct=0.0005)
    buckets = classify_window(bars)
    assert "trend_down" in buckets


@pytest.mark.unit
def test_classify_window_breakout_spike_yields_breakout() -> None:
    """Insert a single big move (>2% intra-bar) into a quiet series."""
    bars = _bars(n=200, start_close=2000.0, drift=0.0, vol_pct=0.0005)
    bars[100]["high"] = bars[100]["close"] * 1.04   # +4% high
    bars[100]["low"] = bars[100]["close"] * 0.99
    buckets = classify_window(bars)
    assert "breakout" in buckets


@pytest.mark.unit
def test_classify_window_weekend_gap_detected() -> None:
    """A 60+h ts gap mid-window counts as a weekend_gap occurrence."""
    bars = _bars(n=200, start_close=2000.0, drift=0.0, vol_pct=0.0005,
                 gap_at=100, gap_hours=65)
    buckets = classify_window(bars)
    assert "weekend_gap" in buckets


@pytest.mark.unit
def test_classify_window_returns_tuple_of_strings() -> None:
    bars = _bars(n=100, start_close=2000.0, drift=0.0, vol_pct=0.0005)
    out = classify_window(bars)
    assert isinstance(out, tuple)
    for b in out:
        assert isinstance(b, str)


@pytest.mark.unit
def test_classify_empty_window_returns_empty_tuple() -> None:
    assert classify_window([]) == ()


# ---------------------------------------------------------------------------
# 3. Determinism + isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_window_deterministic() -> None:
    bars = _bars(n=300, start_close=2000.0, drift=0.0, vol_pct=0.0005)
    a = classify_window(bars)
    b = classify_window(bars)
    assert a == b


@pytest.mark.unit
def test_classify_window_does_not_mutate_input() -> None:
    bars = _bars(n=200, start_close=2000.0, drift=0.0, vol_pct=0.0005)
    snapshot = [dict(b) for b in bars]
    classify_window(bars)
    for orig, after in zip(snapshot, bars):
        assert orig == after
