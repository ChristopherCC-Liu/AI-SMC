"""Phase D-hotfix-1 — replay credibility tests.

Pinned guarantees:
    1. Lookahead-free per-bar windows (H1 strict-prior, H4 closed-bar,
       D1 ATR uses prior closed day only).
    2. envelope.transition_lock_until_ts > ts → no new opens this bar.
    3. Renamed cooldown metrics + post-step equity logging.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.hedgerock.phase_d_walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
)


# ---------------------------------------------------------------------------
# Synthetic lake helpers
# ---------------------------------------------------------------------------


def _ohlcv(
    *, start: datetime, n_bars: int, bar_minutes: int,
    base: float = 2000.0, slope: float = 0.0, noise: float = 5.0,
) -> pl.DataFrame:
    rows = []
    price = base
    for i in range(n_bars):
        ts = start + timedelta(minutes=bar_minutes * i)
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
    return pl.DataFrame(rows).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


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


def _build_lake(*, start: datetime, days: int, slope: float = 0.0,
                spike_at: int | None = None) -> _FakeLake:
    n_h1 = 24 * days
    n_h4 = 6 * days
    n_d1 = days
    h1 = _ohlcv(start=start, n_bars=n_h1, bar_minutes=60, slope=slope)
    if spike_at is not None and 0 <= spike_at < n_h1:
        # Replace bar `spike_at` with extreme OHLC values that would
        # explode the volatility rank if they ever leaked into the
        # trailing window.
        spike_row = h1.row(spike_at, named=True)
        h1 = h1.with_columns([
            pl.when(pl.int_range(pl.len()) == spike_at)
            .then(99999.0).otherwise(pl.col("high")).alias("high"),
            pl.when(pl.int_range(pl.len()) == spike_at)
            .then(-99999.0).otherwise(pl.col("low")).alias("low"),
            pl.when(pl.int_range(pl.len()) == spike_at)
            .then(99999.0).otherwise(pl.col("close")).alias("close"),
        ])
        del spike_row  # noqa: F841
    h4 = _ohlcv(start=start, n_bars=n_h4, bar_minutes=240, slope=slope * 4)
    d1 = _ohlcv(start=start, n_bars=n_d1, bar_minutes=1440, slope=slope * 24)
    return _FakeLake(h1=h1, h4=h4, d1=d1)


# ---------------------------------------------------------------------------
# Patch 1 — no lookahead
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_current_bar_anomaly_does_not_affect_envelope_at_that_bar() -> None:
    """A spike injected at H1 bar K must NOT change the envelope produced
    AT bar K. Bar K's decision is built from the trailing window
    [K-h1_lookback, K) — strictly excluding bar K itself."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days = 30
    spike_idx = 24 * 25  # day 25 — past D1 ATR warmup (~16 days into the run)
    lake_a = _build_lake(start=start, days=days)
    lake_b = _build_lake(start=start, days=days, spike_at=spike_idx)

    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=start,
        end=start + timedelta(days=days),
        h1_lookback=120, h4_lookback=20,
    )
    result_a = run_walk_forward(cfg, lake_a)
    result_b = run_walk_forward(cfg, lake_b)

    # Both runs should have identical-length, identical-ordered envelope
    # logs (same lake skeleton). Find the spike bar by ts equality.
    assert len(result_a.envelope_log) == len(result_b.envelope_log)
    spike_ts_str = (start + timedelta(hours=spike_idx)).isoformat().replace("+00:00", "")
    entry_a = entry_b = None
    for ea, eb in zip(result_a.envelope_log, result_b.envelope_log):
        if spike_ts_str in ea["ts"]:
            entry_a, entry_b = ea, eb
            break
    assert entry_a is not None and entry_b is not None, (
        f"spike bar at {spike_ts_str} not in log; first ts={result_a.envelope_log[0]['ts']!r}"
    )

    # Envelope structural fields at bar K must be identical (only
    # post-step equity / position state can differ).
    structural = ("regime_v2", "confidence", "mode", "risk_tier",
                  "lot_factor", "max_next_lot", "takeprofit_points",
                  "recovery_multiplier", "max_orders_buy", "max_orders_sell",
                  "transition_lock_until_ts", "cooldown_until")
    for key in structural:
        assert entry_a[key] == entry_b[key], (
            f"spike at bar K leaked into bar K's envelope.{key}: "
            f"clean={entry_a[key]} spiked={entry_b[key]}"
        )


@pytest.mark.unit
def test_h1_frame_excludes_current_bar_via_load_data() -> None:
    """Direct introspection: the precomputed H1 frame for bar i must
    not contain bar i's timestamp."""
    from smc.hedgerock.phase_d_walk_forward import _load_data

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=21)
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=start,
        end=start + timedelta(days=21),
        h1_lookback=120, h4_lookback=20,
    )
    h1, atr_per_bar, h4_frames, h1_frames = _load_data(lake, cfg)

    # Pick a bar past warmup.
    i = 24 * 10  # day 10
    frame = h1_frames[i]
    assert frame is not None
    bar_i_ts = h1["ts"][i]
    frame_ts = frame["ts"].to_list()
    assert bar_i_ts not in frame_ts, (
        f"H1 frame for bar {i} (ts={bar_i_ts}) contains its own bar"
    )
    # And the frame's last ts is strictly less than bar_i_ts.
    assert frame_ts[-1] < bar_i_ts


@pytest.mark.unit
def test_h4_frame_excludes_unclosed_h4_bar() -> None:
    """For an H1 bar at 12:30, the H4 bar opened at 12:00 (which closes
    at 16:00) must NOT be in the trailing H4 frame — only H4 bars whose
    period closed strictly before 12:30 are eligible."""
    from smc.hedgerock.phase_d_walk_forward import _load_data

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=21)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start,
        end=start + timedelta(days=21),
        h1_lookback=120, h4_lookback=20,
    )
    h1, _atr, h4_frames, _h1f = _load_data(lake, cfg)

    # Find an H1 bar whose ts falls inside an H4 period (any non-boundary).
    # H4 bars open at 00, 04, 08, 12, 16, 20 UTC. Pick H1 at 14:00 — falls
    # inside the H4 period [12:00, 16:00). The 12:00 H4 bar closes at
    # 16:00, so it must NOT be in the H1 14:00 frame.
    i_target = None
    for i, ts in enumerate(h1["ts"].to_list()):
        if ts.hour == 14 and i >= 24 * 10:  # past warmup
            i_target = i
            break
    assert i_target is not None
    frame = h4_frames[i_target]
    if frame is None:
        pytest.skip("h4 lookback not yet warmed at chosen bar")
    h4_ts_in_frame = frame["ts"].to_list()
    # The H4 bar opening at the same day's 12:00 must NOT appear.
    h1_ts = h1["ts"][i_target]
    in_progress_h4_open = h1_ts.replace(hour=12, minute=0)
    if in_progress_h4_open >= h1_ts:  # boundary — skip
        pytest.skip()
    # Concretely: every H4 open in the frame must satisfy open + 4h ≤ h1_ts.
    for h4_ts in h4_ts_in_frame:
        assert h4_ts + timedelta(hours=4) <= h1_ts, (
            f"h4 frame for {h1_ts} contains unclosed h4 bar at {h4_ts}"
        )


@pytest.mark.unit
def test_d1_atr_lookup_uses_prior_closed_day() -> None:
    """The ATR fed to bar i must come from a D1 bar whose close ts ≤
    h1_ts_i.day_floor() (i.e. yesterday or earlier — never today)."""
    from smc.hedgerock.phase_d_walk_forward import _load_data, _prepare_atr_d1

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=30)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start, end=start + timedelta(days=30),
        h1_lookback=120, h4_lookback=20,
    )
    h1, atr_per_bar, _h4, _h1f = _load_data(lake, cfg)

    # Inverse the lookup: for each H1 bar with a non-None ATR, find the
    # D1 bar that produced that ATR — it must be a day strictly earlier.
    d1 = lake.query("XAUUSD", "D1", start - timedelta(days=60),
                    start + timedelta(days=30))
    atr_lookup = _prepare_atr_d1(d1)

    # Pick a bar past warmup.
    sample_i = 24 * 25
    h1_ts = h1["ts"][sample_i]
    expected_atr = atr_per_bar[sample_i]
    if expected_atr is None:
        pytest.skip("ATR not yet warmed at chosen bar")

    # The earliest day strictly before h1_ts.day_floor() that has the
    # ATR value matches what _load_data returned.
    today = h1_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    day = today - timedelta(days=1)
    found = None
    for _ in range(30):
        if day in atr_lookup:
            found = atr_lookup[day]
            break
        day = day - timedelta(days=1)
    assert found == expected_atr, (
        f"_load_data ATR for {h1_ts} = {expected_atr} but expected the "
        f"prior-closed-day ATR at {day} = {found}"
    )


# ---------------------------------------------------------------------------
# Patch 2 — replay enforces transition_lock
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_replay_does_not_open_during_transition_lock_window() -> None:
    """Synthesize a replay where v2 regime flips → transition_lock fires.
    During the lock window, no new positions may be opened."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=21)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start,
        end=start + timedelta(days=21),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, lake)

    # Find every bar where transition_lock_active is True.
    locked_entries = [e for e in result.envelope_log if e.get("transition_lock_active")]
    if not locked_entries:
        pytest.skip("synthetic data did not produce a regime transition this run")

    for e in locked_entries:
        # During the lock, the harness must have reset effective_mode
        # to observe (no new opens) — even if rule_engine said hedgerock.
        assert e["effective_mode"] in ("observe", "halt"), (
            f"transition_lock active at {e['ts']} but effective_mode="
            f"{e['effective_mode']} (rule_engine mode={e['mode']})"
        )
        assert e["bar_opens"] == 0, (
            f"opened {e['bar_opens']} positions during transition_lock at {e['ts']}"
        )


# ---------------------------------------------------------------------------
# Patch 3 — metric naming + post-step equity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_use_renamed_cooldown_fields() -> None:
    """``cooldown_bars`` and ``cooldown_trigger_count`` are populated
    on TradeMetrics; the old ambiguous ``cooldown_triggers`` is gone."""
    from smc.hedgerock.phase_d_walk_forward import TradeMetrics

    fields = TradeMetrics.__dataclass_fields__
    assert "cooldown_bars" in fields
    assert "cooldown_trigger_count" in fields
    assert "transition_lock_bars" in fields
    assert "cooldown_triggers" not in fields


@pytest.mark.unit
def test_envelope_log_records_post_step_equity_and_bar_opens() -> None:
    """Each envelope_log entry now records bar_opens (positions opened
    during this bar's _step) and post-step equity."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=21)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start, end=start + timedelta(days=21),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, lake)
    assert result.envelope_log
    for e in result.envelope_log:
        assert "bar_opens" in e
        assert e["bar_opens"] >= 0
        assert "transition_lock_active" in e
        assert "transition_lock_until_ts" in e
        assert "effective_mode" in e


@pytest.mark.unit
def test_aggressive_tier_pnl_uses_post_step_delta() -> None:
    """If aggressive tier never fires (cold-start in the small fixture),
    aggressive_tier_pnl must be 0 — never a stale delta from earlier."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lake = _build_lake(start=start, days=21)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start, end=start + timedelta(days=21),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, lake)
    if result.dynamic_metrics.aggressive_tier_bars == 0:
        assert result.dynamic_metrics.aggressive_tier_pnl == 0.0
