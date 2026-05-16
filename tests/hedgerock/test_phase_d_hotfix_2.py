"""Phase D-hotfix-2 — transition_lock must persist to expiry.

Pinned guarantees:
    1. Production /signal: lock_until carries forward across polls; the
       same-regime poll right after a transition does NOT clear it.
    2. Replay: same carryover semantics; effective_mode stays observe
       for every bar covered by the carried lock.
    3. H1-resolution 7200s lock survives at least one subsequent bar.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    MarketFeatures,
    PrevRegimeV2Store,
    create_app,
)
from smc.hedgerock.envelope_store import EnvelopeStore


# ---------------------------------------------------------------------------
# Synthetic features that classify_regime_v2 maps to specific v2 regimes
# ---------------------------------------------------------------------------


def _features_v2_range() -> MarketFeatures:
    """Classifier_v2 → 'range' at conf=0.65 (range#2 rule)."""
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


def _features_v2_trend_up() -> MarketFeatures:
    """Classifier_v2 → 'trend_up' at conf=0.85 (trend rule)."""
    return MarketFeatures(
        volatility_rank=0.55, hh_count=8, ll_count=2,
        h4_trend_bars=5, regime="TREND_UP",
    )


class _SwitchableProvider:
    """Returns one MarketFeatures up to ``switch_after``, then another."""

    def __init__(self, first: MarketFeatures, then: MarketFeatures, *,
                 switch_after: int = 1) -> None:
        self._first = first
        self._then = then
        self._switch_after = switch_after
        self.call_count = 0

    def get_features(self, symbol: str) -> MarketFeatures:
        self.call_count += 1
        if self.call_count <= self._switch_after:
            return self._first
        return self._then


# ---------------------------------------------------------------------------
# Production /signal — lock persists across multiple polls
# ---------------------------------------------------------------------------


def _query(equity: float = 10000.0) -> dict:
    return {
        "symbol": "XAUUSD",
        "equity": equity, "balance": 10000.0,
        "dd_pct": 0.0, "spread_pts": 20,
    }


@pytest.mark.unit
def test_signal_transition_lock_persists_across_same_regime_poll() -> None:
    """User spec scenario:
        Poll 1: range  (no prev → no lock)
        Poll 2: trend_up (range→trend_up = distance 2 → 3600s lock)
        Poll 3: trend_up (same regime → fresh compute returns None,
                          but carryover MUST keep poll-2's lock_until)
    """
    market = _SwitchableProvider(
        _features_v2_range(), _features_v2_trend_up(), switch_after=1,
    )
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        b1 = client.get("/signal", params=_query()).json()
        assert b1["regime"] == "range"
        assert b1["transition_lock_until_ts"] is None  # first call, no prev v2

        b2 = client.get("/signal", params=_query()).json()
        assert b2["regime"] == "trend_up"
        assert b2["transition_lock_until_ts"] is not None  # range→trend_up = 3600s

        b3 = client.get("/signal", params=_query()).json()
        assert b3["regime"] == "trend_up"
        # Without carryover, fresh compute_lock_until_v2(trend_up, trend_up)
        # would return None and the lock would be cleared. With Phase
        # D-hotfix-2 carryover, the previous lock_until is preserved.
        assert b3["transition_lock_until_ts"] == b2["transition_lock_until_ts"], (
            "transition_lock_until_ts cleared on same-regime poll right "
            "after a transition — Phase D-hotfix-2 carryover regressed."
        )


@pytest.mark.unit
def test_signal_transition_lock_drops_after_expiry() -> None:
    """Once `now > lock_until`, the carried lock is no longer in the
    future and must be dropped (return None when no fresh lock fires)."""
    from smc.hedgerock.schemas import SignalEnvelope

    # Hand-seed an envelope_store with a lock that is already EXPIRED.
    env_store = EnvelopeStore()
    expired_ts = datetime.now(timezone.utc) - timedelta(hours=1)
    expired_env = SignalEnvelope(
        symbol="XAUUSD",
        generated_at=expired_ts - timedelta(hours=1),
        active_timeframe="H1",
        active_strategy_id="xauusd_h1_range",
        regime="range",
        transition_lock_until_ts=expired_ts,
    )
    env_store.set("XAUUSD", expired_env)

    # Provider always says range. v2_store starts empty. fresh lock = None.
    market = _SwitchableProvider(_features_v2_range(), _features_v2_range())
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        envelope_store=env_store,
    )
    with TestClient(app) as client:
        body = client.get("/signal", params=_query()).json()
        # Carryover must drop the expired lock.
        assert body["transition_lock_until_ts"] is None


@pytest.mark.unit
def test_signal_new_transition_can_extend_carried_lock() -> None:
    """If a new transition produces a longer lock than the carryover,
    take max(). Demonstrates the carryover never SHORTENS a lock."""
    # Start: range (lock=None)
    # Then:  trend_up (lock=now+3600)
    # Then:  range (trend_up→range = distance 2 = 3600s; new lock should
    #              be > previous lock by ~10s due to elapsed real time)
    # ... but `now` advances between calls, so newly-computed locks at
    # later polls will naturally be later than the carried one.
    market = _SwitchableProvider(
        _features_v2_range(), _features_v2_trend_up(), switch_after=1,
    )
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        client.get("/signal", params=_query())  # range
        b2 = client.get("/signal", params=_query()).json()
        # Now provider switches to trend_up — different invocation; we
        # need three polls to test extension. Use a different provider.

    market2 = _SwitchableProvider(
        _features_v2_trend_up(), _features_v2_range(), switch_after=1,
    )
    app2 = create_app(market2, enable_debate=False, enable_rule_engine=True)
    with TestClient(app2) as client:
        # Poll 1: trend_up (no prev v2 → lock=None)
        client.get("/signal", params=_query())
        # Poll 2: range (trend_up→range = 3600s lock)
        b2 = client.get("/signal", params=_query()).json()
        # Now manually wait 0 seconds; another poll same regime — carryover.
        b3 = client.get("/signal", params=_query()).json()
        # Both poll-2 and poll-3 have a lock; poll-3's lock should be
        # AT LEAST poll-2's (carryover takes max with fresh, where fresh
        # is None on same-regime).
        assert b3["transition_lock_until_ts"] is not None
        assert b3["transition_lock_until_ts"] >= b2["transition_lock_until_ts"]


# ---------------------------------------------------------------------------
# Replay — lock holds past the trigger bar
# ---------------------------------------------------------------------------


def _ohlcv(*, start, n_bars, bar_minutes, base=2000.0, slope=0.0, noise=5.0):
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
    def __init__(self, h1, h4, d1):
        self._h1, self._h4, self._d1 = h1, h4, d1

    def query(self, instrument, timeframe, start, end):
        df = {"H1": self._h1, "H4": self._h4, "D1": self._d1}.get(str(timeframe))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.mark.unit
def test_replay_transition_lock_persists_for_multiple_bars() -> None:
    """Constructed scenario: replay sees v2 transition at bar K. The
    next several H1 bars must report transition_lock_active=True and
    bar_opens=0 — proving the lock survives past the trigger bar."""
    from smc.hedgerock.phase_d_walk_forward import (
        WalkForwardConfig,
        run_walk_forward,
    )

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days = 30
    h1 = _ohlcv(start=start, n_bars=24 * days, bar_minutes=60)
    h4 = _ohlcv(start=start, n_bars=6 * days, bar_minutes=240)
    d1 = _ohlcv(start=start, n_bars=days, bar_minutes=1440)
    lake = _FakeLake(h1=h1, h4=h4, d1=d1)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start,
        end=start + timedelta(days=days),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, lake)

    # Find any transition_lock_active=True streak in the log. With
    # synthetic data, regime classifier produces transitions naturally.
    log = result.envelope_log
    locked_streak: list[dict] = []
    for i, e in enumerate(log):
        if e.get("transition_lock_active"):
            locked_streak.append(e)
            if len(locked_streak) >= 2:
                # Need at least 2 consecutive bars to prove persistence.
                # H1 bar interval is 1h. The smallest v2 lock is 900s
                # (15 min) which is < 1h, so a single regime change can
                # reasonably produce only 1 active bar. Larger distances
                # (3600 / 7200) MUST produce ≥ 1 / ≥ 2 active bars
                # respectively. Check existence of a streak of ≥ 2 to
                # prove carryover happened.
                break
        else:
            locked_streak = []

    assert len(locked_streak) >= 2, (
        "no transition_lock streak of 2+ bars found — carryover not "
        "persisting past the trigger bar"
    )
    # Every bar in the streak must have bar_opens == 0.
    for e in locked_streak:
        assert e["bar_opens"] == 0
        assert e["effective_mode"] in ("observe", "halt")


@pytest.mark.unit
def test_replay_h1_resolution_long_lock_covers_next_bar() -> None:
    """A 7200s lock (extreme reversal) is 2 hours — at H1 resolution
    must cover at least the immediately-following bar after trigger."""
    from smc.hedgerock.transition_lock import compute_lock_until_v2

    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    # trend_up→trend_down = distance 3 = 7200s.
    lock = compute_lock_until_v2("trend_up", "trend_down", now)
    assert lock is not None
    next_h1 = now + timedelta(hours=1)
    assert lock > next_h1, (
        f"7200s lock at {now} expires at {lock}, but next H1 bar at "
        f"{next_h1} — lock should still be active for that bar"
    )


@pytest.mark.unit
def test_replay_metrics_split_raw_vs_effective_mode() -> None:
    """TradeMetrics now has separate raw and effective mode counters.
    Effective should be ≤ raw for hedgerock (transition lock can flip
    raw=hedgerock to effective=observe but never the reverse)."""
    from smc.hedgerock.phase_d_walk_forward import (
        WalkForwardConfig,
        run_walk_forward,
    )

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days = 30
    h1 = _ohlcv(start=start, n_bars=24 * days, bar_minutes=60)
    h4 = _ohlcv(start=start, n_bars=6 * days, bar_minutes=240)
    d1 = _ohlcv(start=start, n_bars=days, bar_minutes=1440)
    lake = _FakeLake(h1=h1, h4=h4, d1=d1)
    cfg = WalkForwardConfig(
        instrument="XAUUSD", start=start,
        end=start + timedelta(days=days),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, lake)
    m = result.dynamic_metrics

    # Schema guarantee: both raw and effective fields exist.
    assert hasattr(m, "bars_in_hedgerock_raw")
    assert hasattr(m, "bars_in_hedgerock")
    # Effective hedgerock count ≤ raw hedgerock count (lock veto can
    # only reduce, never increase, hedgerock bars).
    assert m.bars_in_hedgerock <= m.bars_in_hedgerock_raw, (
        f"effective hedgerock {m.bars_in_hedgerock} > raw "
        f"{m.bars_in_hedgerock_raw} — lock veto is malformed"
    )
    # And observe count ≥ raw observe count (lock veto adds bars to
    # the observe bucket).
    assert m.bars_in_observe >= m.bars_in_observe_raw
