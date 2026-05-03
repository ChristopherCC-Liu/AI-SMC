"""Phase A-closeout — EA → Decision Center reverse data flow tests.

Covers:
- EAState dataclass + -1 sentinel mapping in build_ea_state
- /signal endpoint parses EA query params end-to-end
- /status exposes latest_ea_states
- consec_losses=-1 / recent_closed_pnl=-1 / recent_sample_count=-1 → None
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    EAState,
    EAStateStore,
    MarketFeatures,
    create_app,
)
from smc.hedgerock.ea_state import build_ea_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubFeaturesProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._features = features

    def get_features(self, symbol: str) -> MarketFeatures:
        return self._features


@pytest.fixture
def trend_features() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.5,
        hh_count=8,
        ll_count=0,
        h4_trend_bars=5,
        regime="TREND_UP",
    )


# ---------------------------------------------------------------------------
# build_ea_state — pure parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_ea_state_returns_none_when_no_state_sent() -> None:
    """Legacy EA / curl without query params → None (no state to record)."""
    assert build_ea_state() is None


@pytest.mark.unit
def test_build_ea_state_minimal_equity_or_balance_triggers_record() -> None:
    """Either equity or balance is enough to start recording."""
    s = build_ea_state(equity=10000.0)
    assert s is not None
    assert s.equity == pytest.approx(10000.0)
    assert s.balance is None


@pytest.mark.unit
def test_build_ea_state_only_spread_pts_records_partial_state() -> None:
    """Mini-patch: any single non-None field records a partial EAState.

    A spread-only poll lets the rule engine react to spread spikes
    even before equity is reported.
    """
    s = build_ea_state(spread_pts=25)
    assert s is not None
    assert s.spread_pts == 25
    assert s.equity is None
    assert s.balance is None


@pytest.mark.unit
def test_build_ea_state_only_open_positions_records_partial_state() -> None:
    """Mini-patch: open_positions alone is enough to start a record."""
    s = build_ea_state(open_positions=2)
    assert s is not None
    assert s.open_positions == 2
    assert s.equity is None


@pytest.mark.unit
def test_build_ea_state_recent_sample_count_minus_one_fans_unavailable() -> None:
    """Mini-patch: recent_sample_count=-1 is the canonical sentinel.

    When the EA reports HistorySelect failed, ALL three history-derived
    fields must be None — even if the EA also sent stale numeric values
    in consec_losses / recent_closed_pnl by accident.
    """
    s = build_ea_state(
        equity=10000.0,
        balance=10100.0,
        consec_losses=99,           # would-be stale value
        recent_closed_pnl=-999.0,   # would-be stale value
        recent_sample_count=-1,     # ← THE canonical signal
    )
    assert s is not None
    assert s.consec_losses is None
    assert s.recent_closed_pnl is None
    assert s.recent_sample_count is None


@pytest.mark.unit
def test_build_ea_state_recent_closed_pnl_minus_one_with_valid_sample_kept() -> None:
    """Mini-patch: recent_closed_pnl=-1.0 with sample_count=20 is a REAL loss.

    Without this fix the rule engine would silently lose a 1-USD loser
    to sentinel mapping and treat it as "no data".
    """
    s = build_ea_state(
        equity=10000.0,
        balance=10000.0,
        consec_losses=1,
        recent_closed_pnl=-1.0,    # legit small loss
        recent_sample_count=20,    # history IS available
    )
    assert s is not None
    assert s.recent_closed_pnl == pytest.approx(-1.0)
    assert s.recent_sample_count == 20
    assert s.consec_losses == 1


@pytest.mark.unit
def test_build_ea_state_consec_losses_minus_one_alone_kept() -> None:
    """consec_losses=-1 without recent_sample_count=-1 is unusual but
    not the unavailable sentinel — pass through unchanged.

    (The EA actually never emits this combination; this guards against
    future producers that mis-encode.)
    """
    s = build_ea_state(
        equity=10000.0,
        consec_losses=-1,
        recent_sample_count=20,
    )
    assert s is not None
    assert s.consec_losses == -1
    assert s.recent_sample_count == 20


@pytest.mark.unit
def test_build_ea_state_real_history_passes_through() -> None:
    s = build_ea_state(
        equity=9800.0,
        balance=10000.0,
        dd_pct=0.02,
        consec_losses=3,
        recent_closed_pnl=-120.50,
        recent_sample_count=20,
    )
    assert s is not None
    assert s.consec_losses == 3
    assert s.recent_closed_pnl == pytest.approx(-120.50)
    assert s.recent_sample_count == 20
    assert s.dd_pct == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# EAStateStore — thread-safe per-symbol map
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ea_state_store_set_get_roundtrip() -> None:
    store = EAStateStore()
    s = build_ea_state(equity=10000.0, balance=10000.0)
    assert s is not None
    store.set("XAUUSD", s)
    assert store.get("XAUUSD") == s
    assert store.get("EURUSD") is None


@pytest.mark.unit
def test_ea_state_store_records_recorded_at_default_now() -> None:
    """Phase B-closeout #1: store stamps recorded_at on set()."""
    from datetime import datetime, timezone, timedelta

    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    before = datetime.now(timezone.utc)
    store.set("XAUUSD", s)
    after = datetime.now(timezone.utc)

    rec = store.get_record("XAUUSD")
    assert rec is not None
    assert rec.state == s
    # recorded_at was stamped within the (before, after) window.
    assert before - timedelta(seconds=1) <= rec.recorded_at <= after + timedelta(seconds=1)


@pytest.mark.unit
def test_ea_state_store_explicit_recorded_at_preserved() -> None:
    from datetime import datetime, timezone

    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    fixed = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    store.set("XAUUSD", s, recorded_at=fixed)
    assert store.get_recorded_at("XAUUSD") == fixed


@pytest.mark.unit
def test_ea_state_store_rejects_naive_recorded_at() -> None:
    from datetime import datetime
    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    with pytest.raises(ValueError, match="tz-aware"):
        store.set("XAUUSD", s, recorded_at=datetime(2026, 4, 1, 12, 0))


@pytest.mark.unit
def test_ea_state_store_canonicalizes_symbol_case() -> None:
    """Phase B-closeout #2: write lowercase, read UPPERCASE → same record.

    Without canonicalization a Phase C wiring bug (e.g. ``sym.lower()``
    leak) would create a shadow entry the rule engine misses.
    """
    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    store.set("xauusd", s)  # lowercase write
    # Reads in any case must succeed.
    assert store.get("XAUUSD") == s
    assert store.get("xauusd") == s
    assert store.get("XaUuSd") == s
    assert store.get_record("XAUUSD") is not None
    assert store.get_recorded_at("xauusd") is not None


@pytest.mark.unit
def test_ea_state_store_uppercase_canonical_in_snapshot() -> None:
    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    store.set("xauusd", s)
    snap = store.snapshot()
    assert "XAUUSD" in snap
    assert "xauusd" not in snap


@pytest.mark.unit
def test_ea_state_store_snapshot_records_decoupled() -> None:
    from datetime import datetime, timezone
    store = EAStateStore()
    s = build_ea_state(equity=10000.0)
    store.set("XAUUSD", s, recorded_at=datetime(2026, 4, 1, tzinfo=timezone.utc))
    snap = store.snapshot_records()
    assert "XAUUSD" in snap
    assert snap["XAUUSD"].state == s
    snap.pop("XAUUSD")
    # Live store unaffected
    assert store.get_record("XAUUSD") is not None


@pytest.mark.unit
def test_ea_state_store_snapshot_decoupled_from_cache() -> None:
    store = EAStateStore()
    store.set("XAUUSD", build_ea_state(equity=10000.0))  # type: ignore[arg-type]
    snap = store.snapshot()
    assert "XAUUSD" in snap
    # Mutating snapshot must not affect future reads.
    snap.pop("XAUUSD")
    assert store.get("XAUUSD") is not None


# ---------------------------------------------------------------------------
# /signal — query param parsing end-to-end
# ---------------------------------------------------------------------------


def test_signal_parses_full_ea_state_into_store(
    trend_features: MarketFeatures,
) -> None:
    """The exact contract the user asked for — /signal?...consec_losses=3&recent_closed_pnl=-120
    must record the EA state and expose it via /status."""
    market = _StubFeaturesProvider(trend_features)
    ea_store = EAStateStore()
    app = create_app(market, ea_state_store=ea_store, enable_debate=False)
    with TestClient(app) as client:
        resp = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10234.0,
                "balance": 10500.0,
                "dd_pct": 0.0253,
                "free_margin": 9000.0,
                "margin_level": 850.0,
                "open_lots": 0.50,
                "open_positions": 2,
                "floating_pnl": -50.0,
                "spread_pts": 22,
                "consec_losses": 3,
                "recent_closed_pnl": -120.50,
                "recent_sample_count": 18,
            },
        )
        assert resp.status_code == 200

        # Direct store inspection
        recorded = ea_store.get("XAUUSD")
        assert recorded is not None
        assert recorded.equity == pytest.approx(10234.0)
        assert recorded.dd_pct == pytest.approx(0.0253)
        assert recorded.consec_losses == 3
        assert recorded.recent_closed_pnl == pytest.approx(-120.50)
        assert recorded.recent_sample_count == 18

        # /status exposure — Phase B-closeout #1: now wrapped in
        # {state, recorded_at, age_seconds}.
        status = client.get("/status").json()
        assert status["ea_state_store_attached"] is True
        assert "XAUUSD" in status["latest_ea_states"]
        wrapper = status["latest_ea_states"]["XAUUSD"]
        assert "recorded_at" in wrapper
        assert "age_seconds" in wrapper
        assert wrapper["age_seconds"] >= 0.0
        s = wrapper["state"]
        assert s["equity"] == pytest.approx(10234.0)
        assert s["consec_losses"] == 3
        assert s["recent_closed_pnl"] == pytest.approx(-120.50)


def test_signal_minus_one_sentinels_become_none(
    trend_features: MarketFeatures,
) -> None:
    """EA's -1 sentinel must NOT participate in risk calculations.

    Phase A-closeout requirement: -1 == "unavailable", not "0 losses".
    """
    market = _StubFeaturesProvider(trend_features)
    ea_store = EAStateStore()
    app = create_app(market, ea_state_store=ea_store, enable_debate=False)
    with TestClient(app) as client:
        client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 9000.0,
                "balance": 10000.0,
                "consec_losses": -1,
                "recent_closed_pnl": -1,
                "recent_sample_count": -1,
            },
        )
        recorded = ea_store.get("XAUUSD")
        assert recorded is not None
        # All three must be None (NOT -1, NOT 0).
        assert recorded.consec_losses is None
        assert recorded.recent_closed_pnl is None
        assert recorded.recent_sample_count is None
        # Direct read fields stay as-sent.
        assert recorded.equity == pytest.approx(9000.0)

        # /status JSON form must also serialize as null, not -1.
        status = client.get("/status").json()
        s = status["latest_ea_states"]["XAUUSD"]["state"]
        assert s["consec_losses"] is None
        assert s["recent_closed_pnl"] is None
        assert s["recent_sample_count"] is None


def test_signal_legacy_call_without_state_does_not_clobber_previous_record(
    trend_features: MarketFeatures,
) -> None:
    """A poll that omits all EA-state fields must NOT overwrite the
    previous good state — otherwise a single legacy / health-check call
    would erase the record the rule engine relies on."""
    market = _StubFeaturesProvider(trend_features)
    ea_store = EAStateStore()
    app = create_app(market, ea_state_store=ea_store, enable_debate=False)
    with TestClient(app) as client:
        # First call: full state.
        client.get(
            "/signal",
            params={"symbol": "XAUUSD", "equity": 10000.0, "balance": 10000.0},
        )
        first = ea_store.get("XAUUSD")
        assert first is not None
        # Second call: bare /signal — no state params at all.
        client.get("/signal", params={"symbol": "XAUUSD"})
        # Store still holds the first poll's state.
        assert ea_store.get("XAUUSD") == first


def test_signal_partial_state_still_recorded(
    trend_features: MarketFeatures,
) -> None:
    """If only equity arrives (e.g., very-early-EA poll before history is
    ready), we still record it — partial state > no state."""
    market = _StubFeaturesProvider(trend_features)
    ea_store = EAStateStore()
    app = create_app(market, ea_state_store=ea_store, enable_debate=False)
    with TestClient(app) as client:
        client.get(
            "/signal",
            params={"symbol": "XAUUSD", "equity": 10000.0, "spread_pts": 25},
        )
        recorded = ea_store.get("XAUUSD")
        assert recorded is not None
        assert recorded.equity == pytest.approx(10000.0)
        assert recorded.balance is None
        assert recorded.spread_pts == 25
        assert recorded.consec_losses is None


def test_status_ea_state_store_default_attached_when_omitted(
    trend_features: MarketFeatures,
) -> None:
    """create_app without ea_state_store kwarg still creates a fresh store."""
    market = _StubFeaturesProvider(trend_features)
    app = create_app(market, enable_debate=False)  # no ea_state_store kwarg
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["ea_state_store_attached"] is True
        assert body["latest_ea_states"] == {}
