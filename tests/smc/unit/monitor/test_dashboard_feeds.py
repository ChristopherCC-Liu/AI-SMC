"""Unit tests for smc.monitor.dashboard_feeds (Round 5 O1/O2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from smc.monitor.dashboard_feeds import (
    PNL_LEG_BY_COMPOSITE,
    PNL_LEG_KEYS,
    PNL_LEG_LABELS,
    build_pnl_snapshot,
    reset_pnl_cache,
    tail_regime_events,
)


# ---------------------------------------------------------------------------
# O1: regime tail
# ---------------------------------------------------------------------------


def _write_structured(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for severity, row in rows:
            f.write(f"[{severity}] {json.dumps(row)}\n")


def test_tail_regime_skips_default_source(tmp_path: Path):
    log = tmp_path / "structured.jsonl"
    _write_structured(log, [
        ("INFO", {"ts": "2026-04-20T01:00:00+00:00", "event": "ai_regime_classified",
                  "source": "default", "regime": "TRANSITION", "confidence": 0.3}),
        ("INFO", {"ts": "2026-04-20T02:00:00+00:00", "event": "ai_regime_classified",
                  "source": "atr_fallback", "regime": "CONSOLIDATION", "confidence": 0.7,
                  "reasoning": "ATR tight"}),
        ("INFO", {"ts": "2026-04-20T03:00:00+00:00", "event": "ai_regime_classified",
                  "source": "claude_debate", "regime": "TRENDING", "confidence": 0.85,
                  "direction": "bullish", "reasoning": "SMA50 up + MFE > 0.5R"}),
    ])
    events = tail_regime_events(log, limit=5)
    assert len(events) == 2
    # Newest first.
    assert events[0]["regime"] == "TRENDING"
    assert events[0]["direction"] == "bullish"
    assert events[1]["regime"] == "CONSOLIDATION"
    assert all(e["source"] != "default" for e in events)


def test_tail_regime_truncates_long_reasoning(tmp_path: Path):
    log = tmp_path / "structured.jsonl"
    _write_structured(log, [
        ("INFO", {"ts": "2026-04-20T02:00:00+00:00", "event": "ai_regime_classified",
                  "source": "claude_debate", "regime": "TRENDING", "confidence": 0.9,
                  "reasoning": "x" * 500}),
    ])
    events = tail_regime_events(log, limit=1, reasoning_max_chars=300)
    assert events[0]["reasoning"] is not None
    assert len(events[0]["reasoning"]) == 300


def test_tail_regime_respects_limit(tmp_path: Path):
    log = tmp_path / "structured.jsonl"
    rows = [
        ("INFO",
         {"ts": f"2026-04-20T{i:02d}:00:00+00:00", "event": "ai_regime_classified",
          "source": "claude_debate", "regime": "TRENDING", "confidence": 0.8})
        for i in range(10)
    ]
    _write_structured(log, rows)
    events = tail_regime_events(log, limit=3)
    assert len(events) == 3
    assert events[0]["ts"].startswith("2026-04-20T09")


def test_tail_regime_handles_missing_file(tmp_path: Path):
    assert tail_regime_events(tmp_path / "nonexistent.jsonl") == []


def test_tail_regime_ignores_unrelated_events(tmp_path: Path):
    log = tmp_path / "structured.jsonl"
    _write_structured(log, [
        ("INFO", {"ts": "2026-04-20T02:00:00+00:00", "event": "trade_reconciled", "ticket": 1}),
        ("CRIT", {"ts": "2026-04-20T02:05:00+00:00", "event": "trade_closed", "ticket": 1}),
    ])
    assert tail_regime_events(log) == []


# ---------------------------------------------------------------------------
# O2: P&L snapshot — composite (symbol, magic) grouping + positions_get()
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    """Drop the /api/pnl memo around every test — cache state bleeds across."""
    reset_pnl_cache()
    yield
    reset_pnl_cache()


@dataclass
class FakeDeal:
    entry: int
    magic: int
    symbol: str
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0


@dataclass
class FakePosition:
    symbol: str
    magic: int
    profit: float = 0.0


class FakeMT5:
    """Minimal fake for history_deals_get + positions_get + DEAL_ENTRY_* constants."""
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2

    def __init__(
        self,
        deals: list[FakeDeal] | None = None,
        positions: list[FakePosition] | None = None,
    ) -> None:
        self._deals = deals or []
        self._positions = positions or []

    def history_deals_get(self, from_time, to_time):  # noqa: ARG002
        return self._deals

    def positions_get(self):
        return self._positions


# ---------- Realized aggregation ----------


def test_realized_groups_by_composite_key(tmp_path: Path):
    mt5 = FakeMT5(deals=[
        FakeDeal(entry=1, symbol="XAUUSD", magic=19760418, profit=10.0, commission=-0.5),
        FakeDeal(entry=1, symbol="XAUUSD", magic=19760418, profit=-3.0),
        FakeDeal(entry=1, symbol="BTCUSD", magic=19760419, profit=100.0),
        FakeDeal(entry=1, symbol="XAUUSD", magic=19760428, profit=5.0),
        # Wrong entry type — ignored.
        FakeDeal(entry=0, symbol="XAUUSD", magic=19760418, profit=999.0),
        # Unknown magic — ignored.
        FakeDeal(entry=1, symbol="XAUUSD", magic=99999, profit=999.0),
    ])
    snap = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    legs = snap["legs"]
    assert legs["control_xau"]["realized"] == pytest.approx(6.5)
    assert legs["control_xau"]["trades"] == 2
    assert legs["treatment_xau"]["realized"] == pytest.approx(5.0)
    assert legs["control_btc"]["realized"] == pytest.approx(100.0)
    assert snap["total"]["realized"] == pytest.approx(111.5)
    assert snap["source"] == "mt5_live"


def test_btc_tagged_with_xau_treatment_magic_is_dropped(tmp_path: Path):
    """Anomaly: BTC deal carrying the XAU treatment magic (19760428).

    Must NOT be merged into treatment_xau — there's no btc-treatment leg.
    """
    mt5 = FakeMT5(deals=[
        FakeDeal(entry=1, symbol="BTCUSD", magic=19760428, profit=999.0),
    ])
    snap = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    assert snap["legs"]["treatment_xau"]["realized"] == 0.0
    assert snap["legs"]["treatment_xau"]["trades"] == 0


def test_broker_suffixed_symbol_normalized(tmp_path: Path):
    """TMGM ships bare XAUUSD; other brokers may suffix ("XAUUSD.r")."""
    mt5 = FakeMT5(deals=[
        FakeDeal(entry=1, symbol="XAUUSD.r", magic=19760418, profit=7.0),
        FakeDeal(entry=1, symbol="BTCUSD.x", magic=19760419, profit=11.0),
    ])
    snap = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    assert snap["legs"]["control_xau"]["realized"] == 7.0
    assert snap["legs"]["control_btc"]["realized"] == 11.0


# ---------- Floating aggregation ----------


def test_floating_reads_from_positions_get(tmp_path: Path):
    mt5 = FakeMT5(
        deals=[],
        positions=[
            FakePosition(symbol="XAUUSD", magic=19760418, profit=2.5),
            FakePosition(symbol="XAUUSD", magic=19760428, profit=-1.75),
            FakePosition(symbol="BTCUSD", magic=19760419, profit=30.0),
            # Unknown combo — dropped.
            FakePosition(symbol="BTCUSD", magic=19760428, profit=999.0),
        ],
    )
    snap = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    legs = snap["legs"]
    assert legs["control_xau"]["floating"] == 2.5
    assert legs["treatment_xau"]["floating"] == -1.75
    assert legs["control_btc"]["floating"] == 30.0
    assert snap["total"]["floating"] == pytest.approx(30.75)


def test_floating_graceful_when_positions_get_raises(tmp_path: Path):
    class BoomMT5:
        DEAL_ENTRY_OUT = 1

        def history_deals_get(self, *_a, **_kw):
            return []

        def positions_get(self):
            raise RuntimeError("connection lost")

    snap = build_pnl_snapshot(BoomMT5(), data_root=tmp_path, use_cache=False)
    # Source still reads mt5_live because the mt5 object is not None.
    assert snap["legs"]["control_xau"]["floating"] == 0.0


# ---------- Graceful degradation ----------


def test_build_pnl_none_mt5_returns_unavailable(tmp_path: Path):
    snap = build_pnl_snapshot(None, data_root=tmp_path, use_cache=False)
    assert snap["source"] == "mt5_unavailable"
    for key in PNL_LEG_KEYS:
        assert snap["legs"][key]["realized"] == 0.0
        assert snap["legs"][key]["floating"] == 0.0
        assert snap["legs"][key]["trades"] == 0
    assert snap["total"]["realized"] == 0.0


def test_build_pnl_history_deals_get_raises_still_returns_shape(tmp_path: Path):
    class BoomMT5:
        DEAL_ENTRY_OUT = 1

        def history_deals_get(self, *a, **kw):
            raise RuntimeError("disconnected")

        def positions_get(self):
            return []

    snap = build_pnl_snapshot(BoomMT5(), data_root=tmp_path, use_cache=False)
    assert snap["legs"]["control_xau"]["realized"] == 0.0
    # All legs still rendered.
    for key in PNL_LEG_KEYS:
        assert key in snap["legs"]


def test_build_pnl_response_shape_is_stable(tmp_path: Path):
    snap = build_pnl_snapshot(None, data_root=tmp_path, use_cache=False)
    for k in ("as_of", "today_utc_start", "legs", "total", "source", "cached"):
        assert k in snap, f"missing top-level key {k}"
    for leg in PNL_LEG_KEYS:
        assert leg in snap["legs"], f"missing leg {leg}"
        row = snap["legs"][leg]
        for field in ("magic", "symbol", "label", "key", "realized", "floating", "trades"):
            assert field in row, f"leg {leg} missing {field}"


def test_build_pnl_leg_keys_exclude_btc_treatment(tmp_path: Path):
    """Team-lead 2026-04-20 correction: no BTC-treatment leg exists."""
    snap = build_pnl_snapshot(None, data_root=tmp_path, use_cache=False)
    assert "treatment_btc" not in snap["legs"]
    assert set(snap["legs"].keys()) == {"control_xau", "treatment_xau", "control_btc"}


# ---------- Composite-key constants ----------


def test_composite_map_matches_spec():
    assert PNL_LEG_BY_COMPOSITE[("XAUUSD", 19760418)] == "control_xau"
    assert PNL_LEG_BY_COMPOSITE[("XAUUSD", 19760428)] == "treatment_xau"
    assert PNL_LEG_BY_COMPOSITE[("BTCUSD", 19760419)] == "control_btc"
    assert ("BTCUSD", 19760428) not in PNL_LEG_BY_COMPOSITE
    assert len(PNL_LEG_BY_COMPOSITE) == 3


def test_leg_labels_present_for_every_key():
    for k in PNL_LEG_KEYS:
        assert k in PNL_LEG_LABELS


# ---------- 5-second cache behaviour ----------


def test_cache_serves_repeat_calls_within_ttl(tmp_path: Path):
    call_count = {"deals": 0}

    class CountingMT5:
        DEAL_ENTRY_OUT = 1

        def history_deals_get(self, *_a, **_kw):
            call_count["deals"] += 1
            return [FakeDeal(entry=1, symbol="XAUUSD", magic=19760418, profit=1.0)]

        def positions_get(self):
            return []

    mt5 = CountingMT5()
    reset_pnl_cache()
    first = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=True)
    second = build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=True)

    assert call_count["deals"] == 1  # second call served from cache
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["legs"]["control_xau"]["realized"] == first["legs"]["control_xau"]["realized"]


def test_cache_bypassed_when_use_cache_false(tmp_path: Path):
    call_count = {"deals": 0}

    class CountingMT5:
        DEAL_ENTRY_OUT = 1

        def history_deals_get(self, *_a, **_kw):
            call_count["deals"] += 1
            return []

        def positions_get(self):
            return []

    mt5 = CountingMT5()
    build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=False)
    assert call_count["deals"] == 2


def test_cache_key_not_shared_across_cache_reset(tmp_path: Path):
    call_count = {"deals": 0}

    class CountingMT5:
        DEAL_ENTRY_OUT = 1

        def history_deals_get(self, *_a, **_kw):
            call_count["deals"] += 1
            return []

        def positions_get(self):
            return []

    mt5 = CountingMT5()
    build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=True)
    reset_pnl_cache()
    build_pnl_snapshot(mt5, data_root=tmp_path, use_cache=True)
    assert call_count["deals"] == 2
