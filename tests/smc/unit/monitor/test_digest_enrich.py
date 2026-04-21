"""Unit tests for R5 M1 digest_enrich module (per-leg / regime / AI / handles)."""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from smc.monitor.digest_enrich import (
    build_ai_debate_stats,
    build_leg_breakdown,
    build_regime_distribution,
    count_handle_resets,
    percentile,
)


TARGET = date(2026, 4, 20)


# ---------------------------------------------------------------------------
# percentile helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty(self) -> None:
        assert percentile([], 50) == 0.0

    def test_single(self) -> None:
        assert percentile([42.0], 90) == 42.0

    def test_p50_odd_count(self) -> None:
        assert percentile([10.0, 20.0, 30.0], 50) == pytest.approx(20.0)

    def test_p90_interpolated(self) -> None:
        # 10 items, p90 -> k = 0.9 * 9 = 8.1 -> lo=8, hi=9
        vals = list(map(float, range(1, 11)))  # 1..10
        # srt[8]=9, srt[9]=10; interp -> 9 + 0.1*(10-9) = 9.1
        assert percentile(vals, 90) == pytest.approx(9.1)


# ---------------------------------------------------------------------------
# build_leg_breakdown — journal-driven
# ---------------------------------------------------------------------------


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _mk_row(ts: str, result: float, *, magic: int = 19760418, ticket: int | None = None) -> dict:
    row = {
        "time": ts,
        "action": "BUY",
        "mode": "LIVE_EXEC",
        "direction": "long",
        "magic": magic,
        "result": result,
    }
    if ticket is not None:
        row["ticket"] = ticket
    return row


class TestLegBreakdown:
    def test_empty_journals(self, tmp_path: Path) -> None:
        paths = {"XAUUSD:control": tmp_path / "missing.jsonl"}
        out = build_leg_breakdown(paths, TARGET)
        assert out == []

    def test_control_day1_matches_snapshot(self, tmp_path: Path) -> None:
        """Day-1 synthetic matching team-lead's Control leg: 2 SL + 2 TP = +$31.84."""
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        _write_journal(jpath, [
            _mk_row("2026-04-20T02:15:00+00:00", 68.92),
            _mk_row("2026-04-20T05:42:00+00:00", -53.00),
            _mk_row("2026-04-20T11:07:00+00:00", 68.92),
            _mk_row("2026-04-20T16:23:00+00:00", -53.00),
        ])
        out = build_leg_breakdown({"XAUUSD:control": jpath}, TARGET)
        assert len(out) == 1
        row = out[0]
        assert row["leg"] == "XAUUSD:control"
        assert row["trades"] == 4
        assert row["wins"] == 2
        assert row["losses"] == 2
        assert row["win_rate_pct"] == pytest.approx(50.0)
        assert row["total_pnl_usd"] == pytest.approx(31.84)
        assert row["profit_factor"] == pytest.approx(1.3)
        assert row["avg_win_usd"] == pytest.approx(68.92)
        assert row["avg_loss_usd"] == pytest.approx(-53.00)

    def test_cross_day_rows_excluded(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        _write_journal(jpath, [
            _mk_row("2026-04-19T23:59:00+00:00", 100.0),  # prev day
            _mk_row("2026-04-20T01:00:00+00:00", 50.0),   # target
            _mk_row("2026-04-21T00:00:00+00:00", 200.0),  # next day
        ])
        out = build_leg_breakdown({"X:c": jpath}, TARGET)
        assert out[0]["trades"] == 1
        assert out[0]["total_pnl_usd"] == pytest.approx(50.0)

    def test_hold_and_gated_excluded(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        _write_journal(jpath, [
            {"time": "2026-04-20T02:00:00+00:00", "action": "HOLD", "mode": "LIVE_EXEC"},
            {"time": "2026-04-20T03:00:00+00:00", "action": "BUY", "mode": "MARGIN_GATED"},
            _mk_row("2026-04-20T04:00:00+00:00", 25.0),
        ])
        out = build_leg_breakdown({"X:c": jpath}, TARGET)
        assert out[0]["trades"] == 1

    def test_closures_override_journal_result(self, tmp_path: Path) -> None:
        """structured.jsonl pnl (ticket-matched) takes priority over journal result."""
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        _write_journal(jpath, [
            _mk_row("2026-04-20T02:00:00+00:00", 10.0, ticket=999),
        ])
        # Override: closures reports this ticket as actually -5
        out = build_leg_breakdown(
            {"X:c": jpath}, TARGET, closures_by_ticket={999: -5.0},
        )
        assert out[0]["total_pnl_usd"] == pytest.approx(-5.0)
        assert out[0]["losses"] == 1

    def test_max_drawdown(self, tmp_path: Path) -> None:
        """Peak $100 then −$80 -> max DD $80."""
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        _write_journal(jpath, [
            _mk_row("2026-04-20T01:00:00+00:00", 100.0),
            _mk_row("2026-04-20T02:00:00+00:00", -30.0),
            _mk_row("2026-04-20T03:00:00+00:00", -50.0),
            _mk_row("2026-04-20T04:00:00+00:00", 20.0),
        ])
        out = build_leg_breakdown({"X:c": jpath}, TARGET)
        assert out[0]["max_drawdown_usd"] == pytest.approx(80.0)

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        jpath = tmp_path / "journal" / "live_trades.jsonl"
        jpath.parent.mkdir(parents=True)
        jpath.write_text(
            "{not json\n"
            + json.dumps(_mk_row("2026-04-20T02:00:00+00:00", 10.0))
            + "\n",
            encoding="utf-8",
        )
        out = build_leg_breakdown({"X:c": jpath}, TARGET)
        assert out[0]["trades"] == 1


# ---------------------------------------------------------------------------
# build_regime_distribution
# ---------------------------------------------------------------------------


class TestRegimeDistribution:
    def test_empty(self) -> None:
        buckets = build_regime_distribution([])
        assert set(buckets.keys()) == {
            "TRANSITION", "TREND_UP", "TREND_DOWN", "CONSOLIDATION",
            "ATH_BREAKOUT", "UNKNOWN",
        }
        assert all(v == 0 for v in buckets.values())

    def test_counts_by_regime(self) -> None:
        events = [
            {"event": "ai_regime_classified", "regime": "TRANSITION"},
            {"event": "ai_regime_classified", "regime": "TRANSITION"},
            {"event": "ai_regime_classified", "regime": "CONSOLIDATION"},
            {"event": "ai_regime_classified", "regime": "TREND_UP"},
            {"event": "other_event", "regime": "TREND_UP"},  # excluded
        ]
        buckets = build_regime_distribution(events)
        assert buckets["TRANSITION"] == 2
        assert buckets["CONSOLIDATION"] == 1
        assert buckets["TREND_UP"] == 1
        assert buckets["TREND_DOWN"] == 0
        assert buckets["UNKNOWN"] == 0

    def test_unknown_regime_counted_as_unknown(self) -> None:
        events = [{"event": "ai_regime_classified", "regime": "WEIRDVALUE"}]
        buckets = build_regime_distribution(events)
        assert buckets["UNKNOWN"] == 1


# ---------------------------------------------------------------------------
# build_ai_debate_stats
# ---------------------------------------------------------------------------


class TestAiDebateStats:
    def test_empty(self) -> None:
        stats = build_ai_debate_stats([])
        assert stats["cycles_ran"] == 0
        assert stats["p50_elapsed_ms"] is None
        assert stats["p90_elapsed_ms"] is None
        assert stats["total_cost_usd"] == 0.0

    def test_latency_and_cost(self) -> None:
        events = [
            {"event": "ai_debate_completed", "elapsed_ms": 100.0, "total_cost_usd": 0.01},
            {"event": "ai_debate_completed", "elapsed_ms": 200.0, "total_cost_usd": 0.02},
            {"event": "ai_debate_completed", "elapsed_ms": 300.0, "total_cost_usd": 0.03},
        ]
        stats = build_ai_debate_stats(events)
        assert stats["cycles_ran"] == 3
        assert stats["p50_elapsed_ms"] == pytest.approx(200.0)
        assert stats["total_cost_usd"] == pytest.approx(0.06)

    def test_tolerates_schema_drift(self) -> None:
        """Schema may use latency_ms or duration_ms instead of elapsed_ms, and cost_usd instead of total_cost_usd."""
        events = [
            {"event": "ai_debate_completed", "latency_ms": 50.0, "cost_usd": 0.01},
            {"event": "ai_debate_completed", "duration_ms": 75.0, "cost_usd": 0.02},
        ]
        stats = build_ai_debate_stats(events)
        assert stats["cycles_ran"] == 2
        assert stats["p50_elapsed_ms"] == pytest.approx(62.5)
        assert stats["total_cost_usd"] == pytest.approx(0.03)

    def test_prod_schema_ai_regime_debate_result(self) -> None:
        """VPS prod emits event='ai_regime_debate_result' with elapsed_s (seconds)."""
        events = [
            {"event": "ai_regime_debate_result", "elapsed_s": 144.9, "confidence": 0.72},
            {"event": "ai_regime_debate_result", "elapsed_s": 105.1, "confidence": 0.62},
        ]
        stats = build_ai_debate_stats(events)
        assert stats["cycles_ran"] == 2
        # elapsed_s -> ms: p50 of [144900, 105100] = 125000
        assert stats["p50_elapsed_ms"] == pytest.approx(125000.0)
        assert stats["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# count_handle_resets
# ---------------------------------------------------------------------------


class TestHandleResets:
    def test_empty(self) -> None:
        assert count_handle_resets([]) == 0

    def test_counts_resets(self) -> None:
        events = [
            {"event": "mt5_handle_reset", "reason": "timeout"},
            {"event": "mt5_handle_reset", "reason": "terminal_crash"},
            {"event": "other"},
        ]
        assert count_handle_resets(events) == 2
