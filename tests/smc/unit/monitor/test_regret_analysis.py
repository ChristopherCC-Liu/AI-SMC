"""Unit tests for R5 M3 regret_analysis module."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from smc.monitor.regret_analysis import (
    MAGIC_CONTROL,
    MAGIC_TREATMENT,
    build_regret_records,
    compute_regret_for_ticket,
    compute_regret_row,
    compute_summary,
    count_anti_stack_blocks,
    extract_closures_by_ticket,
    load_day_trades,
    synthesize_unassigned_closures,
)


TARGET = date(2026, 4, 20)


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _trade(ts: str, result: float, *, magic: int = MAGIC_CONTROL, ticket: int | None = None) -> dict:
    row = {
        "time": ts, "action": "BUY", "mode": "LIVE_EXEC",
        "direction": "long", "magic": magic, "result": result,
    }
    if ticket is not None:
        row["ticket"] = ticket
    return row


# ---------------------------------------------------------------------------
# load_day_trades
# ---------------------------------------------------------------------------


class TestLoadDayTrades:
    def test_empty_paths(self) -> None:
        assert load_day_trades([], TARGET) == []

    def test_missing_file(self, tmp_path: Path) -> None:
        assert load_day_trades([tmp_path / "nope.jsonl"], TARGET) == []

    def test_filters_date_and_holds(self, tmp_path: Path) -> None:
        p = tmp_path / "j.jsonl"
        _write_journal(p, [
            _trade("2026-04-19T12:00:00+00:00", 10.0),  # prev day
            _trade("2026-04-20T12:00:00+00:00", 20.0),  # target
            {"time": "2026-04-20T13:00:00+00:00", "action": "HOLD", "mode": "LIVE_EXEC"},
        ])
        out = load_day_trades([p], TARGET)
        assert len(out) == 1
        assert out[0]["result"] == 20.0
        assert out[0]["_journal"] == str(p)


# ---------------------------------------------------------------------------
# count_anti_stack_blocks
# ---------------------------------------------------------------------------


class TestAntiStackBlocks:
    def test_counts_cooldown_and_anti_stack(self) -> None:
        events = [
            {"event": "pre_write_gate_blocked", "blocked_reason": "cooldown:120s"},
            {"event": "pre_write_gate_blocked", "blocked_reason": "anti_stack:max_open"},
            {"event": "pre_write_gate_blocked", "blocked_reason": "margin_cap:exceeded"},
            {"event": "other"},
        ]
        assert count_anti_stack_blocks(events) == 2


class TestExtractClosuresByTicket:
    def test_maps_trade_reconciled_and_closed(self) -> None:
        events = [
            {"event": "trade_reconciled", "ticket": 111, "pnl_usd": 66.31},
            {"event": "trade_closed",     "ticket": 222, "pnl_usd": -25.0},
            {"event": "other",            "ticket": 333, "pnl_usd": 9.0},
        ]
        out = extract_closures_by_ticket(events)
        assert out == {111: 66.31, 222: -25.0}

    def test_dedupes_by_ticket_first_wins(self) -> None:
        events = [
            {"event": "trade_reconciled", "ticket": 111, "pnl_usd": 66.31},
            {"event": "trade_closed",     "ticket": 111, "pnl_usd": 66.31},  # dup
        ]
        assert extract_closures_by_ticket(events) == {111: 66.31}

    def test_ignores_missing_ticket_or_pnl(self) -> None:
        events = [
            {"event": "trade_reconciled", "pnl_usd": 10.0},  # no ticket
            {"event": "trade_reconciled", "ticket": 111},    # no pnl
            {"event": "trade_reconciled", "ticket": "bad", "pnl_usd": 5.0},  # bad int
        ]
        assert extract_closures_by_ticket(events) == {}

    def test_compute_regret_uses_closure_over_journal_result(self) -> None:
        """When ticket is in closures map, use that PnL instead of journal result."""
        trade = _trade("2026-04-20T02:00:00+00:00", 10.0, ticket=262715883)
        closures = {262715883: 66.31}
        row = compute_regret_row(trade, closures_by_ticket=closures)
        assert row["actual_pnl"] == pytest.approx(66.31)
        assert row["confidence"] == "heuristic"

    def test_compute_regret_ticket_field_alternative(self) -> None:
        """Journal rows may use mt5_ticket instead of ticket."""
        trade = {
            "time": "2026-04-20T02:00:00+00:00",
            "action": "BUY", "mode": "LIVE_EXEC",
            "magic": MAGIC_CONTROL, "mt5_ticket": 555,
        }
        closures = {555: 42.0}
        row = compute_regret_row(trade, closures_by_ticket=closures)
        assert row["actual_pnl"] == pytest.approx(42.0)


class TestSynthesizeUnassignedClosures:
    def test_no_closures_returns_empty(self) -> None:
        trades = [_trade("2026-04-20T02:00:00+00:00", 10.0)]
        out = synthesize_unassigned_closures(trades, {}, target_date=TARGET)
        assert out == []

    def test_all_closures_matched_returns_empty(self) -> None:
        """When every closure ticket has a matching journal row, no
        synthetic rows are needed."""
        trades = [_trade("2026-04-20T02:00:00+00:00", 10.0, ticket=111)]
        closures = {111: 15.0}
        out = synthesize_unassigned_closures(trades, closures, target_date=TARGET)
        assert out == []

    def test_unmatched_closures_synthesised_as_unassigned(self) -> None:
        """Closures without journal counterpart should carry the
        <unassigned> journal marker so compute_regret_row tags them
        as leg='unassigned'."""
        trades: list[dict] = []
        closures = {
            262518001: -52.72,
            262784309: 48.08,
            262673858: 74.83,
        }
        synth = synthesize_unassigned_closures(trades, closures, target_date=TARGET)
        assert len(synth) == 3
        for row in synth:
            assert row["_journal"] == "<unassigned>"
            assert row["mode"] == "LIVE_EXEC"
            assert row["result"] in {-52.72, 48.08, 74.83}

        for trade_row in synth:
            r = compute_regret_row(trade_row)
            assert r["leg"] == "unassigned"
            assert r["confidence"] == "heuristic"

    def test_end_to_end_day1_vps_style(self) -> None:
        """Simulate the real VPS scenario: empty journal + 7 broker
        closures.  build_regret_records must show trade_count=7 and an
        honest actual_total, not 0."""
        closures = {
            262518001: -52.72, 262579661: -38.35, 262626659: -30.11,
            262518005: -52.72, 262579669: -38.35, 262784309: 48.08,
            262673858: 74.83,
        }
        trades = synthesize_unassigned_closures([], closures, target_date=TARGET)
        records = build_regret_records(
            trades, closures_by_ticket=closures, anti_stack_blocks=0,
        )
        assert len(records) == 8  # 7 trades + 1 summary
        summary = records[-1]
        assert summary["trade_count"] == 7
        assert summary["actual_total"] == pytest.approx(-89.34)


# ---------------------------------------------------------------------------
# compute_regret_row
# ---------------------------------------------------------------------------


class TestComputeRegretRow:
    def test_missing_result_yields_unknown(self) -> None:
        row = compute_regret_row({"time": "2026-04-20T00:00:00+00:00", "magic": MAGIC_CONTROL})
        assert row["confidence"] == "unknown"
        assert row["actual_pnl"] == 0.0

    def test_control_winner(self) -> None:
        trade = _trade("2026-04-20T02:00:00+00:00", 68.92)
        row = compute_regret_row(trade)
        assert row["leg"] == "control"
        assert row["actual_pnl"] == pytest.approx(68.92)
        assert row["no_macro_pnl"] == pytest.approx(68.92)
        # Winners shrink 15% without reversal_confirm
        assert row["no_reversal_confirm_pnl"] == pytest.approx(68.92 * 0.85, rel=1e-2)
        assert row["confidence"] == "heuristic"

    def test_treatment_loser(self) -> None:
        trade = _trade("2026-04-20T03:00:00+00:00", -35.33, magic=MAGIC_TREATMENT)
        row = compute_regret_row(trade)
        assert row["leg"] == "treatment"
        # Losers grow 10% without reversal_confirm
        assert row["no_reversal_confirm_pnl"] == pytest.approx(-35.33 * 1.10, rel=1e-2)
        assert "treatment leg" in row["notes"]

    def test_trade_id_uses_ticket_when_present(self) -> None:
        row = compute_regret_row(_trade("2026-04-20T02:00:00+00:00", 10.0, ticket=9999))
        assert row["trade_id"] == "ticket:9999"

    def test_zero_pnl_is_neutral(self) -> None:
        row = compute_regret_row(_trade("2026-04-20T02:00:00+00:00", 0.0))
        assert row["no_reversal_confirm_pnl"] == 0.0


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_empty(self) -> None:
        s = compute_summary([])
        assert s["trade_count"] == 0
        assert s["actual_total"] == 0.0
        assert s["no_macro_verdict"] == "neutral"

    def test_verdicts_and_deltas(self) -> None:
        per_trade = [
            compute_regret_row(_trade("2026-04-20T02:00:00+00:00", 100.0)),
            compute_regret_row(_trade("2026-04-20T03:00:00+00:00", -50.0)),
        ]
        s = compute_summary(per_trade)
        assert s["actual_total"] == pytest.approx(50.0)
        # no_reversal: 85 + (-55) = 30 → delta -20, guardrail_helped
        assert s["no_reversal_confirm_delta"] == pytest.approx(-20.0)
        assert s["no_reversal_confirm_verdict"] == "guardrail_helped"

    def test_anti_stack_delta_positive_when_blocks_winning(self) -> None:
        per_trade = [compute_regret_row(_trade("2026-04-20T02:00:00+00:00", 100.0))]
        s = compute_summary(per_trade, anti_stack_blocks=2)
        # imputed = 0.8 * avg(100) * 2 = 160
        assert s["no_anti_stack_total"] == pytest.approx(100.0 + 160.0)
        assert s["no_anti_stack_verdict"] == "guardrail_cost"


# ---------------------------------------------------------------------------
# build_regret_records (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildRegretRecords:
    def test_day1_control_vs_treatment(self) -> None:
        trades = [
            _trade("2026-04-20T02:15:00+00:00", 68.92, magic=MAGIC_CONTROL),
            _trade("2026-04-20T05:42:00+00:00", -53.00, magic=MAGIC_CONTROL),
            _trade("2026-04-20T11:07:00+00:00", 68.92, magic=MAGIC_CONTROL),
            _trade("2026-04-20T16:23:00+00:00", -53.00, magic=MAGIC_CONTROL),
            _trade("2026-04-20T03:01:00+00:00", -35.33, magic=MAGIC_TREATMENT),
            _trade("2026-04-20T07:18:00+00:00", 51.13, magic=MAGIC_TREATMENT),
            _trade("2026-04-20T12:45:00+00:00", -35.33, magic=MAGIC_TREATMENT),
            _trade("2026-04-20T18:52:00+00:00", -35.34, magic=MAGIC_TREATMENT),
        ]
        recs = build_regret_records(trades, anti_stack_blocks=0)
        # 8 per-trade + 1 summary
        assert len(recs) == 9
        summary = recs[-1]
        assert summary["summary"] is True
        assert summary["trade_count"] == 8
        assert summary["actual_total"] == pytest.approx(31.84 + (-54.87))

    def test_summary_at_end_only(self) -> None:
        recs = build_regret_records([_trade("2026-04-20T02:00:00+00:00", 10.0)])
        assert recs[0].get("summary") is not True
        assert recs[-1]["summary"] is True


# ---------------------------------------------------------------------------
# compute_regret_for_ticket (R6 B2 — O3 Telegram scalar)
# ---------------------------------------------------------------------------


class TestComputeRegretForTicket:
    """Four fixtures: helpful / hurtful / neutral / uncomputable regret."""

    def test_control_leg_is_neutral_zero(self) -> None:
        """Control leg by design has no_macro_pnl == actual_pnl → regret 0."""
        closure = {
            "event": "trade_reconciled",
            "ticket": 262715883,
            "pnl_usd": 66.31,
            "magic": MAGIC_CONTROL,
            "direction": "long",
        }
        regret = compute_regret_for_ticket(
            ticket=262715883,
            closure=closure,
        )
        assert regret == 0.0

    def test_treatment_leg_neutral_under_current_heuristic(self) -> None:
        """Treatment leg: current v4 Alt-B heuristic assumes macro doesn't
        re-time entries, so no_macro_pnl == actual_pnl → regret 0 as well.

        This documents the heuristic baseline. If a future version of
        ``compute_regret_row`` moves to a non-trivial macro counterfactual,
        the value will change without needing to change the wrapper API.
        """
        closure = {
            "event": "trade_reconciled",
            "ticket": 262518001,
            "pnl_usd": -52.72,
            "magic": MAGIC_TREATMENT,
            "direction": "short",
        }
        regret = compute_regret_for_ticket(
            ticket=262518001, closure=closure,
        )
        assert regret == 0.0

    def test_helpful_macro_when_counterfactual_worse(self) -> None:
        """If we monkeypatch no_macro_pnl lower than actual → negative regret
        (macro helped). Simulates a future heuristic upgrade path."""
        import smc.monitor.regret_analysis as ra

        def fake_row(trade, closures_by_ticket=None):
            # Simulate "no-macro would have lost $5 more than we actually did"
            actual = 10.0
            no_macro = actual - 5.0  # counterfactual worse by $5
            return {
                "trade_id": "ticket:999",
                "actual_pnl": actual,
                "no_macro_pnl": no_macro,
                "confidence": "heuristic",
            }

        original = ra.compute_regret_row
        ra.compute_regret_row = fake_row  # type: ignore[assignment]
        try:
            closure = {
                "event": "trade_reconciled",
                "ticket": 999, "pnl_usd": 10.0,
                "magic": MAGIC_TREATMENT, "direction": "long",
            }
            regret = compute_regret_for_ticket(ticket=999, closure=closure)
        finally:
            ra.compute_regret_row = original  # type: ignore[assignment]
        assert regret == pytest.approx(-5.0)

    def test_hurtful_macro_when_counterfactual_better(self) -> None:
        """no_macro_pnl > actual_pnl → positive regret (macro hurt)."""
        import smc.monitor.regret_analysis as ra

        def fake_row(trade, closures_by_ticket=None):
            actual = -10.0
            no_macro = actual + 15.0  # would have made $5 without macro
            return {
                "trade_id": "ticket:888",
                "actual_pnl": actual,
                "no_macro_pnl": no_macro,
                "confidence": "heuristic",
            }

        original = ra.compute_regret_row
        ra.compute_regret_row = fake_row  # type: ignore[assignment]
        try:
            closure = {
                "event": "trade_reconciled",
                "ticket": 888, "pnl_usd": -10.0,
                "magic": MAGIC_TREATMENT, "direction": "long",
            }
            regret = compute_regret_for_ticket(ticket=888, closure=closure)
        finally:
            ra.compute_regret_row = original  # type: ignore[assignment]
        assert regret == pytest.approx(15.0)

    def test_uncomputable_missing_magic_returns_none(self) -> None:
        """No magic in closure → cannot classify leg → None."""
        closure = {
            "event": "trade_reconciled",
            "ticket": 777,
            "pnl_usd": 10.0,
            # no magic
            "direction": "long",
        }
        assert compute_regret_for_ticket(ticket=777, closure=closure) is None

    def test_uncomputable_unknown_magic_returns_none(self) -> None:
        """Unknown magic (not control/treatment) → None."""
        closure = {
            "event": "trade_reconciled",
            "ticket": 666, "pnl_usd": 10.0,
            "magic": 99999999, "direction": "long",
        }
        assert compute_regret_for_ticket(ticket=666, closure=closure) is None

    def test_uncomputable_missing_pnl_returns_none(self) -> None:
        """No pnl_usd and ticket not in closures_by_ticket → None."""
        closure = {
            "event": "trade_reconciled",
            "ticket": 555,
            "magic": MAGIC_CONTROL,
            "direction": "long",
        }
        assert compute_regret_for_ticket(ticket=555, closure=closure) is None

    def test_uses_closures_by_ticket_when_closure_pnl_absent(self) -> None:
        """Pre-indexed closures_by_ticket map supplies pnl when closure dict
        doesn't carry it directly."""
        closure = {
            "event": "trade_reconciled",
            "ticket": 444,
            "magic": MAGIC_CONTROL,
            "direction": "long",
        }
        regret = compute_regret_for_ticket(
            ticket=444,
            closure=closure,
            closures_by_ticket={444: 25.0},
        )
        # Control leg → 0 even when pnl came from the external map.
        assert regret == 0.0

    def test_bad_ticket_string_returns_none(self) -> None:
        """Non-coercible ticket → None (defensive)."""
        closure = {
            "event": "trade_reconciled", "ticket": "abc",
            "pnl_usd": 10.0, "magic": MAGIC_CONTROL, "direction": "long",
        }
        assert compute_regret_for_ticket(ticket="abc", closure=closure) is None  # type: ignore[arg-type]
