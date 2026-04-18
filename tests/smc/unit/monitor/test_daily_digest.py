"""Tests for Round 3 Sprint 1 daily_digest builder.

Covers empty day / 1 win / 1 loss + halt / yesterday's halt reset today /
cross-UTC-day boundary / corrupt journal line / margin_cap trip / invalid date.

Data sources simulated via synthetic files in ``tmp_path``:
  - ``data/{SYMBOL}/journal/live_trades.jsonl``
  - ``data/{SYMBOL}/{consec_loss,phase1a_breaker,asian_range_quota}_state.json``
  - ``data/{SYMBOL}/live_state.json``
  - ``data/{SYMBOL}/live_demo.pid``
  - ``logs/structured.jsonl``
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from smc.monitor.daily_digest import build_daily_digest


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _mk_tree(tmp_path: Path, symbol: str = "XAUUSD") -> tuple[Path, Path]:
    data_root = tmp_path / "data" / symbol
    (data_root / "journal").mkdir(parents=True)
    log_root = tmp_path / "logs"
    log_root.mkdir()
    return data_root, log_root


def _journal_line(**fields) -> str:
    return json.dumps(fields)


def _structured_line(severity: str, event: str, **fields) -> str:
    payload = {"ts": fields.pop("ts"), "event": event, **fields}
    return f"[{severity}] {json.dumps(payload)}"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


TARGET = date(2026, 4, 18)
NOON_UTC = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Case 1: Empty day — no data sources at all
# ---------------------------------------------------------------------------


class TestEmptyDay:
    def test_no_data_sources_returns_zeros_and_warnings(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["symbol"] == "XAUUSD"
        assert digest["date"] == "2026-04-18"
        # decision-reviewer request: cycles_today_note disclaimer must be
        # present so dashboard tooltip can set operator expectations.
        assert "cycles_today_note" in digest
        assert "process-start" in digest["cycles_today_note"]
        assert digest["trades_opened"] == 0
        assert digest["trades_closed"] == 0
        assert digest["wins"] == 0
        assert digest["losses"] == 0
        assert digest["gross_pnl_usd"] == 0.0
        assert digest["win_rate_pct"] is None
        assert digest["avg_win_usd"] is None
        assert digest["avg_loss_usd"] is None
        assert digest["margin_cap_gate_trips"] == 0
        assert digest["mt5_order_fails"] == 0
        assert digest["consec_loss_halt_tripped"] is False
        assert digest["phase1a_breaker_tripped"] is False
        assert digest["last_state_ts"] is None
        assert digest["uptime_hours"] is None
        # Must flag missing inputs without crashing
        assert "journal_missing" in digest["warnings"]
        assert "structured_log_missing" in digest["warnings"]
        # Shape sanity
        assert "guards_current" in digest
        assert digest["guards_current"]["consec_loss"]["tripped"] is False


# ---------------------------------------------------------------------------
# Case 2: 1 trade opened + closed win
# ---------------------------------------------------------------------------


class TestOneTradeWin:
    def test_one_open_one_close_win(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # journal: one PAPER open at 10:00Z
        _write_lines(
            data_root / "journal" / "live_trades.jsonl",
            [_journal_line(
                time="2026-04-18T10:00:00+00:00",
                cycle=12,
                mode="PAPER",
                direction="long",
                entry=2350.0,
                position_size_lots=0.33,
            )],
        )
        # structured: one trade_reconciled with +$5 at 11:30Z
        _write_lines(
            log_root / "structured.jsonl",
            [_structured_line(
                "INFO", "trade_reconciled",
                ts="2026-04-18T11:30:00+00:00",
                ticket=999,
                pnl_usd=5.0,
                running_daily_pnl=5.0,
            )],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["trades_opened"] == 1
        assert digest["trades_closed"] == 1
        assert digest["wins"] == 1
        assert digest["losses"] == 0
        assert digest["gross_pnl_usd"] == 5.0
        assert digest["win_rate_pct"] == 100.0
        assert digest["avg_win_usd"] == 5.0
        assert digest["avg_loss_usd"] is None


# ---------------------------------------------------------------------------
# Case 3: 1 trade lost + consec_halt tripped today
# ---------------------------------------------------------------------------


class TestLossWithHalt:
    def test_loss_and_halt_tripped_today(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        _write_lines(
            data_root / "journal" / "live_trades.jsonl",
            [_journal_line(
                time="2026-04-18T09:00:00+00:00",
                mode="PAPER",
                direction="long",
                entry=2350.0,
            )],
        )
        _write_lines(
            log_root / "structured.jsonl",
            [_structured_line(
                "INFO", "trade_reconciled",
                ts="2026-04-18T10:00:00+00:00",
                ticket=555,
                pnl_usd=-20.0,
            )],
        )
        # consec_loss_state tripped today
        (data_root / "consec_loss_state.json").write_text(json.dumps({
            "consec_losses": 3,
            "tripped": True,
            "tripped_at": "2026-04-18T10:05:00+00:00",
            "last_reset_date": "2026-04-18",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["losses"] == 1
        assert digest["gross_pnl_usd"] == -20.0
        assert digest["avg_loss_usd"] == -20.0
        assert digest["consec_loss_halt_tripped"] is True
        assert digest["consec_loss_halt_tripped_at"] == "2026-04-18T10:05:00+00:00"
        assert digest["guards_current"]["consec_loss"]["consec_losses"] == 3


# ---------------------------------------------------------------------------
# Case 4: Yesterday's halt still in state file but reset today
# ---------------------------------------------------------------------------


class TestYesterdayHaltResetToday:
    def test_yesterday_tripped_reset_today_not_counted(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # Daily reset already fired: tripped=False, last_reset_date=today
        (data_root / "consec_loss_state.json").write_text(json.dumps({
            "consec_losses": 0,
            "tripped": False,
            "tripped_at": None,
            "last_reset_date": "2026-04-18",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["consec_loss_halt_tripped"] is False
        assert digest["consec_loss_halt_tripped_at"] is None

    def test_tripped_yesterday_still_true_but_not_today(self, tmp_path: Path):
        """Edge: state file says tripped, but tripped_at is YESTERDAY.

        Reflects broker's current trading-disabled state, but digest field
        `consec_loss_halt_tripped` asks "tripped TODAY?" → False.
        """
        data_root, log_root = _mk_tree(tmp_path)
        (data_root / "consec_loss_state.json").write_text(json.dumps({
            "consec_losses": 3,
            "tripped": True,
            "tripped_at": "2026-04-17T22:00:00+00:00",  # yesterday
            "last_reset_date": "2026-04-17",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["consec_loss_halt_tripped"] is False  # not tripped TODAY
        # Snapshot still shows current state (for operator visibility)
        assert digest["guards_current"]["consec_loss"]["tripped"] is True


# ---------------------------------------------------------------------------
# Case 5: Cross UTC-day boundary
# ---------------------------------------------------------------------------


class TestDayBoundary:
    def test_entries_outside_target_day_excluded(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        _write_lines(
            data_root / "journal" / "live_trades.jsonl",
            [
                # Yesterday 23:30 — should NOT count
                _journal_line(time="2026-04-17T23:30:00+00:00", mode="PAPER"),
                # Today 00:30 — SHOULD count
                _journal_line(time="2026-04-18T00:30:00+00:00", mode="PAPER"),
                # Tomorrow 00:30 — should NOT count
                _journal_line(time="2026-04-19T00:30:00+00:00", mode="PAPER"),
            ],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["trades_opened"] == 1


# ---------------------------------------------------------------------------
# Case 6: Corrupt journal line tolerated
# ---------------------------------------------------------------------------


class TestCorruptJournal:
    def test_malformed_line_skipped_and_warned(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        _write_lines(
            data_root / "journal" / "live_trades.jsonl",
            [
                _journal_line(time="2026-04-18T10:00:00+00:00", mode="PAPER"),
                "{this is not valid json",
                _journal_line(time="2026-04-18T11:00:00+00:00", mode="PAPER"),
            ],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["trades_opened"] == 2
        assert any("journal_parse_errors" in w for w in digest["warnings"])


# ---------------------------------------------------------------------------
# Case 7: margin_cap gate trip
# ---------------------------------------------------------------------------


class TestMarginCapTrip:
    def test_margin_gated_journal_entry_counted(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        _write_lines(
            data_root / "journal" / "live_trades.jsonl",
            [_journal_line(
                time="2026-04-18T10:00:00+00:00",
                mode="MARGIN_GATED",
                margin_gated=True,
                margin_reason="cap_exceeded: 45.2% > 40.0%",
            )],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["margin_cap_gate_trips"] == 1
        assert digest["trades_opened"] == 0  # MARGIN_GATED is not a "successful open"

    def test_pre_write_gate_block_counted_from_structured_log(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        _write_lines(
            log_root / "structured.jsonl",
            [
                _structured_line(
                    "WARN", "pre_write_gate_blocked",
                    ts="2026-04-18T10:00:00+00:00",
                    blocked_reason="margin_cap:exceeded:43%>40%",
                ),
                _structured_line(
                    "WARN", "pre_write_gate_blocked",
                    ts="2026-04-18T11:00:00+00:00",
                    blocked_reason="asian_quota:exhausted_today",
                ),
            ],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["pre_write_gate_blocks"] == 2
        assert digest["margin_blocks_count"] == 1
        assert digest["asian_quota_blocks_count"] == 1


# ---------------------------------------------------------------------------
# Case 8: dedup trade_reconciled+trade_closed on same ticket
# ---------------------------------------------------------------------------


class TestTickerDedup:
    def test_same_ticket_only_counted_once(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # trade_reconciled + trade_closed both emitted with same ticket
        _write_lines(
            log_root / "structured.jsonl",
            [
                _structured_line(
                    "INFO", "trade_reconciled",
                    ts="2026-04-18T10:00:00+00:00",
                    ticket=777,
                    pnl_usd=10.0,
                ),
                _structured_line(
                    "CRIT", "trade_closed",
                    ts="2026-04-18T10:00:01+00:00",
                    ticket=777,
                    pnl_usd=10.0,
                ),
            ],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        # Only the trade_reconciled event contributes (event name scoped).
        assert digest["wins"] == 1
        assert digest["gross_pnl_usd"] == 10.0


# ---------------------------------------------------------------------------
# Case 9: freshness + uptime
# ---------------------------------------------------------------------------


class TestFreshnessAndUptime:
    def test_live_state_age_and_cycle(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # live_state.json 2 min old relative to NOON_UTC
        (data_root / "live_state.json").write_text(json.dumps({
            "cycle": 48,
            "timestamp": "2026-04-18T11:58:00+00:00",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["last_state_ts"] == "2026-04-18T11:58:00+00:00"
        assert digest["last_state_age_sec"] == 120
        assert digest["cycles_today"] == 48
        assert "state_stale_over_5min" not in digest["warnings"]

    def test_stale_state_adds_warning(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # 10 min stale
        (data_root / "live_state.json").write_text(json.dumps({
            "cycle": 1,
            "timestamp": "2026-04-18T11:50:00+00:00",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["last_state_age_sec"] == 600
        assert "state_stale_over_5min" in digest["warnings"]

    def test_pid_file_mtime_yields_uptime_hours(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        pid = data_root / "live_demo.pid"
        pid.write_text("12345")
        # Set mtime 3h before NOON_UTC
        three_h_ago_ts = (NOON_UTC - timedelta(hours=3)).timestamp()
        import os
        os.utime(pid, (three_h_ago_ts, three_h_ago_ts))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["uptime_hours"] is not None
        assert 2.9 <= digest["uptime_hours"] <= 3.1


# ---------------------------------------------------------------------------
# Case 10: asian_quota exhausted_today
# ---------------------------------------------------------------------------


class TestAsianQuota:
    def test_exhausted_today_true_when_date_matches(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        (data_root / "asian_range_quota_state.json").write_text(json.dumps({
            "last_open_date": "2026-04-18",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["guards_current"]["asian_quota"]["exhausted_today"] is True

    def test_exhausted_today_false_when_date_yesterday(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        (data_root / "asian_range_quota_state.json").write_text(json.dumps({
            "last_open_date": "2026-04-17",
        }))
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["guards_current"]["asian_quota"]["exhausted_today"] is False


# ---------------------------------------------------------------------------
# Case 11: Rotated structured.jsonl.YYYY-MM-DD
# ---------------------------------------------------------------------------


class TestRotatedLog:
    def test_rotated_file_for_target_date_scanned(self, tmp_path: Path):
        data_root, log_root = _mk_tree(tmp_path)
        # Entry is in the rotated file (structured.jsonl.2026-04-18), not the current one
        _write_lines(
            log_root / "structured.jsonl.2026-04-18",
            [_structured_line(
                "INFO", "trade_reconciled",
                ts="2026-04-18T10:00:00+00:00",
                ticket=100,
                pnl_usd=3.0,
            )],
        )
        digest = build_daily_digest(
            "XAUUSD", TARGET, data_root=data_root, log_root=log_root, now=NOON_UTC,
        )
        assert digest["wins"] == 1
        assert digest["gross_pnl_usd"] == 3.0
