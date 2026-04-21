"""Unit tests for R5 M1 digest_report module (aggregator + CSV/MD writer)."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from smc.monitor.digest_report import (
    build_multi_symbol_digest,
    collect_journal_paths,
    render_digest_csv,
    render_digest_markdown,
    scan_structured_events,
    write_digest_report,
)


TARGET = date(2026, 4, 20)
NOW = datetime(2026, 4, 21, 0, 5, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _mk_layout(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    data_root.mkdir()
    log_root.mkdir()
    return data_root, log_root


def _write_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _journal_row(ts: str, result: float, *, magic: int = 19760418) -> dict:
    return {
        "time": ts,
        "action": "BUY",
        "mode": "LIVE_EXEC",
        "direction": "long",
        "magic": magic,
        "result": result,
    }


def _struct_line(severity: str, event: str, ts: str, **extra) -> str:
    payload = {"ts": ts, "event": event, **extra}
    return f"[{severity}] {json.dumps(payload)}"


def _write_structured(log_root: Path, lines: list[str]) -> None:
    (log_root / "structured.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# collect_journal_paths
# ---------------------------------------------------------------------------


class TestCollectJournalPaths:
    def test_empty_data_root(self, tmp_path: Path) -> None:
        data_root, _ = _mk_layout(tmp_path)
        paths = collect_journal_paths(data_root)
        assert paths == {}

    def test_xau_control_only(self, tmp_path: Path) -> None:
        data_root, _ = _mk_layout(tmp_path)
        (data_root / "XAUUSD" / "journal").mkdir(parents=True)
        paths = collect_journal_paths(data_root)
        assert list(paths.keys()) == ["XAUUSD:control"]

    def test_all_three_legs(self, tmp_path: Path) -> None:
        data_root, _ = _mk_layout(tmp_path)
        (data_root / "XAUUSD" / "journal").mkdir(parents=True)
        (data_root / "XAUUSD" / "journal_macro").mkdir(parents=True)
        (data_root / "BTCUSD" / "journal").mkdir(parents=True)
        paths = collect_journal_paths(data_root)
        assert set(paths.keys()) == {
            "XAUUSD:control", "XAUUSD:treatment", "BTCUSD:control",
        }


# ---------------------------------------------------------------------------
# scan_structured_events
# ---------------------------------------------------------------------------


class TestScanStructuredEvents:
    def test_missing_log_root(self, tmp_path: Path) -> None:
        # Log root that doesn't exist
        events = scan_structured_events(tmp_path / "nope", TARGET)
        assert events == []

    def test_filters_by_date(self, tmp_path: Path) -> None:
        _, log_root = _mk_layout(tmp_path)
        _write_structured(log_root, [
            _struct_line("INFO", "ai_regime_classified", "2026-04-19T23:00:00+00:00", regime="TRANSITION"),
            _struct_line("INFO", "ai_regime_classified", "2026-04-20T12:00:00+00:00", regime="TREND_UP"),
            _struct_line("INFO", "ai_regime_classified", "2026-04-21T01:00:00+00:00", regime="CONSOLIDATION"),
        ])
        events = scan_structured_events(log_root, TARGET)
        assert len(events) == 1
        assert events[0]["regime"] == "TREND_UP"

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        _, log_root = _mk_layout(tmp_path)
        (log_root / "structured.jsonl").write_text(
            "garbage line\n"
            "[INFO] {not json}\n"
            + _struct_line("INFO", "ai_regime_classified", "2026-04-20T12:00:00+00:00", regime="TRANSITION")
            + "\n",
            encoding="utf-8",
        )
        events = scan_structured_events(log_root, TARGET)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# build_multi_symbol_digest
# ---------------------------------------------------------------------------


class TestBuildMultiSymbolDigest:
    def test_day1_full_shape(self, tmp_path: Path) -> None:
        data_root, log_root = _mk_layout(tmp_path)

        # XAUUSD control — 2 SL + 2 TP = +$31.84
        _write_journal(
            data_root / "XAUUSD" / "journal" / "live_trades.jsonl",
            [
                _journal_row("2026-04-20T02:15:00+00:00", 68.92),
                _journal_row("2026-04-20T05:42:00+00:00", -53.00),
                _journal_row("2026-04-20T11:07:00+00:00", 68.92),
                _journal_row("2026-04-20T16:23:00+00:00", -53.00),
            ],
        )
        # XAUUSD treatment — 1 TP + 3 SL = -$54.87
        _write_journal(
            data_root / "XAUUSD" / "journal_macro" / "live_trades.jsonl",
            [
                _journal_row("2026-04-20T03:01:00+00:00", -35.33, magic=19760428),
                _journal_row("2026-04-20T07:18:00+00:00", 51.13, magic=19760428),
                _journal_row("2026-04-20T12:45:00+00:00", -35.33, magic=19760428),
                _journal_row("2026-04-20T18:52:00+00:00", -35.34, magic=19760428),
            ],
        )
        # Regime + AI + handle_reset events
        _write_structured(log_root, [
            _struct_line("INFO", "ai_regime_classified", "2026-04-20T02:00:00+00:00", regime="TRANSITION"),
            _struct_line("INFO", "ai_regime_classified", "2026-04-20T03:00:00+00:00", regime="TRANSITION"),
            _struct_line("INFO", "ai_regime_classified", "2026-04-20T04:00:00+00:00", regime="CONSOLIDATION"),
            _struct_line("INFO", "ai_debate_completed", "2026-04-20T05:00:00+00:00", elapsed_ms=1500.0, total_cost_usd=0.05),
            _struct_line("INFO", "ai_debate_completed", "2026-04-20T06:00:00+00:00", elapsed_ms=2000.0, total_cost_usd=0.07),
            _struct_line("WARN", "mt5_handle_reset", "2026-04-20T08:00:00+00:00", reason="timeout"),
        ])

        digest = build_multi_symbol_digest(
            TARGET,
            data_root=data_root, log_root=log_root,
            symbols=("XAUUSD", "BTCUSD"),
            now=NOW,
        )

        # Shape
        assert digest["date"] == "2026-04-20"
        assert digest["generated_at"] == NOW.isoformat()
        assert set(digest.keys()) == {
            "date", "generated_at", "per_symbol", "per_leg",
            "regime_distribution", "ai_debate", "handle_resets",
        }

        # per_symbol has XAUUSD only (BTCUSD skipped — no dir)
        assert "XAUUSD" in digest["per_symbol"]
        assert "BTCUSD" not in digest["per_symbol"]
        # Legacy fields still present
        xau = digest["per_symbol"]["XAUUSD"]
        assert xau["symbol"] == "XAUUSD"
        assert xau["date"] == "2026-04-20"
        assert "guards_current" in xau

        # per_leg — 2 entries
        assert len(digest["per_leg"]) == 2
        legs = {row["leg"]: row for row in digest["per_leg"]}
        assert legs["XAUUSD:control"]["total_pnl_usd"] == pytest.approx(31.84)
        assert legs["XAUUSD:treatment"]["total_pnl_usd"] == pytest.approx(-54.87)

        # Regime distribution
        assert digest["regime_distribution"]["TRANSITION"] == 2
        assert digest["regime_distribution"]["CONSOLIDATION"] == 1

        # AI debate
        assert digest["ai_debate"]["cycles_ran"] == 2
        assert digest["ai_debate"]["p50_elapsed_ms"] == pytest.approx(1750.0)
        assert digest["ai_debate"]["total_cost_usd"] == pytest.approx(0.12)

        # Handle resets
        assert digest["handle_resets"] == 1

    def test_backward_compat_legacy_digest_shape(self, tmp_path: Path) -> None:
        """Per-symbol blob must still carry all keys the dashboard API expects."""
        data_root, log_root = _mk_layout(tmp_path)
        (data_root / "XAUUSD" / "journal").mkdir(parents=True)
        digest = build_multi_symbol_digest(
            TARGET, data_root=data_root, log_root=log_root,
            symbols=("XAUUSD",), now=NOW,
        )
        xau = digest["per_symbol"]["XAUUSD"]
        legacy_keys = {
            "symbol", "date", "trades_opened", "trades_closed", "wins",
            "losses", "breakeven", "gross_pnl_usd", "win_rate_pct",
            "avg_win_usd", "avg_loss_usd", "pre_write_gate_blocks",
            "margin_blocks_count", "asian_quota_blocks_count",
            "margin_cap_gate_trips", "mt5_order_fails",
            "consec_loss_halt_tripped", "consec_loss_halt_tripped_at",
            "phase1a_breaker_tripped", "phase1a_breaker_tripped_at",
            "drawdown_halt_active", "drawdown_halt_reason",
            "guards_current", "last_state_ts", "last_state_age_sec",
            "uptime_hours", "cycles_today", "cycles_today_note", "warnings",
        }
        assert legacy_keys.issubset(set(xau.keys()))


# ---------------------------------------------------------------------------
# CSV / Markdown rendering
# ---------------------------------------------------------------------------


class TestRender:
    def test_csv_empty(self) -> None:
        csv_text = render_digest_csv({"date": "2026-04-20", "per_leg": []})
        assert csv_text.splitlines()[0].startswith("leg,trades")
        # Must not raise; must contain comment-prefixed sections
        assert "# Regime distribution" in csv_text
        assert "# AI debate" in csv_text
        assert "# Handle resets" in csv_text

    def test_csv_with_legs(self) -> None:
        digest = {
            "date": "2026-04-20",
            "per_leg": [
                {
                    "leg": "XAUUSD:control",
                    "trades": 4, "wins": 2, "losses": 2,
                    "win_rate_pct": 50.0,
                    "total_pnl_usd": 31.84, "max_drawdown_usd": 53.0,
                    "avg_win_usd": 68.92, "avg_loss_usd": -53.0,
                    "payoff_ratio": 1.3, "profit_factor": 1.3,
                },
            ],
            "regime_distribution": {"TRANSITION": 2, "TREND_UP": 0},
            "ai_debate": {
                "cycles_ran": 2, "p50_elapsed_ms": 1500.0,
                "p90_elapsed_ms": 1900.0, "total_cost_usd": 0.12,
            },
            "handle_resets": 1,
        }
        csv_text = render_digest_csv(digest)
        assert "XAUUSD:control,4,2,2" in csv_text
        assert "# TRANSITION,2" in csv_text
        assert "# Handle resets: 1" in csv_text

    def test_markdown_max_dd_unsigned(self) -> None:
        """Max drawdown should render without a +/- sign (non-negative by def)."""
        digest = {
            "date": "2026-04-20",
            "generated_at": "2026-04-21T00:05:00+00:00",
            "per_symbol": {},
            "per_leg": [{
                "leg": "X:c", "trades": 2, "wins": 1, "losses": 1,
                "win_rate_pct": 50.0, "total_pnl_usd": 10.0,
                "max_drawdown_usd": 80.0,
                "avg_win_usd": 50.0, "avg_loss_usd": -40.0,
                "payoff_ratio": 1.25, "profit_factor": 1.25,
            }],
            "regime_distribution": {},
            "ai_debate": {},
            "handle_resets": 0,
        }
        md = render_digest_markdown(digest)
        # Max DD column shows unsigned $80.00, not $+80.00
        assert "| $80.00 |" in md

    def test_markdown_rendering(self) -> None:
        digest = {
            "date": "2026-04-20",
            "generated_at": "2026-04-21T00:05:00+00:00",
            "per_symbol": {},
            "per_leg": [
                {
                    "leg": "XAUUSD:control",
                    "trades": 4, "wins": 2, "losses": 2,
                    "win_rate_pct": 50.0,
                    "total_pnl_usd": 31.84, "max_drawdown_usd": 53.0,
                    "avg_win_usd": 68.92, "avg_loss_usd": -53.0,
                    "payoff_ratio": 1.3, "profit_factor": 1.3,
                },
            ],
            "regime_distribution": {"TRANSITION": 2},
            "ai_debate": {"cycles_ran": 2, "p50_elapsed_ms": 1500.0, "p90_elapsed_ms": 1900.0, "total_cost_usd": 0.12},
            "handle_resets": 1,
        }
        md = render_digest_markdown(digest)
        assert "# AI-SMC Daily Digest — 2026-04-20" in md
        assert "| XAUUSD:control |" in md
        assert "$+31.84" in md
        assert "Cycles ran:** 2" in md
        assert "MT5 handle resets today:** 1" in md

    def test_markdown_empty_per_leg(self) -> None:
        digest = {
            "date": "2026-04-20",
            "generated_at": "2026-04-21T00:05:00+00:00",
            "per_symbol": {},
            "per_leg": [],
            "regime_distribution": {},
            "ai_debate": {},
            "handle_resets": 0,
        }
        md = render_digest_markdown(digest)
        assert "*No closed trades today across any leg.*" in md
        assert "*No ai_regime_classified events today.*" in md


# ---------------------------------------------------------------------------
# write_digest_report
# ---------------------------------------------------------------------------


class TestWriteDigestReport:
    def test_writes_csv_and_md(self, tmp_path: Path) -> None:
        digest = {
            "date": "2026-04-20",
            "generated_at": NOW.isoformat(),
            "per_symbol": {},
            "per_leg": [],
            "regime_distribution": {},
            "ai_debate": {},
            "handle_resets": 0,
        }
        reports_root = tmp_path / "data" / "reports"
        csv_path, md_path = write_digest_report(digest, reports_root=reports_root)

        assert csv_path == reports_root / "digest_2026-04-20.csv"
        assert md_path == reports_root / "digest_2026-04-20.md"
        assert csv_path.read_text().startswith("leg,trades")
        assert md_path.read_text().startswith("# AI-SMC Daily Digest")

    def test_missing_date_uses_today(self, tmp_path: Path) -> None:
        """Digest without explicit date -> today."""
        reports_root = tmp_path / "reports"
        csv_path, _ = write_digest_report(
            {"per_leg": [], "regime_distribution": {}, "ai_debate": {}, "handle_resets": 0},
            reports_root=reports_root,
        )
        assert csv_path.name.startswith("digest_")
        assert csv_path.name.endswith(".csv")
