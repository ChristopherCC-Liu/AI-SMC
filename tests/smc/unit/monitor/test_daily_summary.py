"""Unit tests for R5 M4 daily_summary module."""
from __future__ import annotations

import pytest

from smc.monitor.daily_summary import MAX_TELEGRAM_CHARS, format_daily_summary


def _digest(**overrides) -> dict:
    base = {
        "date": "2026-04-20",
        "generated_at": "2026-04-21T00:05:00+00:00",
        "per_symbol": {},
        "per_leg": [],
        "regime_distribution": {},
        "ai_debate": {},
        "handle_resets": 0,
    }
    base.update(overrides)
    return base


class TestFormatDailySummary:
    def test_minimal_digest(self) -> None:
        msg = format_daily_summary(_digest())
        assert "AI-SMC Daily Summary - 2026-04-20" in msg
        assert "Trades: 0" in msg
        assert "PF: n/a" in msg
        assert "Handle resets: 0" in msg

    def test_day1_full_shape(self) -> None:
        digest = _digest(
            per_leg=[
                {
                    "leg": "XAUUSD:control", "trades": 4, "wins": 2, "losses": 2,
                    "win_rate_pct": 50.0, "total_pnl_usd": 31.84,
                    "avg_win_usd": 68.92, "avg_loss_usd": -53.0,
                    "profit_factor": 1.3, "payoff_ratio": 1.3,
                    "max_drawdown_usd": 53.0,
                },
                {
                    "leg": "XAUUSD:treatment", "trades": 4, "wins": 1, "losses": 3,
                    "win_rate_pct": 25.0, "total_pnl_usd": -54.87,
                    "avg_win_usd": 51.13, "avg_loss_usd": -35.33,
                    "profit_factor": 0.48, "payoff_ratio": 1.45,
                    "max_drawdown_usd": 70.67,
                },
            ],
            regime_distribution={
                "TRANSITION": 2, "TREND_UP": 0, "TREND_DOWN": 0,
                "CONSOLIDATION": 1, "ATH_BREAKOUT": 0, "UNKNOWN": 0,
            },
            ai_debate={
                "cycles_ran": 2, "p50_elapsed_ms": 1500.0,
                "p90_elapsed_ms": 1900.0, "total_cost_usd": 0.12,
            },
            handle_resets=1,
        )
        msg = format_daily_summary(digest)
        # Content
        assert "Trades: 8" in msg
        assert "-$23.03" in msg or "+$-23" in msg  # net = 31.84 - 54.87 = -23.03
        assert "XAU control" in msg
        assert "XAU treatment" in msg
        assert "d(C-T): +$86.71" in msg
        assert "Top regime: TRANSITION (2)" in msg
        assert "AI: 2 cycles, p50 1500ms, $0.12" in msg
        assert "Handle resets: 1" in msg

    def test_skips_zero_trade_legs(self) -> None:
        digest = _digest(per_leg=[
            {"leg": "BTCUSD:control", "trades": 0, "wins": 0, "losses": 0,
             "total_pnl_usd": 0.0, "profit_factor": None,
             "avg_win_usd": None, "avg_loss_usd": None},
        ])
        msg = format_daily_summary(digest)
        assert "BTC control" not in msg
        assert "Trades: 0" in msg

    def test_warnings_rendered_and_capped(self) -> None:
        digest = _digest(per_symbol={
            "XAUUSD": {"warnings": ["journal_missing", "structured_log_missing", "state_stale_over_5min", "foo"]},
        })
        msg = format_daily_summary(digest)
        # Only first 3 rendered
        assert "journal_missing" in msg
        assert "structured_log_missing" in msg
        assert "state_stale_over_5min" in msg
        assert "foo" not in msg

    def test_no_top_regime_when_all_zero(self) -> None:
        digest = _digest(regime_distribution={"TRANSITION": 0, "TREND_UP": 0})
        msg = format_daily_summary(digest)
        assert "Top regime:" not in msg

    def test_under_telegram_limit(self) -> None:
        digest = _digest(per_leg=[
            {"leg": "XAUUSD:control", "trades": 4, "wins": 2, "losses": 2,
             "total_pnl_usd": 31.84, "profit_factor": 1.3,
             "avg_win_usd": 68.92, "avg_loss_usd": -53.0},
        ])
        msg = format_daily_summary(digest)
        assert len(msg) < MAX_TELEGRAM_CHARS

    def test_oversized_message_truncated(self) -> None:
        # Craft an artificially huge warnings set (but capped at 3 rendered);
        # simulate actual truncation by injecting a giant per_leg legname.
        digest = _digest(per_leg=[
            {"leg": "X" * (MAX_TELEGRAM_CHARS + 100), "trades": 1, "wins": 1, "losses": 0,
             "total_pnl_usd": 10.0, "profit_factor": 2.0,
             "avg_win_usd": 10.0, "avg_loss_usd": None},
        ])
        msg = format_daily_summary(digest)
        assert len(msg) <= MAX_TELEGRAM_CHARS
