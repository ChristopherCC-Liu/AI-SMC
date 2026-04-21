"""Unit tests for smc.monitor.trade_close_enrichment (Round 5 O3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.monitor.trade_close_enrichment import (
    LEG_BY_COMPOSITE,
    MAGIC_TO_LEG_LEGACY,
    TradeCloseContext,
    build_trade_close_context,
    format_trade_close_telegram,
    resolve_leg_label,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def journal_dir(tmp_path: Path) -> Path:
    """Simulate data/XAUUSD/journal/live_trades.jsonl + journal_macro/."""
    root = tmp_path / "data" / "XAUUSD"
    (root / "journal").mkdir(parents=True)
    (root / "journal_macro").mkdir(parents=True)
    return root


def _write_journal(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_structured(path: Path, rows: list[tuple[str, dict]]) -> None:
    """Write structured.jsonl lines in the repo's actual format: ``[SEV] {...}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for severity, row in rows:
            f.write(f"[{severity}] {json.dumps(row)}\n")


# ---------------------------------------------------------------------------
# Journal lookup
# ---------------------------------------------------------------------------


def test_build_context_resolves_fields_from_journal(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {
            "time": "2026-04-20T14:30:00+00:00",
            "mt5_ticket": 12345,
            "entry": 4743.51,
            "sl": 4720.00,
            "tp1": 4800.00,
            "direction": "long",
            "trigger": "support_bounce",
            "magic": 19760418,
        },
    ])
    structured = tmp_path / "structured.jsonl"
    _write_structured(structured, [])

    ctx = build_trade_close_context(
        ticket=12345,
        pnl_usd=74.83,
        exit_price=4818.34,
        close_time=datetime(2026, 4, 20, 15, 0, tzinfo=timezone.utc),
        journal_paths=[journal_path],
        structured_log_path=structured,
    )

    assert ctx.ticket == 12345
    assert ctx.entry_price == 4743.51
    assert ctx.exit_price == 4818.34
    assert ctx.sl == 4720.00
    assert ctx.direction == "long"
    assert ctx.trigger == "support_bounce"
    assert ctx.leg_label == "Control-XAU"
    assert ctx.magic == 19760418


def test_build_context_long_rr_positive_on_profit(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {
            "time": "2026-04-20T14:30:00+00:00",
            "mt5_ticket": 1,
            "entry": 100.0,
            "sl": 90.0,
            "direction": "long",
            "magic": 19760418,
        },
    ])
    ctx = build_trade_close_context(
        ticket=1, pnl_usd=20.0, exit_price=120.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    # risk=10, profit=20 → RR = +2.0
    assert ctx.rr_realized == 2.0


def test_build_context_short_rr_positive_on_profit(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {
            "time": "2026-04-20T14:30:00+00:00",
            "mt5_ticket": 2,
            "entry": 100.0,
            "sl": 110.0,
            "direction": "short",
            "magic": 19760418,
        },
    ])
    ctx = build_trade_close_context(
        ticket=2, pnl_usd=15.0, exit_price=85.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    # short: entry=100, exit=85 → gain 15, risk=10 → RR = +1.5
    assert ctx.rr_realized == 1.5


def test_build_context_missing_ticket_degrades_gracefully(tmp_path: Path):
    journal_path = tmp_path / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [])
    ctx = build_trade_close_context(
        ticket=9999, pnl_usd=-5.0, exit_price=100.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    # All contextual fields should be None; pnl preserved.
    assert ctx.entry_price is None
    assert ctx.rr_realized is None
    assert ctx.leg_label == "Unknown"
    assert ctx.pnl_usd == -5.0


def test_build_context_scans_both_legs(journal_dir: Path, tmp_path: Path):
    """Control journal has the ticket; treatment journal doesn't."""
    control = journal_dir / "journal" / "live_trades.jsonl"
    treatment = journal_dir / "journal_macro" / "live_trades.jsonl"
    _write_journal(treatment, [
        {"time": "2026-04-20T14:00", "mt5_ticket": 999, "entry": 1, "sl": 0.9,
         "direction": "long", "magic": 19760428},
    ])
    _write_journal(control, [
        {"time": "2026-04-20T14:30", "mt5_ticket": 555, "entry": 100, "sl": 90,
         "direction": "long", "trigger": "support_bounce", "magic": 19760418},
    ])
    ctx = build_trade_close_context(
        ticket=555, pnl_usd=10.0, exit_price=105.0, close_time=None,
        journal_paths=[treatment, control],  # treatment scanned first
        structured_log_path=tmp_path / "missing.jsonl",
    )
    assert ctx.trigger == "support_bounce"
    assert ctx.leg_label == "Control-XAU"


# ---------------------------------------------------------------------------
# Structured log regime lookup
# ---------------------------------------------------------------------------


def test_find_regime_skips_default_source(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 1,
         "entry": 100, "sl": 90, "direction": "long", "magic": 19760418},
    ])
    structured = tmp_path / "structured.jsonl"
    _write_structured(structured, [
        # Too late (after entry) — must be ignored.
        ("INFO", {"ts": "2026-04-20T14:45:00+00:00",
                  "event": "ai_regime_classified", "source": "atr_fallback",
                  "regime": "TRENDING", "confidence": 0.8}),
        # Before entry but fallback — must be ignored per spec.
        ("INFO", {"ts": "2026-04-20T14:25:00+00:00",
                  "event": "ai_regime_classified", "source": "default",
                  "regime": "TRANSITION", "confidence": 0.3}),
        # The one we want.
        ("INFO", {"ts": "2026-04-20T14:20:00+00:00",
                  "event": "ai_regime_classified", "source": "atr_fallback",
                  "regime": "CONSOLIDATION", "confidence": 0.75}),
    ])

    ctx = build_trade_close_context(
        ticket=1, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=structured,
    )
    assert ctx.regime_at_entry == "CONSOLIDATION"
    assert ctx.regime_confidence == 0.75


def test_find_regime_returns_none_when_no_match(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 1,
         "entry": 100, "sl": 90, "direction": "long", "magic": 19760418},
    ])
    structured = tmp_path / "structured.jsonl"
    # Only default-source rows — none qualify.
    _write_structured(structured, [
        ("INFO", {"ts": "2026-04-20T14:20:00+00:00",
                  "event": "ai_regime_classified", "source": "default",
                  "regime": "TRANSITION", "confidence": 0.3}),
    ])
    ctx = build_trade_close_context(
        ticket=1, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[journal_path], structured_log_path=structured,
    )
    assert ctx.regime_at_entry is None
    assert ctx.regime_confidence is None


# ---------------------------------------------------------------------------
# Telegram formatter
# ---------------------------------------------------------------------------


def test_format_telegram_matches_spec_example():
    ctx = TradeCloseContext(
        ticket=12345,
        pnl_usd=74.83,
        magic=19760418,
        leg_label="Control-XAU",
        symbol="XAUUSD",
        direction="long",
        entry_price=4743.51,
        exit_price=4818.34,
        sl=4720.00,
        tp1=4800.00,
        trigger="support_bounce",
        rr_realized=2.88,
        regime_at_entry="TRANSITION",
        regime_confidence=0.72,
        regret_delta=None,
    )
    text = format_trade_close_telegram(ctx)
    # Header + 4 expected lines.
    assert "[CLOSED] XAUUSD BUY Control-XAU" in text
    assert "Entry: 4743.51" in text
    assert "Exit: 4818.34" in text
    assert "+$74.83" in text
    assert "+2.88R" in text
    assert "TRANSITION" in text
    assert "conf 0.72" in text
    assert "support_bounce" in text


def test_format_telegram_loss_formatted_correctly():
    ctx = TradeCloseContext(
        ticket=2, pnl_usd=-23.50, magic=19760419, leg_label="Control-BTC",
        symbol="BTCUSD", direction="short",
        entry_price=65000.0, exit_price=65500.0, sl=64500.0, tp1=64000.0,
        trigger="resistance_rejection", rr_realized=-1.0,
        regime_at_entry="TRENDING", regime_confidence=0.88, regret_delta=None,
    )
    text = format_trade_close_telegram(ctx)
    assert "SELL Control-BTC" in text
    assert "-$23.50" in text
    assert "-1.00R" in text


def test_format_telegram_handles_missing_fields():
    ctx = TradeCloseContext(
        ticket=3, pnl_usd=0.0, magic=None, leg_label="Unknown",
        symbol=None, direction=None, entry_price=None, exit_price=None,
        sl=None, tp1=None, trigger=None, rr_realized=None,
        regime_at_entry=None, regime_confidence=None, regret_delta=None,
    )
    text = format_trade_close_telegram(ctx)
    # Dashes for missing values; no unhandled-None crashes.
    assert "?" in text or "—" in text
    assert "unknown" in text.lower()


def test_format_telegram_respects_plain_text_rule():
    """2026-04-20 parse_mode fix: no HTML/Markdown special chars."""
    ctx = TradeCloseContext(
        ticket=4, pnl_usd=1.0, magic=19760418, leg_label="Control-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=1.0,
        regime_at_entry="TRENDING", regime_confidence=0.9, regret_delta=None,
    )
    text = format_trade_close_telegram(ctx)
    for bad in ("<b>", "<i>", "*", "`"):
        assert bad not in text, f"plain-text rule broken by {bad!r}"


def test_format_telegram_under_1000_chars():
    ctx = TradeCloseContext(
        ticket=5, pnl_usd=1.0, magic=19760418, leg_label="Control-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="x" * 500, rr_realized=1.0,
        regime_at_entry="TRENDING", regime_confidence=0.9, regret_delta=5.5,
    )
    text = format_trade_close_telegram(ctx)
    assert len(text) <= 1000


def test_format_telegram_includes_regret_when_present():
    # R6 B2: regret_delta is USD, not R. Negative means macro helped
    # (no-macro counterfactual would have been $0.40 worse).
    ctx = TradeCloseContext(
        ticket=6, pnl_usd=1.0, magic=19760428, leg_label="Treatment-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=1.0,
        regime_at_entry="TRENDING", regime_confidence=0.9,
        regret_delta=-0.4,
    )
    text = format_trade_close_telegram(ctx)
    assert "Regret" in text
    assert "-$0.40" in text
    assert "no-macro counterfactual" in text


def test_format_telegram_omits_regret_when_zero():
    # Control leg always produces regret_delta=0.0 — omit to save chars.
    ctx = TradeCloseContext(
        ticket=7, pnl_usd=1.0, magic=19760418, leg_label="Control-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=1.0,
        regime_at_entry="TRENDING", regime_confidence=0.9,
        regret_delta=0.0,
    )
    text = format_trade_close_telegram(ctx)
    assert "Regret" not in text


def test_format_telegram_includes_regret_positive_sign():
    # Positive regret_delta means macro HURT this trade.
    ctx = TradeCloseContext(
        ticket=8, pnl_usd=-10.0, magic=19760428, leg_label="Treatment-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=-0.5,
        regime_at_entry="TRENDING", regime_confidence=0.9,
        regret_delta=12.34,
    )
    text = format_trade_close_telegram(ctx)
    assert "+$12.34" in text


# ---------------------------------------------------------------------------
# Leg mapping (composite-key + legacy fallback)
# ---------------------------------------------------------------------------


def test_leg_by_composite_contains_known_triples():
    # Team-lead 2026-04-20 correction: BTC has no macro/treatment leg.
    assert LEG_BY_COMPOSITE[("XAUUSD", 19760418)] == "Control-XAU"
    assert LEG_BY_COMPOSITE[("XAUUSD", 19760428)] == "Treatment-XAU"
    assert LEG_BY_COMPOSITE[("BTCUSD", 19760419)] == "Control-BTC"
    # Explicitly NO ("BTCUSD", 19760428) — must stay absent.
    assert ("BTCUSD", 19760428) not in LEG_BY_COMPOSITE


def test_legacy_magic_only_map_excludes_treatment():
    # 19760428 needs symbol context — legacy map must not resolve it alone.
    assert 19760428 not in MAGIC_TO_LEG_LEGACY
    assert MAGIC_TO_LEG_LEGACY[19760418] == "Control-XAU"
    assert MAGIC_TO_LEG_LEGACY[19760419] == "Control-BTC"


def test_resolve_leg_label_with_symbol():
    assert resolve_leg_label("XAUUSD", 19760418) == "Control-XAU"
    assert resolve_leg_label("XAUUSD", 19760428) == "Treatment-XAU"
    assert resolve_leg_label("BTCUSD", 19760419) == "Control-BTC"


def test_resolve_leg_label_rejects_btc_treatment_combo():
    # A BTC deal tagged with the XAU treatment magic is an anomaly — must
    # not silently resolve to "Treatment".
    assert resolve_leg_label("BTCUSD", 19760428) == "Unknown"


def test_resolve_leg_label_without_symbol_falls_back():
    # Legacy callers may not have symbol context.
    assert resolve_leg_label(None, 19760418) == "Control-XAU"
    assert resolve_leg_label(None, 19760419) == "Control-BTC"
    # Treatment alone is ambiguous without symbol.
    assert resolve_leg_label(None, 19760428) == "Unknown"


def test_build_context_treatment_leg_resolves_with_path_hint(journal_dir: Path, tmp_path: Path):
    treatment = journal_dir / "journal_macro" / "live_trades.jsonl"
    _write_journal(treatment, [
        {"time": "2026-04-20T14:30", "mt5_ticket": 777, "entry": 100, "sl": 90,
         "direction": "long", "magic": 19760428},
    ])
    ctx = build_trade_close_context(
        ticket=777, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[treatment],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    assert ctx.symbol == "XAUUSD"
    assert ctx.leg_label == "Treatment-XAU"


# ---------------------------------------------------------------------------
# A2 trailing SL fields (Task #8 integration)
# ---------------------------------------------------------------------------


def test_format_telegram_includes_trail_line_when_set():
    ctx = TradeCloseContext(
        ticket=10, pnl_usd=1.0, magic=19760418, leg_label="Control-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=1.0,
        regime_at_entry="TREND_UP", regime_confidence=0.9, regret_delta=None,
        trail_activate_r=0.3, trail_distance_r=0.5,
    )
    text = format_trade_close_telegram(ctx)
    assert "Trail" in text
    assert "0.30R" in text
    assert "0.50R" in text


def test_format_telegram_omits_trail_line_when_none():
    """Pre-A2 closures have trail fields None — line must be absent."""
    ctx = TradeCloseContext(
        ticket=11, pnl_usd=1.0, magic=19760418, leg_label="Control-XAU",
        symbol="XAUUSD", direction="long", entry_price=4000, exit_price=4010,
        sl=3990, tp1=4020, trigger="breakout", rr_realized=1.0,
        regime_at_entry="TREND_UP", regime_confidence=0.9, regret_delta=None,
    )
    text = format_trade_close_telegram(ctx)
    assert "Trail" not in text


def test_build_context_reads_trail_params_from_journal(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 42,
         "entry": 100.0, "sl": 90.0, "direction": "long", "magic": 19760418,
         "trail_activate_r": 0.3, "trail_distance_r": 0.5},
    ])
    ctx = build_trade_close_context(
        ticket=42, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    assert ctx.trail_activate_r == 0.3
    assert ctx.trail_distance_r == 0.5


def test_build_context_trail_defaults_to_none_when_missing(journal_dir: Path, tmp_path: Path):
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 43,
         "entry": 100.0, "sl": 90.0, "direction": "long", "magic": 19760418},
    ])
    ctx = build_trade_close_context(
        ticket=43, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    assert ctx.trail_activate_r is None
    assert ctx.trail_distance_r is None


# ---------------------------------------------------------------------------
# R6 B2 — regret_delta wiring
# ---------------------------------------------------------------------------


def test_build_context_regret_delta_none_without_closure(journal_dir: Path, tmp_path: Path):
    """Pre-R6 callers pass no closure_event — regret_delta stays None."""
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 100,
         "entry": 100.0, "sl": 90.0, "direction": "long", "magic": 19760418},
    ])
    ctx = build_trade_close_context(
        ticket=100, pnl_usd=5.0, exit_price=105.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
    )
    assert ctx.regret_delta is None


def test_build_context_regret_delta_populated_from_closure_event(journal_dir: Path, tmp_path: Path):
    """R6 caller supplies closure_event → regret_delta is computed."""
    journal_path = journal_dir / "journal" / "live_trades.jsonl"
    _write_journal(journal_path, [
        {"time": "2026-04-20T14:30:00+00:00", "mt5_ticket": 101,
         "entry": 100.0, "sl": 90.0, "direction": "long", "magic": 19760418},
    ])
    closure = {
        "event": "trade_reconciled",
        "ticket": 101, "pnl_usd": 66.31,
        "magic": 19760418, "direction": "long",
    }
    ctx = build_trade_close_context(
        ticket=101, pnl_usd=66.31, exit_price=105.0, close_time=None,
        journal_paths=[journal_path],
        structured_log_path=tmp_path / "missing.jsonl",
        closure_event=closure,
    )
    # Control leg: heuristic yields 0.0 — non-None proves the wire worked.
    assert ctx.regret_delta == 0.0
