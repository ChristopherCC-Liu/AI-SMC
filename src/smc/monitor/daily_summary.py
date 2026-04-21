"""R5 M4: compose the short plain-text daily summary for Telegram.

Pure function module — no I/O.  The scheduled task (``scripts/send_daily_summary.py``)
feeds a multi-symbol digest in, gets back a single string (~500 chars),
and hands it to ``alert_critical(send_telegram=True)``.

Message layout (kept under Telegram's 4096-char limit; typically < 600 chars)::

    AI-SMC Daily Summary — 2026-04-20
    Trades: 8  Net: -$23.03  PF: 0.86
    Control: 4 trades, +$31.84, PF 1.30
    Treatment: 4 trades, -$54.87, PF 0.48
    Δ(C-T): +$86.71
    Top regime: TRANSITION (2)
    AI: 2 cycles, p50 1500ms, $0.12
    Handle resets: 1
"""
from __future__ import annotations

from typing import Any


__all__ = ["format_daily_summary", "MAX_TELEGRAM_CHARS"]


MAX_TELEGRAM_CHARS = 4000  # Telegram's limit is 4096; leave buffer for suffix.


def format_daily_summary(digest: dict[str, Any]) -> str:
    """Render the digest as a compact plain-text Telegram message.

    Accepts the shape returned by ``digest_report.build_multi_symbol_digest``.
    Missing optional sections render as a short "n/a" line; the function
    never raises.
    """
    date_str = digest.get("date", "?")
    lines: list[str] = [f"AI-SMC Daily Summary - {date_str}"]

    per_leg = digest.get("per_leg") or []
    total_trades = sum(int(l.get("trades", 0)) for l in per_leg)
    total_pnl = sum(float(l.get("total_pnl_usd", 0) or 0) for l in per_leg)
    gross_wins = sum(
        float(l.get("avg_win_usd") or 0) * int(l.get("wins", 0))
        for l in per_leg if l.get("avg_win_usd") is not None
    )
    gross_losses = sum(
        abs(float(l.get("avg_loss_usd") or 0)) * int(l.get("losses", 0))
        for l in per_leg if l.get("avg_loss_usd") is not None
    )
    pf = (gross_wins / gross_losses) if gross_losses > 0 else None

    lines.append(
        f"Trades: {total_trades}  Net: {_fmt_usd(total_pnl)}  PF: {_fmt_num(pf, 2)}"
    )

    # Per-leg rows — skip legs with 0 trades to keep message tight.
    control_pnl = None
    treatment_pnl = None
    for leg in per_leg:
        name = str(leg.get("leg", ""))
        trades = int(leg.get("trades", 0))
        if trades == 0:
            continue
        pnl = float(leg.get("total_pnl_usd", 0) or 0)
        lines.append(
            f"{_short_leg(name)}: {trades} trades, {_fmt_usd(pnl)}, PF {_fmt_num(leg.get('profit_factor'), 2)}"
        )
        if name.endswith(":control") and "XAUUSD" in name:
            control_pnl = pnl
        elif name.endswith(":treatment") and "XAUUSD" in name:
            treatment_pnl = pnl

    if control_pnl is not None and treatment_pnl is not None:
        lines.append(f"d(C-T): {_fmt_usd(control_pnl - treatment_pnl)}")

    # Regime — top bucket
    dist = digest.get("regime_distribution") or {}
    if dist:
        top_name, top_count = max(dist.items(), key=lambda kv: kv[1])
        if top_count > 0:
            lines.append(f"Top regime: {top_name} ({top_count})")

    # AI debate
    ai = digest.get("ai_debate") or {}
    cycles = ai.get("cycles_ran", 0)
    if cycles:
        p50 = ai.get("p50_elapsed_ms")
        cost = ai.get("total_cost_usd", 0)
        p50_str = f"p50 {int(p50)}ms" if p50 is not None else "p50 n/a"
        lines.append(f"AI: {cycles} cycles, {p50_str}, ${cost:.2f}")

    # Stability
    handles = digest.get("handle_resets", 0)
    lines.append(f"Handle resets: {handles}")

    # Warnings (cap at 3)
    warnings_flat: list[str] = []
    for sym, blob in (digest.get("per_symbol") or {}).items():
        for w in blob.get("warnings", []) or []:
            warnings_flat.append(f"{sym}:{w}")
    if warnings_flat:
        lines.append("Warnings: " + ", ".join(warnings_flat[:3]))

    msg = "\n".join(lines)
    if len(msg) > MAX_TELEGRAM_CHARS:
        msg = msg[: MAX_TELEGRAM_CHARS - 3] + "..."
    return msg


def _short_leg(name: str) -> str:
    """Trim 'XAUUSD:control' -> 'XAU control', 'BTCUSD:control' -> 'BTC control'."""
    if ":" not in name:
        return name
    sym, rest = name.split(":", 1)
    short = {"XAUUSD": "XAU", "BTCUSD": "BTC"}.get(sym, sym)
    return f"{short} {rest}"


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{'+' if value >= 0 else '-'}${abs(value):.2f}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"
