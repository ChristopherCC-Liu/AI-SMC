"""Enrich a ``trade_closed`` event with the context老板 wants on Telegram.

Round 5 O3: the existing ``alert_critical("trade_closed", ...)`` call in
``scripts/live_demo.py`` only carries ``pnl_usd`` and ``running_daily_pnl``.
That's not enough to judge whether the close matched the strategy's intent.
This module fills in:

- ``entry_price``, ``sl``, ``tp1``, ``trigger``, ``direction`` — looked up from
  the per-leg journal (``data/{SYMBOL}/journal{suffix}/live_trades.jsonl``)
  by MT5 ticket.
- ``rr_realized`` — ``(exit - entry) / (entry - sl)`` with direction sign.
- ``regime_at_entry`` + ``regime_confidence`` — last non-default
  ``ai_regime_classified`` event in ``logs/structured.jsonl`` before the
  entry timestamp.
- ``regret_delta`` — scalar counterfactual PnL difference in USD
  (``no_macro_pnl - actual_pnl``). Populated by
  :func:`smc.monitor.regret_analysis.compute_regret_for_ticket` when the
  caller supplies a ``closure_event`` or ``closures_by_ticket`` map;
  ``None`` otherwise so downstream formatters can omit the line.

Everything here is best-effort: missing data (no matching journal row, no
regime event, I/O error) returns ``None``/fallback values — the trade_closed
message must never block on enrichment.

Public API:
    - :func:`build_trade_close_context` — main entry point
    - :func:`format_trade_close_telegram` — turn the context dict into the
      ≤1000-char plain-text body we push to Telegram
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from smc.monitor.regret_analysis import compute_regret_for_ticket

logger = logging.getLogger(__name__)


# (symbol, magic) → display leg mapping (audit-r4 v5 Option B dual-magic).
# Team-lead 2026-04-20 correction: BTC has no macro/treatment leg — the
# 19760428 magic is only valid on XAUUSD. Keying on the composite catches
# anomalies (e.g. a BTC deal tagged 19760428) instead of silently merging
# them into the XAU-treatment bucket.
LEG_BY_COMPOSITE: dict[tuple[str, int], str] = {
    ("XAUUSD", 19760418): "Control-XAU",
    ("XAUUSD", 19760428): "Treatment-XAU",
    ("BTCUSD", 19760419): "Control-BTC",
}

# Backward-compat shim: some callers / tests look up by magic alone. We
# expose a magic→label helper that returns "Unknown" if BTC-treatment
# would be implied — never silently map 19760428 to just "Treatment".
MAGIC_TO_LEG_LEGACY: dict[int, str] = {
    19760418: "Control-XAU",
    19760419: "Control-BTC",
    # Intentionally NO 19760428 → treatment needs symbol context.
}


def resolve_leg_label(symbol: str | None, magic: int | None) -> str:
    """Resolve (symbol, magic) → human-readable leg label."""
    if symbol and magic is not None:
        label = LEG_BY_COMPOSITE.get((symbol, int(magic)))
        if label:
            return label
    if magic is not None:
        # Fall back to legacy single-magic map (won't resolve 19760428).
        legacy = MAGIC_TO_LEG_LEGACY.get(int(magic))
        if legacy:
            return legacy
    return "Unknown"


@dataclass(frozen=True)
class TradeCloseContext:
    """Immutable enrichment payload for a single closed trade.

    Any field may be ``None`` when source data was missing — downstream code
    must tolerate that (Telegram formatter prints ``—``).
    """
    ticket: int
    pnl_usd: float
    magic: int | None
    leg_label: str
    symbol: str | None
    direction: str | None            # "long" / "short"
    entry_price: float | None
    exit_price: float | None
    sl: float | None
    tp1: float | None
    trigger: str | None
    rr_realized: float | None
    regime_at_entry: str | None
    regime_confidence: float | None
    regret_delta: float | None       # M3 fills this later
    # A2 (alpha-lead Task #8): regime-aware trailing SL parameters active
    # on this ticket at open time. Optional — older journals won't have
    # them, and the EA falls back to its global TrailActivateR / TrailDistanceR
    # inputs when these are absent. Format the telegram body to surface
    # them only when present so old closures don't show a bogus "Trail: — @ —".
    trail_activate_r: float | None = None
    trail_distance_r: float | None = None


# ---------------------------------------------------------------------------
# Journal lookup
# ---------------------------------------------------------------------------


def _scan_journal_for_ticket(
    journal_paths: Iterable[Path],
    ticket: int,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Return ``(row, source_path)`` for the latest journal row matching ticket.

    Scans all candidate paths (Control + Treatment per symbol). A ticket can
    only belong to one leg on the broker side, but we scan both so callers
    don't need to know the mapping upfront. Returns ``(None, None)`` when
    no match — the source_path is returned so callers can infer symbol
    from the path layout (data/XAUUSD/journal_macro/... etc.).
    """
    latest: dict[str, Any] | None = None
    latest_ts: str = ""
    latest_path: Path | None = None
    for path in journal_paths:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.debug("journal read fail %s: %s", path, exc)
            continue
        # Scan tail-first for efficiency (most recent trades matter).
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_ticket = row.get("mt5_ticket")
            if row_ticket != ticket:
                continue
            ts = str(row.get("time") or "")
            if ts >= latest_ts:
                latest = row
                latest_ts = ts
                latest_path = path
            break  # only one match per file (first from tail = most recent)
    return latest, latest_path


# ---------------------------------------------------------------------------
# Structured log lookup
# ---------------------------------------------------------------------------


def _parse_structured_line(line: str) -> dict[str, Any] | None:
    """Parse a ``[SEV] {...json...}`` line from structured.jsonl."""
    line = line.strip()
    if not line:
        return None
    # strip severity prefix "[CRIT] " / "[INFO] " / ...
    if line.startswith("["):
        close = line.find("]")
        if close != -1:
            line = line[close + 1:].strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _find_regime_at(
    structured_log_path: Path,
    entry_ts: datetime,
    *,
    max_scan_lines: int = 4000,
) -> tuple[str | None, float | None]:
    """Return (regime, confidence) from last non-fallback ai_regime_classified
    before ``entry_ts``.

    Scans the tail of ``structured.jsonl`` — we never need to go back farther
    than a few thousand lines because regime events fire every M15 cycle.
    Fallback rows (``source=="default"``) are skipped because they carry no
    real reasoning — we want the real regime call, not the "insufficient data"
    default that fires before first cycle.
    """
    if not structured_log_path.exists():
        return None, None
    try:
        # O(lines) read is fine — structured.jsonl caps at a few 10k lines/day.
        lines = structured_log_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.debug("structured log read fail %s: %s", structured_log_path, exc)
        return None, None

    # Scan tail-first, bounded, for any ai_regime_classified with ts <= entry_ts.
    scanned = 0
    for line in reversed(lines):
        if scanned >= max_scan_lines:
            break
        scanned += 1
        row = _parse_structured_line(line)
        if row is None or row.get("event") != "ai_regime_classified":
            continue
        if row.get("source") == "default":
            continue
        ts_raw = row.get("ts")
        try:
            row_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if row_ts > entry_ts:
            continue
        regime = row.get("regime")
        conf = row.get("confidence")
        return (
            str(regime) if regime is not None else None,
            float(conf) if conf is not None else None,
        )
    return None, None


# ---------------------------------------------------------------------------
# RR math
# ---------------------------------------------------------------------------


def _compute_rr_realized(
    entry: float | None,
    exit_price: float | None,
    sl: float | None,
    direction: str | None,
) -> float | None:
    """RR in risk-units. Positive = profit relative to 1R risk, negative = loss.

    Sign is direction-aware: for longs ``exit > entry`` is positive; for
    shorts ``exit < entry`` is positive.
    Returns ``None`` when inputs are missing or risk is zero.
    """
    if entry is None or exit_price is None or sl is None or direction is None:
        return None
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    raw = exit_price - entry
    if direction == "short":
        raw = -raw
    return round(raw / risk, 2)


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_trade_close_context(
    *,
    ticket: int,
    pnl_usd: float,
    exit_price: float | None,
    close_time: datetime | None,
    journal_paths: Iterable[Path],
    structured_log_path: Path,
    closure_event: dict[str, Any] | None = None,
    closures_by_ticket: dict[int, float] | None = None,
) -> TradeCloseContext:
    """Assemble the full enrichment payload for a closed trade.

    Callers pass what they already know (ticket + pnl from
    :func:`fetch_closed_pnl_since`, exit_price if they can read the deal) and
    this function fills in everything else from the journals + structured
    log. Never raises — on any failure, relevant fields come back ``None``.

    ``closure_event`` and ``closures_by_ticket`` are optional hooks for R6 B2
    ``regret_delta`` computation: when either is provided, we pass them to
    :func:`compute_regret_for_ticket` so the Telegram body can surface a
    ``Regret: ±$X.XX`` line. Both ``None`` preserves the pre-R6 behaviour of
    leaving ``regret_delta=None``.
    """
    row_match, source_path = _scan_journal_for_ticket(journal_paths, ticket)
    row = row_match or {}

    entry_price = _safe_float(row.get("entry"))
    sl = _safe_float(row.get("sl"))
    tp1 = _safe_float(row.get("tp1"))
    direction = str(row.get("direction")) if row.get("direction") else None
    trigger = str(row.get("trigger")) if row.get("trigger") else None
    magic = row.get("magic")
    if magic is not None:
        try:
            magic = int(magic)
        except (TypeError, ValueError):
            magic = None
    # Symbol is inferred from the journal path (data/XAUUSD/journal/... etc.);
    # this avoids trusting a broker-specific symbol string in the deal row.
    symbol = _infer_symbol_from_path(source_path) if source_path else None
    leg_label = resolve_leg_label(symbol, magic)

    entry_ts_str = row.get("time")
    regime = None
    regime_conf = None
    if entry_ts_str:
        try:
            entry_ts = datetime.fromisoformat(str(entry_ts_str).replace("Z", "+00:00"))
            regime, regime_conf = _find_regime_at(structured_log_path, entry_ts)
        except ValueError:
            pass

    rr_realized = _compute_rr_realized(entry_price, exit_price, sl, direction)

    # A2 (Task #8): trailing SL preset at open time. alpha-lead writes
    # these into the journal per-trade once /signal carries regime presets;
    # older rows return None and the formatter omits the line.
    trail_activate_r = _safe_float(row.get("trail_activate_r"))
    trail_distance_r = _safe_float(row.get("trail_distance_r"))

    regret_delta = _resolve_regret_delta(
        ticket=ticket,
        pnl_usd=pnl_usd,
        magic=magic,
        direction=direction,
        closure_event=closure_event,
        closures_by_ticket=closures_by_ticket,
    )

    return TradeCloseContext(
        ticket=ticket,
        pnl_usd=round(float(pnl_usd), 2),
        magic=magic,
        leg_label=leg_label,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        sl=sl,
        tp1=tp1,
        trigger=trigger,
        rr_realized=rr_realized,
        regime_at_entry=regime,
        regime_confidence=regime_conf,
        regret_delta=regret_delta,
        trail_activate_r=trail_activate_r,
        trail_distance_r=trail_distance_r,
    )


def _resolve_regret_delta(
    *,
    ticket: int,
    pnl_usd: float,
    magic: int | None,
    direction: str | None,
    closure_event: dict[str, Any] | None,
    closures_by_ticket: dict[int, float] | None,
) -> float | None:
    """Call compute_regret_for_ticket with best-effort inputs.

    When the caller didn't pass a closure event, synthesise a minimal one
    from the fields we already know so downstream heuristics still fire.
    Returns ``None`` if nothing useful can be computed.
    """
    if closure_event is None and closures_by_ticket is None:
        return None
    closure = closure_event if closure_event is not None else {
        "event": "trade_reconciled",
        "ticket": ticket,
        "pnl_usd": pnl_usd,
        "magic": magic,
        "direction": direction,
    }
    try:
        return compute_regret_for_ticket(
            ticket=ticket,
            closure=closure,
            closures_by_ticket=closures_by_ticket,
        )
    except Exception as exc:  # never let enrichment block on regret failure
        logger.debug("regret compute failed for ticket %s: %s", ticket, exc)
        return None


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _infer_symbol_from_path(path: Path | str) -> str | None:
    """Infer "XAUUSD" / "BTCUSD" from a path like data/XAUUSD/journal/...jsonl."""
    try:
        parts = Path(path).parts
        for p in parts:
            if p in ("XAUUSD", "BTCUSD"):
                return p
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Telegram formatter
# ---------------------------------------------------------------------------


def format_trade_close_telegram(ctx: TradeCloseContext) -> str:
    """Format a TradeCloseContext into ≤1000-char plain text for Telegram.

    Format (from O3 spec):
        [CLOSED] XAUUSD BUY Control
        Entry: 4743.51 → Exit: 4818.34
        PnL: +$74.83 | RR: +2.88R
        Regime: TRANSITION (conf 0.72)
        Trigger: support_bounce
    """
    sym = ctx.symbol or "?"
    dir_tag = "BUY" if ctx.direction == "long" else "SELL" if ctx.direction == "short" else "?"
    leg = ctx.leg_label or "?"

    header = f"[CLOSED] {sym} {dir_tag} {leg}"

    entry_s = _fmt_price(ctx.entry_price)
    exit_s = _fmt_price(ctx.exit_price)
    prices = f"Entry: {entry_s} -> Exit: {exit_s}"

    if ctx.pnl_usd >= 0:
        pnl_s = f"+${ctx.pnl_usd:.2f}"
    else:
        pnl_s = f"-${abs(ctx.pnl_usd):.2f}"
    rr_s = _fmt_rr(ctx.rr_realized)
    pnl_line = f"PnL: {pnl_s} | RR: {rr_s}"

    regime_s = ctx.regime_at_entry or "unknown"
    conf_s = f" (conf {ctx.regime_confidence:.2f})" if ctx.regime_confidence is not None else ""
    regime_line = f"Regime: {regime_s}{conf_s}"

    trigger_line = f"Trigger: {ctx.trigger or 'unknown'}"

    lines = [header, prices, pnl_line, regime_line, trigger_line]

    # A2 Task #8: surface trailing preset only when the journal row carried
    # it (post-A2 deploys). Older closures omit this line instead of
    # rendering "Trail: — @ —".
    if ctx.trail_activate_r is not None and ctx.trail_distance_r is not None:
        lines.append(
            f"Trail: activate {ctx.trail_activate_r:.2f}R @ distance "
            f"{ctx.trail_distance_r:.2f}R"
        )

    # R6 B2: regret_delta in USD ("no-macro counterfactual minus actual").
    # Omit the line when exactly 0 (Control leg or neutral Treatment) to keep
    # the Telegram body terse — a $0.00 line carries no signal.
    if ctx.regret_delta is not None and abs(ctx.regret_delta) >= 0.01:
        sign_char = "+" if ctx.regret_delta >= 0 else "-"
        lines.append(
            f"Regret: {sign_char}${abs(ctx.regret_delta):.2f} vs no-macro counterfactual"
        )

    msg = "\n".join(lines)
    return msg[:1000]


def _fmt_price(p: float | None) -> str:
    if p is None:
        return "—"
    # XAU in 2 decimals, BTC in 1 — 2 dp is sufficient for both display-wise.
    return f"{p:.2f}"


def _fmt_rr(rr: float | None) -> str:
    if rr is None:
        return "—"
    sign = "+" if rr >= 0 else ""
    return f"{sign}{rr:.2f}R"


__all__ = [
    "TradeCloseContext",
    "LEG_BY_COMPOSITE",
    "MAGIC_TO_LEG_LEGACY",
    "resolve_leg_label",
    "build_trade_close_context",
    "format_trade_close_telegram",
]
