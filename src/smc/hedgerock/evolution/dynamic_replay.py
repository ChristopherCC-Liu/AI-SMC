"""XAUUSD-only dynamic replay against the real rule_engine.

Read-only Tier-1 unseal. This module imports ``rule_engine``,
``market_state``, ``regime_classifier_v2``, ``decision_server`` and
``transition_lock`` to drive a true bar-by-bar replay; it NEVER calls
the live ``/signal`` endpoint, NEVER mutates EAState stores, NEVER
writes under ``policy_registry/``.

Honesty contract:

  * Closed-bar features only — every per-bar decision uses bars
    strictly STRICTLY before bar i. Bar i's high/low/close cannot
    influence the decision at i.
  * Live gates honoured — when ``derive_envelope_params`` returns
    ``mode == "observe"`` or ``mode == "halt"``, no entry. When
    ``cooldown_until`` is in the future relative to the bar's close
    timestamp, no entry. When the transition lock is active, no
    entry. When ``lot_factor == 0``, no entry (counted as a veto).
  * XAUUSD-only by hard assertion.
  * No lookahead even at exit: trades are simulated using only the
    NEXT bar's high / low / close.

All metrics required by the orchestrator's ``DynamicReplayStats`` are
returned in the ``ReplayResult`` dataclass.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence


SYMBOL = "XAUUSD"

# Conservative defaults — operator can override via replay_config.
_DEFAULT_LOOKBACK_H1 = 24
_DEFAULT_LOOKBACK_H4 = 12
_DEFAULT_HOLD_BARS = 8
_DEFAULT_INIT_EQUITY = 10_000.0
_POINT = 0.1  # XAUUSD point convention used in the live EA / decision server


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    """Frozen output. Maps 1:1 onto orchestrator's DynamicReplayStats."""

    available: bool
    reason: str
    pnl_pct: float | None
    max_drawdown_pct: float | None
    sharpe_annualised: float | None
    trade_count: int | None
    entry_count: int | None
    exit_count: int | None
    win_rate: float | None
    veto_reasons: dict[str, int] | None
    cooldown_reasons: dict[str, int] | None
    observe_reasons: dict[str, int] | None
    halt_reasons: dict[str, int] | None
    risk_tier_distribution: dict[str, int] | None
    lot_factor_distribution: dict[str, int] | None
    transition_lock_states: dict[str, int] | None
    transition_lock_events: int | None
    cooldown_events: int | None


def _empty_result(reason: str) -> ReplayResult:
    return ReplayResult(
        available=False, reason=reason,
        pnl_pct=None, max_drawdown_pct=None, sharpe_annualised=None,
        trade_count=None, entry_count=None, exit_count=None, win_rate=None,
        veto_reasons=None, cooldown_reasons=None,
        observe_reasons=None, halt_reasons=None,
        risk_tier_distribution=None, lot_factor_distribution=None,
        transition_lock_states=None,
        transition_lock_events=None, cooldown_events=None,
    )


# ---------------------------------------------------------------------------
# Closed-bar feature extraction. Reuses the simple shape from
# phase_d_walk_forward._features_from_frames; intentionally NOT
# importing it to avoid coupling to a stub that may evolve.
# ---------------------------------------------------------------------------


def _closed_bar_features(
    *,
    h1_window: Sequence[Mapping[str, float]],
    h4_window: Sequence[Mapping[str, float]],
) -> tuple[float, int, int, int]:
    """Return (volatility_rank, hh_count, ll_count, h4_trend_bars)
    using ONLY the bars in the windows. Caller MUST exclude bar i
    before invoking — this function does not police that."""
    if not h1_window:
        return 0.5, 0, 0, 0
    closes = [float(b["close"]) for b in h1_window]
    highs = [float(b["high"]) for b in h1_window]
    lows = [float(b["low"]) for b in h1_window]
    if len(closes) < 4:
        return 0.5, 0, 0, 0

    recent_h = highs[-24:] if len(highs) >= 24 else highs
    recent_l = lows[-24:] if len(lows) >= 24 else lows
    recent = max(recent_h) - min(recent_l)
    full = max(highs) - min(lows) if max(highs) > min(lows) else 1.0
    vol_rank = max(0.0, min(1.0, recent / full)) if full > 0 else 0.5

    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])

    h4_trend = 0
    if h4_window:
        h4_closes = [float(b["close"]) for b in h4_window]
        if h4_closes:
            mid = sum(h4_closes) / len(h4_closes)
            for c in reversed(h4_closes):
                if (c >= mid and h4_closes[-1] >= mid) or (
                    c < mid and h4_closes[-1] < mid
                ):
                    h4_trend += 1
                else:
                    break
    return vol_rank, hh, ll, h4_trend


def _parse_ts(value: Any) -> datetime:
    """Tolerant ts parser — accepts ISO string, datetime, or None."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        # Strip trailing 'Z' if present.
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return datetime.now(timezone.utc)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Trade simulation primitive — closed-bar only.
# ---------------------------------------------------------------------------


@dataclass
class _OpenTrade:
    side: str  # "long" only for now (HedgeRock range mode)
    entry_ts: datetime
    entry_price: float
    tp_pts: int
    sl_pts: int
    bars_held: int = 0


def _simulate_one_bar_exit(
    *, trade: _OpenTrade, next_bar: Mapping[str, float],
    max_hold_bars: int,
) -> tuple[bool, float | None, str]:
    """Decide whether the trade closes on ``next_bar``. Returns
    (closed, pnl_quote, exit_reason). Closed-bar only — relies on the
    next bar's HIGH / LOW / CLOSE which are observable AFTER the bar
    has closed."""
    nh = float(next_bar["high"])
    nl = float(next_bar["low"])
    nc = float(next_bar["close"])
    tp_price = trade.entry_price + trade.tp_pts * _POINT
    sl_price = trade.entry_price - trade.sl_pts * _POINT

    # Conservative tie-break: if both TP and SL are within range, take SL.
    if nl <= sl_price:
        return True, (sl_price - trade.entry_price), "stop_loss"
    if nh >= tp_price:
        return True, (tp_price - trade.entry_price), "take_profit"

    trade.bars_held += 1
    if trade.bars_held >= max_hold_bars:
        return True, (nc - trade.entry_price), "time_exit"
    return False, None, ""


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def replay_via_walk_forward(
    *,
    lake_root: Any,
    instrument: str = SYMBOL,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ReplayResult:
    """Drive the official ``phase_d_walk_forward.run_walk_forward``
    harness against the real lake and project its TradeMetrics +
    envelope_log onto the 13 ``DynamicReplayStats`` fields.

    Returns ``ReplayResult(available=False, ...)`` when imports fail or
    the harness emits no useful data — gracefully falls back so the
    orchestrator can still drive the closed-bar replay path."""
    if instrument != SYMBOL:
        return _empty_result(f"walk-forward is XAUUSD-only; got {instrument!r}")
    try:
        from smc.data.lake import ForexDataLake
        from smc.hedgerock.phase_d_walk_forward import (
            WalkForwardConfig, run_walk_forward,
        )
    except Exception as e:  # pragma: no cover — defensive
        return _empty_result(f"walk_forward imports failed: {e!r}")
    try:
        lake = ForexDataLake(lake_root)
        cfg = WalkForwardConfig(
            instrument=instrument,
            start=start or datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=end or datetime(2024, 12, 31, tzinfo=timezone.utc),
        )
        result = run_walk_forward(cfg, lake)
    except Exception as e:
        return _empty_result(f"run_walk_forward raised: {e!r}")

    tm = result.dynamic_metrics
    log = list(result.envelope_log or [])

    veto: dict[str, int] = {}
    cooldown: dict[str, int] = {}
    observe: dict[str, int] = {}
    halt: dict[str, int] = {}
    risk_tier: dict[str, int] = {}
    lot_factor: dict[str, int] = {}
    transition_lock_states: dict[str, int] = {}

    for row in log:
        mode = str(row.get("mode") or row.get("effective_mode") or "")
        reason = str(row.get("reason") or row.get("decision_reason") or "")
        if mode == "halt":
            halt[reason or "halt"] = halt.get(reason or "halt", 0) + 1
        elif mode == "observe":
            observe[reason or "observe"] = observe.get(reason or "observe", 0) + 1
        rt = row.get("risk_tier")
        if rt is not None:
            risk_tier[str(rt)] = risk_tier.get(str(rt), 0) + 1
        lf = row.get("lot_factor")
        if lf is not None:
            key = f"{float(lf):g}"
            lot_factor[key] = lot_factor.get(key, 0) + 1
        if row.get("transition_lock_active"):
            transition_lock_states["locked"] = (
                transition_lock_states.get("locked", 0) + 1
            )
        else:
            transition_lock_states["unlocked"] = (
                transition_lock_states.get("unlocked", 0) + 1
            )

    # TradeMetrics → 13 fields.
    return ReplayResult(
        available=True,
        reason=(
            "phase_d_walk_forward.run_walk_forward harness over real "
            "XAUUSD lake (envelope_log + TradeMetrics)"
        ),
        pnl_pct=round(float(tm.total_return_pct or 0.0), 4),
        max_drawdown_pct=round(float(tm.max_dd_pct or 0.0), 4),
        sharpe_annualised=0.0,  # TradeMetrics doesn't expose Sharpe; left at 0.
        trade_count=int(tm.n_trades or 0),
        entry_count=int(tm.n_trades or 0),
        exit_count=int(tm.n_trades or 0),
        win_rate=round(float(tm.win_rate or 0.0), 4),
        veto_reasons=veto,
        cooldown_reasons=cooldown,
        observe_reasons=observe,
        halt_reasons=halt,
        risk_tier_distribution=risk_tier,
        lot_factor_distribution=lot_factor,
        transition_lock_states=transition_lock_states,
        transition_lock_events=int(tm.transition_lock_bars or 0),
        cooldown_events=int(tm.cooldown_trigger_count or 0),
    )


def replay_xauusd_h1(
    *,
    symbol: str,
    h1_bars: Sequence[Mapping[str, Any]],
    h4_bars: Sequence[Mapping[str, Any]] | None = None,
    d1_bars: Sequence[Mapping[str, Any]] | None = None,
    h1_lookback: int = _DEFAULT_LOOKBACK_H1,
    h4_lookback: int = _DEFAULT_LOOKBACK_H4,
    max_hold_bars: int = _DEFAULT_HOLD_BARS,
) -> ReplayResult:
    """Replay HedgeRock decisions over closed H1 bars.

    Returns ``ReplayResult(available=False, ...)`` when imports of
    rule_engine + classifier fail (graceful degradation), or when the
    bar count is too small to warm up.
    """
    if symbol != SYMBOL:
        return _empty_result(f"replay is XAUUSD-only; got {symbol!r}")
    if not h1_bars or len(h1_bars) <= h1_lookback + 2:
        return _empty_result(
            f"insufficient H1 bars ({len(h1_bars) if h1_bars else 0}) "
            f"for lookback={h1_lookback}"
        )

    # Read-only Tier-1 imports happen here so callers without these
    # deps can still construct the orchestrator.
    try:
        from smc.hedgerock.decision_server import MarketFeatures
        from smc.hedgerock.market_state import aggregate_market_state
        from smc.hedgerock.regime_classifier_v2 import classify_regime_v2
        from smc.hedgerock.rule_engine import derive_envelope_params
        from smc.hedgerock.transition_lock import compute_lock_until_v2
    except Exception as e:  # pragma: no cover — defensive
        return _empty_result(f"rule_engine imports failed: {e!r}")

    # Bucket H4 bars by ts for trailing-window construction.
    h4_sorted = sorted(
        list(h4_bars or []),
        key=lambda b: _parse_ts(b.get("ts")),
    )
    h4_ts = [_parse_ts(b.get("ts")) for b in h4_sorted]

    veto_reasons: Counter = Counter()
    cooldown_reasons: Counter = Counter()
    observe_reasons: Counter = Counter()
    halt_reasons: Counter = Counter()
    risk_tier_distribution: Counter = Counter()
    lot_factor_distribution: Counter = Counter()
    transition_lock_states: Counter = Counter()

    transition_lock_events = 0
    cooldown_events = 0

    trades_pnl_quote: list[float] = []
    entry_count = 0
    exit_count = 0
    open_trade: _OpenTrade | None = None

    cooldown_until: datetime | None = None
    transition_lock_until: datetime | None = None
    prev_regime: str | None = None
    prev_envelope = None

    init_price = float(h1_bars[h1_lookback].get("close", 1.0)) or 1.0
    equity = _DEFAULT_INIT_EQUITY
    peak = _DEFAULT_INIT_EQUITY
    max_dd = 0.0
    realised_returns: list[float] = []

    n = len(h1_bars)
    for i in range(h1_lookback, n - 1):
        bar = h1_bars[i]
        next_bar = h1_bars[i + 1]
        ts = _parse_ts(bar.get("ts"))

        # Closed-bar windows — bars strictly BEFORE i.
        h1_window = list(h1_bars[max(0, i - h1_lookback): i])
        h4_window: list[Mapping[str, Any]] = []
        if h4_ts:
            cut = ts
            j = len(h4_ts) - 1
            while j >= 0 and h4_ts[j] >= cut:
                j -= 1
            if j >= 0:
                lo = max(0, j + 1 - h4_lookback)
                h4_window = h4_sorted[lo: j + 1]

        # 1) Closed-bar features.
        vol_rank, hh, ll, h4_trend = _closed_bar_features(
            h1_window=h1_window, h4_window=h4_window,
        )

        # 2) Regime classification (closed-bar features only).
        ra = classify_regime_v2(
            volatility_rank=vol_rank,
            h4_trend_bars=h4_trend,
            hh_count=hh,
            ll_count=ll,
        )

        # 3) MarketFeatures + MarketState — no EAState in replay.
        features = MarketFeatures(
            volatility_rank=vol_rank, hh_count=hh, ll_count=ll,
            h4_trend_bars=h4_trend, regime=ra.regime,
        )
        state = aggregate_market_state(
            symbol=SYMBOL, now=ts,
            features=features, regime_assessment=ra,
            ea_state=None, ea_state_recorded_at=None,
        )

        # 4) Derive envelope params — REAL rule_engine call.
        try:
            params = derive_envelope_params(state, prev_envelope)
        except Exception as e:  # pragma: no cover
            veto_reasons[f"derive_failed:{type(e).__name__}"] += 1
            continue

        # 5) Maintain transition lock from regime transition.
        if prev_regime is not None and prev_regime != ra.regime:
            try:
                transition_lock_until = compute_lock_until_v2(
                    now=ts, prev_regime=prev_regime,
                    new_regime=ra.regime,
                )
            except Exception:
                transition_lock_until = None
            if transition_lock_until is not None:
                transition_lock_events += 1
        prev_regime = ra.regime

        # 6) Track the params distributions regardless of gating —
        #    operator wants to SEE the full per-bar shape.
        risk_tier_distribution[str(params.risk_tier)] += 1
        lot_factor_distribution[f"{params.lot_factor:g}"] += 1

        lock_active = (
            transition_lock_until is not None
            and ts < transition_lock_until
        )
        cooldown_active = (
            params.cooldown_until is not None
            and ts < params.cooldown_until
        )
        if lock_active:
            transition_lock_states["locked"] += 1
        else:
            transition_lock_states["unlocked"] += 1

        # 7) Resolve any open trade against the next bar.
        if open_trade is not None:
            closed, pnl_quote, exit_reason = _simulate_one_bar_exit(
                trade=open_trade, next_bar=next_bar,
                max_hold_bars=max_hold_bars,
            )
            if closed:
                trades_pnl_quote.append(pnl_quote or 0.0)
                # Convert to a return on init_price for sharpe / dd.
                ret = (pnl_quote or 0.0) / init_price
                realised_returns.append(ret)
                equity *= (1.0 + ret)
                peak = max(peak, equity)
                dd = (equity / peak) - 1.0
                max_dd = min(max_dd, dd)
                exit_count += 1
                open_trade = None

        # 8) Live-gate decisions for new entries.
        if open_trade is not None:
            # already in a trade; do not double-up.
            prev_envelope = None  # _apply_cooldown_carryover ignores None
            continue

        if str(params.mode) == "halt":
            halt_reasons[params.reason] += 1
            continue
        if str(params.mode) == "observe":
            observe_reasons[params.reason] += 1
            continue
        if cooldown_active:
            cooldown_events += 1
            cooldown_reasons[params.reason or "cooldown_carryover"] += 1
            continue
        if lock_active:
            veto_reasons["transition_lock_active"] += 1
            continue
        if params.lot_factor <= 0.0:
            veto_reasons["lot_factor_zero"] += 1
            continue

        # 9) Open a new long trade at next_bar.open (no lookahead — we
        #    decided at bar i's close, entry at bar i+1's open).
        entry_price = float(next_bar["open"])
        open_trade = _OpenTrade(
            side="long",
            entry_ts=_parse_ts(next_bar.get("ts")),
            entry_price=entry_price,
            tp_pts=int(params.takeprofit_points),
            sl_pts=int(params.stoploss_points),
        )
        entry_count += 1

    # Aggregate metrics.
    trade_count = len(trades_pnl_quote)
    pnl_pct = ((equity / _DEFAULT_INIT_EQUITY) - 1.0) * 100.0
    max_drawdown_pct = max_dd * 100.0
    win_rate = (
        sum(1 for p in trades_pnl_quote if p > 0) / trade_count
        if trade_count > 0 else 0.0
    )

    if len(realised_returns) >= 2:
        mu = statistics.mean(realised_returns)
        sigma = statistics.pstdev(realised_returns)
        # Approx annualisation factor: trades-per-year for H1 ≈ entries
        # times bars-per-year/total-bars. Fall back to the H1 bar
        # constant when trades are sparse.
        trades_per_year = max(
            1.0, len(realised_returns) * (24 * 365) / max(1, n),
        )
        sharpe = (
            (mu / sigma) * math.sqrt(trades_per_year) if sigma > 0 else 0.0
        )
    else:
        sharpe = 0.0

    return ReplayResult(
        available=True,
        reason="rule_engine-backed walk-forward replay over closed H1 bars",
        pnl_pct=round(pnl_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        sharpe_annualised=round(sharpe, 4),
        trade_count=trade_count,
        entry_count=entry_count,
        exit_count=exit_count,
        win_rate=round(win_rate, 4),
        veto_reasons=dict(veto_reasons),
        cooldown_reasons=dict(cooldown_reasons),
        observe_reasons=dict(observe_reasons),
        halt_reasons=dict(halt_reasons),
        risk_tier_distribution=dict(risk_tier_distribution),
        lot_factor_distribution=dict(lot_factor_distribution),
        transition_lock_states=dict(transition_lock_states),
        transition_lock_events=transition_lock_events,
        cooldown_events=cooldown_events,
    )
