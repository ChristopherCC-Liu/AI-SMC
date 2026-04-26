"""Range detection and mean-reversion setup generation for Mode B (ranging).

Detects horizontal consolidation ranges on H1 using two methods:
  - Method A: OB boundaries (highest bearish OB.high + lowest bullish OB.low)
  - Method B: Swing extremes fallback (max swing high + min swing low, last 50 bars)

Generates support_bounce / resistance_rejection setups when price is near
a boundary, confirmed by M15 CHoCH in the corresponding zone.

SL is ATR-adaptive (reuses entry_trigger._compute_sl_buffer).
TP conservative = midpoint, TP aggressive = opposite boundary minus 10%.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import polars as pl

from smc.instruments.types import InstrumentConfig
from smc.monitor.state_io import atomic_write_json, load_json
from smc.smc_core.constants import XAUUSD_POINT_SIZE  # retained; Stage 6 will deprecate
from smc.smc_core.types import OrderBlock, SMCSnapshot, SwingPoint
from smc.strategy.entry_trigger import _compute_sl_buffer, _find_choch_in_zone
from smc.strategy.range_types import RangeBounds, RangeSetup
from smc.strategy.types import BiasDirection, TradeZone

if TYPE_CHECKING:
    from smc.ai.models import AIRegimeAssessment

_log = logging.getLogger(__name__)

__all__ = [
    "RangeTrader",
    "check_range_guards",
    "check_bounds_only_guards",
    "get_last_guards_diagnostic",
    "_min_range_width_resolved",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SWING_POINTS_FOR_RANGE = 2
_SWING_LOOKBACK_BARS = 50
_OB_BOUNDARY_CONFIDENCE = 0.8
_SWING_EXTREME_CONFIDENCE = 0.6
_DONCHIAN_CONFIDENCE = 0.5  # Round 4.5: lowest trust — pure statistical fallback
# Round 4.6-A: lookback 24 → 48 (2 trading days). Asian single-session window
# too narrow to form detectable channel on low-volatility days — widen to span
# Asian + prior London/NY session for more stable upper/lower bounds.
_DONCHIAN_LOOKBACK_BARS = 48

# Round 4.6-B: session-aware Guard 1 (width) & Guard 4 (duration) thresholds.
# Asian core market 低波动 — H1 48-bar channel 实测 400-700 pts 居多,
# 旧 800/12 threshold 把 Asian 全 reject. Downstream CircuitBreaker +
# RangeQuotaTracker 仍作 risk 兜底.
_ASIAN_SESSIONS: frozenset[str] = frozenset(
    {"ASIAN_CORE", "ASIAN_LONDON_TRANSITION"}
)
_GUARD_WIDTH_MIN_DEFAULT = 800.0
_GUARD_WIDTH_MIN_ASIAN = 400.0
_GUARD_DURATION_MIN_DEFAULT = 12
_GUARD_DURATION_MIN_ASIAN = 8

# Round 4.6-F (USER DIRECTIVE): Asian 专用 boundary_pct.
# 用户期望 5-10 笔/天; 48-bar Donchian 下 15% band 自然到达概率极低.
# Asian 30% 让 price 在 range 中 60% 内就触发 near_lower/upper.
# 风险: PF 可能 <1, CircuitBreaker + RangeQuota 仍兜底.
#
# Round 4.6-I (USER DIRECTIVE 延续): UTC 08:00 cycle 显示 London 切换后 15%
# 过严 (price $4784 距 lower $4768 差 $16.38, 15% band 仅 $10.60 不触发).
# 扩展到所有 ranging sessions 统一 30%. default 0.15 保留给未知 session.
_BOUNDARY_PCT_DEFAULT = 0.15
_BOUNDARY_PCT_ASIAN = 0.30  # retained for naming symmetry / history
_BOUNDARY_PCT_WIDE = 0.30

_WIDE_BAND_SESSIONS: frozenset[str] = frozenset(
    {
        "ASIAN_CORE",
        "ASIAN_LONDON_TRANSITION",
        "LONDON",
        "LONDON/NY OVERLAP",
        "NEW YORK",
        "LATE NY",
    }
)
_SECONDS_PER_H1_BAR = 3600

# ---------------------------------------------------------------------------
# Round 9 P0 — AI regime / D1 trend filter constants
# ---------------------------------------------------------------------------

# Minimum AIRegimeAssessment.confidence required for an AI-driven gate to fire.
# Below this, the assessment is treated as informational only (fall through).
# Aligned with mode_router._TRANSITION_TREND_DIR_CONF and SMCConfig
# `ai_regime_trust_threshold` so all AI-aware gates use a single floor.
_AI_REGIME_GATE_CONF_FLOOR: float = 0.6

# P0-A: D1 SMA50 trend-filter thresholds. Mirror of regime_classifier
# `_SLOPE_TREND_MIN_ABS = 0.05` so detection logic stays consistent across
# modules; the close-vs-SMA50 floor of 1% picked from the 2026-04-21 Tuesday
# regression fixture (D1 close 2.1% below SMA50 should block).
_TREND_FILTER_SLOPE_THRESHOLD: float = 0.05  # %/bar — must match _SLOPE_TREND_MIN_ABS
_TREND_FILTER_CLOSE_THRESHOLD: float = 1.0   # % — close vs SMA50 distance
_TREND_FILTER_MIN_BARS: int = 55             # need 50 SMA + 5 prior bars

# P0-B: trend regimes that block opposite range direction; ATH_BREAKOUT
# is treated as a strong-uptrend variant (blocks shorts).
_BLOCK_LONG_REGIMES: frozenset[str] = frozenset({"TREND_DOWN"})
_BLOCK_SHORT_REGIMES: frozenset[str] = frozenset({"TREND_UP", "ATH_BREAKOUT"})

# P0-B: TRANSITION regime tightens RR floor from 1.2 → 2.0 to demand
# a higher reward when regime conviction is mid (transition is genuinely
# ambiguous; the only setups worth taking must have outsized payoff).
_TRANSITION_MIN_RR_RATIO: float = 2.0

# P0-C: regimes that invalidate any detected Donchian/OB/swing range.
# Same roster as the union of _BLOCK_LONG_REGIMES + _BLOCK_SHORT_REGIMES.
_RANGE_INVALIDATING_REGIMES: frozenset[str] = (
    _BLOCK_LONG_REGIMES | _BLOCK_SHORT_REGIMES
)

# R10 P2.2 — ai_direction entry-veto threshold. When the DirectionEngine
# signals a direction with confidence at/above this floor AND the proposed
# setup direction opposes it, _build_setup rejects the setup. Tagged
# ``_PHASE2_CALIBRATION_PENDING`` until live-bake gate_funnel.json validates.
_AI_DIRECTION_ENTRY_GATE_CONF_FLOOR: float = 0.55  # _PHASE2_CALIBRATION_PENDING


def _ai_direction_opposes(ai_direction: str | None, setup_direction: str) -> bool:
    """True iff the AI directional view directly contradicts the setup direction.

    ``neutral`` and ``None`` never oppose (no veto). Strict mapping: bullish
    opposes shorts, bearish opposes longs.
    """
    if ai_direction == "bullish" and setup_direction == "short":
        return True
    if ai_direction == "bearish" and setup_direction == "long":
        return True
    return False


def _min_range_width_resolved(cfg: InstrumentConfig, current_price: float) -> float:
    """Resolve min range width: XAU uses absolute points, BTC uses pct of price.

    XAU path: cfg.min_range_width_points is not None → return points * point_size
      (note: _validate_bounds uses raw points, not price; this returns price units
       so callers that need price-domain comparison can use it directly).
    BTC path: cfg.min_range_width_pct is not None → return (pct/100) * current_price.
      pct is stored as 2.0 meaning 2% (BTC config).
    Stage 5 will wire price flow for live BTC use; this helper is exposed now for
    unit testing and future integration.
    """
    if cfg.min_range_width_points is not None:
        return cfg.min_range_width_points * cfg.point_size
    assert cfg.min_range_width_pct is not None, (
        f"{cfg.symbol}: both min_range_width_points and min_range_width_pct are None"
    )
    return (cfg.min_range_width_pct / 100.0) * current_price


def _h1_bars_between(earlier: datetime, later: datetime) -> int:
    """Round 4.5.1: approximate H1 bars elapsed between two timestamps.

    Used by Method A/B to populate RangeBounds.duration_bars so Guard 4 (>=12)
    can evaluate actual range age instead of seeing 0 (silent reject).
    """
    if earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=timezone.utc)
    if later.tzinfo is None:
        later = later.replace(tzinfo=timezone.utc)
    delta_seconds = (later - earlier).total_seconds()
    if delta_seconds <= 0:
        return 0
    return int(delta_seconds // _SECONDS_PER_H1_BAR)


# ---------------------------------------------------------------------------
# Range guard functions (module-level)
# ---------------------------------------------------------------------------

# Round 4.6-C2 extended: module-level latest-call diagnostic.
# check_range_guards is a pure function without self, so we stash its
# per-call decision trace here for live_demo to harvest each cycle.
_LAST_GUARDS_CHECK: dict[str, object] = {}


def get_last_guards_diagnostic() -> dict[str, object]:
    """Return a copy of the most recent check_range_guards trace."""
    return dict(_LAST_GUARDS_CHECK)


def check_bounds_only_guards(
    bounds: RangeBounds,
    session: str,
    h1_df: pl.DataFrame,
    cfg: InstrumentConfig | None = None,
) -> bool:
    """Round 4.6-E: bounds-level subset of check_range_guards (no setup yet).

    Used in live_demo BEFORE route_trading_mode so `guards_passed` can be
    supplied to mode_router without having generated setups. Covers Guards
    1 (width), 3 (touches), 4 (duration) — the bounds-dependent checks.
    Guard 2 (RR) and Guard 5 (lot) are setup/execution-level and remain
    enforced downstream in _build_setup / _determine_ranging.

    Same session-aware thresholds as check_range_guards (Round 4.6-B).
    Writes its own trace into _LAST_GUARDS_CHECK so the existing diagnostic
    surface keeps working even before a setup exists.

    cfg: InstrumentConfig to use. Defaults to XAUUSD config when None.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")

    global _LAST_GUARDS_CHECK

    is_asian = session in cfg.asian_sessions
    min_width = cfg.guard_width_low if is_asian else cfg.guard_width_high
    min_duration = (
        cfg.guard_duration_low if is_asian else cfg.guard_duration_high
    )

    width_pass = bounds.width_points >= min_width
    touches = _count_boundary_touches(h1_df, bounds, tolerance_ratio=0.05, cfg=cfg)
    touches_pass = touches >= 2
    duration_pass = bounds.duration_bars >= min_duration

    all_passed = width_pass and touches_pass and duration_pass

    _LAST_GUARDS_CHECK = {
        "stage": "bounds_only",
        "session": session,
        "is_asian_profile": is_asian,
        "min_width_required": min_width,
        "min_duration_required": min_duration,
        "width_points": bounds.width_points,
        "width_pass": width_pass,
        "touches_count": touches,
        "touches_pass": touches_pass,
        "duration_bars": bounds.duration_bars,
        "duration_pass": duration_pass,
        # rr + lot deferred to setup-level check_range_guards
        "rr_pass": None,
        "all_passed": all_passed,
    }

    return all_passed


def check_range_guards(
    bounds: RangeBounds,
    setup: RangeSetup,
    session: str,
    h1_df: pl.DataFrame,
    htf_bias: Optional["BiasDirection"] = None,
    cfg: InstrumentConfig | None = None,
) -> bool:
    """Return True if all guards pass. See v3.0 plan.

    Round 4.6-B: Guard 1 (width) and Guard 4 (duration) are session-aware.
    Asian sessions use relaxed thresholds (width 400/duration 8) because
    Asian low-volatility ranges rarely reach London/NY-calibrated 800/12.
    Guards 2 (RR>=1.2), 3 (touches>=2), 5 (lot) stay uniform to preserve
    quality floor.

    Round 4.6-C2: records per-call diagnostic in module _LAST_GUARDS_CHECK
    so live_demo can surface why a given setup was rejected.

    Round 5 T0 (P0-9): Guard 6 — HTF bias alignment.  Only active when
    htf_bias.confidence >= 0.7 (Round 5 T5 tweak: raised from 0.5 so only
    Tier 1 multi-TF confirmed bias blocks range mean-reversion; weaker
    biases are pass-through to preserve range double-sided trading).

    cfg: InstrumentConfig to use. Defaults to XAUUSD config when None.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")

    global _LAST_GUARDS_CHECK

    is_asian = session in cfg.asian_sessions
    min_width = cfg.guard_width_low if is_asian else cfg.guard_width_high
    min_duration = (
        cfg.guard_duration_low if is_asian else cfg.guard_duration_high
    )

    width_pass = bounds.width_points >= min_width
    rr_pass = setup.rr_ratio >= cfg.guard_rr_min
    touches = _count_boundary_touches(h1_df, bounds, tolerance_ratio=0.05, cfg=cfg)
    touches_pass = touches >= 2
    duration_pass = bounds.duration_bars >= min_duration

    # Guard 6: HTF bias alignment (P0-9 — Round 5 T0; threshold raised T5)
    # Only activated when bias confidence >= 0.7 (Tier 1 multi-TF confirmed).
    # Below that threshold, range mean-reversion remains double-sided; this
    # avoids turning a 40-50% conf bearish regime into SELL-only trading.
    guard6_pass = True
    guard6_reason: Optional[str] = None
    if htf_bias is not None and htf_bias.confidence >= 0.7:
        bias_dir = htf_bias.direction  # "bullish" / "bearish" / "neutral"
        if bias_dir == "bullish" and setup.direction == "short":
            guard6_pass = False
            guard6_reason = "htf_bias_opposed"
        elif bias_dir == "bearish" and setup.direction == "long":
            guard6_pass = False
            guard6_reason = "htf_bias_opposed"

    all_passed = (
        width_pass and rr_pass and touches_pass and duration_pass and guard6_pass
    )

    _LAST_GUARDS_CHECK = {
        "session": session,
        "is_asian_profile": is_asian,
        "min_width_required": min_width,
        "min_duration_required": min_duration,
        "width_points": bounds.width_points,
        "width_pass": width_pass,
        "rr_ratio": setup.rr_ratio,
        "rr_pass": rr_pass,
        "touches_count": touches,
        "touches_pass": touches_pass,
        "duration_bars": bounds.duration_bars,
        "duration_pass": duration_pass,
        # Guard 5 (lot_multiplier) enforced downstream (live_demo); logged separately.
        # Guard 6: HTF alignment
        "guard6_pass": guard6_pass,
        "guard6_reason": guard6_reason,
        "all_passed": all_passed,
    }

    return all_passed


def _count_boundary_touches(
    h1_df: pl.DataFrame,
    bounds: RangeBounds,
    tolerance_ratio: float = 0.05,
    cfg: InstrumentConfig | None = None,
) -> int:
    """Count bars that touched either upper or lower boundary within tolerance.

    cfg: InstrumentConfig for point_size. Defaults to XAUUSD config when None.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")
    tolerance_pts = bounds.width_points * tolerance_ratio
    tol_price = tolerance_pts * cfg.point_size

    highs = h1_df["high"].to_list()
    lows = h1_df["low"].to_list()
    touches = 0
    for h, l in zip(highs, lows):
        if abs(h - bounds.upper) <= tol_price or abs(l - bounds.lower) <= tol_price:
            touches += 1
    return touches


_SOFT_REVERSAL_SWING_WINDOW: int = 3
_SOFT_REVERSAL_RECENCY: timedelta = timedelta(minutes=30)

# Audit R3 J2 — mid-range sub-classification cutoff.
# Price within this fraction of range_width from a boundary is "almost_near",
# otherwise "middle". 5% chosen as balanced operating point; raw distances
# are also surfaced in diagnostic for 3%/5%/10% post-hoc bucketing.
_ALMOST_NEAR_PCT: float = 0.05


def _classify_mid_range(
    distance_to_upper_pct: float | None,
    distance_to_lower_pct: float | None,
) -> str:
    """Return reason_if_zero label for price in mid-range region.

    Both distances are fractions of range width in (0, 1). Since the caller
    only invokes this when both `near_upper` and `near_lower` are False, we
    know the price is in the interior; the narrower side wins.
    """
    if distance_to_upper_pct is None or distance_to_lower_pct is None:
        return "price_mid_range"
    if distance_to_upper_pct < _ALMOST_NEAR_PCT:
        return "almost_near_upper"
    if distance_to_lower_pct < _ALMOST_NEAR_PCT:
        return "almost_near_lower"
    return "middle"


def _soft_reversal_3bar(
    m15_snapshot: SMCSnapshot,
    direction: str,
) -> bool:
    """Soft reversal fallback when strict M15 CHoCH is absent.

    Require structure break (Check 1) OR recent + fresh swing match (Check 2);
    no structure → reject.

    Round 4.6-U + V: widened swing look-back to last 5 points.
    Round 5 T0 (P0-9): removed unconditional Check 3 `return True` — that
    bypass allowed structureless setups through with no market evidence.
    Audit R2 S1: tightened Check 2 — window narrowed to last 3 swings AND
    matching swing must land within 30 min (≤2 M15 bars) of snapshot ts.
    The prior `[-5:]` window with no recency check was near-100% pass
    because any "low" in recent history satisfied a long-direction soft
    reversal regardless of freshness.
    """
    target = "bearish" if direction == "short" else "bullish"

    # Check 1: most recent structure_break must match target direction.
    for brk in reversed(m15_snapshot.structure_breaks):
        if brk.direction == target:
            return True
        break

    # Check 2 (R2 S1): swing must be in last N points AND fresh (ts-recency).
    target_swing = "high" if direction == "short" else "low"
    recency_cutoff = m15_snapshot.ts - _SOFT_REVERSAL_RECENCY
    for sw in reversed(m15_snapshot.swing_points[-_SOFT_REVERSAL_SWING_WINDOW:]):
        if sw.swing_type == target_swing and sw.ts >= recency_cutoff:
            return True

    # No structure break and no fresh matching swing → reject (P0-9 fix).
    return False


def _last_bar_reversal_confirm(
    m15_df: object,
    direction: str,
) -> bool:
    """Round 4 v5 (Task #52): confirm last closed M15 bar shows reversal bias.

    The existing ``_find_choch_in_zone`` + ``_soft_reversal_3bar`` let today's
    5 stacked-BUY disasters through because structure signals can form at a
    local bottom that is immediately violated by continued selling. Add a
    cheap bar-close sanity check: the most recent closed M15 bar must itself
    confirm the intended direction.

    Conditions for acceptance:

      long setup → last bar close > open (bullish candle) AND close > prior close
      short setup → last bar close < open (bearish candle) AND close < prior close

    Returns True when the M15 DataFrame is missing or has <2 bars (fail-open
    to keep backward compat when callers don't thread the frame through).
    """
    if m15_df is None:
        return True
    if direction not in ("long", "short"):
        return True
    try:
        n = len(m15_df)
    except TypeError:
        return True
    if n < 2:
        return True

    # Last CLOSED bar is the penultimate row (live_demo strips the in-progress bar)
    # but defensively accept either: check whichever row sits one before the end.
    last = m15_df[-1]
    prior = m15_df[-2]
    try:
        last_open = float(last["open"][0]) if hasattr(last["open"], "__getitem__") else float(last["open"])
        last_close = float(last["close"][0]) if hasattr(last["close"], "__getitem__") else float(last["close"])
        prior_close = float(prior["close"][0]) if hasattr(prior["close"], "__getitem__") else float(prior["close"])
    except (KeyError, TypeError, ValueError, IndexError):
        return True  # fail-open on unexpected schema

    if direction == "long":
        return last_close > last_open and last_close > prior_close
    return last_close < last_open and last_close < prior_close


def _d1_sma50_trend_metrics(d1_df: object) -> tuple[float, float] | None:
    """Round 9 P0-A: compute D1 SMA50 slope %/bar and close-vs-SMA50 %.

    Returns (slope_pct_per_bar, close_vs_sma_pct) when d1_df has at least
    ``_TREND_FILTER_MIN_BARS`` (55) rows, else None. The slope is computed
    against an SMA50 5 bars in the past so a 5-bar look-back captures the
    persistent drift used by the trend-filter gate.

    Mirror of ``regime_classifier._sma50_direction_and_slope`` arithmetic
    but inlined here to avoid importing the AI module from the strategy
    layer (would create a circular dependency once classify_regime_ai
    starts threading more strategy state). Both formulas are pure
    `(sma_now - sma_5ago) / sma_now * 100 / 5` and verified equivalent
    in the regression fixture.
    """
    if d1_df is None:
        return None
    try:
        n_rows = len(d1_df)
    except TypeError:
        return None
    if n_rows < _TREND_FILTER_MIN_BARS:
        return None
    try:
        closes = d1_df["close"].to_list()
    except Exception:
        return None
    if len(closes) < _TREND_FILTER_MIN_BARS:
        return None

    sma50_now = sum(closes[-50:]) / 50
    sma50_5ago = sum(closes[-55:-5]) / 50
    if sma50_now <= 0:
        return None

    slope_pct = (sma50_now - sma50_5ago) / sma50_now * 100.0 / 5.0
    close_vs_sma_pct = (closes[-1] - sma50_now) / sma50_now * 100.0
    return slope_pct, close_vs_sma_pct


def _trend_filter_should_block(
    direction: str,
    slope_pct: float,
    close_vs_sma_pct: float,
) -> bool:
    """Round 9 P0-A: decide whether the D1 trend filter blocks the setup.

    Long support_bounce blocked when D1 trend is materially down:
        slope <= -0.05%/bar AND close <= -1.0% below SMA50
    Short resistance_rejection blocked symmetrically when trend is up.

    Both conditions must agree to avoid false positives from a single-bar
    drift or a transient close-spike against an otherwise flat SMA50.
    """
    if direction == "long":
        return (
            slope_pct <= -_TREND_FILTER_SLOPE_THRESHOLD
            and close_vs_sma_pct <= -_TREND_FILTER_CLOSE_THRESHOLD
        )
    if direction == "short":
        return (
            slope_pct >= _TREND_FILTER_SLOPE_THRESHOLD
            and close_vs_sma_pct >= _TREND_FILTER_CLOSE_THRESHOLD
        )
    return False


# ---------------------------------------------------------------------------
# RangeTrader class
# ---------------------------------------------------------------------------


class RangeTrader:
    """Detects horizontal ranges and generates mean-reversion setups.

    Parameters
    ----------
    min_range_width:
        Minimum range width in points to qualify as a valid range.
    max_range_width:
        Maximum range width in points (reject overly wide ranges).
    boundary_pct:
        Fraction of range width that defines the "boundary zone" at each edge.
    """

    def __init__(
        self,
        *,
        cfg: InstrumentConfig | None = None,
        min_range_width: float | None = None,
        max_range_width: float | None = None,
        boundary_pct: float | None = None,
        cooldown_state_path: Optional[Path] = None,
        reversal_confirm_enabled: bool = False,
        trend_filter_enabled: bool = False,
        ai_regime_gate_enabled: bool = False,
        require_regime_valid: bool = False,
        ai_direction_entry_gate_enabled: bool = False,
    ) -> None:
        from smc.instruments import get_instrument_config
        if cfg is None:
            cfg = get_instrument_config("XAUUSD")
        self._cfg = cfg
        # Points-based (XAU): use cfg directly; pct-based (BTC): None pending price resolution
        self._min_range_width: float = (
            min_range_width if min_range_width is not None
            else (cfg.min_range_width_points if cfg.min_range_width_points is not None else 0.0)
        )
        self._max_range_width: float = (
            max_range_width if max_range_width is not None
            else (cfg.max_range_width_points if cfg.max_range_width_points is not None else float("inf"))
        )
        self._boundary_pct: float = boundary_pct if boundary_pct is not None else cfg.boundary_pct_default
        # Round 5 T0 (P0-4): persist cooldown state across process restarts.
        self._cooldown_state_path: Path = (
            cooldown_state_path
            if cooldown_state_path is not None
            else Path("data/range_cooldown_state.json")
        )
        # Round 4.6-C2: measure-first diagnostic.
        # live_demo writes this into journal['range_diagnostic'] each cycle
        # so we can see which method/condition caused range_bounds=None.
        self._last_diagnostic: dict[str, object] = {}
        self._last_setups_diagnostic: dict[str, object] = {}
        # Round 4.6-W: same-direction cooldown (30 min) to prevent over-trading.
        # Tracks last setup timestamp per direction — blocks _build_setup from
        # returning a non-None setup within cooldown window. Solves UTC 16:45-18:00
        # 6-SHORT same-zone stacking (quality dropped Grade A → C, RR 2.31 → 1.45).
        # Round 5 T0 (P0-4): loaded from / persisted to JSON so restarts don't reset.
        self._last_setup_ts: dict[str, datetime] = {}
        self._load_cooldown_state()
        # Round 4 v5 (Task #52): reversal-candle confirmation gate.
        self._reversal_confirm_enabled: bool = reversal_confirm_enabled
        # Round 9 P0-A/B/C: AI-aware gates default OFF for production safety.
        # All three are independent — caller can flip any subset.
        self._trend_filter_enabled: bool = trend_filter_enabled
        self._ai_regime_gate_enabled: bool = ai_regime_gate_enabled
        self._require_regime_valid: bool = require_regime_valid
        # R10 P2.2: ai_direction entry-veto gate (default OFF for live safety).
        self._ai_direction_entry_gate_enabled: bool = ai_direction_entry_gate_enabled

    # ------------------------------------------------------------------
    # Cooldown persistence helpers (Round 5 T0, P0-4)
    # ------------------------------------------------------------------

    def _load_cooldown_state(self) -> None:
        """Load per-direction cooldown timestamps from JSON (fail-open).

        Populates self._last_setup_ts from {direction: iso_timestamp} dict.
        Missing or corrupt file is silently ignored (fresh cooldown state).
        """
        raw: dict[str, object] = load_json(self._cooldown_state_path, default={})
        ts_map: dict[str, datetime] = {}
        for direction, iso_str in raw.items():
            if not isinstance(iso_str, str):
                continue
            try:
                ts = datetime.fromisoformat(iso_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts_map[direction] = ts
            except Exception:
                pass
        self._last_setup_ts = ts_map

    def _persist_cooldown_state(self) -> None:
        """Persist per-direction cooldown timestamps to JSON (best-effort).

        Writes {direction: iso_timestamp} dict atomically.  Failure is logged
        but never raised — in-memory state remains authoritative.
        """
        payload = {
            direction: ts.isoformat()
            for direction, ts in self._last_setup_ts.items()
        }
        try:
            atomic_write_json(self._cooldown_state_path, payload)
        except Exception as exc:
            _log.warning(
                "RangeTrader: failed to persist cooldown state to %s: %s",
                self._cooldown_state_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Range detection
    # ------------------------------------------------------------------

    def detect_range(
        self,
        h1_df: pl.DataFrame,
        h1_snapshot: SMCSnapshot,
        *,
        ai_regime_assessment: "AIRegimeAssessment | None" = None,
    ) -> RangeBounds | None:
        """Detect a horizontal range from H1 data.

        Tries Method A (OB boundaries), then Method B (swing extremes),
        then Method D (Donchian channel — Round 4.5 hotfix for Asian core
        low-volatility fallback). Returns None if no valid range is found.

        Round 4.6-C2: populates self._last_diagnostic so live_demo can surface
        per-cycle reasoning in the journal (measure-first debugging).

        Round 9 P0-C: when ``self._require_regime_valid`` is True AND a
        non-None ``ai_regime_assessment`` clears the confidence floor
        (>= 0.6) AND its regime is one of TREND_UP / TREND_DOWN /
        ATH_BREAKOUT, return None — a Donchian on a trend is a false range.
        Diagnostic key ``range_invalidated_by_regime`` records the regime
        label that caused the invalidation.
        """
        now = datetime.now(tz=timezone.utc)

        # Counts useful for diagnostic regardless of which method succeeds.
        n_bearish_ob = sum(
            1 for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bearish" and not ob.mitigated
        )
        n_bullish_ob = sum(
            1 for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bullish" and not ob.mitigated
        )
        n_swing_high = sum(1 for s in h1_snapshot.swing_points if s.swing_type == "high")
        n_swing_low = sum(1 for s in h1_snapshot.swing_points if s.swing_type == "low")
        h1_bars_count = h1_df.height if h1_df is not None else 0

        method_a = self._detect_from_ob_boundaries(h1_snapshot, now)
        method_b = self._detect_from_swing_extremes(h1_snapshot, now) if method_a is None else None
        method_d = (
            self._detect_from_donchian_channel(h1_df, now)
            if method_a is None and method_b is None
            else None
        )

        # Compute Donchian candidate width for diagnostic (regardless of success).
        donchian_width_pts: float | None = None
        if h1_bars_count >= self._cfg.donchian_lookback:
            recent = h1_df.tail(self._cfg.donchian_lookback)
            try:
                upper = float(recent["high"].max())
                lower = float(recent["low"].min())
                if upper > lower:
                    donchian_width_pts = round((upper - lower) / self._cfg.point_size, 1)
            except Exception:
                donchian_width_pts = None

        bounds = method_a or method_b or method_d

        # Round 9 P0-C: invalidate any detected range when AI says trend.
        # Only fires when the gate is enabled AND assessment is confident.
        # Assessment + confidence are recorded in diagnostic regardless of
        # gate state so live_demo can show "would have invalidated" telemetry.
        regime_invalidation: str | None = None
        ai_regime_label: str | None = None
        ai_regime_conf: float | None = None
        if ai_regime_assessment is not None:
            ai_regime_label = ai_regime_assessment.regime
            ai_regime_conf = float(ai_regime_assessment.confidence)
            if (
                self._require_regime_valid
                and bounds is not None
                and ai_regime_conf >= _AI_REGIME_GATE_CONF_FLOOR
                and ai_regime_label in _RANGE_INVALIDATING_REGIMES
            ):
                regime_invalidation = ai_regime_label
                bounds = None

        self._last_diagnostic = {
            "h1_bars_count": h1_bars_count,
            "donchian_lookback_required": self._cfg.donchian_lookback,
            "n_bearish_ob": n_bearish_ob,
            "n_bullish_ob": n_bullish_ob,
            "n_swing_high": n_swing_high,
            "n_swing_low": n_swing_low,
            "donchian_width_pts": donchian_width_pts,
            "min_range_width_required": self._min_range_width,
            "max_range_width_required": self._max_range_width,
            "method_a_hit": method_a is not None,
            "method_b_hit": method_b is not None,
            "method_d_hit": method_d is not None,
            "final_source": bounds.source if bounds is not None else None,
            "ai_regime_label": ai_regime_label,
            "ai_regime_confidence": ai_regime_conf,
            "range_invalidated_by_regime": regime_invalidation,
        }
        return bounds

    # ------------------------------------------------------------------
    # Setup generation
    # ------------------------------------------------------------------

    def generate_range_setups(
        self,
        h1_snapshot: SMCSnapshot,
        m15_snapshot: SMCSnapshot,
        current_price: float,
        bounds: RangeBounds,
        h1_atr: float = 0.0,
        session: str = "",
        *,
        m15_df: object = None,
        d1_df: object = None,
        ai_regime_assessment: "AIRegimeAssessment | None" = None,
        ai_direction: str | None = None,
        ai_direction_confidence: float = 0.0,
    ) -> tuple[RangeSetup, ...]:
        """Generate mean-reversion setups at range boundaries.

        Returns at most 2 setups (one long at lower boundary, one short at
        upper boundary).  Each requires M15 CHoCH confirmation.

        Round 4.6-C2: populates self._last_setups_diagnostic for live_demo to
        surface reasons when no setups are generated (price away from boundary
        / no M15 CHoCH / ...).

        Round 4.6-F (USER DIRECTIVE): session-aware boundary_pct. Asian
        sessions use 30% (wider entry window) to meet user's 5-10 trades/day
        target. Non-Asian sessions keep constructor/default value.

        Round 9 P0-A: ``d1_df`` enables D1 SMA50 trend filter inside
        ``_build_setup``. Default None preserves prior behavior.

        Round 9 P0-B: ``ai_regime_assessment`` enables per-direction regime
        gates. When ``self._ai_regime_gate_enabled`` is True AND assessment
        confidence >= 0.6:
        - TREND_DOWN forces ``near_lower=False`` (no long support bounce in
          a downtrend).
        - TREND_UP / ATH_BREAKOUT force ``near_upper=False`` (no short
          resistance rejection in an uptrend).
        - CONSOLIDATION leaves both untouched (range is its native regime).
        - TRANSITION leaves both untouched but ``_build_setup`` raises the
          minimum RR floor from 1.2 to 2.0 to demand higher payoff for the
          ambiguous regime.

        R10 P2.2: ``ai_direction`` (raw "bullish"/"bearish"/"neutral" string
        from the DirectionEngine) plus ``ai_direction_confidence`` route
        through to ``_build_setup``. When the entry-gate flag is on AND the
        AI direction confidently opposes a setup direction, that setup is
        rejected with a ``{direction}_ai_direction_blocked`` diagnostic key.
        Default behavior (gate off) is byte-identical to R9.
        """
        setups: list[RangeSetup] = []
        # Round 9 P0-A: clear stale trend-filter diagnostic keys from prior
        # cycle so this call's _build_setup writes are guaranteed fresh.
        # The final dict assignment at the bottom only sets the structural
        # keys; we explicitly preserve `*_trend_filter_blocked` keys after.
        self._last_setups_diagnostic = {}
        # Round 4.6-I: all active ranging sessions use wide band.
        # Unknown/empty session falls back to constructor value (default boundary_pct_default).
        effective_boundary_pct = (
            self._cfg.boundary_pct_wide if session in self._cfg.wide_band_sessions
            else self._boundary_pct
        )
        boundary_width_price = (
            bounds.width_points * self._cfg.point_size * effective_boundary_pct
        )

        near_lower = current_price <= bounds.lower + boundary_width_price
        near_upper = current_price >= bounds.upper - boundary_width_price

        # Round 9 P0-B: per-direction gates from confident AI regime view.
        ai_regime_gate_block: dict[str, str] = {}
        if (
            self._ai_regime_gate_enabled
            and ai_regime_assessment is not None
            and ai_regime_assessment.confidence >= _AI_REGIME_GATE_CONF_FLOOR
        ):
            regime = ai_regime_assessment.regime
            if regime in _BLOCK_LONG_REGIMES and near_lower:
                ai_regime_gate_block["long"] = regime
                near_lower = False
            if regime in _BLOCK_SHORT_REGIMES and near_upper:
                ai_regime_gate_block["short"] = regime
                near_upper = False

        long_setup: RangeSetup | None = None
        short_setup: RangeSetup | None = None

        # --- Lower boundary: support bounce (long) ---
        if near_lower:
            long_setup = self._build_setup(
                direction="long",
                trigger="support_bounce",
                current_price=current_price,
                bounds=bounds,
                m15_snapshot=m15_snapshot,
                h1_atr=h1_atr,
                boundary_pct=effective_boundary_pct,
                m15_df=m15_df,
                d1_df=d1_df,
                ai_regime_assessment=ai_regime_assessment,
                ai_direction=ai_direction,
                ai_direction_confidence=ai_direction_confidence,
            )
            if long_setup is not None:
                setups.append(long_setup)

        # --- Upper boundary: resistance rejection (short) ---
        if near_upper:
            short_setup = self._build_setup(
                direction="short",
                trigger="resistance_rejection",
                current_price=current_price,
                bounds=bounds,
                m15_snapshot=m15_snapshot,
                h1_atr=h1_atr,
                boundary_pct=effective_boundary_pct,
                m15_df=m15_df,
                d1_df=d1_df,
                ai_regime_assessment=ai_regime_assessment,
                ai_direction=ai_direction,
                ai_direction_confidence=ai_direction_confidence,
            )
            if short_setup is not None:
                setups.append(short_setup)

        # Audit R3 J2: sub-classify mid-range into almost_near_upper /
        # almost_near_lower / middle so journal reason_if_zero distribution
        # can guide future band-widening decisions. Also surface raw
        # distance_to_{upper,lower}_pct floats so post-hoc bucketing at
        # 3%/5%/10% cuts is possible without code change.
        range_price = bounds.width_points * self._cfg.point_size
        distance_to_upper_pct = (
            (bounds.upper - current_price) / range_price if range_price > 0 else None
        )
        distance_to_lower_pct = (
            (current_price - bounds.lower) / range_price if range_price > 0 else None
        )

        reason_if_zero: str | None = None
        if not setups:
            if not near_lower and not near_upper:
                reason_if_zero = _classify_mid_range(
                    distance_to_upper_pct,
                    distance_to_lower_pct,
                )
            elif near_lower and long_setup is None and not near_upper:
                reason_if_zero = "no_m15_choch_at_lower"
            elif near_upper and short_setup is None and not near_lower:
                reason_if_zero = "no_m15_choch_at_upper"
            else:
                reason_if_zero = "no_m15_choch_any_boundary"

        # Preserve any P0-A trend-filter and R10 P2.2 ai-direction keys that
        # _build_setup may have written during this call (it writes before
        # returning None, then we'd otherwise overwrite below).
        prior_trend_filter_keys = {
            k: v
            for k, v in self._last_setups_diagnostic.items()
            if k.endswith("_trend_filter_blocked")
            or k.endswith("_ai_direction_blocked")
        }
        self._last_setups_diagnostic = {
            "current_price": current_price,
            "lower_boundary": bounds.lower,
            "upper_boundary": bounds.upper,
            "boundary_band_price": boundary_width_price,
            "boundary_pct_applied": effective_boundary_pct,
            "session": session,
            "near_lower": near_lower,
            "near_upper": near_upper,
            "long_setup_built": long_setup is not None,
            "short_setup_built": short_setup is not None,
            "setup_count": len(setups),
            "reason_if_zero": reason_if_zero,
            # R3 J2 — raw distances for post-hoc bucketing (None if range zero)
            "distance_to_upper_pct": distance_to_upper_pct,
            "distance_to_lower_pct": distance_to_lower_pct,
            # Round 9 P0-B: which directions the AI regime gate blocked,
            # if any. Empty dict means gate did not fire (off, no AI, low
            # conf, or non-blocking regime).
            "ai_regime_gate_block": dict(ai_regime_gate_block),
            "ai_regime_label": (
                ai_regime_assessment.regime
                if ai_regime_assessment is not None else None
            ),
            "ai_regime_confidence": (
                float(ai_regime_assessment.confidence)
                if ai_regime_assessment is not None else None
            ),
            **prior_trend_filter_keys,
        }

        return tuple(setups[:2])

    # ------------------------------------------------------------------
    # Private: range detection methods
    # ------------------------------------------------------------------

    def _detect_from_ob_boundaries(
        self,
        h1_snapshot: SMCSnapshot,
        now: datetime,
    ) -> RangeBounds | None:
        """Method A: Use highest bearish OB.high and lowest bullish OB.low."""
        bearish_obs = tuple(
            ob for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bearish" and not ob.mitigated
        )
        bullish_obs = tuple(
            ob for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bullish" and not ob.mitigated
        )

        if not bearish_obs or not bullish_obs:
            return None

        upper = max(ob.high for ob in bearish_obs)
        lower = min(ob.low for ob in bullish_obs)

        if upper <= lower:
            return None

        # Round 4.5.1 fix: compute duration from earliest boundary-defining OB.
        # Previously defaulted to 0 → Guard 4 (>=12) silently rejected Method A.
        earliest_ts = min(ob.ts_start for ob in (*bearish_obs, *bullish_obs))
        duration_bars = _h1_bars_between(earliest_ts, now)

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="ob_boundaries",
            confidence=_OB_BOUNDARY_CONFIDENCE,
            now=now,
            duration_bars=duration_bars,
        )

    def _detect_from_swing_extremes(
        self,
        h1_snapshot: SMCSnapshot,
        now: datetime,
    ) -> RangeBounds | None:
        """Method B: Use max swing high + min swing low from recent swings."""
        swings = h1_snapshot.swing_points
        if len(swings) < _MIN_SWING_POINTS_FOR_RANGE:
            return None

        # Take last N swing points (proxy for lookback window)
        recent = swings[-_SWING_LOOKBACK_BARS:]

        highs = tuple(s for s in recent if s.swing_type == "high")
        lows = tuple(s for s in recent if s.swing_type == "low")

        if not highs or not lows:
            return None

        upper = max(s.price for s in highs)
        lower = min(s.price for s in lows)

        if upper <= lower:
            return None

        # Round 4.5.1 fix: compute duration from earliest swing defining bounds.
        # Previously defaulted to 0 → Guard 4 (>=12) silently rejected Method B.
        earliest_ts = min(s.ts for s in (*highs, *lows))
        duration_bars = _h1_bars_between(earliest_ts, now)

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="swing_extremes",
            confidence=_SWING_EXTREME_CONFIDENCE,
            now=now,
            duration_bars=duration_bars,
        )

    def _detect_from_donchian_channel(
        self,
        h1_df: pl.DataFrame,
        now: datetime,
    ) -> RangeBounds | None:
        """Method D (Round 4.5): N-bar high/low channel (Donchian).

        Pure statistical fallback when Method A/B fail (Asian core low
        volatility → OB/swing sparse). Uses last _DONCHIAN_LOOKBACK_BARS
        (24 H1 bars = 1 trading day) to define upper/lower channel.

        Width must exceed _min_range_width (inline equivalent of Guard 1).
        Downstream 5 guards (width/RR/touches/duration/lot) still enforce
        full quality gate — this only opens candidate detection.
        """
        if h1_df.is_empty() or h1_df.height < self._cfg.donchian_lookback:
            return None

        recent = h1_df.tail(self._cfg.donchian_lookback)
        upper = float(recent["high"].max())
        lower = float(recent["low"].min())

        if upper <= lower:
            return None

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="donchian_channel",
            confidence=_DONCHIAN_CONFIDENCE,
            now=now,
            duration_bars=self._cfg.donchian_lookback,
        )

    def _validate_bounds(
        self,
        *,
        upper: float,
        lower: float,
        source: str,
        confidence: float,
        now: datetime,
        duration_bars: int = 0,
    ) -> RangeBounds | None:
        """Shared width validation for both detection methods."""
        width_points = (upper - lower) / self._cfg.point_size

        if width_points < self._min_range_width:
            return None
        if width_points > self._max_range_width:
            return None

        return RangeBounds(
            upper=round(upper, 2),
            lower=round(lower, 2),
            width_points=round(width_points, 1),
            midpoint=round((upper + lower) / 2.0, 2),
            detected_at=now,
            source=source,  # type: ignore[arg-type]
            confidence=confidence,
            duration_bars=duration_bars,
        )

    # ------------------------------------------------------------------
    # Private: setup building
    # ------------------------------------------------------------------

    def _build_setup(
        self,
        *,
        direction: str,
        trigger: str,
        current_price: float,
        bounds: RangeBounds,
        m15_snapshot: SMCSnapshot,
        h1_atr: float,
        boundary_pct: float | None = None,
        m15_df: object = None,
        d1_df: object = None,
        ai_regime_assessment: "AIRegimeAssessment | None" = None,
        ai_direction: str | None = None,
        ai_direction_confidence: float = 0.0,
    ) -> RangeSetup | None:
        """Build a single mean-reversion setup with M15 CHoCH confirmation.

        Round 4.6-G (skeptic catch): boundary_pct now passed from caller so
        the synthetic M15 CHoCH search zone matches the Asian-wide band
        (30%) used in generate_range_setups. Previously hard-coded
        self._boundary_pct (15%) silently truncated the CHoCH search
        window for prices in the 15-30% region — setup builds never fired.

        Round 9 P0-A: ``d1_df`` lets us pre-compute D1 SMA50 slope and
        close-vs-SMA50 distance. When ``self._trend_filter_enabled`` is True
        AND the trend points materially against the setup direction, the
        build is rejected and ``_last_setups_diagnostic`` records the
        slope / close-vs-SMA percentages for post-hoc analysis.

        Round 9 P0-B: ``ai_regime_assessment`` adjusts the minimum RR
        threshold. TRANSITION + confidence >= 0.6 raises the floor from
        1.2 to 2.0; all other regimes keep the legacy floor.

        R10 P2.2: ``ai_direction`` + ``ai_direction_confidence`` carry the
        DirectionEngine's primary read.  When the entry-gate flag is on AND
        the AI direction confidently opposes ``direction`` (per
        ``_ai_direction_opposes``), the build is rejected and a
        ``{direction}_ai_direction_blocked`` diagnostic key records the AI
        view.  Independent signal source from R9 P0-B (regime classifier);
        a setup may pass the regime gate but still fail this veto.
        """
        bp = boundary_pct if boundary_pct is not None else self._boundary_pct

        # Round 4.6-W: same-direction cooldown (30 min). Prevents over-trading
        # of same-zone same-direction setups (UTC 16:45-18:00 saw 6 SHORT in 90min
        # stacking risk Grade A→C). Instance-level only; restart resets cooldown.
        last_ts = self._last_setup_ts.get(direction)
        if last_ts is not None:
            elapsed = (datetime.now(tz=timezone.utc) - last_ts).total_seconds()
            if elapsed < 1800:  # 30 minutes
                return None

        # R10 P2.2: ai_direction entry-veto — runs before the D1 trend filter
        # because the DirectionEngine signal is computed cheaply upstream and
        # we want to short-circuit obvious contradictions first. Default-OFF;
        # when ON requires conf >= 0.55 AND a direct opposition.
        if (
            self._ai_direction_entry_gate_enabled
            and ai_direction is not None
            and ai_direction_confidence >= _AI_DIRECTION_ENTRY_GATE_CONF_FLOOR
            and _ai_direction_opposes(ai_direction, direction)
        ):
            self._last_setups_diagnostic[
                f"{direction}_ai_direction_blocked"
            ] = (
                f"ai_dir={ai_direction} conf={ai_direction_confidence:.2f}"
            )
            return None

        # Round 9 P0-A: D1 SMA50 trend filter — block obvious counter-trend
        # range entries. Only fires when the gate is enabled AND we have
        # >= 55 bars of D1 data. Early-out preserves prior behavior on
        # short histories or when caller did not thread d1_df.
        if self._trend_filter_enabled:
            metrics = _d1_sma50_trend_metrics(d1_df)
            if metrics is not None:
                slope_pct, close_vs_sma_pct = metrics
                if _trend_filter_should_block(direction, slope_pct, close_vs_sma_pct):
                    self._last_setups_diagnostic[
                        f"{direction}_trend_filter_blocked"
                    ] = (
                        f"slope={slope_pct:.3f} "
                        f"close_vs_sma={close_vs_sma_pct:.2f}"
                    )
                    return None

        # Create a synthetic TradeZone at the boundary for CHoCH check
        if direction == "long":
            boundary_width = bounds.width_points * self._cfg.point_size * bp
            zone = TradeZone(
                zone_high=round(bounds.lower + boundary_width, 2),
                zone_low=round(bounds.lower, 2),
                zone_type="ob",
                direction="long",
                timeframe=m15_snapshot.timeframe,
                confidence=bounds.confidence,
            )
        else:
            boundary_width = bounds.width_points * self._cfg.point_size * bp
            zone = TradeZone(
                zone_high=round(bounds.upper, 2),
                zone_low=round(bounds.upper - boundary_width, 2),
                zone_type="ob",
                direction="short",
                timeframe=m15_snapshot.timeframe,
                confidence=bounds.confidence,
            )

        # Require M15 CHoCH confirmation in the synthetic zone
        # Round 4.6-U (USER 解决到开仓): fallback "3-bar soft reversal" when
        # strict CHoCH missing. In high-volatility / trending market 结构性
        # CHoCH 常缺, but 3-bar momentum reversal 可近似表达 reversal intent.
        if not _find_choch_in_zone(m15_snapshot, zone):
            if not _soft_reversal_3bar(m15_snapshot, direction):
                return None

        # Round 4 v5 (Task #52): bar-close reversal confirmation.
        # 2026-04-20 post-mortem — 5 stacked BUYs passed the CHoCH/3-bar
        # checks but all had MFE < 0.10R. The current M15 bar was itself
        # bearish (price still falling into the support zone). Require the
        # most recent closed bar to actually show the direction we intend.
        # Fail-open if m15_df not threaded (backward compat).
        if self._reversal_confirm_enabled and m15_df is not None:
            if not _last_bar_reversal_confirm(m15_df, direction):
                return None

        # SL: boundary +/- ATR-adaptive buffer
        sl_buffer = _compute_sl_buffer(h1_atr) * self._cfg.point_size
        if direction == "long":
            stop_loss = bounds.lower - sl_buffer
        else:
            stop_loss = bounds.upper + sl_buffer

        entry_price = current_price
        risk_points = abs(entry_price - stop_loss) / self._cfg.point_size
        if risk_points == 0:
            return None

        # TP conservative: midpoint
        take_profit = bounds.midpoint

        # TP aggressive: opposite boundary minus 10% inset
        range_price = bounds.width_points * self._cfg.point_size
        inset = range_price * 0.10
        if direction == "long":
            take_profit_ext = bounds.upper - inset
        else:
            take_profit_ext = bounds.lower + inset

        # Round 4.6-O [USER CATCH]: rr_ratio 用 take_profit_ext (对立边界)
        # 而非 midpoint. 原因: midpoint TP 在 narrow range 里 reward 太小,
        # RR 永远 < 1.2, Guard 2 reject all trades (UTC 08:30-11:30 共 12 cycle
        # 全 HOLD). Mean reversion 经典做法 = 持仓到对立边界. TP_ext 是真实
        # 策略目标 (TP1=midpoint 保留作 "conservative partial exit" 概念).
        reward_points = abs(take_profit_ext - entry_price) / self._cfg.point_size
        rr_ratio = reward_points / risk_points if risk_points > 0 else 0.0

        # Round 4.6-K: enforce Guard 2 (RR >= 1.2) at setup build time.
        # Round 9 P0-B: TRANSITION regime + confidence >= 0.6 raises the
        # floor to 2.0 — ambiguous regime demands outsized payoff.
        _MIN_RR_RATIO = 1.2
        if (
            self._ai_regime_gate_enabled
            and ai_regime_assessment is not None
            and ai_regime_assessment.regime == "TRANSITION"
            and ai_regime_assessment.confidence >= _AI_REGIME_GATE_CONF_FLOOR
        ):
            _MIN_RR_RATIO = _TRANSITION_MIN_RR_RATIO
        if rr_ratio < _MIN_RR_RATIO:
            return None

        grade = self._grade_setup(bounds, rr_ratio)

        # Round 4.6-W / Round 5 T0 (P0-4 + P0-5): record setup timestamp for
        # unconditional per-direction 30-min cooldown, then persist to disk.
        self._last_setup_ts[direction] = datetime.now(tz=timezone.utc)
        self._persist_cooldown_state()

        return RangeSetup(
            direction=direction,  # type: ignore[arg-type]
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            take_profit_ext=round(take_profit_ext, 2),
            risk_points=round(risk_points, 1),
            reward_points=round(reward_points, 1),
            rr_ratio=round(rr_ratio, 2),
            range_bounds=bounds,
            confidence=bounds.confidence,
            trigger=trigger,  # type: ignore[arg-type]
            grade=grade,  # type: ignore[arg-type]
        )

    @staticmethod
    def _grade_setup(bounds: RangeBounds, rr_ratio: float) -> str:
        """Assign A/B/C grade based on detection confidence and RR."""
        score = 0.0

        # Source confidence
        if bounds.source == "ob_boundaries":
            score += 0.5
        else:
            score += 0.3

        # RR scoring
        if rr_ratio >= 2.0:
            score += 0.4
        elif rr_ratio >= 1.5:
            score += 0.3
        else:
            score += 0.1

        if score >= 0.7:
            return "A"
        if score >= 0.5:
            return "B"
        return "C"
