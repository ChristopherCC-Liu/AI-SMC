"""AI regime classification with automatic fallback chain.

Public API:
    classify_regime_ai()  — AI debate → ATR fallback → default TRANSITION

Feature extraction is deterministic and fast (~1ms). It runs unconditionally
to prepare both the AI debate input and the ATR fallback.

Fallback chain:
    1. AI debate pipeline (if LLM available + budget remaining + confidence >= threshold)
    2. ATR-based classifier (existing regime.py logic + SMA50 direction)
    3. Default TRANSITION regime (safest default)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import polars as pl

from smc.ai.models import (
    AIRegimeAssessment,
    ExternalContext,
    MarketRegimeAI,
)
from smc.ai.param_router import route
from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.strategy.regime import classify_regime

__all__ = ["classify_regime_ai", "extract_regime_context", "RegimeContext"]

# ---------------------------------------------------------------------------
# Pre-computed feature snapshot
# ---------------------------------------------------------------------------

_SMA50_PERIOD = 50
_ATR_PERIOD = 14
_SWING_LOOKBACK = 10  # count HH/LL in last N swing points
_RANGE_LOOKBACK = 20  # bars for recent range calculation
_TREND_SLOPE_FLAT_THRESHOLD = 0.02  # % slope below which SMA50 is "flat"


@dataclass(frozen=True)
class RegimeContext:
    """Pre-computed features fed into the AI debate pipeline.

    All fields are deterministic and computed from raw OHLCV data.
    This snapshot is the sole input for both the AI path and the
    ATR fallback path — ensuring consistency between the two.
    """

    # D1 features
    d1_atr_pct: float | None
    d1_sma50_direction: Literal["up", "down", "flat"]
    d1_sma50_slope: float  # normalized % slope per bar
    d1_close_vs_sma50: float  # % distance: (close - sma50) / sma50 * 100
    d1_recent_range_pct: float  # 20-bar range as % of price
    d1_higher_highs: int  # count of HH in last N swing-like moves
    d1_lower_lows: int  # count of LL in last N swing-like moves

    # H4 features
    h4_atr_pct: float | None
    h4_trend_bars: int  # consecutive bars closing in same direction vs SMA50
    h4_volatility_rank: float  # percentile rank of current ATR vs 100-bar history

    # Price context
    current_price: float
    ath_distance_pct: float  # % below all-time high in available data
    price_52w_percentile: float  # where in 52-week range (0.0 = low, 1.0 = high)

    # Macro (optional — may be None if external_context.py not configured)
    external: ExternalContext | None

    # ATR regime from existing classifier (used by fallback)
    atr_regime: Literal["trending", "transitional", "ranging"]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _compute_atr_pct(df: pl.DataFrame, period: int = _ATR_PERIOD) -> float | None:
    """Compute ATR(period) as % of price. Returns None if insufficient data."""
    if df is None or len(df) < period + 1:
        return None

    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()

    tr_values: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_values.append(max(hl, hc, lc))

    if len(tr_values) < period:
        return None

    atr = sum(tr_values[-period:]) / period
    latest_close = close[-1]
    if latest_close <= 0:
        return None

    return round((atr / latest_close) * 100.0, 4)


def _compute_sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of the last `period` values."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _sma50_direction_and_slope(
    closes: list[float],
) -> tuple[Literal["up", "down", "flat"], float]:
    """Compute SMA50 direction and normalized slope.

    Slope = (SMA50_current - SMA50_5bars_ago) / SMA50_current * 100 / 5
    This gives a per-bar % change rate.
    """
    if len(closes) < _SMA50_PERIOD + 5:
        return "flat", 0.0

    sma_now = _compute_sma(closes, _SMA50_PERIOD)
    sma_5ago = _compute_sma(closes[:-5], _SMA50_PERIOD)

    if sma_now is None or sma_5ago is None or sma_now <= 0:
        return "flat", 0.0

    slope_pct = (sma_now - sma_5ago) / sma_now * 100.0 / 5.0

    if slope_pct > _TREND_SLOPE_FLAT_THRESHOLD:
        direction: Literal["up", "down", "flat"] = "up"
    elif slope_pct < -_TREND_SLOPE_FLAT_THRESHOLD:
        direction = "down"
    else:
        direction = "flat"

    return direction, round(slope_pct, 4)


def _close_vs_sma50(closes: list[float]) -> float:
    """% distance of current close from SMA50."""
    sma = _compute_sma(closes, _SMA50_PERIOD)
    if sma is None or sma <= 0:
        return 0.0
    return round((closes[-1] - sma) / sma * 100.0, 4)


def _recent_range_pct(df: pl.DataFrame, lookback: int = _RANGE_LOOKBACK) -> float:
    """Recent N-bar range as % of current price."""
    if len(df) < lookback:
        return 0.0

    recent = df[-lookback:]
    high_max = recent["high"].max()
    low_min = recent["low"].min()
    close = df["close"][-1]

    if close <= 0 or high_max is None or low_min is None:
        return 0.0

    return round((high_max - low_min) / close * 100.0, 4)


def _count_hh_ll(df: pl.DataFrame, lookback: int = _SWING_LOOKBACK) -> tuple[int, int]:
    """Count higher-highs and lower-lows in recent bars.

    Uses a simplified approach: compare consecutive local extremes
    in the last `lookback * 3` bars (enough for ~10 swing-like moves).
    A HH occurs when a local high exceeds the previous local high.
    A LL occurs when a local low is below the previous local low.
    """
    n_bars = min(len(df), lookback * 3)
    if n_bars < 6:
        return 0, 0

    highs = df["high"][-n_bars:].to_list()
    lows = df["low"][-n_bars:].to_list()

    # Find local highs and lows using 2-bar lookback/forward
    local_highs: list[float] = []
    local_lows: list[float] = []

    for i in range(2, len(highs) - 2):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i - 2] and \
           highs[i] >= highs[i + 1] and highs[i] >= highs[i + 2]:
            local_highs.append(highs[i])
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and \
           lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            local_lows.append(lows[i])

    # Count HH: each local high that exceeds its predecessor
    hh_count = 0
    for i in range(1, min(len(local_highs), lookback)):
        if local_highs[-(i)] > local_highs[-(i + 1)]:
            hh_count += 1

    # Count LL: each local low below its predecessor
    ll_count = 0
    for i in range(1, min(len(local_lows), lookback)):
        if local_lows[-(i)] < local_lows[-(i + 1)]:
            ll_count += 1

    return hh_count, ll_count


def _h4_trend_bars(df: pl.DataFrame) -> int:
    """Count consecutive H4 bars closing on the same side of SMA50."""
    closes = df["close"].to_list()
    if len(closes) < _SMA50_PERIOD + 1:
        return 0

    sma = _compute_sma(closes, _SMA50_PERIOD)
    if sma is None:
        return 0

    # Count from most recent bar backwards
    above = closes[-1] > sma
    count = 0
    for i in range(len(closes) - 1, _SMA50_PERIOD - 1, -1):
        bar_sma = _compute_sma(closes[: i + 1], _SMA50_PERIOD)
        if bar_sma is None:
            break
        if (closes[i] > bar_sma) == above:
            count += 1
        else:
            break

    return count


def _h4_volatility_rank(df: pl.DataFrame) -> float:
    """Percentile rank of current ATR vs last 100 H4 bars' ATR values."""
    lookback = 100
    if len(df) < _ATR_PERIOD + lookback:
        return 0.5  # default to median if insufficient data

    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()

    # Compute rolling ATR for last `lookback` positions
    atr_values: list[float] = []
    for end_idx in range(len(close) - lookback, len(close)):
        start = max(1, end_idx - _ATR_PERIOD)
        trs: list[float] = []
        for j in range(start, end_idx + 1):
            hl = high[j] - low[j]
            hc = abs(high[j] - close[j - 1])
            lc = abs(low[j] - close[j - 1])
            trs.append(max(hl, hc, lc))
        if trs:
            atr_values.append(sum(trs) / len(trs))

    if len(atr_values) < 2:
        return 0.5

    current_atr = atr_values[-1]
    rank = sum(1 for v in atr_values if v <= current_atr) / len(atr_values)
    return round(rank, 4)


def _ath_distance_pct(df: pl.DataFrame, current_price: float) -> float:
    """% below all-time high in available D1 data."""
    if df is None or df.is_empty():
        return 0.0

    ath = df["high"].max()
    if ath is None or ath <= 0:
        return 0.0

    return round(max(0.0, (ath - current_price) / ath * 100.0), 4)


def _price_52w_percentile(df: pl.DataFrame, current_price: float) -> float:
    """Where the current price sits in the 52-week range (0.0=low, 1.0=high)."""
    # 52 weeks ~ 252 trading days of D1 data
    n_bars = min(len(df), 252)
    if n_bars < 10:
        return 0.5

    recent = df[-n_bars:]
    high_52w = recent["high"].max()
    low_52w = recent["low"].min()

    if high_52w is None or low_52w is None or high_52w == low_52w:
        return 0.5

    return round(
        max(0.0, min(1.0, (current_price - low_52w) / (high_52w - low_52w))),
        4,
    )


# ---------------------------------------------------------------------------
# Public feature extraction
# ---------------------------------------------------------------------------


def extract_regime_context(
    d1_df: pl.DataFrame | None,
    h4_df: pl.DataFrame | None,
    external_ctx: ExternalContext | None = None,
) -> RegimeContext:
    """Extract a deterministic feature snapshot from raw OHLCV data.

    This is the sole input preparation step for both the AI debate path
    and the ATR fallback path — ensuring feature consistency.

    Parameters
    ----------
    d1_df:
        D1 OHLCV DataFrame (must include high, low, close columns).
    h4_df:
        H4 OHLCV DataFrame.
    external_ctx:
        Optional macro data snapshot from external_context.py.

    Returns
    -------
    RegimeContext
        Frozen dataclass with all pre-computed features.
    """
    # Current price from D1 close (preferred) or H4 close
    current_price = 0.0
    if d1_df is not None and not d1_df.is_empty():
        current_price = float(d1_df["close"][-1])
    elif h4_df is not None and not h4_df.is_empty():
        current_price = float(h4_df["close"][-1])

    # D1 features
    d1_atr_pct = _compute_atr_pct(d1_df) if d1_df is not None else None

    d1_closes = d1_df["close"].to_list() if d1_df is not None and not d1_df.is_empty() else []
    d1_sma_dir, d1_sma_slope = _sma50_direction_and_slope(d1_closes)
    d1_close_vs = _close_vs_sma50(d1_closes)
    d1_range = _recent_range_pct(d1_df) if d1_df is not None and not d1_df.is_empty() else 0.0
    d1_hh, d1_ll = _count_hh_ll(d1_df) if d1_df is not None and not d1_df.is_empty() else (0, 0)

    # H4 features
    h4_atr_pct = _compute_atr_pct(h4_df) if h4_df is not None else None
    h4_tb = _h4_trend_bars(h4_df) if h4_df is not None and not h4_df.is_empty() else 0
    h4_vol_rank = _h4_volatility_rank(h4_df) if h4_df is not None and not h4_df.is_empty() else 0.5

    # ATR regime from existing classifier
    atr_regime = classify_regime(d1_df)

    return RegimeContext(
        d1_atr_pct=d1_atr_pct,
        d1_sma50_direction=d1_sma_dir,
        d1_sma50_slope=d1_sma_slope,
        d1_close_vs_sma50=d1_close_vs,
        d1_recent_range_pct=d1_range,
        d1_higher_highs=d1_hh,
        d1_lower_lows=d1_ll,
        h4_atr_pct=h4_atr_pct,
        h4_trend_bars=h4_tb,
        h4_volatility_rank=h4_vol_rank,
        current_price=current_price,
        ath_distance_pct=_ath_distance_pct(d1_df, current_price) if d1_df is not None else 0.0,
        price_52w_percentile=_price_52w_percentile(d1_df, current_price) if d1_df is not None else 0.5,
        external=external_ctx,
        atr_regime=atr_regime,
    )


# ---------------------------------------------------------------------------
# ATR fallback mapping
# ---------------------------------------------------------------------------


def _atr_fallback(ctx: RegimeContext) -> AIRegimeAssessment:
    """Map the existing ATR regime to a MarketRegimeAI enum using SMA50 direction.

    Mapping:
        trending + SMA50 up   → TREND_UP
        trending + SMA50 down → TREND_DOWN
        trending + SMA50 flat → TRANSITION (ambiguous momentum)
        transitional           → TRANSITION
        ranging                → CONSOLIDATION

    ATH_BREAKOUT is NOT reachable via ATR fallback — it requires
    macro context that only the AI debate path can assess.
    """
    regime: MarketRegimeAI
    trend_dir: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

    if ctx.atr_regime == "trending":
        if ctx.d1_sma50_direction == "up":
            regime = "TREND_UP"
            trend_dir = "bullish"
            confidence = min(0.8, 0.5 + ctx.d1_sma50_slope * 5)
            reasoning = (
                f"ATR trending ({ctx.d1_atr_pct:.2f}%), SMA50 up "
                f"(slope {ctx.d1_sma50_slope:+.4f}%), "
                f"HH={ctx.d1_higher_highs} LL={ctx.d1_lower_lows}"
            )
        elif ctx.d1_sma50_direction == "down":
            regime = "TREND_DOWN"
            trend_dir = "bearish"
            confidence = min(0.8, 0.5 + abs(ctx.d1_sma50_slope) * 5)
            reasoning = (
                f"ATR trending ({ctx.d1_atr_pct:.2f}%), SMA50 down "
                f"(slope {ctx.d1_sma50_slope:+.4f}%), "
                f"HH={ctx.d1_higher_highs} LL={ctx.d1_lower_lows}"
            )
        else:
            # Trending ATR but flat SMA50 — ambiguous
            regime = "TRANSITION"
            trend_dir = "neutral"
            confidence = 0.4
            reasoning = (
                f"ATR trending ({ctx.d1_atr_pct:.2f}%) but SMA50 flat "
                f"(slope {ctx.d1_sma50_slope:+.4f}%) — ambiguous momentum"
            )

    elif ctx.atr_regime == "ranging":
        regime = "CONSOLIDATION"
        trend_dir = "neutral"
        confidence = min(0.75, 0.5 + (1.0 - (ctx.d1_atr_pct or 0.5)) * 0.5)
        reasoning = (
            f"ATR ranging ({ctx.d1_atr_pct:.2f}%), "
            f"range {ctx.d1_recent_range_pct:.2f}% of price"
        )

    else:
        # transitional
        regime = "TRANSITION"
        trend_dir = "neutral"
        if ctx.d1_sma50_direction == "up":
            trend_dir = "bullish"
        elif ctx.d1_sma50_direction == "down":
            trend_dir = "bearish"
        confidence = 0.45
        reasoning = (
            f"ATR transitional ({ctx.d1_atr_pct:.2f}%), SMA50 {ctx.d1_sma50_direction} "
            f"(slope {ctx.d1_sma50_slope:+.4f}%) — regime ambiguous"
        )

    return AIRegimeAssessment(
        regime=regime,
        trend_direction=trend_dir,
        confidence=round(confidence, 4),
        param_preset=route(regime),
        reasoning=reasoning[:300],
        assessed_at=datetime.now(tz=timezone.utc),
        source="atr_fallback",
        cost_usd=0.0,
    )


def _default_assessment() -> AIRegimeAssessment:
    """Return the safest default: TRANSITION regime with low confidence."""
    return AIRegimeAssessment(
        regime="TRANSITION",
        trend_direction="neutral",
        confidence=0.3,
        param_preset=route("TRANSITION"),
        reasoning="Default fallback — insufficient data for classification",
        assessed_at=datetime.now(tz=timezone.utc),
        source="default",
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_regime_ai(
    d1_df: pl.DataFrame | None,
    h4_df: pl.DataFrame | None,
    external_ctx: ExternalContext | None = None,
    *,
    ai_enabled: bool = False,
    min_confidence: float = 0.5,
    cache: "RegimeCacheLookup | None" = None,
    cache_ts: datetime | None = None,
) -> AIRegimeAssessment:
    """AI-powered regime classification with automatic fallback.

    Fallback chain:
        0. Pre-computed cache lookup (if ``cache`` provided — backtest mode)
        1. AI debate pipeline (if ``ai_enabled`` + LLM available + budget remaining)
        2. ATR-based classifier + SMA50 direction mapping
        3. Default TRANSITION regime

    Parameters
    ----------
    d1_df:
        D1 OHLCV DataFrame. None triggers immediate default fallback.
    h4_df:
        H4 OHLCV DataFrame. Can be None (reduced feature set).
    external_ctx:
        Optional macro data snapshot for the AI debate path.
    ai_enabled:
        Feature flag for AI debate pipeline. When False (default),
        the system uses ATR fallback only — identical to Sprint 5+
        behaviour with enhanced parameter routing.
    min_confidence:
        Minimum AI confidence threshold. Below this, the ATR fallback
        activates even if the AI debate produced a result.
    cache:
        Optional ``RegimeCacheLookup`` instance for backtest mode.
        When provided, classifications are read from a pre-computed
        parquet file instead of running the AI/ATR pipeline.
    cache_ts:
        Timestamp to look up in the cache. Required when ``cache``
        is provided. Typically the current bar's timestamp.

    Returns
    -------
    AIRegimeAssessment
        Frozen assessment with regime, parameters, and metadata.
    """
    import time as _time

    start_ts = _time.monotonic()
    result: AIRegimeAssessment

    # Step 0: Cache lookup (backtest mode — zero computation cost)
    if cache is not None and cache_ts is not None:
        cached = cache.lookup(cache_ts)
        if cached is not None:
            result = cached
            _emit_telemetry(result, start_ts, ai_enabled_flag=ai_enabled)
            return result
        # Cache miss (ts before cache range) → fall through to live path

    # Insufficient data → immediate default
    if d1_df is None or d1_df.is_empty():
        result = _default_assessment()
        _emit_telemetry(result, start_ts, ai_enabled_flag=ai_enabled)
        return result

    # Step 1: Extract features (deterministic, ~1ms)
    ctx = extract_regime_context(d1_df, h4_df, external_ctx)

    # Step 2: Try AI debate path
    if ai_enabled:
        try:
            from smc.ai.debate.pipeline import run_regime_debate

            ai_result = run_regime_debate(ctx)
            if ai_result.confidence >= min_confidence:
                _emit_telemetry(ai_result, start_ts, ai_enabled_flag=ai_enabled)
                return ai_result
            # Low confidence → fall through to ATR
        except (ImportError, RuntimeError):
            # LLM not available or debate failed → fall through
            pass

    # Step 3: ATR fallback (always available)
    result = _atr_fallback(ctx)
    _emit_telemetry(result, start_ts, ai_enabled_flag=ai_enabled)
    return result


def _emit_telemetry(
    result: AIRegimeAssessment,
    start_ts: float,
    *,
    ai_enabled_flag: bool,
) -> None:
    """Emit one ai_regime_classified event per classify_regime_ai call.

    Never raises — telemetry must not break the classifier.
    """
    try:
        import time as _time

        from smc.monitor.structured_log import info as _log_info

        _log_info(
            "ai_regime_classified",
            regime=result.regime,
            source=result.source,
            direction=result.trend_direction,
            confidence=round(result.confidence, 3),
            elapsed_ms=int((_time.monotonic() - start_ts) * 1000),
            cost_usd=round(result.cost_usd, 4),
            ai_enabled=ai_enabled_flag,
        )
    except Exception:
        pass
