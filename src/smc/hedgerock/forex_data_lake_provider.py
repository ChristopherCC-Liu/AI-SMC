"""ForexDataLake-backed :class:`MarketFeaturesProvider` for live HedgeRock.

Phase B step 2 — replaces the Phase 1 ``StaticMockProvider`` with a real
implementation that reads OHLCV bars from the data-lake parquet store
and computes the features the decision_server needs:

    - ``volatility_rank``: percentile rank of current H1 ATR vs lookback
    - ``hh_count`` / ``ll_count``: count of higher highs / lower lows
      across recent H4 swings
    - ``h4_trend_bars``: consecutive H4 closes on the same side of the
      mid-window mean (proxy for SMA50 direction)
    - ``regime``: legacy ``MarketRegimeAI`` enum (UPPERCASE);
      decision_server.build_envelope maps to the v2 lowercase enum

Design rules:
    1. Pull bars via ``ForexDataLake.query`` — deterministic and
       cache-free in this module so a stale lake cannot serve stale
       features. Internal LRU cache by ``(symbol, bar_close_ts)`` is
       OK since bar timestamps make natural cache keys.
    2. Raise :class:`FeaturesUnavailable` rather than returning an
       arbitrary "safe" snapshot when the lake is empty / cold. The
       endpoint downgrades to 503; the EA keeps running its safe-mode
       inputs untouched.
    3. NO LLM, NO external context — Phase B is the deterministic,
       cheap path. AI debate is bolted on by ``regime_classifier.py``
       in Phase 3 / 4 (already exists in ``smc.ai.regime_classifier``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Final

import polars as pl

from smc.ai.models import MarketRegimeAI
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.hedgerock.decision_server import (
    FeaturesUnavailable,
    MarketFeatures,
)

__all__ = [
    "DEFAULT_VOLATILITY_LOOKBACK",
    "ForexDataLakeMarketFeaturesProvider",
    "compute_market_features",
]


Clock = Callable[[], datetime]
"""Override hook for tests — defaults to ``datetime.now(timezone.utc)``."""


# Bars per timeframe — minimum samples required for stable feature calc.
_H4_LOOKBACK_BARS: Final[int] = 60          # ~10 days of H4
_H1_LOOKBACK_BARS: Final[int] = 240         # 10 days of H1
_VOLATILITY_LOOKBACK_BARS: Final[int] = 100  # for percentile rank
_TREND_FLAT_PCT: Final[float] = 0.001       # 0.1% — below = flat regime
_TREND_BAR_THRESHOLD: Final[int] = 3         # consecutive H4 bars to call it trending

DEFAULT_VOLATILITY_LOOKBACK = _VOLATILITY_LOOKBACK_BARS


@dataclass(frozen=True)
class _CacheKey:
    """Cache key — symbol + most-recent CLOSED H1 bar boundary.

    Phase B-closeout #3: cache invalidates only when a fresh H1 bar
    closes. Within an open H1 (e.g. 12:34 reads against the H1 bar
    that opened at 12:00 — still in flight) every read returns the
    same cached features.

    H4 boundaries are coarser; the H1 boundary is the binding
    constraint for cache freshness because the H1 ATR percentile
    drives ``volatility_rank``.
    """

    symbol: str
    last_closed_h1_floor_utc: datetime


def _h1_closed_floor(now: datetime) -> datetime:
    """The most recent UTC time strictly *before now* that is on an H1
    boundary. The H1 bar opened at this time has already closed."""
    return now.replace(minute=0, second=0, microsecond=0)


def _h4_closed_floor(now: datetime) -> datetime:
    """The most recent UTC H4 boundary (00, 04, 08, 12, 16, 20) strictly
    before ``now``. The H4 bar opened at this time has already closed."""
    h = (now.hour // 4) * 4
    return now.replace(hour=h, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Pure feature computation — separate so tests can drive it with synthetic
# DataFrames without a lake at all.
# ---------------------------------------------------------------------------


def _atr(df: pl.DataFrame, period: int = 14) -> float | None:
    """ATR(period) on a sorted-ascending OHLC frame. ``None`` if too few rows."""
    if df.is_empty() or len(df) < period + 1:
        return None
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()
    trs: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        trs.append(max(hl, hc, lc))
    if len(trs) < period:
        return None
    # Simple moving average of TR — matches MT5's iATR default.
    return sum(trs[-period:]) / period


def _atr_series(df: pl.DataFrame, period: int = 14) -> list[float]:
    """Rolling ATR series (length = len(df) - period). Used for percentile rank."""
    if df.is_empty() or len(df) < period + 1:
        return []
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()
    trs: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        trs.append(max(hl, hc, lc))
    out: list[float] = []
    for i in range(period, len(trs) + 1):
        out.append(sum(trs[i - period : i]) / period)
    return out


def _percentile_rank(series: list[float], value: float) -> float:
    """Percentile rank of ``value`` within ``series``, in [0, 1]."""
    if not series:
        return 0.5
    below_or_equal = sum(1 for v in series if v <= value)
    return below_or_equal / len(series)


def _swing_counts(h4_df: pl.DataFrame, lookback: int = 20) -> tuple[int, int]:
    """Count higher-highs and lower-lows over the last ``lookback`` H4 bars.

    A "higher high" = bar's high strictly exceeds the previous bar's high.
    A "lower low"  = bar's low strictly under the previous bar's low.
    Light-weight proxy for the swing-detection logic in
    ``smc.ai.regime_classifier`` — sufficient for the regime router.
    """
    if h4_df.is_empty() or len(h4_df) < 2:
        return (0, 0)
    tail = h4_df.tail(min(lookback, len(h4_df)))
    highs = tail["high"].to_list()
    lows = tail["low"].to_list()
    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
    return (hh, ll)


def _h4_trend_bars(h4_df: pl.DataFrame) -> int:
    """Count of consecutive H4 closes on the same side of the recent mean.

    Uses the 50-bar rolling mean (or shorter window if the frame has
    fewer bars). Returns the streak length at the most recent bar.
    """
    if h4_df.is_empty() or len(h4_df) < 5:
        return 0
    closes = h4_df["close"].to_list()
    window = min(50, len(closes))
    mean_recent = sum(closes[-window:]) / window
    last_side = 1 if closes[-1] > mean_recent else (-1 if closes[-1] < mean_recent else 0)
    if last_side == 0:
        return 0
    streak = 1
    for i in range(len(closes) - 2, -1, -1):
        side = 1 if closes[i] > mean_recent else (-1 if closes[i] < mean_recent else 0)
        if side != last_side:
            break
        streak += 1
    return streak


def _classify_regime(
    *,
    volatility_rank: float,
    h4_trend_bars: int,
    hh_count: int,
    ll_count: int,
) -> MarketRegimeAI:
    """Map computed features to the legacy ``MarketRegimeAI`` enum.

    Trivial rule book — Phase B is "real data, simple rules". Phase C+
    can plug in :func:`smc.ai.regime_classifier.classify_regime_ai`
    when LLM budget allows.

    Rules:
      - ATR percentile < 0.30 → CONSOLIDATION
      - h4_trend_bars >= threshold + HH dominant → TREND_UP
      - h4_trend_bars >= threshold + LL dominant → TREND_DOWN
      - HH and LL both elevated → TRANSITION (whipsaw)
      - default → TRANSITION
    """
    if volatility_rank < 0.30:
        return "CONSOLIDATION"

    if h4_trend_bars >= _TREND_BAR_THRESHOLD:
        if hh_count > ll_count + 2:
            return "TREND_UP"
        if ll_count > hh_count + 2:
            return "TREND_DOWN"
    return "TRANSITION"


def compute_market_features(
    *,
    h4_df: pl.DataFrame,
    h1_df: pl.DataFrame,
) -> MarketFeatures:
    """Pure computation: OHLCV → MarketFeatures.

    Raises ``FeaturesUnavailable`` when the input frames are too small
    to produce meaningful features (callers degrade to 503).
    """
    if h4_df is None or h4_df.is_empty():
        raise FeaturesUnavailable("h4 frame is empty")
    if h1_df is None or h1_df.is_empty():
        raise FeaturesUnavailable("h1 frame is empty")
    if len(h1_df) < _VOLATILITY_LOOKBACK_BARS // 2:
        raise FeaturesUnavailable(
            f"h1 frame too short ({len(h1_df)}) for volatility rank"
        )
    if len(h4_df) < 5:
        raise FeaturesUnavailable(f"h4 frame too short ({len(h4_df)}) for swing/trend")

    # Volatility rank — current H1 ATR vs rolling H1 ATR series
    current_atr = _atr(h1_df, period=14)
    atr_series = _atr_series(h1_df, period=14)
    if current_atr is None or not atr_series:
        raise FeaturesUnavailable("ATR computation failed (insufficient data)")
    volatility_rank = _percentile_rank(atr_series, current_atr)

    # H4 swing counts
    hh_count, ll_count = _swing_counts(h4_df, lookback=20)

    # H4 trend bar streak
    h4_trend = _h4_trend_bars(h4_df)

    # Regime classification
    regime = _classify_regime(
        volatility_rank=volatility_rank,
        h4_trend_bars=h4_trend,
        hh_count=hh_count,
        ll_count=ll_count,
    )

    return MarketFeatures(
        volatility_rank=volatility_rank,
        hh_count=hh_count,
        ll_count=ll_count,
        h4_trend_bars=h4_trend,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ForexDataLakeMarketFeaturesProvider:
    """Live :class:`MarketFeaturesProvider` backed by a parquet data lake.

    On each ``get_features`` call:
      1. Resolve the current UTC time via the injected ``clock``.
      2. Query the lake for the last ~10 days of H4 + H1 bars.
      3. Cache the result keyed by the timestamp of the most recent H1
         close — invalidates automatically on the next bar.
      4. Pass the frames into :func:`compute_market_features`.

    Args:
        lake: A ForexDataLake (or duck-typed equivalent with ``query``).
        instrument_whitelist: Symbols this provider can serve. Anything
            else raises :class:`FeaturesUnavailable`.
        clock: Optional override for ``datetime.now(timezone.utc)``.
        h4_lookback_bars: Tail length for the H4 query.
        h1_lookback_bars: Tail length for the H1 query.
    """

    def __init__(
        self,
        lake: ForexDataLake,
        *,
        instrument_whitelist: frozenset[str] = frozenset({"XAUUSD"}),
        clock: Clock | None = None,
        h4_lookback_bars: int = _H4_LOOKBACK_BARS,
        h1_lookback_bars: int = _H1_LOOKBACK_BARS,
    ) -> None:
        self._lake = lake
        self._whitelist = instrument_whitelist
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._h4_lookback_bars = h4_lookback_bars
        self._h1_lookback_bars = h1_lookback_bars
        self._lock = Lock()
        self._cache: dict[str, tuple[_CacheKey, MarketFeatures]] = {}

    def get_features(self, symbol: str) -> MarketFeatures:
        sym = symbol.upper()
        if sym not in self._whitelist:
            raise FeaturesUnavailable(
                f"symbol {sym!r} not in lake provider whitelist"
            )
        now = self._clock()
        # Phase B-closeout #3: only consider CLOSED bars. The current
        # in-flight H1 / H4 bar must NOT participate in ATR / trend.
        h1_end = _h1_closed_floor(now)
        h4_end = _h4_closed_floor(now)
        key = _CacheKey(symbol=sym, last_closed_h1_floor_utc=h1_end)

        # Cache check FIRST — invalidates only when a fresh H1 closes.
        with self._lock:
            cached = self._cache.get(sym)
            if cached and cached[0] == key:
                return cached[1]

        # Lookback windows scaled to the closed-bar end. H4 needs more
        # bars than H1 because it ticks 4× slower, so the back-window
        # math here is independent.
        h4_start = h4_end - timedelta(hours=4 * (self._h4_lookback_bars + 5))
        h1_start = h1_end - timedelta(hours=1 * (self._h1_lookback_bars + 5))

        # Lake queries use the closed-bar end as exclusive upper bound —
        # any in-flight bar at `now` is excluded.
        h4_df = self._lake.query(sym, Timeframe.H4, h4_start, h4_end)
        h1_df = self._lake.query(sym, Timeframe.H1, h1_start, h1_end)

        if h1_df.is_empty():
            raise FeaturesUnavailable(
                f"lake returned no closed H1 bars for {sym} before {h1_end.isoformat()}"
            )
        if h4_df.is_empty():
            raise FeaturesUnavailable(
                f"lake returned no closed H4 bars for {sym} before {h4_end.isoformat()}"
            )

        features = compute_market_features(h4_df=h4_df, h1_df=h1_df)
        with self._lock:
            self._cache[sym] = (key, features)
        return features
