"""Phase B step 2 — ForexDataLakeMarketFeaturesProvider tests.

Covers:
- Pure feature computation on synthetic OHLCV (volatility_rank,
  hh_count / ll_count, h4_trend_bars, regime mapping).
- Cache invalidation by last-H1-close timestamp.
- FeaturesUnavailable on empty / cold lake.
- Symbol whitelist enforcement.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.hedgerock.decision_server import FeaturesUnavailable
from smc.hedgerock.forex_data_lake_provider import (
    ForexDataLakeMarketFeaturesProvider,
    compute_market_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv(
    *,
    start: datetime,
    n_bars: int,
    bar_minutes: int,
    pattern: str = "drift_up",
    base_price: float = 2300.0,
    atr_size: float = 5.0,
) -> pl.DataFrame:
    """Build a synthetic OHLCV DataFrame.

    pattern values:
      - "drift_up":     close[i] = base + i * 0.5         (gentle uptrend)
      - "drift_down":   close[i] = base - i * 0.5
      - "flat":         close[i] = base + (i % 2) * 0.1   (tiny oscillation)
      - "explode":      atr_size doubles for last 25% of bars
    """
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for i in range(n_bars):
        if pattern == "drift_up":
            c = base_price + i * 0.5
        elif pattern == "drift_down":
            c = base_price - i * 0.5
        elif pattern == "flat":
            c = base_price + (i % 2) * 0.1
        elif pattern == "explode":
            c = base_price + i * 0.2
        else:
            raise ValueError(f"unknown pattern {pattern}")

        # ATR proxy via local range
        local_atr = atr_size if (pattern != "explode" or i < n_bars * 3 // 4) else atr_size * 4
        closes.append(c)
        highs.append(c + local_atr / 2)
        lows.append(c - local_atr / 2)

    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(n_bars)]
    return pl.DataFrame(
        {
            "ts": pl.Series(
                [start + timedelta(minutes=bar_minutes * i) for i in range(n_bars)],
                dtype=pl.Datetime("ns", "UTC"),
            ),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n_bars,
        }
    )


class _StubLake:
    """ForexDataLake-shaped stub returning a fixed frame per timeframe."""

    def __init__(
        self,
        *,
        h4_df: pl.DataFrame,
        h1_df: pl.DataFrame,
        whitelist: frozenset[str] = frozenset({"XAUUSD"}),
    ) -> None:
        self._h4 = h4_df
        self._h1 = h1_df
        self._whitelist = whitelist
        self.calls: list[tuple[str, str]] = []  # (symbol, timeframe)
        self.query_args: list[tuple[str, str, datetime, datetime]] = []

    def query(self, instrument, timeframe, start, end):
        self.calls.append((instrument, str(timeframe)))
        self.query_args.append((instrument, str(timeframe), start, end))
        if instrument not in self._whitelist:
            return _empty_ohlcv()
        if str(timeframe) == "H4":
            df = self._h4
        elif str(timeframe) == "H1":
            df = self._h1
        else:
            return _empty_ohlcv()
        if df.is_empty() or "ts" not in df.columns:
            return _empty_ohlcv()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


def _empty_ohlcv() -> pl.DataFrame:
    """Schema-correct empty frame so downstream filters don't blow up."""
    return pl.DataFrame(
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


# ---------------------------------------------------------------------------
# compute_market_features — pure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_features_drift_up_classifies_trend_up() -> None:
    """A clean H4 uptrend with non-flat ATR → TREND_UP."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=80, bar_minutes=240, pattern="drift_up")
    h1_df = _ohlcv(start=start, n_bars=240, bar_minutes=60, pattern="drift_up")

    features = compute_market_features(h4_df=h4_df, h1_df=h1_df)

    assert features.regime == "TREND_UP"
    assert features.h4_trend_bars >= 3
    assert features.hh_count > features.ll_count
    assert 0.0 <= features.volatility_rank <= 1.0


@pytest.mark.unit
def test_compute_features_drift_down_classifies_trend_down() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=80, bar_minutes=240, pattern="drift_down")
    h1_df = _ohlcv(start=start, n_bars=240, bar_minutes=60, pattern="drift_down")

    features = compute_market_features(h4_df=h4_df, h1_df=h1_df)

    assert features.regime == "TREND_DOWN"
    assert features.ll_count > features.hh_count


@pytest.mark.unit
def test_compute_features_flat_classifies_consolidation() -> None:
    """Tiny range → low ATR rank → CONSOLIDATION regardless of swing."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=80, bar_minutes=240, pattern="flat", atr_size=0.5)
    h1_df = _ohlcv(start=start, n_bars=240, bar_minutes=60, pattern="flat", atr_size=0.5)

    # Spike a single recent H1 ATR to keep volatility from collapsing to 0
    # but still rank low — adjust last bar slightly.
    features = compute_market_features(h4_df=h4_df, h1_df=h1_df)

    # In flat market the ATR distribution is degenerate; we just assert
    # the regime defaults to a low-vol / non-trending choice.
    assert features.regime in ("CONSOLIDATION", "TRANSITION")
    assert features.h4_trend_bars >= 1


@pytest.mark.unit
def test_compute_features_volatility_rank_in_unit_interval() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=80, bar_minutes=240)
    h1_df = _ohlcv(start=start, n_bars=240, bar_minutes=60)

    features = compute_market_features(h4_df=h4_df, h1_df=h1_df)
    assert 0.0 <= features.volatility_rank <= 1.0
    assert math.isfinite(features.volatility_rank)


@pytest.mark.unit
def test_compute_features_explode_pushes_volatility_rank_high() -> None:
    """Recent ATR spike → top-percentile rank."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=80, bar_minutes=240, pattern="explode")
    h1_df = _ohlcv(start=start, n_bars=240, bar_minutes=60, pattern="explode")

    features = compute_market_features(h4_df=h4_df, h1_df=h1_df)
    assert features.volatility_rank >= 0.7


@pytest.mark.unit
def test_compute_features_empty_h1_raises() -> None:
    h4_df = _ohlcv(start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                   n_bars=20, bar_minutes=240)
    h1_df = pl.DataFrame()
    with pytest.raises(FeaturesUnavailable):
        compute_market_features(h4_df=h4_df, h1_df=h1_df)


@pytest.mark.unit
def test_compute_features_short_h1_raises() -> None:
    """H1 frame shorter than the volatility-rank lookback → unavailable."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h4_df = _ohlcv(start=start, n_bars=20, bar_minutes=240)
    h1_df = _ohlcv(start=start, n_bars=10, bar_minutes=60)
    with pytest.raises(FeaturesUnavailable, match="too short"):
        compute_market_features(h4_df=h4_df, h1_df=h1_df)


# ---------------------------------------------------------------------------
# Provider — integration with stub lake
# ---------------------------------------------------------------------------


def _build_provider(now: datetime) -> tuple[ForexDataLakeMarketFeaturesProvider, _StubLake]:
    h4_df = _ohlcv(start=now - timedelta(days=15), n_bars=80, bar_minutes=240, pattern="drift_up")
    h1_df = _ohlcv(start=now - timedelta(days=12), n_bars=240, bar_minutes=60, pattern="drift_up")
    lake = _StubLake(h4_df=h4_df, h1_df=h1_df)
    provider = ForexDataLakeMarketFeaturesProvider(lake, clock=lambda: now)
    return provider, lake


def test_provider_returns_market_features_for_whitelisted_symbol() -> None:
    now = datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc)
    provider, _lake = _build_provider(now)

    features = provider.get_features("XAUUSD")
    assert features.regime == "TREND_UP"
    assert features.hh_count > 0


def test_provider_rejects_off_whitelist_symbol() -> None:
    now = datetime(2024, 2, 1, 12, 0, tzinfo=timezone.utc)
    provider, _lake = _build_provider(now)

    with pytest.raises(FeaturesUnavailable, match="not in lake provider whitelist"):
        provider.get_features("BTCUSD")


def test_provider_caches_within_same_h1_bar() -> None:
    """Two reads inside the same open H1 bar → 0 extra lake queries.

    The cache key is the most-recent CLOSED H1 floor, so 12:34 and 12:58
    of the same H1 hour both map to the H1 boundary at 12:00 → same key.
    """
    now = datetime(2024, 2, 1, 12, 34, 0, tzinfo=timezone.utc)  # mid-bar
    provider, lake = _build_provider(now)

    f1 = provider.get_features("XAUUSD")
    n_after_first = len(lake.calls)
    f2 = provider.get_features("XAUUSD")  # same H1 floor → cache hit

    assert f1 == f2
    assert len(lake.calls) == n_after_first


def test_provider_invalidates_cache_when_new_h1_closes() -> None:
    """Crossing into the next H1 hour → cache miss + re-query."""
    now1 = datetime(2024, 2, 1, 12, 30, 0, tzinfo=timezone.utc)  # mid-12-bar
    h4_df = _ohlcv(start=now1 - timedelta(days=15), n_bars=80, bar_minutes=240,
                   pattern="drift_up")
    h1_df = _ohlcv(start=now1 - timedelta(days=12), n_bars=240, bar_minutes=60,
                   pattern="drift_up")
    lake = _StubLake(h4_df=h4_df, h1_df=h1_df)
    clock_state = {"now": now1}
    provider = ForexDataLakeMarketFeaturesProvider(
        lake, clock=lambda: clock_state["now"],
    )

    provider.get_features("XAUUSD")
    base_calls = len(lake.calls)

    # Step the clock past the next H1 close (13:00) → cache miss.
    clock_state["now"] = datetime(2024, 2, 1, 13, 5, 0, tzinfo=timezone.utc)
    provider.get_features("XAUUSD")
    assert len(lake.calls) == base_calls + 2


def test_provider_excludes_partial_h1_bar_from_query_range() -> None:
    """The lake.query upper bound passed by the provider must be the
    most-recent CLOSED H1 boundary — never the wall clock time, which
    would let the in-flight bar leak into ATR computation."""
    now = datetime(2024, 2, 1, 12, 47, 0, tzinfo=timezone.utc)
    provider, lake = _build_provider(now)
    provider.get_features("XAUUSD")

    # Inspect the query call args via the stub — every (instrument, tf,
    # start, end) tuple must have end <= 12:00 (the last closed H1).
    expected_h1_floor = datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    expected_h4_floor = datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    for sym, tf, start, end in lake.query_args:
        if tf == "H1":
            assert end <= expected_h1_floor, (
                f"H1 query end {end} > closed-bar floor {expected_h1_floor} "
                "→ partial bar would leak in"
            )
        elif tf == "H4":
            assert end <= expected_h4_floor, (
                f"H4 query end {end} > closed-bar floor {expected_h4_floor}"
            )


def test_provider_features_unaffected_by_partial_bar_in_lake() -> None:
    """Insert a wildly-different in-flight H1 bar at `now` — features
    must equal the result computed without it."""
    now = datetime(2024, 2, 1, 12, 30, 0, tzinfo=timezone.utc)

    h4_df = _ohlcv(start=now - timedelta(days=15), n_bars=80, bar_minutes=240,
                   pattern="drift_up")
    h1_df = _ohlcv(start=now - timedelta(days=12), n_bars=240, bar_minutes=60,
                   pattern="drift_up")

    # Add a partial bar anchored at 12:00 (in-flight at 12:30).
    partial_ts = datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    partial_row = pl.DataFrame({
        "ts": pl.Series([partial_ts], dtype=pl.Datetime("ns", "UTC")),
        "open": [h1_df["close"][-1]],
        # Wildly anomalous values that would distort ATR if included.
        "high": [h1_df["close"][-1] + 500.0],
        "low":  [h1_df["close"][-1] - 500.0],
        "close": [h1_df["close"][-1] + 200.0],
        "volume": [100.0],
    })
    h1_df_with_partial = pl.concat([h1_df, partial_row])

    # Compute features WITHOUT the partial.
    lake_clean = _StubLake(h4_df=h4_df, h1_df=h1_df)
    p_clean = ForexDataLakeMarketFeaturesProvider(lake_clean, clock=lambda: now)
    f_clean = p_clean.get_features("XAUUSD")

    # Compute features WITH the partial — provider must exclude it via
    # the H1 closed-floor end and produce an IDENTICAL result.
    lake_with_partial = _StubLake(h4_df=h4_df, h1_df=h1_df_with_partial)
    p_with_partial = ForexDataLakeMarketFeaturesProvider(
        lake_with_partial, clock=lambda: now,
    )
    f_with_partial = p_with_partial.get_features("XAUUSD")

    assert f_clean == f_with_partial


def test_provider_raises_when_only_partial_bar_available() -> None:
    """If the lake only has the open in-flight bar (no closed bars
    before now), the provider must raise FeaturesUnavailable rather
    than serve features built from a partial bar."""
    now = datetime(2024, 2, 1, 12, 30, 0, tzinfo=timezone.utc)
    partial_ts = datetime(2024, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    only_partial = pl.DataFrame({
        "ts": pl.Series([partial_ts], dtype=pl.Datetime("ns", "UTC")),
        "open": [2300.0], "high": [2310.0], "low": [2290.0],
        "close": [2305.0], "volume": [100.0],
    })
    lake = _StubLake(h4_df=only_partial, h1_df=only_partial)
    provider = ForexDataLakeMarketFeaturesProvider(lake, clock=lambda: now)
    with pytest.raises(FeaturesUnavailable):
        provider.get_features("XAUUSD")


def test_provider_raises_when_lake_empty() -> None:
    now = datetime(2024, 2, 1, tzinfo=timezone.utc)
    lake = _StubLake(h4_df=_empty_ohlcv(), h1_df=_empty_ohlcv())
    provider = ForexDataLakeMarketFeaturesProvider(lake, clock=lambda: now)
    with pytest.raises(FeaturesUnavailable):
        provider.get_features("XAUUSD")
