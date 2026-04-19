"""TIPS real-yield fetcher for macro overlay (Alt-B round 4).

Fetches daily DFII10 (10-Year Treasury Inflation-Indexed Security,
Constant Maturity) observations from the FRED API, computes a
20-day directional change, and maps it to a bias float in
[-0.10, +0.10].

Primary source:
    FRED API:   https://api.stlouisfed.org/fred/series/observations
    Series:     DFII10
    Auth:       FRED_API_KEY environment variable
                (free registration: https://fred.stlouisfed.org/docs/api/api_key.html)
    Cost:       Free.

Fallback source (no API key needed):
    yfinance TIP ETF ("TIP", iShares TIPS Bond ETF).  When FRED is
    unreachable or no key is configured, the fetcher pulls 30 trading
    days of TIP closes and converts them into a synthetic real-yield
    series.  TIP price is inversely correlated with real yields, so we
    invert the change to preserve the DFII10 sign convention.

Signal math (per plan §3):
    recent_5d_avg  = mean(newest 5 valid observations)
    older_20d_avg  = mean(observations at indices 15–24 when sorted newest-first)
    real_yield_change = recent_5d_avg - older_20d_avg

    Thresholds (percentage points):
        change <= -0.25  →  +0.10  (yields falling hard → gold bullish)
        change <= -0.10  →  +0.05
        change >=  0.25  →  -0.10  (yields rising hard  → gold bearish)
        change >=  0.10  →  -0.05
        else             →   0.0   (neutral)

Cache strategy:
    Parquet file at ``data/macro/tips_history.parquet``.
    Refreshed once per UTC calendar day.  The cache stores a ``source``
    marker ("fred" or "tip_etf_proxy") so operators can tell which
    backend produced the on-disk series.

Usage::

    fetcher = TIPSFetcher(cache_path=Path("data/macro/tips_history.parquet"))
    history = fetcher.fetch_history(days=30)   # list[float], newest-first
    bias    = compute_tips_bias(history)        # float in [-0.10, +0.10]
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

__all__ = ["TIPSFetcher", "compute_tips_bias", "compute_tips_yield_change"]

logger = logging.getLogger(__name__)

# FRED endpoint and series
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_SERIES_ID = "DFII10"

# How many observations to fetch from FRED (newest-first)
_FETCH_LIMIT = 30

# Average-window sizes (indices into newest-first list)
_RECENT_WINDOW_START = 0
_RECENT_WINDOW_END = 5      # indices [0, 5)  = 5 observations
_OLDER_WINDOW_START = 15
_OLDER_WINDOW_END = 25      # indices [15, 25) = up to 10 observations

# Minimum observations required for each window
_RECENT_MIN = 3
_OLDER_MIN = 3

# Bias thresholds (percentage points of real yield change)
_STRONG_FALL_THRESHOLD = -0.25   # yields down 25bp → gold strongly bullish
_MILD_FALL_THRESHOLD = -0.10
_MILD_RISE_THRESHOLD = 0.10
_STRONG_RISE_THRESHOLD = 0.25    # yields up 25bp → gold strongly bearish

# Cache refresh: once per UTC calendar day
_CACHE_TTL_HOURS = 24

# yfinance fallback: TIP ETF (iShares TIPS Bond Fund).  Price is inversely
# correlated with real yields — when yields fall, TIP rises.  We invert the
# percentage-point change so the synthetic series points in the same
# direction as DFII10 (yield-level delta in pp).
_TIP_ETF_TICKER = "TIP"
# Magnitude scale: TIP daily % changes run ~0.05–0.5%.  Mapping ``-1 * pct``
# to pp units keeps the downstream thresholds (±0.10, ±0.25) meaningful —
# a 0.3% TIP price move typically corresponds to ~0.10 pp real-yield move
# on the 10Y.  This is an intentionally conservative proxy, not a replacement
# for the true DFII10 series.
_TIP_ETF_HISTORY_PERIOD = "30d"

# Cache provenance markers
_SOURCE_FRED = "fred"
_SOURCE_TIP_PROXY = "tip_etf_proxy"


def compute_tips_yield_change(history: list[float]) -> float | None:
    """Compute the 20-day real-yield change from a newest-first value list.

    Parameters
    ----------
    history:
        DFII10 values in newest-first order, missing values already removed.

    Returns
    -------
    float | None
        ``recent_5d_avg - older_20d_avg`` in percentage points.
        Returns None if history has insufficient length for either window.
    """
    recent_slice = history[_RECENT_WINDOW_START:_RECENT_WINDOW_END]
    older_slice = history[_OLDER_WINDOW_START:_OLDER_WINDOW_END]

    if len(recent_slice) < _RECENT_MIN or len(older_slice) < _OLDER_MIN:
        return None

    recent_avg = mean(recent_slice)
    older_avg = mean(older_slice)
    return recent_avg - older_avg


def compute_tips_bias(history: list[float]) -> float:
    """Convert a DFII10 history to a gold-directional bias.

    Parameters
    ----------
    history:
        DFII10 values in newest-first order, missing values removed.

    Returns
    -------
    float
        Bias in [-0.10, +0.10].  0.0 if history is insufficient.
    """
    change = compute_tips_yield_change(history)
    if change is None:
        return 0.0
    return _bias_from_change(change)


def _bias_from_change(change: float) -> float:
    """Map a real-yield 20d change to a directional bias float.

    Gold is classically inversely correlated with real yields:
    yields falling → gold bullish (positive bias)
    yields rising  → gold bearish (negative bias)

    Parameters
    ----------
    change:
        recent_5d_avg − older_20d_avg, in percentage points.

    Returns
    -------
    float
        One of {-0.10, -0.05, 0.0, +0.05, +0.10}.
    """
    if change <= _STRONG_FALL_THRESHOLD:
        return +0.10
    if change <= _MILD_FALL_THRESHOLD:
        return +0.05
    if change >= _STRONG_RISE_THRESHOLD:
        return -0.10
    if change >= _MILD_RISE_THRESHOLD:
        return -0.05
    return 0.0


class TIPSFetcher:
    """Fetches DFII10 real-yield history from FRED with Parquet cache.

    The cache is refreshed once per UTC calendar day.  Falls back to
    stale cache on network failure; returns an empty list on total failure
    (no key, no cache, no network).

    Parameters
    ----------
    cache_path:
        Path to the Parquet cache file.  Parent directory is created
        if it does not exist.
    fred_api_key:
        FRED API key.  Falls back to ``FRED_API_KEY`` env var.
    fetch_timeout:
        HTTP request timeout in seconds.
    days:
        Number of observations to fetch and cache (newest-first).
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        fred_api_key: str | None = None,
        fetch_timeout: int = 10,
        days: int = _FETCH_LIMIT,
    ) -> None:
        self._cache_path = cache_path or Path("data/macro/tips_history.parquet")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self._fetch_timeout = fetch_timeout
        self._days = days

    def fetch_history(self, days: int | None = None) -> list[float]:
        """Return DFII10 values in newest-first order.

        Loads from Parquet cache when still fresh (same UTC day).  On
        cache miss or stale data, attempts the following sources in order:

        1. FRED DFII10 — requires ``FRED_API_KEY``.
        2. yfinance TIP ETF proxy — always attempted if FRED is
           unavailable.  Produces a synthetic real-yield series with the
           same sign convention as DFII10.
        3. Stale on-disk cache (if present) as last resort.

        Parameters
        ----------
        days:
            Number of observations to return.  Defaults to ``self._days``.

        Returns
        -------
        list[float]
            Values in newest-first order, with ``"."`` missing entries
            already removed.  May be shorter than ``days`` when data
            is unavailable.  Empty list only when every source failed.
        """
        limit = days or self._days

        cached = self._load_cache()
        if cached is not None and not self._cache_needs_refresh(cached):
            logger.debug("TIPS: returning fresh cache (%d rows)", len(cached))
            return self._extract_values(cached, limit)

        # --- Primary: FRED DFII10 -----------------------------------------
        if self._api_key:
            logger.debug("TIPS: fetching from FRED API (series=%s, limit=%d)", _SERIES_ID, limit)
            fresh = self._fetch_from_fred(limit)
            if fresh:
                self._save_cache(fresh, source=_SOURCE_FRED)
                return fresh
            logger.warning("TIPS: FRED returned no usable data; attempting yfinance TIP ETF fallback")
        else:
            logger.warning(
                "TIPS: FRED_API_KEY not configured; using yfinance TIP ETF fallback. "
                "Register a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
                "for a more precise real-yield signal."
            )

        # --- Fallback: yfinance TIP ETF proxy -----------------------------
        proxy = self._fetch_from_tip_etf(limit)
        if proxy:
            self._save_cache(proxy, source=_SOURCE_TIP_PROXY)
            return proxy

        # --- Last resort: stale cache -------------------------------------
        if cached is not None:
            logger.warning("TIPS: all live fetchers failed; returning stale cache (%d rows)", len(cached))
            return self._extract_values(cached, limit)

        logger.warning("TIPS: all sources failed (FRED + TIP ETF + cache); returning empty list")
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_fred(self, limit: int) -> list[float] | None:
        """Hit FRED API and return newest-first cleaned value list."""
        try:
            import requests

            params = {
                "series_id": _SERIES_ID,
                "api_key": self._api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": str(limit),
            }
            resp = requests.get(_FRED_URL, params=params, timeout=self._fetch_timeout)
            resp.raise_for_status()
            parsed = self._parse_fred_response(resp.json())
            if parsed is None:
                logger.warning("TIPS: FRED response parsed to no usable observations")
            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("TIPS: FRED fetch failed (series=%s): %s", _SERIES_ID, exc)
            return None

    def _fetch_from_tip_etf(self, limit: int) -> list[float] | None:
        """Return a synthetic DFII10-like series derived from TIP ETF closes.

        TIP (iShares TIPS Bond ETF) closes move inversely with real yields.
        We convert each daily close into a day-over-day percentage change,
        invert the sign so a rising real-yield maps to a positive number
        (matching FRED convention), and return the newest-first list.

        The resulting series is expressed in percentage points of daily
        real-yield delta — small but ordinally comparable to DFII10 daily
        changes.  The downstream :func:`compute_tips_yield_change` 5d/10d
        averaging amplifies the signal into the ±0.10 / ±0.25 pp bands.

        Returns
        -------
        list[float] | None
            Newest-first list, or None when yfinance is unavailable /
            returned insufficient rows.
        """
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except Exception as exc:  # noqa: BLE001
            logger.warning("TIPS: yfinance unavailable for TIP-ETF fallback: %s", exc)
            return None

        try:
            ticker = yf.Ticker(_TIP_ETF_TICKER)
            hist = ticker.history(period=_TIP_ETF_HISTORY_PERIOD)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TIPS: yfinance TIP history fetch failed: %s", exc)
            return None

        if hist is None or getattr(hist, "empty", True) or "Close" not in hist.columns:
            logger.warning("TIPS: yfinance returned empty TIP ETF history")
            return None

        closes = [float(c) for c in hist["Close"].tolist() if c is not None]
        if len(closes) < 2:
            logger.warning("TIPS: TIP ETF history too short (%d rows)", len(closes))
            return None

        # closes are oldest-first from yfinance; convert to day-over-day
        # percentage change then invert sign so "yields up → positive".
        deltas: list[float] = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev <= 0:
                continue
            pct_change = (curr - prev) / prev * 100.0  # percent of price
            # Invert sign: TIP up ⇒ yields down ⇒ negative yield delta
            deltas.append(-pct_change)

        if len(deltas) < 2:
            logger.warning("TIPS: TIP ETF delta series too short after cleaning")
            return None

        # Return newest-first to match FRED's sort order.
        newest_first = list(reversed(deltas))
        return newest_first[:limit]

    @staticmethod
    def _parse_fred_response(data: dict) -> list[float] | None:
        """Parse FRED JSON and return newest-first cleaned float list.

        Parameters
        ----------
        data:
            Parsed JSON from FRED API (must have ``observations`` key).

        Returns
        -------
        list[float] | None
            Cleaned values newest-first, or None if observations missing/empty.
        """
        observations = data.get("observations", [])
        if not observations:
            return None
        values = []
        for obs in observations:
            raw = obs.get("value", ".")
            if raw == ".":
                continue  # FRED uses "." for missing values
            try:
                values.append(float(raw))
            except (ValueError, TypeError):
                continue
        return values if values else None

    def _load_cache(self) -> pl.DataFrame | None:
        """Load Parquet cache, returning None if absent or corrupt."""
        if not self._cache_path.exists():
            return None
        try:
            import polars as pl

            return pl.read_parquet(self._cache_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TIPS: cache read failed at %s: %s", self._cache_path, exc)
            return None

    @staticmethod
    def _cache_needs_refresh(df: pl.DataFrame) -> bool:
        """Return True if cached data is from a previous UTC day."""
        if "fetched_at_utc" not in df.columns or df.is_empty():
            return True
        try:
            # fetched_at_utc stored as a string ISO timestamp in the single-row metadata
            raw = str(df["fetched_at_utc"][0])
            fetched_at = datetime.fromisoformat(raw)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            # Refresh if not the same UTC calendar day
            cutoff = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            return fetched_at < cutoff
        except Exception:
            return True

    def _save_cache(self, values: list[float], *, source: str = _SOURCE_FRED) -> None:
        """Persist values list to Parquet with a fetch-timestamp row.

        Parameters
        ----------
        values:
            Newest-first numeric series to persist.
        source:
            Origin marker — either :data:`_SOURCE_FRED` or
            :data:`_SOURCE_TIP_PROXY`.  Stored in a column alongside the
            values so operators can inspect provenance via ``polars`` or
            ``parquet-tools``.
        """
        try:
            import polars as pl

            now_str = datetime.now(timezone.utc).isoformat()
            df = pl.DataFrame(
                {
                    "value": values,
                    "fetched_at_utc": [now_str] * len(values),
                    "source": [source] * len(values),
                }
            )
            df.write_parquet(self._cache_path)
            logger.debug(
                "TIPS: saved %d rows to cache at %s (source=%s)",
                len(values),
                self._cache_path,
                source,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TIPS: cache write failed at %s: %s", self._cache_path, exc)

    @staticmethod
    def _extract_values(df: pl.DataFrame, limit: int) -> list[float]:
        """Extract up to ``limit`` float values from a cached DataFrame."""
        if "value" not in df.columns:
            return []
        col = df["value"]
        return [float(v) for v in col.to_list()[:limit]]
