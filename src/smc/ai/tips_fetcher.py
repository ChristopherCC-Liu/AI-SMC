"""TIPS real-yield fetcher for macro overlay (Alt-B round 4).

Fetches daily DFII10 (10-Year Treasury Inflation-Indexed Security,
Constant Maturity) observations from the FRED API, computes a
20-day directional change, and maps it to a bias float in
[-0.10, +0.10].

Data source:
    FRED API:   https://api.stlouisfed.org/fred/series/observations
    Series:     DFII10
    Auth:       FRED_API_KEY environment variable (free key)
    Cost:       Free.

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
    Refreshed once per UTC calendar day.

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

        Loads from Parquet cache when still fresh (same UTC day).
        On cache miss or stale, fetches from FRED.  On any FRED failure,
        returns stale cache if available, else empty list.

        Parameters
        ----------
        days:
            Number of observations to return.  Defaults to ``self._days``.

        Returns
        -------
        list[float]
            Values in newest-first order, with ``"."`` missing entries
            already removed.  May be shorter than ``days`` when data
            is unavailable.
        """
        limit = days or self._days

        cached = self._load_cache()
        if cached is not None and not self._cache_needs_refresh(cached):
            logger.debug("TIPS: returning fresh cache (%d rows)", len(cached))
            return self._extract_values(cached, limit)

        if not self._api_key:
            logger.debug("TIPS: no FRED_API_KEY; returning stale cache or empty list")
            if cached is not None:
                return self._extract_values(cached, limit)
            return []

        logger.debug("TIPS: fetching from FRED API (series=%s, limit=%d)", _SERIES_ID, limit)
        fresh = self._fetch_from_fred(limit)
        if fresh is None:
            if cached is not None:
                logger.warning("TIPS: FRED fetch failed; returning stale cache (%d rows)", len(cached))
                return self._extract_values(cached, limit)
            return []

        self._save_cache(fresh)
        return fresh

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
            return self._parse_fred_response(resp.json())
        except Exception:
            logger.debug("TIPS: FRED fetch failed", exc_info=True)
            return None

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
        except Exception:
            logger.debug("TIPS: cache read failed", exc_info=True)
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

    def _save_cache(self, values: list[float]) -> None:
        """Persist values list to Parquet with a fetch-timestamp row."""
        try:
            import polars as pl

            now_str = datetime.now(timezone.utc).isoformat()
            df = pl.DataFrame(
                {
                    "value": values,
                    "fetched_at_utc": [now_str] * len(values),
                }
            )
            df.write_parquet(self._cache_path)
            logger.debug("TIPS: saved %d rows to cache at %s", len(values), self._cache_path)
        except Exception:
            logger.debug("TIPS: cache write failed", exc_info=True)

    @staticmethod
    def _extract_values(df: pl.DataFrame, limit: int) -> list[float]:
        """Extract up to ``limit`` float values from a cached DataFrame."""
        if "value" not in df.columns:
            return []
        col = df["value"]
        return [float(v) for v in col.to_list()[:limit]]
