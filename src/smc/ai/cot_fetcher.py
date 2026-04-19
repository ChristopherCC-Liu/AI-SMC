"""CFTC COT (Commitment of Traders) fetcher for gold futures.

Fetches weekly COT data from the CFTC Socrata Open Data API
(no auth required) and computes a net non-commercial positioning
percentage for XAU/COMEX futures.

Data source:
    Socrata endpoint: https://publicreporting.cftc.gov/resource/gpe5-46if.json
    Market code:      088691 (GOLD - COMMODITY EXCHANGE INC.)
    Release cadence:  Tuesday ~3:30pm ET for prior-Tuesday data (3-day lag)
    Cost:             Free, no API key, no published rate limit.

Cache strategy:
    Parquet file at ``data/macro/cot_gold_history.parquet``.
    Refresh weekly: fetcher checks if the latest cached row is older than 7
    days; if so, pulls the most recent 104 rows and appends new records.

Usage::

    fetcher = COTFetcher(cache_path=Path("data/macro/cot_gold_history.parquet"))
    df = fetcher.fetch()          # returns polars DataFrame or None
    bias = compute_cot_bias(df)   # returns float in [-0.10, +0.10] or None
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

__all__ = ["COTFetcher", "compute_cot_bias", "compute_cot_net_long_pct"]

logger = logging.getLogger(__name__)

# CFTC Socrata Open Data endpoint — Financial Futures Disaggregated
_CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"

# COMEX Gold CFTC market code
_GOLD_MARKET_CODE = "088691"

# Rolling window for percentile calculation (104 weeks ≈ 2 years)
_ROLLING_WINDOW = 104

# Minimum rows required to compute a reliable rolling percentile
_MIN_ROWS_FOR_SIGNAL = 8

# Cache refresh threshold (COT is weekly; 7 days is conservative)
_CACHE_TTL_DAYS = 7

# Maximum rows to fetch per request (keeps response small)
_FETCH_LIMIT = 120


def compute_cot_net_long_pct(
    noncomm_long: int | float,
    noncomm_short: int | float,
    open_interest: int | float,
) -> float | None:
    """Compute net non-commercial long percentage relative to open interest.

    Parameters
    ----------
    noncomm_long:
        Non-commercial long positions (large speculator longs).
    noncomm_short:
        Non-commercial short positions (large speculator shorts).
    open_interest:
        Total open interest for the contract.

    Returns
    -------
    float | None
        ``(long - short) / open_interest * 100``.  Returns None if
        open_interest is zero or negative.
    """
    if open_interest <= 0:
        return None
    return (float(noncomm_long) - float(noncomm_short)) / float(open_interest) * 100.0


def compute_cot_bias(history_df: pl.DataFrame) -> float:
    """Compute COT bias from a history DataFrame.

    Uses a 104-week rolling percentile rank with contrarian logic at
    extremes and momentum logic in the mid-range.

    Percentile thresholds (from plan §2):
        >0.90  → contrarian bearish (crowded long) → -0.10
        >0.70  → mild momentum support             → +0.05
        <0.10  → contrarian bullish (crowded short) → +0.10
        <0.30  → mild bearish lean                 → -0.05
        else   → neutral                           → 0.0

    Parameters
    ----------
    history_df:
        Polars DataFrame with at least column ``cot_net_long_pct``,
        sorted ascending by date (oldest first).

    Returns
    -------
    float
        Bias value in [-0.10, +0.10].

    Raises
    ------
    ValueError
        If DataFrame is missing required column or has fewer rows than
        ``_MIN_ROWS_FOR_SIGNAL``.
    """
    import polars as pl

    if "cot_net_long_pct" not in history_df.columns:
        raise ValueError("DataFrame missing required column: cot_net_long_pct")

    values = history_df["cot_net_long_pct"].drop_nulls()
    n = len(values)

    if n < _MIN_ROWS_FOR_SIGNAL:
        raise ValueError(
            f"Insufficient data for COT signal: {n} rows < minimum {_MIN_ROWS_FOR_SIGNAL}"
        )

    window = min(_ROLLING_WINDOW, n)
    window_values = values.slice(max(0, n - window), window)

    latest = float(values[-1])
    sorted_window = window_values.sort()
    window_n = len(sorted_window)

    # Compute percentile rank of latest value within the window
    # rank = fraction of window values strictly less than the latest
    n_below = int((sorted_window < latest).sum())
    pct_rank = n_below / window_n

    return _bias_from_percentile(pct_rank)


def _bias_from_percentile(pct_rank: float) -> float:
    """Convert a 0–1 percentile rank to a bias float per plan §2.

    Parameters
    ----------
    pct_rank:
        Percentile rank in [0, 1] of the latest COT net long pct.

    Returns
    -------
    float
        Bias in {-0.10, -0.05, 0.0, +0.05, +0.10}.
    """
    if pct_rank > 0.90:
        # Overcrowded long — contrarian bearish signal
        return -0.10
    if pct_rank > 0.70:
        # Moderately extended long — mild momentum support for gold
        return +0.05
    if pct_rank < 0.10:
        # Overcrowded short — contrarian bullish signal
        return +0.10
    if pct_rank < 0.30:
        # Moderately extended short — mild bearish lean
        return -0.05
    # Mid-range neutral zone (30th–70th percentile)
    return 0.0


class COTFetcher:
    """Fetches and caches CFTC COT data for COMEX Gold.

    Parameters
    ----------
    cache_path:
        Path to the Parquet cache file.  Parent directory is created
        if it does not exist.
    fetch_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        fetch_timeout: int = 15,
    ) -> None:
        self._cache_path = cache_path or Path("data/macro/cot_gold_history.parquet")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._fetch_timeout = fetch_timeout

    def fetch(self) -> pl.DataFrame | None:
        """Return COT history as a Polars DataFrame, using cache when fresh.

        Returns
        -------
        polars.DataFrame | None
            DataFrame with columns:
            ``report_date``, ``noncomm_long``, ``noncomm_short``,
            ``open_interest``, ``cot_net_long_pct``.
            Sorted ascending by ``report_date`` (oldest first).
            Returns None on network failure or parse error.
        """
        # Try to load from cache
        cached = self._load_cache()
        if cached is not None and not self._cache_needs_refresh(cached):
            logger.debug("COT: returning fresh cache (%d rows)", len(cached))
            return cached

        # Fetch fresh data
        logger.debug("COT: fetching from CFTC Socrata API")
        fresh = self._fetch_from_cftc()
        if fresh is None:
            # Network failed — return stale cache if we have it
            if cached is not None:
                logger.warning(
                    "COT: network fetch failed; returning stale cache (%d rows)", len(cached)
                )
                return cached
            return None

        # Merge fresh data with cache and persist
        merged = self._merge_and_save(cached, fresh)
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> pl.DataFrame | None:
        """Load cached Parquet, or None if absent/corrupt."""
        if not self._cache_path.exists():
            return None
        try:
            import polars as pl

            return pl.read_parquet(self._cache_path)
        except Exception:
            logger.debug("COT: cache read failed", exc_info=True)
            return None

    def _cache_needs_refresh(self, df: pl.DataFrame) -> bool:
        """Return True if the latest row is older than _CACHE_TTL_DAYS."""
        import polars as pl

        if "report_date" not in df.columns or df.is_empty():
            return True
        try:
            latest_date = df["report_date"].max()
            if latest_date is None:
                return True
            # report_date stored as Date type
            if hasattr(latest_date, "date"):
                latest_dt = datetime(
                    latest_date.year,
                    latest_date.month,
                    latest_date.day,
                    tzinfo=timezone.utc,
                )
            else:
                # Already a datetime
                latest_dt = latest_date
                if latest_dt.tzinfo is None:
                    latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            cutoff = datetime.now(timezone.utc) - timedelta(days=_CACHE_TTL_DAYS)
            return latest_dt < cutoff
        except Exception:
            return True

    def _fetch_from_cftc(self) -> pl.DataFrame | None:
        """Fetch raw COT rows from CFTC Socrata, return parsed DataFrame."""
        try:
            import requests

            params = {
                "$where": f"cftc_commodity_code='{_GOLD_MARKET_CODE}'",
                "$order": "report_date_as_yyyy_mm_dd DESC",
                "$limit": str(_FETCH_LIMIT),
                "$select": ",".join([
                    "report_date_as_yyyy_mm_dd",
                    "noncomm_positions_long_all",
                    "noncomm_positions_short_all",
                    "open_interest_all",
                ]),
            }
            resp = requests.get(_CFTC_URL, params=params, timeout=self._fetch_timeout)
            resp.raise_for_status()
            rows = resp.json()
            return self._parse_rows(rows)
        except Exception:
            logger.debug("COT: CFTC fetch failed", exc_info=True)
            return None

    def _parse_rows(self, rows: list[dict]) -> pl.DataFrame | None:
        """Parse a list of Socrata JSON rows into a cleaned DataFrame.

        Parameters
        ----------
        rows:
            List of dicts from CFTC Socrata API.

        Returns
        -------
        polars.DataFrame | None
            Parsed and cleaned DataFrame, or None if rows is empty or
            all rows fail parsing.
        """
        import polars as pl

        if not rows:
            return None

        records = []
        for row in rows:
            parsed = self._parse_single_row(row)
            if parsed is not None:
                records.append(parsed)

        if not records:
            return None

        df = pl.DataFrame(
            {
                "report_date": [r["report_date"] for r in records],
                "noncomm_long": [r["noncomm_long"] for r in records],
                "noncomm_short": [r["noncomm_short"] for r in records],
                "open_interest": [r["open_interest"] for r in records],
            }
        )
        df = df.with_columns(
            pl.col("report_date").cast(pl.Date)
        )

        # Compute net long pct
        df = df.with_columns(
            (
                (pl.col("noncomm_long").cast(pl.Float64) - pl.col("noncomm_short").cast(pl.Float64))
                / pl.col("open_interest").cast(pl.Float64)
                * 100.0
            ).alias("cot_net_long_pct")
        )

        # Sort ascending by date (oldest first)
        df = df.sort("report_date", descending=False)
        return df

    def _parse_single_row(self, row: dict) -> dict | None:
        """Parse a single Socrata JSON row. Returns None if invalid."""
        try:
            date_str = row.get("report_date_as_yyyy_mm_dd", "")
            if not date_str:
                return None
            report_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()

            noncomm_long = int(row.get("noncomm_positions_long_all", 0))
            noncomm_short = int(row.get("noncomm_positions_short_all", 0))
            open_interest = int(row.get("open_interest_all", 0))

            if open_interest <= 0:
                return None

            return {
                "report_date": report_date,
                "noncomm_long": noncomm_long,
                "noncomm_short": noncomm_short,
                "open_interest": open_interest,
            }
        except (KeyError, ValueError, TypeError):
            return None

    def _merge_and_save(
        self,
        cached: pl.DataFrame | None,
        fresh: pl.DataFrame,
    ) -> pl.DataFrame:
        """Merge fresh rows into cached DataFrame, deduplicate, save to Parquet."""
        import polars as pl

        if cached is not None and not cached.is_empty():
            combined = pl.concat([cached, fresh], how="diagonal")
        else:
            combined = fresh

        # Deduplicate by date, keep latest parse
        combined = combined.unique(subset=["report_date"], keep="last")
        combined = combined.sort("report_date", descending=False)

        try:
            combined.write_parquet(self._cache_path)
            logger.debug("COT: saved %d rows to cache at %s", len(combined), self._cache_path)
        except Exception:
            logger.debug("COT: cache write failed", exc_info=True)

        return combined
