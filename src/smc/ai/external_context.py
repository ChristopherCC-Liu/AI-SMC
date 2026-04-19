"""External macro data aggregation for AI regime classification.

Fetches DXY, VIX, US 10Y real rate, and COT data from free APIs with
in-memory caching and graceful degradation.  Never raises — macro data
is an optional enhancement for the debate pipeline.

Data sources:
    - VIX:          yfinance (^VIX)
    - DXY:          yfinance (DX-Y.NYB)
    - 10Y Real Rate: FRED API (DFII10) — requires FRED_API_KEY env var
    - COT:          Not yet implemented (placeholder)
    - Central Bank:  Not yet implemented (placeholder)

Usage::

    fetcher = ExternalContextFetcher(cache_ttl_minutes=60)
    ctx = fetcher.fetch()
    assert ctx.source_quality in ("live", "cached", "unavailable")
"""

from __future__ import annotations

import io
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from smc.ai.models import ExternalContext

__all__ = ["ExternalContextFetcher", "fetch_external_context"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VIX regime classification thresholds
# ---------------------------------------------------------------------------

_VIX_LOW = 15.0
_VIX_NORMAL = 20.0
_VIX_ELEVATED = 30.0

# ---------------------------------------------------------------------------
# DXY direction detection
# ---------------------------------------------------------------------------

_DXY_LOOKBACK_DAYS = 5
_DXY_DIRECTION_THRESHOLD_PCT = 0.3

# yfinance ticker fallback chain for DXY.
#   DX-Y.NYB — ICE Dollar Index spot (primary, 1st-party Yahoo listing)
#   DX=F     — ICE Dollar Index front-month future (backup, sometimes delisted)
#   UUP      — Invesco DB US Dollar Index Bullish Fund ETF (ETF proxy,
#              positively correlated with DXY; used only as last resort).
#
# If a VPS IP is rate-limited or blocked from one Yahoo backend (404 /
# empty frame), the next ticker is attempted.  Each failure is logged at
# WARNING so operators can diagnose which specific source broke.
_DXY_TICKERS: tuple[str, ...] = ("DX-Y.NYB", "DX=F", "UUP")


def _classify_vix_regime(vix: float) -> str:
    """Classify VIX level into a regime category."""
    if vix < _VIX_LOW:
        return "low"
    if vix < _VIX_NORMAL:
        return "normal"
    if vix < _VIX_ELEVATED:
        return "elevated"
    return "extreme"


# ---------------------------------------------------------------------------
# Individual fetchers (each returns a value or None on failure)
# ---------------------------------------------------------------------------


def _fetch_vix() -> tuple[float | None, str | None]:
    """Fetch current VIX level and regime via yfinance.

    Returns (vix_level, vix_regime) or (None, None) on failure.
    """
    try:
        import yfinance as yf  # noqa: F811

        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="1d")
        if hist.empty:
            logger.warning("VIX fetch: yfinance returned empty history for ^VIX")
            return None, None
        vix = float(hist["Close"].iloc[-1])
        return vix, _classify_vix_regime(vix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VIX fetch failed (yfinance ^VIX): %s", exc)
        return None, None


def _fetch_dxy_single(ticker_symbol: str) -> tuple[float | None, str]:
    """Attempt a DXY fetch from one yfinance ticker.

    Returns (value, direction).  Value is None if the ticker produced
    no rows.  Direction follows the same 5-day lookback rule as
    :func:`_fetch_dxy`.
    """
    import yfinance as yf  # noqa: F811

    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="10d")
    if hist.empty or len(hist) < 2:
        return None, "flat"
    current = float(hist["Close"].iloc[-1])
    lookback_idx = max(0, len(hist) - _DXY_LOOKBACK_DAYS - 1)
    previous = float(hist["Close"].iloc[lookback_idx])
    if previous <= 0:
        return current, "flat"
    pct_change = ((current - previous) / previous) * 100.0
    if pct_change > _DXY_DIRECTION_THRESHOLD_PCT:
        direction = "strengthening"
    elif pct_change < -_DXY_DIRECTION_THRESHOLD_PCT:
        direction = "weakening"
    else:
        direction = "flat"
    return current, direction


def _fetch_dxy() -> tuple[float | None, str]:
    """Fetch DXY value and direction via yfinance with ticker fallback chain.

    Tries tickers in :data:`_DXY_TICKERS` order (DX-Y.NYB → DX=F → UUP).
    The first ticker to return a non-empty history wins.  Each failure
    is logged at WARNING so a VPS-side outage can be diagnosed from logs.

    Fallback chain:
        1. yfinance DX-Y.NYB
        2. yfinance DX=F
        3. yfinance UUP
        4. stooq.com CSV (may require API key — logs warning on failure)
        5. exchangerate.host synthetic DXY (no auth, no key required)

    Returns (dxy_value, dxy_direction).  Direction is ``"flat"`` when all
    sources fail, mirroring legacy behaviour so downstream callers never
    have to handle a new error state.
    """
    last_exc: str | None = None
    for ticker_symbol in _DXY_TICKERS:
        try:
            value, direction = _fetch_dxy_single(ticker_symbol)
            if value is not None:
                logger.debug("DXY fetch: %s → value=%.4f direction=%s", ticker_symbol, value, direction)
                return value, direction
            logger.warning(
                "DXY fetch: yfinance returned empty history for ticker %s",
                ticker_symbol,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = f"{ticker_symbol}: {exc}"
            logger.warning("DXY fetch failed (yfinance %s): %s", ticker_symbol, exc)

    if last_exc is not None:
        logger.warning("DXY fetch: all tickers failed (last error: %s)", last_exc)
    else:
        logger.warning("DXY fetch: all tickers returned empty histories")

    # Fallback 4: stooq.com CSV endpoint (no auth, free — but may need API key on some IPs)
    stooq_value, stooq_direction = _fetch_dxy_stooq()
    if stooq_value is not None:
        logger.debug(
            "DXY fetch: stooq fallback succeeded → value=%.4f direction=%s",
            stooq_value,
            stooq_direction,
        )
        return stooq_value, stooq_direction

    logger.warning("DXY fetch: stooq fallback failed; trying exchangerate.host synthetic DXY")

    # Fallback 5: exchangerate.host synthetic DXY (free, no auth required)
    exr_value, exr_direction = _fetch_dxy_exchangerate_host()
    if exr_value is not None:
        logger.debug(
            "DXY fetch: exchangerate.host synthetic fallback succeeded → value=%.4f direction=%s",
            exr_value,
            exr_direction,
        )
        return exr_value, exr_direction

    logger.warning("DXY fetch: all sources failed (including exchangerate.host); returning (None, 'flat')")
    return None, "flat"


def _fetch_dxy_stooq() -> tuple[float | None, str]:
    """Fetch DXY value and direction from stooq.com CSV endpoint.

    Used as the ultimate fallback after all yfinance tickers fail.
    No authentication required.

    Endpoint::

        https://stooq.com/q/d/l/?s=^dxy&d1=<YYYYMMDD>&d2=<YYYYMMDD>&i=d

    Returns
    -------
    (value, direction)
        ``value`` is the latest Close, ``direction`` is derived from the
        last-two-close comparison (``"rising"`` / ``"falling"`` / ``"flat"``).
        Returns ``(None, "flat")`` on any failure.
    """
    try:
        import csv

        import requests

        today = datetime.now(tz=timezone.utc)
        date_from = (today - timedelta(days=_DXY_LOOKBACK_DAYS + 2)).strftime("%Y%m%d")
        date_to = today.strftime("%Y%m%d")
        url = (
            f"https://stooq.com/q/d/l/?s=^dxy"
            f"&d1={date_from}&d2={date_to}&i=d"
        )
        headers = {
            "User-Agent": (
                "AI-SMC/1.0 (macro-data-fetcher; github.com/AI-SMC; "
                "contact: operator)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        reader = csv.DictReader(io.StringIO(resp.text))
        rows = [row for row in reader if row.get("Close", "").strip()]
        if len(rows) < 1:
            logger.warning("DXY stooq: CSV returned no usable rows")
            return None, "flat"

        latest_close = float(rows[-1]["Close"])

        if len(rows) >= 2:
            prev_close = float(rows[-2]["Close"])
            if prev_close > 0:
                pct_change = ((latest_close - prev_close) / prev_close) * 100.0
                if pct_change > _DXY_DIRECTION_THRESHOLD_PCT:
                    direction = "rising"
                elif pct_change < -_DXY_DIRECTION_THRESHOLD_PCT:
                    direction = "falling"
                else:
                    direction = "flat"
            else:
                direction = "flat"
        else:
            direction = "flat"

        return latest_close, direction
    except Exception as exc:  # noqa: BLE001
        logger.warning("DXY stooq fetch failed: %s", exc)
        return None, "flat"


def _fetch_dxy_exchangerate_host() -> tuple[float | None, str]:
    """Compute a synthetic DXY from exchangerate.host free API (no auth required).

    Uses the official ICE DXY formula::

        DXY = 50.14348112
              × EUR^(-0.576)
              × JPY^(0.136)
              × GBP^(-0.119)
              × CAD^(0.091)
              × SEK^(0.042)
              × CHF^(0.036)

    Where the exponent operands are the USD-quoted rates for each currency
    pair (USD/JPY, USD/CAD, USD/SEK, USD/CHF) or the inverse (EUR/USD and
    GBP/USD).

    exchangerate.host returns ``{base: "USD", rates: {EUR: X, JPY: Y, ...}}``
    where each value is "how many units of the foreign currency equal 1 USD".
    For the DXY formula this means:
    - EUR and GBP appear as USD-priced (EUR/USD = 1 / rates["EUR"]).
    - JPY, CAD, SEK, CHF are already in USD-base form (USD/CCY = rates["CCY"]).

    Accuracy note:
        Synthetic DXY is typically within ±0.5% of the official ICE index,
        which is sufficient for signal-level bias detection.

    Returns
    -------
    (value, direction)
        ``value`` is the synthetic DXY level for the latest available day.
        ``direction`` is derived from a two-day comparison using the same
        :data:`_DXY_DIRECTION_THRESHOLD_PCT` threshold as other fetchers.
        Returns ``(None, "flat")`` on any failure.
    """
    try:
        import math

        import requests

        today = datetime.now(tz=timezone.utc)
        # Request the last 5 calendar days to ensure we have at least 2 business days
        start_date = (today - timedelta(days=_DXY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        url = "https://api.exchangerate.host/timeseries"
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "base": "USD",
            "symbols": "EUR,JPY,GBP,CAD,SEK,CHF",
        }
        headers = {
            "User-Agent": (
                "AI-SMC/1.0 (macro-data-fetcher; synthetic-DXY; "
                "contact: operator)"
            )
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        if not data.get("success", False):
            logger.warning("DXY exchangerate.host: API returned success=false")
            return None, "flat"

        rates_by_date: dict = data.get("rates", {})
        if not rates_by_date:
            logger.warning("DXY exchangerate.host: no rates in response")
            return None, "flat"

        sorted_dates = sorted(rates_by_date.keys())
        if len(sorted_dates) < 1:
            logger.warning("DXY exchangerate.host: empty dates list")
            return None, "flat"

        def _compute_dxy(rates: dict) -> float | None:
            """Apply ICE DXY formula to a single day's rates dict."""
            try:
                eur_usd = 1.0 / float(rates["EUR"])   # EUR/USD (inverse of USD/EUR)
                usd_jpy = float(rates["JPY"])          # USD/JPY direct
                gbp_usd = 1.0 / float(rates["GBP"])   # GBP/USD (inverse of USD/GBP)
                usd_cad = float(rates["CAD"])          # USD/CAD direct
                usd_sek = float(rates["SEK"])          # USD/SEK direct
                usd_chf = float(rates["CHF"])          # USD/CHF direct

                dxy = (
                    50.14348112
                    * math.pow(eur_usd, -0.576)
                    * math.pow(usd_jpy, 0.136)
                    * math.pow(gbp_usd, -0.119)
                    * math.pow(usd_cad, 0.091)
                    * math.pow(usd_sek, 0.042)
                    * math.pow(usd_chf, 0.036)
                )
                return dxy
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                logger.warning("DXY exchangerate.host: formula error for rates=%s: %s", rates, exc)
                return None

        latest_rates = rates_by_date[sorted_dates[-1]]
        latest_dxy = _compute_dxy(latest_rates)
        if latest_dxy is None:
            return None, "flat"

        # Compute direction using previous day if available
        if len(sorted_dates) >= 2:
            prev_rates = rates_by_date[sorted_dates[-2]]
            prev_dxy = _compute_dxy(prev_rates)
            if prev_dxy is not None and prev_dxy > 0:
                pct_change = ((latest_dxy - prev_dxy) / prev_dxy) * 100.0
                if pct_change > _DXY_DIRECTION_THRESHOLD_PCT:
                    direction = "strengthening"
                elif pct_change < -_DXY_DIRECTION_THRESHOLD_PCT:
                    direction = "weakening"
                else:
                    direction = "flat"
            else:
                direction = "flat"
        else:
            direction = "flat"

        return latest_dxy, direction
    except Exception as exc:  # noqa: BLE001
        logger.warning("DXY exchangerate.host fetch failed: %s", exc)
        return None, "flat"


def _fetch_real_rate() -> float | None:
    """Fetch US 10Y real rate from FRED (DFII10 series).

    Requires ``FRED_API_KEY`` environment variable (free registration at
    https://fred.stlouisfed.org/docs/api/api_key.html).  Returns None if
    the key is missing or FRED is unreachable — callers that need a
    yields-adjacent signal should use :class:`smc.ai.tips_fetcher.TIPSFetcher`
    which provides a yfinance TIP-ETF fallback.
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        logger.debug("FRED real rate: FRED_API_KEY env var not set; skipping")
        return None
    try:
        import requests

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "DFII10",
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        if not observations:
            logger.warning("FRED real rate: DFII10 response had no observations")
            return None
        value = observations[0].get("value", ".")
        if value == ".":
            return None
        return float(value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED real rate fetch failed (DFII10): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fetcher with caching
# ---------------------------------------------------------------------------


class ExternalContextFetcher:
    """Fetches external macro data with in-memory TTL cache.

    Thread-safe.  Never raises from ``fetch()`` — returns an
    ``ExternalContext`` with ``source_quality="unavailable"`` on
    total failure.

    Parameters
    ----------
    cache_ttl_minutes:
        How long cached data remains valid.  Default 60 minutes.
        DXY and rates don't change fast enough to justify per-H4 fetches.
    """

    def __init__(self, cache_ttl_minutes: int = 60) -> None:
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._lock = threading.Lock()
        self._cached: ExternalContext | None = None
        self._cached_at: datetime | None = None

    def fetch(self) -> ExternalContext:
        """Fetch external context, returning cached data if still valid.

        Returns
        -------
        ExternalContext
            Always returns a valid object.  Check ``source_quality``
            to determine freshness.
        """
        with self._lock:
            now = datetime.now(tz=timezone.utc)

            # Return cached if still valid
            if (
                self._cached is not None
                and self._cached_at is not None
                and (now - self._cached_at) < self._cache_ttl
            ):
                return ExternalContext(
                    dxy_direction=self._cached.dxy_direction,
                    dxy_value=self._cached.dxy_value,
                    vix_level=self._cached.vix_level,
                    vix_regime=self._cached.vix_regime,
                    real_rate_10y=self._cached.real_rate_10y,
                    cot_net_spec=self._cached.cot_net_spec,
                    central_bank_stance=self._cached.central_bank_stance,
                    fetched_at=self._cached.fetched_at,
                    source_quality="cached",
                )

        # Fetch outside lock to avoid blocking
        ctx = self._fetch_all()

        with self._lock:
            if ctx.source_quality != "unavailable":
                self._cached = ctx
                self._cached_at = datetime.now(tz=timezone.utc)

        return ctx

    def invalidate(self) -> None:
        """Force-invalidate the cache.  Next ``fetch()`` will re-query."""
        with self._lock:
            self._cached = None
            self._cached_at = None

    @staticmethod
    def _fetch_all() -> ExternalContext:
        """Attempt to fetch all data sources.  Never raises."""
        now = datetime.now(tz=timezone.utc)

        vix_level, vix_regime = _fetch_vix()
        dxy_value, dxy_direction = _fetch_dxy()
        real_rate = _fetch_real_rate()

        # Determine overall quality
        has_any = vix_level is not None or dxy_value is not None or real_rate is not None
        quality = "live" if has_any else "unavailable"

        return ExternalContext(
            dxy_direction=dxy_direction,  # type: ignore[arg-type]
            dxy_value=dxy_value,
            vix_level=vix_level,
            vix_regime=vix_regime,  # type: ignore[arg-type]
            real_rate_10y=real_rate,
            cot_net_spec=None,
            central_bank_stance=None,
            fetched_at=now,
            source_quality=quality,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_DEFAULT_FETCHER: ExternalContextFetcher | None = None
_DEFAULT_LOCK = threading.Lock()


def fetch_external_context(cache_ttl_minutes: int = 60) -> ExternalContext:
    """Module-level convenience function with a shared singleton fetcher.

    Creates the fetcher on first call.  Thread-safe.
    """
    global _DEFAULT_FETCHER
    with _DEFAULT_LOCK:
        if _DEFAULT_FETCHER is None:
            _DEFAULT_FETCHER = ExternalContextFetcher(cache_ttl_minutes=cache_ttl_minutes)
    return _DEFAULT_FETCHER.fetch()
