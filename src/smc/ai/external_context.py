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

    Returns (dxy_value, dxy_direction).  Direction is ``"flat"`` when all
    tickers fail, mirroring legacy behaviour so downstream callers never
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
