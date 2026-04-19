"""Macro overlay scoring for SMC confluence boost.

Computes a single bias float in [-0.3, +0.3] from 3 orthogonal macro
sources: COT positioning, TIPS real yield, DXY.  Injected into
``score_confluence`` as an additive bonus.

Design principles:
    - Orthogonal to SMC: adds new signal dimensions, not re-tunes existing.
    - Graceful degradation: any fetcher failure → that source = 0.0.
    - Conservative: max absolute bias = 0.3 (cannot override strong SMC).
    - Cache-first: 7-day Parquet cache for COT; daily Parquet for TIPS; 24h in-memory for DXY.

Implementation status (Alt-B Round 4 W1D1-D5, 2026-04-19):
    - DXY source: implemented (reuses ``_fetch_dxy`` from external_context).
    - COT source: implemented via ``COTFetcher`` + 104-week rolling percentile.
      Contrarian at extremes (>90th → -0.10, <10th → +0.10), mild momentum
      in the 70th–90th / 10th–30th bands (±0.05), neutral in the middle.
    - TIPS yield source: implemented via ``TIPSFetcher`` + FRED DFII10 series.
      Computes ``recent_5d_avg - older_20d_avg`` in pp; inverse-correlation
      mapping: yield rise → bearish gold, yield fall → bullish gold.
      Thresholds: ±0.10 pp mild, ±0.25 pp strong.  1-day Parquet cache.

COT branch behaviour:
    The COT component calls ``COTFetcher(cache_path=<cache_dir>/cot_gold_history.parquet).fetch()``
    to obtain up to 104 weeks of COMEX Gold non-commercial net positioning data.
    ``compute_cot_bias()`` converts the latest data point to a percentile rank
    within the rolling window and maps it to a bias float per plan §2:

    +-----------------------------------------------+--------+
    | Condition (rolling 104-week percentile)        | Bias   |
    +===============================================+========+
    | pct_rank > 0.90  (crowded long)               | -0.10  |
    | 0.70 < pct_rank ≤ 0.90  (extended long)       | +0.05  |
    | 0.30 ≤ pct_rank ≤ 0.70  (neutral mid-range)   |  0.00  |
    | 0.10 ≤ pct_rank < 0.30  (extended short)      | -0.05  |
    | pct_rank < 0.10  (crowded short)              | +0.10  |
    +-----------------------------------------------+--------+

    Network failures or parse errors degrade to 0.0 (never raise).

TIPS real-yield branch behaviour:
    The TIPS component calls ``TIPSFetcher(cache_path=<cache_dir>/tips_history.parquet).fetch_history(30)``
    to obtain up to 30 daily DFII10 observations.  Computes::

        real_yield_change = mean(history[0:5]) - mean(history[15:25])

    where history is sorted newest-first and ``"."`` missing values are removed.
    Gold is inversely correlated with real yields (Erb & Harvey 2013):

    +-----------------------------------+--------+
    | Condition (pp change)             | Bias   |
    +===================================+========+
    | change ≤ -0.25 (sharp fall)       | +0.10  |
    | change ≤ -0.10 (mild fall)        | +0.05  |
    | change ≥ +0.25 (sharp rise)       | -0.10  |
    | change ≥ +0.10 (mild rise)        | -0.05  |
    | else (neutral)                    |  0.00  |
    +-----------------------------------+--------+

    Requires ``FRED_API_KEY`` env var.  Missing key or network failure
    degrades to 0.0 (never raise).  1-day Parquet cache at
    ``data/macro/tips_history.parquet``.

Usage::

    layer = MacroLayer()
    bias = layer.compute_macro_bias(instrument="XAUUSD")
    # Use bias.total_bias as input to score_confluence(...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from smc.ai.cot_fetcher import COTFetcher, compute_cot_bias
from smc.ai.external_context import ExternalContextFetcher
from smc.ai.tips_fetcher import TIPSFetcher, compute_tips_bias

logger = logging.getLogger(__name__)

__all__ = ["MacroLayer", "MacroBias"]

# Hard cap: macro overlay cannot contribute more than 0.3 to confluence.
# This preserves SMC as the primary signal source.
_MAX_ABS_BIAS = 0.3

# DXY thresholds (5-day % change). Gold is inversely correlated with DXY.
_DXY_STRONG_WEAKNESS_PCT = -1.0  # USD down ≥ 1% → strongly bullish gold
_DXY_MILD_WEAKNESS_PCT = -0.3
_DXY_MILD_STRENGTH_PCT = 0.3
_DXY_STRONG_STRENGTH_PCT = 1.0

# Direction thresholds for the aggregated bias
_DIRECTION_THRESHOLD = 0.05


@dataclass(frozen=True)
class MacroBias:
    """Breakdown of macro overlay contribution for logging and diagnostics.

    All bias components are in [-0.10, +0.10].  The aggregated ``total_bias``
    is clamped to [-_MAX_ABS_BIAS, +_MAX_ABS_BIAS].  Positive values are
    bullish for the instrument; negative are bearish.
    """

    cot_bias: float
    yield_bias: float
    dxy_bias: float
    total_bias: float
    sources_available: int
    direction: Literal["bullish", "bearish", "neutral"]
    timestamp_utc: str


class MacroLayer:
    """Orchestrates macro bias computation across COT, TIPS, DXY.

    Parameters
    ----------
    cache_dir:
        Directory for per-source Parquet caches.  Created if missing.
    fred_api_key:
        FRED API key for TIPS real yield.  If None, yield source is skipped.
    cache_ttl_hours:
        TTL for the underlying ``ExternalContextFetcher``.  Default 24h.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        fred_api_key: str | None = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        self._cache_dir = cache_dir or Path("data/macro")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._external = ExternalContextFetcher(
            cache_ttl_minutes=cache_ttl_hours * 60,
        )
        self._fred_api_key = fred_api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_macro_bias(self, instrument: str = "XAUUSD") -> MacroBias:
        """Return the aggregated macro bias for the given instrument.

        Guaranteed never to raise — on total fetcher failure returns a
        neutral ``MacroBias`` with ``sources_available=0``.
        """
        cot = self._safe_cot_bias(instrument)
        yield_b = self._safe_yield_bias()
        dxy = self._safe_dxy_bias()

        sources_available = sum(1 for b in (cot, yield_b, dxy) if b is not None)
        cot_v = cot if cot is not None else 0.0
        yield_v = yield_b if yield_b is not None else 0.0
        dxy_v = dxy if dxy is not None else 0.0
        total = cot_v + yield_v + dxy_v
        total = max(-_MAX_ABS_BIAS, min(_MAX_ABS_BIAS, total))

        direction: Literal["bullish", "bearish", "neutral"]
        if total > _DIRECTION_THRESHOLD:
            direction = "bullish"
        elif total < -_DIRECTION_THRESHOLD:
            direction = "bearish"
        else:
            direction = "neutral"

        return MacroBias(
            cot_bias=round(cot_v, 4),
            yield_bias=round(yield_v, 4),
            dxy_bias=round(dxy_v, 4),
            total_bias=round(total, 4),
            sources_available=sources_available,
            direction=direction,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Safe wrappers — never raise
    # ------------------------------------------------------------------

    def _safe_cot_bias(self, instrument: str) -> float | None:
        try:
            return self._cot_bias(instrument)
        except Exception:  # noqa: BLE001
            logger.debug("COT bias fetch failed", exc_info=True)
            return None

    def _safe_yield_bias(self) -> float | None:
        try:
            return self._yield_bias()
        except Exception:  # noqa: BLE001
            logger.debug("Real yield bias fetch failed", exc_info=True)
            return None

    def _safe_dxy_bias(self) -> float | None:
        try:
            return self._dxy_bias()
        except Exception:  # noqa: BLE001
            logger.debug("DXY bias fetch failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Per-source implementations
    # ------------------------------------------------------------------

    def _cot_bias(self, instrument: str) -> float:
        """Fetch COT data and compute contrarian/momentum bias.

        Fetches up to 104 weeks of COMEX Gold (XAU) non-commercial net
        positioning from CFTC via ``COTFetcher``.  Maps the latest data
        point's 104-week percentile rank to a bias float per plan §2:

        - >90th percentile (crowded long)  → -0.10  contrarian bearish
        - >70th percentile (extended long) → +0.05  mild momentum
        - <10th percentile (crowded short) → +0.10  contrarian bullish
        - <30th percentile (extended short) → -0.05 mild bearish lean
        - 30th–70th (neutral mid-range)    →  0.0   no signal

        Only supports XAUUSD.  Returns 0.0 without error for other
        instruments.  Cache lives at ``<cache_dir>/cot_gold_history.parquet``,
        refreshed when latest cached row is older than 7 days.

        Parameters
        ----------
        instrument:
            Trading instrument identifier (e.g. "XAUUSD").

        Returns
        -------
        float
            COT bias in [-0.10, +0.10].

        Raises
        ------
        RuntimeError
            If COT data fetch returns no rows or bias computation fails.
        """
        if instrument.upper() != "XAUUSD":
            logger.debug("COT bias only supported for XAUUSD; got %s", instrument)
            return 0.0

        cache_path = self._cache_dir / "cot_gold_history.parquet"
        fetcher = COTFetcher(cache_path=cache_path)
        history_df = fetcher.fetch()

        if history_df is None or history_df.is_empty():
            raise RuntimeError("COT fetch returned no data")

        return compute_cot_bias(history_df)

    def _yield_bias(self) -> float:
        """Fetch DFII10 history and compute 20-day real-yield change bias.

        Fetches up to 30 daily DFII10 observations from FRED via
        ``TIPSFetcher``.  Computes the difference between the 5-day
        recent average and a 10-day older average (indices 15–24 newest-first)
        and maps it to a bias per plan §3.

        Only meaningful for gold (XAUUSD).  Non-gold instruments can call
        this safely — the signal is instrument-agnostic at fetch time and
        callers are responsible for XAUUSD gating.

        Returns
        -------
        float
            Bias in [-0.10, +0.10].  Positive = real yields falling =
            bullish gold.

        Raises
        ------
        RuntimeError
            If TIPS fetcher returns an empty list (no API key, no cache,
            and no network).
        """
        cache_path = self._cache_dir / "tips_history.parquet"
        fetcher = TIPSFetcher(
            cache_path=cache_path,
            fred_api_key=self._fred_api_key,
        )
        history = fetcher.fetch_history()

        if not history:
            raise RuntimeError("TIPS fetch returned no data")

        return compute_tips_bias(history)

    def _dxy_bias(self) -> float:
        """Compute DXY change bias by reusing ExternalContext fetcher.

        Uses the external context's ``dxy_direction`` plus a 5-day % change
        heuristic.  The fetcher itself is cached (24h TTL by default), so
        this is cheap on repeated calls within a day.

        Returns
        -------
        float
            Bias in [-0.10, +0.10].  Positive = USD weakening = gold bullish.

        Raises
        ------
        RuntimeError
            If external fetcher reports ``source_quality="unavailable"``.
        """
        ctx = self._external.fetch()
        if ctx.source_quality == "unavailable" or ctx.dxy_value is None:
            raise RuntimeError("DXY data unavailable")

        # ExternalContext only exposes 3-tier direction, not raw % change.
        # For the MVP we map direction strings to conservative bias values.
        # The next iteration (per plan §4) will fetch magnitude directly.
        direction = ctx.dxy_direction
        if direction == "weakening":
            return 0.05
        if direction == "strengthening":
            return -0.05
        return 0.0
