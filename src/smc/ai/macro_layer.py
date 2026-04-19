"""Macro overlay scoring for SMC confluence boost.

Computes a single bias float in [-0.3, +0.3] from 3 orthogonal macro
sources: COT positioning, TIPS real yield, DXY.  Injected into
``score_confluence`` as an additive bonus.

Design principles:
    - Orthogonal to SMC: adds new signal dimensions, not re-tunes existing.
    - Graceful degradation: any fetcher failure → that source = 0.0.
    - Conservative: max absolute bias = 0.3 (cannot override strong SMC).
    - Cache-first: reuses ``ExternalContextFetcher`` in-memory cache.

Implementation status (Alt-B Round 4 MVP, 2026-04-19):
    - DXY source: implemented (reuses ``_fetch_dxy`` from external_context).
    - COT source: stub (NotImplementedError) — see §2 of the plan doc.
    - TIPS yield source: stub — see §3 of the plan doc.

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

from smc.ai.external_context import ExternalContextFetcher

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
        except NotImplementedError:
            # Expected during MVP phase — log once at debug, not warning.
            logger.debug("COT source not yet implemented; contributing 0.0")
            return None
        except Exception:  # noqa: BLE001
            logger.debug("COT bias fetch failed", exc_info=True)
            return None

    def _safe_yield_bias(self) -> float | None:
        try:
            return self._yield_bias()
        except NotImplementedError:
            logger.debug("Yield source not yet implemented; contributing 0.0")
            return None
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
        """Compute COT positioning bias.  Not yet implemented (see plan §2)."""
        raise NotImplementedError("COT fetcher scheduled for W1D1–D2")

    def _yield_bias(self) -> float:
        """Compute real-yield change bias.  Not yet implemented (see plan §3)."""
        raise NotImplementedError("Real yield fetcher scheduled for W1D3")

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
