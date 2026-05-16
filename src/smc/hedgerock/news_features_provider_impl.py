"""Production :class:`NewsFeaturesProvider` wrapping :class:`NewsEngine`.

Phase 5.2 Stage C — fold the Phase 3 building blocks (NewsEngine
crawler + classify_for_xauusd) into a single provider that
``decision_server`` can plug in. The flow per call:

    1. ``find_active_event`` (async, hits ForexFactory or its TTL cache)
    2. If no event → return None.
    3. Otherwise: query the injected ExposureProvider for current lot
       direction (long/short/flat) — classify_for_xauusd needs this
       to decide ``with`` / ``against`` / ``neutral``.
    4. Run ``classify_for_xauusd`` and return the
       :class:`NewsClassification`.

Async handling: ``NewsEngine.find_active_event`` is async because the
underlying ``httpx.AsyncClient`` is. ``decision_server.get_news_classification``
is sync (the ``/signal`` endpoint is fully sync). We bridge with
``asyncio.run`` per call — acceptable because each /signal poll is a
fresh sync request and there's no event loop already running on the
calling thread.

If the calling context already has a running event loop (e.g. async
FastAPI handlers), the caller should use the async helper
``async_get_news_classification`` directly to avoid
``asyncio.run`` raising "cannot be called from a running event loop".
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Final

from smc.hedgerock.decision_server import ExposureProvider, NewsUnavailable
from smc.hedgerock.news_classifier import (
    DEFAULT_SURPRISE_THRESHOLD,
    ExposureDirection,
    NewsClassification,
    classify_for_xauusd,
)
from smc.hedgerock.news_engine import NewsEngine

__all__ = [
    "DEFAULT_FLAT_THRESHOLD_LOTS",
    "NewsEngineFeaturesProvider",
    "exposure_lots_to_direction",
]


logger = logging.getLogger(__name__)

DEFAULT_FLAT_THRESHOLD_LOTS: Final[float] = 0.005
"""Lots-magnitude under this threshold count as ``"flat"``.

Mirrors ``exit_decider._exposure_sign``'s flat threshold so the news
classifier's view of exposure agrees with the exit decider's view.
"""


def exposure_lots_to_direction(
    lots: float,
    *,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD_LOTS,
) -> ExposureDirection:
    """Convert signed lot count to ``"long" | "short" | "flat"``."""
    if abs(lots) <= flat_threshold:
        return "flat"
    return "long" if lots > 0 else "short"


class NewsEngineFeaturesProvider:
    """Production ``NewsFeaturesProvider`` impl.

    Holds:
    - a configured :class:`NewsEngine` (the FF crawler + classifier
      orchestrator)
    - an :class:`ExposureProvider` so that ``classify_for_xauusd`` can
      see the EA's current direction.

    Optional knobs:
    - ``window_before_seconds`` / ``window_after_seconds`` forwarded to
      ``find_active_event`` — defaults match the EA's STOP_BEFORE_/
      START_AFTER_ news window.
    - ``surprise_threshold`` forwarded to ``classify_for_xauusd``.
    - ``clock`` callable returning UTC datetime — allows tests to
      pin "now" without monkeypatching ``datetime``.

    Errors are surfaced as :class:`NewsUnavailable` so the
    decision_server's existing ``_safe_get_news_classification`` wrapper
    can downgrade to ``None`` and keep the envelope flowing.
    """

    def __init__(
        self,
        engine: NewsEngine,
        exposure_provider: ExposureProvider,
        *,
        window_before_seconds: int = 30 * 60,
        window_after_seconds: int = 30 * 60,
        surprise_threshold: float = DEFAULT_SURPRISE_THRESHOLD,
        sentiment_classifier=None,
        clock=None,
    ) -> None:
        self._engine = engine
        self._exposure = exposure_provider
        self._window_before = window_before_seconds
        self._window_after = window_after_seconds
        self._surprise_threshold = surprise_threshold
        self._sentiment_classifier = sentiment_classifier
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

    async def async_get_news_classification(
        self, symbol: str
    ) -> NewsClassification | None:
        """Async variant — call from inside an existing event loop."""
        try:
            event = await self._engine.find_active_event(
                now=self._clock(),
                window_before_seconds=self._window_before,
                window_after_seconds=self._window_after,
            )
        except Exception as exc:  # noqa: BLE001 — crawler must not crash polling
            logger.warning("news engine fetch failed for %s: %s", symbol, exc)
            raise NewsUnavailable(str(exc)) from exc

        if event is None:
            return None

        # Look up exposure direction so classify_for_xauusd can resolve
        # 'with' vs 'against' vs 'neutral'.
        try:
            lots = self._exposure.get_exposure_lots(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "exposure provider failed for %s; defaulting flat: %s", symbol, exc
            )
            lots = 0.0
        direction = exposure_lots_to_direction(lots)

        return classify_for_xauusd(
            event,
            current_exposure_direction=direction,
            surprise_threshold=self._surprise_threshold,
            sentiment_classifier=self._sentiment_classifier,
        )

    def get_news_classification(self, symbol: str) -> NewsClassification | None:
        """Sync entry point — bridges to the async engine via ``asyncio.run``.

        Caveat: only use this from a thread without an active event
        loop. FastAPI sync handlers (the decision_server default) are
        fine; async handlers should call ``async_get_news_classification``
        directly.
        """
        try:
            return asyncio.run(self.async_get_news_classification(symbol))
        except NewsUnavailable:
            raise
        except RuntimeError as exc:
            # asyncio.run raises RuntimeError if a loop is already
            # running. Surface as NewsUnavailable so the
            # decision_server's safe wrapper degrades gracefully.
            logger.warning(
                "asyncio.run failed (already in a loop?) for %s: %s", symbol, exc
            )
            raise NewsUnavailable(f"async loop conflict: {exc}") from exc
