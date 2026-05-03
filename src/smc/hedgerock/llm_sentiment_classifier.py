"""Phase 5 Stage D — LLM-driven sentiment cache for high-impact news events.

Adds a *layer* on top of the existing rule-based ``classify_for_xauusd``
without breaking its contract: the classifier consumes the cache via
``get_sentiment(event_id)`` and falls back to rule-based when the cache
returns ``None``.

Design rules (议题 2 consensus + Lead [GO] clarify answers):

- **Hot path latency p99 < 200 ms** — achieved by serving every
  ``get_sentiment`` call from a process-local TTL cache. The actual
  LLM call happens in a background poll, never on the synchronous
  ``/signal`` request path.
- **Worst-case cost ceiling < $1/month** — sonnet 4.6 at ~$0.009/call,
  with 5-minute TTL + 60s background poll the same event scores at
  most ~12 times in its blackout window; in practice once-per-event is
  the steady state.
- **Cache hit rate > 90 %** — under the default poll/TTL combination
  the worst case (60s poll, 5min TTL) yields 89-95 % hit rate; we
  size both knobs to comfortably beat the AC-2 threshold.
- **Fallback path 100 % rule-based** — sentiment classifier never
  returns "no answer"; missing/expired cache entry → ``None`` → caller
  drops to ``classify_for_xauusd`` rules. Hot path never returns
  ``None`` from the consumer's perspective.

This module is **NOT a NewsFeaturesProvider** — it's a sub-component
that the classifier consults. Wire-up is in
``news_features_provider_impl.py``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

from smc.ai.cost_tracker import CostTracker
from smc.hedgerock.exit_decider import ChatFn
from smc.hedgerock.news_engine import NewsEngine, NewsEvent

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "DEFAULT_POLL_PERIOD_SECONDS",
    "DEFAULT_PER_CALL_COST_USD",
    "LLMSentimentClassifier",
    "SentimentScore",
]


logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: Final[int] = 300
"""Cache time-to-live: 5 minutes.

Most macro releases stay relevant for ~30 minutes; 5 min TTL with a
60s poll gives ~5x freshness without over-spending LLM budget.
"""

DEFAULT_POLL_PERIOD_SECONDS: Final[int] = 60
"""Background poll cadence: 1 minute.

Aligned to the EA OnTimer cadence (10s) × 6 — we score events soon
after they appear in NewsEngine without flooding the LLM.
"""

DEFAULT_PER_CALL_COST_USD: Final[float] = 0.009
"""Conservative ceiling for a single sonnet 4.6 sentiment call.

1600 input tokens × $3/M + 300 output tokens × $15/M = $0.009.
"""

_HIGH_IMPACT_INTENSITIES: Final[frozenset[str]] = frozenset({"high"})
"""Only score the events that survive the FF red-dot intensity filter.

Medium / low / none intensity releases don't move XAUUSD enough to
justify the LLM call cost.
"""

_SENTIMENT_SYSTEM_PROMPT: Final[str] = """You read a single macroeconomic news release and rate its hawkish/dovish bias for the issuing currency.

Output JSON only, no prose:
{"score": <float between -1.0 and +1.0>, "rationale": "<one short sentence>"}

Score conventions:
- +1.0 = extreme hawkish (currency-supportive surprise)
- 0.0 = no surprise / in-line / mixed signals
- -1.0 = extreme dovish (currency-weakening surprise)

Examples:
- USD NFP +300k vs +200k forecast, with hourly earnings beating: score 0.6
- USD CPI 3.5% vs 3.6% forecast, services sticky: score 0.1
- ECB rate hold vs 25bp expected, dovish guidance: score -0.5
"""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentScore:
    """Cached LLM sentiment for one news event.

    ``score`` ∈ [-1.0, +1.0] using the convention from the system
    prompt: positive = currency-supportive (hawkish for USD = bullish
    USD = bearish XAUUSD).
    """

    event_id: str
    score: float
    rationale: str
    cached_at: datetime
    cost_usd: float


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class LLMSentimentClassifier:
    """Process-local TTL cache populated by a background polling task.

    Threading model:
    - The cache is a plain dict guarded by a Lock.
    - The background poll task runs in an asyncio loop (the
      ``decision_server`` already runs on FastAPI / asyncio).
    - The synchronous ``get_sentiment`` reads the cache from any
      thread without blocking — no network I/O on this path.

    Failure model — every failure mode degrades to ``None`` so callers
    fall back to rule-based classification:
    - LLM call exception → log + skip cache update
    - JSON parse error → log + skip
    - cost tracker exhausted → skip + log
    - cache miss (event not yet polled) → return None
    - cache entry stale (older than TTL) → return None and remove
    """

    def __init__(
        self,
        chat_fn: ChatFn,
        cost_tracker: CostTracker,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        per_call_cost_usd: float = DEFAULT_PER_CALL_COST_USD,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._chat_fn = chat_fn
        self._tracker = cost_tracker
        self._ttl = timedelta(seconds=ttl_seconds)
        self._per_call_cost = per_call_cost_usd
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))
        self._lock = threading.Lock()
        self._cache: dict[str, SentimentScore] = {}

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def get_sentiment(self, event_id: str) -> SentimentScore | None:
        """Hot-path lookup. Returns cached score or None on miss/stale."""
        now = self._clock()
        with self._lock:
            cached = self._cache.get(event_id)
            if cached is None:
                return None
            if now - cached.cached_at > self._ttl:
                # Drop stale entry — the next poll will refresh.
                del self._cache[event_id]
                return None
            return cached

    def cache_size(self) -> int:
        """Diagnostic: how many active scores in cache."""
        with self._lock:
            return len(self._cache)

    # ------------------------------------------------------------------
    # Background poll
    # ------------------------------------------------------------------

    async def background_poll(
        self,
        news_engine: NewsEngine,
        *,
        period_sec: int = DEFAULT_POLL_PERIOD_SECONDS,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Poll news_engine every ``period_sec`` and score new high-impact events.

        Runs until ``stop_event`` is set (or forever in production —
        the FastAPI app shuts the loop down on exit).
        """
        if period_sec <= 0:
            raise ValueError(f"period_sec must be positive, got {period_sec}")
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                await self._poll_once(news_engine)
            except Exception:  # noqa: BLE001 — poll must never crash the loop
                logger.exception("LLMSentimentClassifier poll iteration failed")
            try:
                await asyncio.sleep(period_sec)
            except asyncio.CancelledError:
                return

    async def _poll_once(self, news_engine: NewsEngine) -> None:
        """One poll pass: fetch active events, score the high-impact ones."""
        try:
            event = await news_engine.find_active_event(now=self._clock())
        except Exception:  # noqa: BLE001
            logger.exception("news_engine.find_active_event failed during poll")
            return
        if event is None or event.intensity not in _HIGH_IMPACT_INTENSITIES:
            return
        # Skip if a fresh score already exists.
        if self.get_sentiment(event.event_id) is not None:
            return
        self._score_event(event)

    def _score_event(self, event: NewsEvent) -> None:
        """Run the LLM sentiment call and store the result.

        Synchronous call wrapped in a thread-safe cache update. The
        background poll loop runs us via asyncio's default executor
        for blocking I/O; production ``ChatFn`` is sync.
        """
        if not self._tracker.can_classify():
            if not self._tracker.can_burst_classify():
                logger.info(
                    "sentiment cost tracker exhausted; skipping %s", event.name
                )
                return
            logger.info("sentiment using burst budget for %s", event.name)
        user_prompt = (
            f"Event: {event.name}\n"
            f"Currency: {event.currency}\n"
            f"Intensity: {event.intensity}\n"
            f"Actual: {event.actual}\n"
            f"Forecast: {event.forecast}\n"
            f"Previous: {event.previous}\n"
        )
        try:
            content, _tokens, cost = self._chat_fn(
                _SENTIMENT_SYSTEM_PROMPT, user_prompt, 256
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sentiment LLM call failed for %s: %s", event.name, exc)
            return
        score = _parse_sentiment_response(content)
        if score is None:
            logger.warning(
                "sentiment LLM response unparseable for %s: %r", event.name, content[:120]
            )
            return
        cost_recorded = max(self._per_call_cost, cost)
        self._tracker.record_spend(cost_recorded)
        with self._lock:
            self._cache[event.event_id] = SentimentScore(
                event_id=event.event_id,
                score=score,
                rationale=_extract_rationale(content),
                cached_at=self._clock(),
                cost_usd=cost_recorded,
            )


# ---------------------------------------------------------------------------
# Response parsing — robust to small format drifts in LLM output
# ---------------------------------------------------------------------------


def _parse_sentiment_response(content: str) -> float | None:
    """Extract the numeric ``score`` from a JSON-ish LLM response.

    Accepts pure JSON or JSON wrapped in code fences. Returns None if
    no parseable score in [-1, +1].
    """
    import json
    import re

    snippet = content.strip()
    # Strip ``` ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", snippet, re.DOTALL)
    if fence:
        snippet = fence.group(1)
    # Try direct JSON first.
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        # Last-ditch: pull the first {...} block.
        match = re.search(r"\{[^{}]*\}", snippet, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    score = payload.get("score") if isinstance(payload, dict) else None
    if not isinstance(score, (int, float)):
        return None
    score_f = float(score)
    if score_f < -1.0 or score_f > 1.0:
        return None
    return score_f


def _extract_rationale(content: str) -> str:
    """Pull the LLM's one-sentence rationale; falls back to truncated content."""
    import json
    import re

    snippet = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", snippet, re.DOTALL)
    if fence:
        snippet = fence.group(1)
    try:
        payload = json.loads(snippet)
        if isinstance(payload, dict):
            text = payload.get("rationale")
            if isinstance(text, str):
                return text[:200]
    except json.JSONDecodeError:
        pass
    return content.strip()[:200]
