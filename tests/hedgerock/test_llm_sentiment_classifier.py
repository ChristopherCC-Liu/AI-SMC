"""Tests for ``smc.hedgerock.llm_sentiment_classifier``.

Coverage targets:

- TTL cache: hit / miss / stale eviction
- Background poll: high-impact events scored, others skipped
- Cost tracker integration: respect budget exhaustion + burst path
- Robustness: LLM exception, malformed JSON, out-of-range score → no
  cache poisoning
- Response parsing: JSON, JSON-in-fences, malformed inputs
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from smc.ai.cost_tracker import CostTracker
from smc.hedgerock.llm_sentiment_classifier import (
    DEFAULT_PER_CALL_COST_USD,
    DEFAULT_TTL_SECONDS,
    LLMSentimentClassifier,
    SentimentScore,
    _extract_rationale,
    _parse_sentiment_response,
)
from smc.hedgerock.news_engine import NewsEvent


_NOW: Final[datetime] = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    *,
    event_id: str = "evt-nfp",
    name: str = "Non-Farm Payrolls",
    intensity: str = "high",
    currency: str = "USD",
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        name=name,
        currency=currency,
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=_NOW,
        actual=300_000.0,
        forecast=200_000.0,
        previous=180_000.0,
    )


def _make_chat_fn(content: str, cost: float = 0.009):
    """Return a ``ChatFn`` that always replies with the canned content."""
    calls = []

    def chat(system: str, user: str, max_tokens: int):
        calls.append((system, user))
        return content, 100, cost

    chat.calls = calls  # type: ignore[attr-defined]
    return chat


class _StubEngine:
    def __init__(self, event: NewsEvent | None) -> None:
        self._event = event

    async def find_active_event(self, *, now, **kwargs):  # noqa: ANN001
        return self._event


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


def test_parse_pure_json() -> None:
    assert _parse_sentiment_response('{"score": 0.6, "rationale": "ok"}') == 0.6


def test_parse_json_in_code_fence() -> None:
    payload = '```json\n{"score": -0.4, "rationale": "dovish"}\n```'
    assert _parse_sentiment_response(payload) == pytest.approx(-0.4)


def test_parse_score_out_of_range_returns_none() -> None:
    assert _parse_sentiment_response('{"score": 2.0, "rationale": "??"}') is None
    assert _parse_sentiment_response('{"score": -2.0}') is None


def test_parse_malformed_returns_none() -> None:
    assert _parse_sentiment_response("totally not json") is None


def test_parse_score_zero_accepted() -> None:
    assert _parse_sentiment_response('{"score": 0.0, "rationale": "flat"}') == 0.0


def test_extract_rationale_pulls_from_json() -> None:
    text = '{"score": 0.5, "rationale": "USD strong on hot CPI"}'
    assert _extract_rationale(text) == "USD strong on hot CPI"


def test_extract_rationale_falls_back_to_truncated_content() -> None:
    text = "no json here"
    assert _extract_rationale(text) == "no json here"


# ---------------------------------------------------------------------------
# get_sentiment / cache TTL
# ---------------------------------------------------------------------------


def test_get_sentiment_returns_none_on_empty_cache() -> None:
    chat = _make_chat_fn('{"score": 0.5, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    assert classifier.get_sentiment("evt-nope") is None
    assert classifier.cache_size() == 0


def test_get_sentiment_returns_cached_score_then_evicts_when_stale() -> None:
    chat = _make_chat_fn('{"score": 0.5, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    clock_value = _NOW

    def clock():
        return clock_value

    classifier = LLMSentimentClassifier(
        chat, tracker, ttl_seconds=300, clock=clock,
    )
    # Manually insert a fresh entry.
    classifier._cache["evt-1"] = SentimentScore(
        event_id="evt-1",
        score=0.7,
        rationale="hawkish",
        cached_at=clock_value,
        cost_usd=DEFAULT_PER_CALL_COST_USD,
    )
    # Hit while fresh.
    hit = classifier.get_sentiment("evt-1")
    assert hit is not None and hit.score == pytest.approx(0.7)

    # Advance clock past TTL.
    clock_value = _NOW + timedelta(seconds=400)
    miss = classifier.get_sentiment("evt-1")
    assert miss is None
    # Stale entry has been evicted.
    assert classifier.cache_size() == 0


def test_constructor_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        LLMSentimentClassifier(_make_chat_fn(""), CostTracker(), ttl_seconds=0)


# ---------------------------------------------------------------------------
# Background poll: scores + skips + budget
# ---------------------------------------------------------------------------


def _run_one_poll(classifier: LLMSentimentClassifier, engine: _StubEngine) -> None:
    asyncio.run(classifier._poll_once(engine))


def test_poll_scores_high_impact_event() -> None:
    chat = _make_chat_fn('{"score": 0.6, "rationale": "hawkish NFP"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)

    cached = classifier.get_sentiment("evt-nfp")
    assert cached is not None
    assert cached.score == pytest.approx(0.6)
    assert "hawkish" in cached.rationale
    assert cached.cost_usd >= DEFAULT_PER_CALL_COST_USD


def test_poll_skips_medium_intensity_event() -> None:
    """Intensity != high → no LLM call, no cache write."""
    chat = _make_chat_fn('{"score": 0.6, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event(intensity="medium"))

    _run_one_poll(classifier, engine)

    assert classifier.cache_size() == 0
    assert chat.calls == []  # type: ignore[attr-defined]


def test_poll_skips_when_event_already_cached() -> None:
    """Subsequent polls don't re-score a fresh cached event."""
    chat = _make_chat_fn('{"score": 0.6, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)
    _run_one_poll(classifier, engine)

    assert len(chat.calls) == 1  # type: ignore[attr-defined]


def test_poll_handles_engine_exception() -> None:
    """Engine failure → poll iteration logs and returns, no crash."""

    class _BrokenEngine:
        async def find_active_event(self, *, now, **kwargs):  # noqa: ANN001
            raise RuntimeError("FF unreachable")

    chat = _make_chat_fn('{"score": 0.5, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)

    asyncio.run(classifier._poll_once(_BrokenEngine()))  # type: ignore[arg-type]

    assert classifier.cache_size() == 0


def test_poll_handles_llm_exception() -> None:
    """LLM exception → no cache write, log only."""

    def chat(system: str, user: str, max_tokens: int):
        raise RuntimeError("LLM 500")

    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)

    assert classifier.cache_size() == 0


def test_poll_handles_malformed_llm_response() -> None:
    chat = _make_chat_fn("definitely not json")
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)

    assert classifier.cache_size() == 0


def test_poll_uses_burst_budget_when_daily_exhausted() -> None:
    chat = _make_chat_fn('{"score": 0.7, "rationale": "x"}')
    # Pre-spend the daily budget, leave burst available.
    tracker = CostTracker(daily_budget_usd=0.5, burst_budget_usd=2.0)
    tracker.record_spend(0.5)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)

    # Burst path → still scored.
    assert classifier.cache_size() == 1


def test_poll_skipped_when_both_budgets_exhausted() -> None:
    chat = _make_chat_fn('{"score": 0.7, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=0.01, burst_budget_usd=0.01)
    tracker.record_spend(0.05)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    _run_one_poll(classifier, engine)

    assert classifier.cache_size() == 0
    assert chat.calls == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# background_poll loop control
# ---------------------------------------------------------------------------


def test_background_poll_stops_when_event_set() -> None:
    """Setting stop_event before run → returns immediately without polling."""
    chat = _make_chat_fn('{"score": 0.7, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)
    engine = _StubEngine(_make_event())

    async def runner():
        stop = asyncio.Event()
        stop.set()  # signal stop before loop entry
        await classifier.background_poll(
            engine, period_sec=1, stop_event=stop,
        )

    asyncio.run(runner())
    assert chat.calls == []  # type: ignore[attr-defined]


def test_background_poll_rejects_non_positive_period() -> None:
    chat = _make_chat_fn('{"score": 0.5, "rationale": "x"}')
    tracker = CostTracker(daily_budget_usd=10.0)
    classifier = LLMSentimentClassifier(chat, tracker, clock=lambda: _NOW)

    async def runner():
        await classifier.background_poll(
            _StubEngine(None), period_sec=0,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="period_sec must be positive"):
        asyncio.run(runner())


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


def test_sentiment_score_is_immutable() -> None:
    score = SentimentScore(
        event_id="x", score=0.5, rationale="r",
        cached_at=_NOW, cost_usd=0.009,
    )
    with pytest.raises(Exception):
        score.score = 0.9  # type: ignore[misc]
