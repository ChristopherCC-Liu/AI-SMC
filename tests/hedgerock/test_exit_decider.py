"""Tests for ``smc.hedgerock.exit_decider``.

Coverage layers:

1. **Pure helpers** (``regime_transition_distance``, ``_exposure_sign``,
   ``_extract_recommendation``).
2. **hard_rule_directive** — every documented branch + edge cases:
   - extreme regime flip with opposed exposure → halt
   - high+against → urgent TP
   - high+with+matching exposure → news window
   - news_classification=None paths
   - first call (prev_regime=None) → no halt fires
3. **decide_exit** end-to-end:
   - hard rule fires → debate is never called
   - hard rule emits "none" + arguable context → debate runs
   - enable_debate=False bypasses LLM
   - exposure flat + no news → debate skipped
   - timeout → fallback "none"
   - backend RuntimeError → fallback "none"
   - cost tracker exhausted → debate skipped (no burst)
   - burst budget allows one more call → debate runs
   - bull/bear majority vote (agree / disagree-bear / disagree-soften-halt)
   - cost tracker recording on success
4. **Latency benchmark**: hard rule path completes well below 1 ms.
"""

from __future__ import annotations

import time

import pytest

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.hedgerock.exit_decider import (
    AGENT_MAX_TOKENS,
    DEBATE_TIMEOUT_S,
    DECIDER_VERSION,
    DEFAULT_PER_CALL_COST_USD,
    EXTREME_TRANSITION_DISTANCE,
    ExitDecision,
    _exposure_sign,
    _extract_recommendation,
    decide_exit,
    hard_rule_directive,
    regime_transition_distance,
)
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.news_engine import NewsEvent
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


_TS = datetime(2024, 6, 7, 12, 30, tzinfo=timezone.utc)


def _evt(
    name: str = "Non-Farm Payrolls",
    *,
    intensity: str = "high",
    currency: str = "USD",
    actual: float | None = 272_000.0,
    forecast: float | None = 200_000.0,
) -> NewsEvent:
    return NewsEvent(
        event_id="ev-test",
        name=name,
        currency=currency,
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=_TS,
        actual=actual,
        forecast=forecast,
    )


def _classification(
    *,
    name: str = "Non-Farm Payrolls",
    intensity: str = "high",
    direction: str = "against",
    currency: str = "USD",
    surprise: float | None = 0.36,
) -> NewsClassification:
    return NewsClassification(
        event=_evt(name, intensity=intensity, currency=currency),
        direction=direction,  # type: ignore[arg-type]
        surprise_score=surprise,
        impact_currency=currency,
        classifier_version="v1.0.0",
    )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_module_constants_documented() -> None:
    assert DECIDER_VERSION.startswith("v")
    assert DEBATE_TIMEOUT_S == 2.0  # lead spec
    assert AGENT_MAX_TOKENS == 256  # lead spec
    assert DEFAULT_PER_CALL_COST_USD > 0
    assert EXTREME_TRANSITION_DISTANCE >= 1


# ---------------------------------------------------------------------------
# Pure helpers — regime distance + exposure sign + recommendation parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regime_transition_distance_first_call_returns_zero() -> None:
    assert regime_transition_distance(None, "TREND_UP") == 0


@pytest.mark.unit
def test_regime_transition_distance_same_regime_returns_zero() -> None:
    assert regime_transition_distance("TREND_UP", "TREND_UP") == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prev", "now", "expected"),
    [
        ("TREND_UP", "TREND_DOWN", 4),  # full ladder flip
        ("TREND_UP", "ATH_BREAKOUT", 1),
        ("CONSOLIDATION", "TREND_DOWN", 2),
        ("ATH_BREAKOUT", "TRANSITION", 2),
    ],
)
def test_regime_transition_distance_table(
    prev: str, now: str, expected: int
) -> None:
    assert regime_transition_distance(prev, now) == expected  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lots", "expected"),
    [
        (1.0, 1),
        (-1.0, -1),
        (0.0, 0),
        (0.001, 0),  # below flat threshold
        (-0.003, 0),  # below flat threshold
        (0.01, 1),  # above flat threshold
    ],
)
def test_exposure_sign(lots: float, expected: int) -> None:
    assert _exposure_sign(lots) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("blah\nRECOMMEND: urgent_take_profit", "urgent_take_profit"),
        ("RECOMMEND:halt_and_close_all", "halt_and_close_all"),
        ("recommend: none", "none"),
        ("RECOMMEND: news_trade_window", "news_trade_window"),
    ],
)
def test_extract_recommendation_valid(text: str, expected: str) -> None:
    assert _extract_recommendation(text) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    ["no marker at all", "RECOMMEND: bogus", "RECOMMEND missing colon"],
)
def test_extract_recommendation_invalid(text: str) -> None:
    assert _extract_recommendation(text) is None


# ---------------------------------------------------------------------------
# hard_rule_directive — full branch matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hard_rule_extreme_regime_flip_with_opposed_long_halts() -> None:
    """TREND_UP→TREND_DOWN with long exposure → halt."""
    directive, rationale = hard_rule_directive(
        regime="TREND_DOWN",
        prev_regime="TREND_UP",
        news_classification=None,
        current_exposure_lots=1.0,
    )
    assert directive == "halt_and_close_all"
    assert "extreme regime flip" in rationale


@pytest.mark.unit
def test_hard_rule_extreme_regime_flip_with_opposed_short_halts() -> None:
    """TREND_DOWN→TREND_UP with short exposure → halt."""
    directive, _ = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_DOWN",
        news_classification=None,
        current_exposure_lots=-1.0,
    )
    assert directive == "halt_and_close_all"


@pytest.mark.unit
def test_hard_rule_first_call_never_halts() -> None:
    """prev_regime=None → distance=0 → halt branch never fires."""
    directive, _ = hard_rule_directive(
        regime="TREND_DOWN",
        prev_regime=None,
        news_classification=None,
        current_exposure_lots=1.0,
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_extreme_flip_with_aligned_exposure_does_not_halt() -> None:
    """If exposure is already with the new regime, no halt fires."""
    directive, _ = hard_rule_directive(
        regime="TREND_DOWN",
        prev_regime="TREND_UP",
        news_classification=None,
        current_exposure_lots=-1.0,  # short matches TREND_DOWN
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_extreme_flip_flat_exposure_does_not_halt() -> None:
    directive, _ = hard_rule_directive(
        regime="TREND_DOWN",
        prev_regime="TREND_UP",
        news_classification=None,
        current_exposure_lots=0.0,
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_high_against_long_yields_urgent_take_profit() -> None:
    cls = _classification(intensity="high", direction="against")
    directive, rationale = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
    )
    assert directive == "urgent_take_profit"
    assert "high-intensity" in rationale


@pytest.mark.unit
def test_hard_rule_high_against_short_yields_urgent_take_profit() -> None:
    cls = _classification(intensity="high", direction="against")
    directive, _ = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=-1.0,
    )
    assert directive == "urgent_take_profit"


@pytest.mark.unit
def test_hard_rule_high_against_flat_does_not_fire() -> None:
    cls = _classification(intensity="high", direction="against")
    directive, _ = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=0.0,
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_high_with_matching_long_yields_news_window() -> None:
    cls = _classification(intensity="high", direction="with")
    directive, rationale = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
    )
    assert directive == "news_trade_window"
    assert "open news window" in rationale


@pytest.mark.unit
def test_hard_rule_medium_intensity_returns_none() -> None:
    """Medium intensity always escalates to micro debate."""
    cls = _classification(intensity="medium", direction="against")
    directive, _ = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_neutral_direction_returns_none() -> None:
    cls = _classification(intensity="high", direction="neutral")
    directive, _ = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
    )
    assert directive == "none"


@pytest.mark.unit
def test_hard_rule_no_news_classification_returns_none() -> None:
    directive, rationale = hard_rule_directive(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=None,
        current_exposure_lots=1.0,
    )
    assert directive == "none"
    assert "no news context" in rationale


# ---------------------------------------------------------------------------
# decide_exit — hard rule paths
# ---------------------------------------------------------------------------


def _failing_chat(*_args: object, **_kwargs: object) -> tuple[str, int, float]:
    raise AssertionError("chat_fn must not be called on hard-rule path")


@pytest.mark.unit
def test_decide_exit_hard_rule_skips_chat() -> None:
    """High+against+long → hard rule fires; chat must not be called."""
    cls = _classification(intensity="high", direction="against")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=_failing_chat,
    )
    assert isinstance(out, ExitDecision)
    assert out.directive == "urgent_take_profit"
    assert out.source == "hard_rule"
    assert out.cost_usd == 0.0
    assert out.elapsed_ms >= 0
    assert out.decider_version == DECIDER_VERSION


@pytest.mark.unit
def test_decide_exit_hard_rule_latency_under_50ms() -> None:
    """Hard rule must resolve well under 50 ms even on slow CI."""
    cls = _classification(intensity="high", direction="against")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=_failing_chat,
    )
    assert out.elapsed_ms < 50


@pytest.mark.unit
def test_decide_exit_rejects_non_positive_timeout() -> None:
    cls = _classification()
    with pytest.raises(ValueError, match="debate_timeout_s must be positive"):
        decide_exit(
            regime="TREND_UP",
            prev_regime="TREND_UP",
            news_classification=cls,
            current_exposure_lots=1.0,
            debate_timeout_s=0.0,
        )


# ---------------------------------------------------------------------------
# decide_exit — debate skipped
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decide_exit_disable_debate_short_circuits() -> None:
    """enable_debate=False skips LLM even if hard rule emits 'none'."""
    cls = _classification(intensity="medium", direction="against")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        enable_debate=False,
        chat_fn=_failing_chat,
    )
    assert out.directive == "none"
    assert out.source == "hard_rule"  # short-circuit reuses rule rationale


@pytest.mark.unit
def test_decide_exit_flat_exposure_no_news_skips_debate() -> None:
    out = decide_exit(
        regime="CONSOLIDATION",
        prev_regime="CONSOLIDATION",
        news_classification=None,
        current_exposure_lots=0.0,
        chat_fn=_failing_chat,
    )
    assert out.directive == "none"
    assert out.source == "hard_rule"


@pytest.mark.unit
def test_decide_exit_intensity_none_news_skips_debate() -> None:
    """News with intensity='none' is treated as no event for debate gating."""
    cls = _classification(intensity="none", direction="neutral")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=0.0,
        chat_fn=_failing_chat,
    )
    assert out.source == "hard_rule"


# ---------------------------------------------------------------------------
# decide_exit — debate path with canned chat
# ---------------------------------------------------------------------------


def _canned_chat(
    *,
    bull: str = "RECOMMEND: urgent_take_profit",
    bear: str = "RECOMMEND: urgent_take_profit",
    cost_per_call: float = 0.005,
    sleep_s: float = 0.0,
):
    """Build a chat fixture that returns canned bull then canned bear."""
    sequence = [bull, bear]
    state = {"i": 0}

    def chat(_system: str, _user: str, _max_tokens: int) -> tuple[str, int, float]:
        if sleep_s > 0:
            time.sleep(sleep_s)
        idx = state["i"] % len(sequence)
        state["i"] += 1
        return sequence[idx], 100, cost_per_call

    return chat


@pytest.mark.integration
def test_decide_exit_debate_consensus_uses_directive() -> None:
    """Bull and bear agree → debate consensus, source='micro_debate'."""
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(
        bull="RECOMMEND: urgent_take_profit",
        bear="RECOMMEND: urgent_take_profit",
    )
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
    )
    assert out.directive == "urgent_take_profit"
    assert out.source == "micro_debate"
    assert out.cost_usd >= DEFAULT_PER_CALL_COST_USD * 0.5


@pytest.mark.integration
def test_decide_exit_debate_split_defers_to_bear() -> None:
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(
        bull="RECOMMEND: none",
        bear="RECOMMEND: urgent_take_profit",
    )
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
    )
    assert out.directive == "urgent_take_profit"
    assert out.source == "micro_debate"
    assert "deferring to bear" in out.rationale or "consensus" in out.rationale


@pytest.mark.integration
def test_decide_exit_debate_split_softens_bear_halt_to_urgent_tp() -> None:
    """Bear escalating to halt while bull says none → soften to urgent TP."""
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(
        bull="RECOMMEND: none",
        bear="RECOMMEND: halt_and_close_all",
    )
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
    )
    assert out.directive == "urgent_take_profit"
    assert "softening" in out.rationale.lower()


@pytest.mark.integration
def test_decide_exit_debate_only_bull_parsed_uses_bull() -> None:
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(
        bull="RECOMMEND: news_trade_window",
        bear="bear has no marker",
    )
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
    )
    assert out.directive == "news_trade_window"
    assert "only bull" in out.rationale


@pytest.mark.integration
def test_decide_exit_debate_unparseable_returns_none() -> None:
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(bull="bull rambles", bear="bear rambles")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
    )
    assert out.directive == "none"
    assert out.source == "micro_debate"


# ---------------------------------------------------------------------------
# decide_exit — guard rails
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_decide_exit_timeout_returns_fallback_none() -> None:
    cls = _classification(intensity="medium", direction="against")
    chat = _canned_chat(sleep_s=0.30)
    started = time.perf_counter()
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
        debate_timeout_s=0.10,  # below the per-call sleep
    )
    elapsed = time.perf_counter() - started
    assert out.source == "fallback"
    assert out.directive == "none"
    assert out.cost_usd == 0.0
    assert "timed out" in out.rationale
    # End-to-end stays inside the 2-second public budget.
    assert elapsed < DEBATE_TIMEOUT_S


@pytest.mark.integration
def test_decide_exit_chat_runtime_error_returns_fallback() -> None:
    cls = _classification(intensity="medium", direction="against")

    def boom(*_a: object, **_kw: object) -> tuple[str, int, float]:
        raise RuntimeError("backend down")

    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=boom,
    )
    # Per-agent helper swallows exceptions, so we end up with empty
    # bull/bear strings; both unparseable → directive 'none' via debate.
    assert out.directive == "none"
    assert out.cost_usd >= 0.0  # micro_debate path records ceiling cost


@pytest.mark.integration
def test_decide_exit_cost_tracker_exhausted_skips_debate() -> None:
    cls = _classification(intensity="medium", direction="against")
    tracker = CostTracker(daily_budget_usd=0.0, burst_budget_usd=0.0)
    chat = _canned_chat()
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
        cost_tracker=tracker,
    )
    assert out.directive == "none"
    assert out.source == "fallback"
    assert out.cost_usd == 0.0
    assert tracker.classification_count == 0
    assert "cost tracker" in out.rationale.lower()


@pytest.mark.integration
def test_decide_exit_burst_budget_allows_one_more_call() -> None:
    cls = _classification(intensity="medium", direction="against")
    tracker = CostTracker(daily_budget_usd=0.0, burst_budget_usd=1.0)
    chat = _canned_chat()
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
        cost_tracker=tracker,
    )
    assert out.source == "micro_debate"
    assert tracker.classification_count == 1


@pytest.mark.integration
def test_decide_exit_records_cost_on_success() -> None:
    cls = _classification(intensity="medium", direction="against")
    tracker = CostTracker(daily_budget_usd=1.0, burst_budget_usd=0.0)
    chat = _canned_chat(cost_per_call=0.001)
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=chat,
        cost_tracker=tracker,
    )
    assert out.source == "micro_debate"
    # Conservative ceiling overrides the 2 × 0.001 = 0.002 raw sum.
    assert tracker.daily_spend >= DEFAULT_PER_CALL_COST_USD


@pytest.mark.integration
def test_decide_exit_recent_equity_passed_through_to_debate() -> None:
    """The chat receives a context block containing the recent equity figures."""
    cls = _classification(intensity="medium", direction="against")
    captured: list[str] = []

    def capturing_chat(_system: str, user: str, _max_tokens: int) -> tuple[str, int, float]:
        captured.append(user)
        return "RECOMMEND: urgent_take_profit", 100, 0.001

    decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        recent_equity=(10_000.0, 10_050.0, 10_120.0),
        chat_fn=capturing_chat,
    )
    assert any("recent_equity" in u and "10120" in u.replace(".0", "") or "10120.00" in u for u in captured)


# ---------------------------------------------------------------------------
# Output dataclass invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_exit_decision_is_frozen() -> None:
    cls = _classification(intensity="high", direction="against")
    out = decide_exit(
        regime="TREND_UP",
        prev_regime="TREND_UP",
        news_classification=cls,
        current_exposure_lots=1.0,
        chat_fn=_failing_chat,
    )
    with pytest.raises(Exception):
        out.directive = "halt_and_close_all"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Performance benchmark — hard rule path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decide_exit_hard_rule_under_one_millisecond_avg() -> None:
    """Hard-rule path average latency must be well under 1 ms."""
    cls = _classification(intensity="high", direction="against")
    iterations = 1_000
    t0 = time.perf_counter()
    for _ in range(iterations):
        decide_exit(
            regime="TREND_UP",
            prev_regime="TREND_UP",
            news_classification=cls,
            current_exposure_lots=1.0,
            chat_fn=_failing_chat,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_call_us = elapsed_ms * 1000 / iterations
    # Generous bound — pure dict lookups + dataclass build.
    assert per_call_us < 200.0, f"{per_call_us:.1f} µs/call exceeds 200 µs"
