"""Tests for ``smc.hedgerock.decision_replay``.

Coverage targets the four [GO]-mandated scenarios verbatim:

1. 1-week synthetic data, no news, flat exposure → near-100% ``"none"``.
2. NFP fixture injected on day 4 → at least one ``"urgent_take_profit"``.
3. Extreme regime flip + opposed exposure → at least one ``"halt_and_close_all"``.
4. ``enable_debate=True`` with a canned chat_fn → ``cost_tracker`` records the
   debate spend and the directive is not stuck on hard-rule ``"none"``.

Plus utility coverage on:
- :func:`format_directive_distribution_table` table layout.
- :func:`run_decision_replay` argument validation.
- 30-day perf smoke (< 60s budget from [GO]).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.backtest.walk_forward import Grain
from smc.hedgerock.decision_replay import (
    DecisionReplayConfig,
    DecisionReplayResult,
    ReplayDataSource,
    ReplayObservation,
    _empty_distribution,
    _empty_pct,
    _expected_window_count,
    format_directive_distribution_table,
    run_decision_replay,
)
from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.news_engine import NewsEvent
from smc.hedgerock.schemas import EXIT_DIRECTIVES


# ---------------------------------------------------------------------------
# Constants & builders
# ---------------------------------------------------------------------------


_INSTRUMENT: Final[str] = "XAUUSD"
_BASE_TS: Final[datetime] = datetime(2024, 3, 1, tzinfo=timezone.utc)


def _features(regime: MarketRegimeAI = "TREND_UP") -> MarketFeatures:
    """Default benign MarketFeatures.

    Regime defaults to TREND_UP so the directional bias is +1 — useful
    for the urgent_take_profit / halt scenarios which require an
    opposed exposure direction.
    """
    return MarketFeatures(
        volatility_rank=0.4,
        hh_count=3,
        ll_count=1,
        h4_trend_bars=4,
        regime=regime,
    )


def _news_event(
    *,
    name: str = "Non-Farm Payrolls",
    intensity: str = "high",
    scheduled_at: datetime,
) -> NewsEvent:
    return NewsEvent(
        event_id="test-event",
        name=name,
        currency="USD",
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=scheduled_at,
        actual=300_000.0,
        forecast=200_000.0,
        previous=180_000.0,
    )


def _news_classification(
    *,
    direction: str,
    surprise: float | None = 0.5,
    intensity: str = "high",
    scheduled_at: datetime,
) -> NewsClassification:
    return NewsClassification(
        event=_news_event(intensity=intensity, scheduled_at=scheduled_at),
        direction=direction,  # type: ignore[arg-type]
        surprise_score=surprise,
        impact_currency="USD",
        classifier_version="test",
    )


class _ListSource:
    """Trivial in-memory :class:`ReplayDataSource` for tests."""

    def __init__(self, observations: Sequence[ReplayObservation]) -> None:
        self._obs = tuple(observations)

    def iter_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        grain: Grain,
        train_grains: int,
        test_grains: int,
        step_grains: int,
    ) -> Sequence[ReplayObservation]:
        return self._obs


# ---------------------------------------------------------------------------
# Scenario 1 — synthetic baseline → almost all "none"
# ---------------------------------------------------------------------------


def test_synthetic_week_no_news_yields_all_none() -> None:
    """Without news + flat exposure, hard rule must always emit 'none'."""
    obs = tuple(
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=i),
            features=_features(regime="TREND_UP"),
            prev_regime="TREND_UP",
            news_classification=None,
            current_exposure_lots=0.0,
        )
        for i in range(7)
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=7),
        grain="day",
        train_grains=1,
        test_grains=1,
        step_grains=1,
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))

    assert result.total_decisions == 7
    assert result.directive_distribution["none"] == 7
    assert result.directive_pct["none"] == pytest.approx(100.0)
    assert result.total_cost_usd == 0.0
    # Every other directive must read zero — the parallel dict invariant.
    for directive in EXIT_DIRECTIVES:
        if directive != "none":
            assert result.directive_distribution[directive] == 0
            assert result.directive_pct[directive] == 0.0


# ---------------------------------------------------------------------------
# Scenario 2 — NFP injection → at least one urgent_take_profit
# ---------------------------------------------------------------------------


def test_nfp_against_long_emits_urgent_take_profit() -> None:
    """With a high-intensity USD NFP 'against' a +1 lot long, hard rule
    must emit ``urgent_take_profit`` for that window."""
    nfp_ts = _BASE_TS + timedelta(days=3)
    benign = ReplayObservation(
        ts=_BASE_TS,
        features=_features(),
        prev_regime="TREND_UP",
    )
    nfp_obs = ReplayObservation(
        ts=nfp_ts,
        features=_features(regime="TREND_UP"),
        prev_regime="TREND_UP",
        news_classification=_news_classification(
            direction="against",
            scheduled_at=nfp_ts,
        ),
        current_exposure_lots=1.0,  # long, opposed to USD-strengthening NFP
    )
    obs = (benign, nfp_obs, benign, benign)

    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=7),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))

    assert result.directive_distribution["urgent_take_profit"] >= 1
    assert result.directive_distribution["none"] == 3


def test_nfp_with_long_emits_news_trade_window() -> None:
    """Favourable NFP + matching exposure → news_trade_window."""
    nfp_ts = _BASE_TS + timedelta(days=2)
    obs = (
        ReplayObservation(
            ts=nfp_ts,
            features=_features(regime="TREND_UP"),
            prev_regime="TREND_UP",
            news_classification=_news_classification(
                direction="with",
                scheduled_at=nfp_ts,
            ),
            current_exposure_lots=0.5,
        ),
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=7),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))
    assert result.directive_distribution["news_trade_window"] == 1


# ---------------------------------------------------------------------------
# Scenario 3 — extreme regime flip → halt_and_close_all
# ---------------------------------------------------------------------------


def test_extreme_regime_flip_with_opposed_exposure_emits_halt() -> None:
    """ATH_BREAKOUT → TREND_DOWN (distance 4) with long exposure → halt."""
    obs = (
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=1),
            features=_features(regime="TREND_DOWN"),
            prev_regime="ATH_BREAKOUT",
            current_exposure_lots=2.0,  # long, opposes new TREND_DOWN bias
        ),
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=2),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))
    assert result.directive_distribution["halt_and_close_all"] == 1
    decision = result.raw_decisions[0]
    assert decision.directive == "halt_and_close_all"
    assert decision.source == "hard_rule"


# ---------------------------------------------------------------------------
# Scenario 4 — debate path with canned chat_fn → cost recorded
# ---------------------------------------------------------------------------


def test_debate_enabled_with_canned_chat_fn_records_cost() -> None:
    """When the hard rule returns 'none' the debate kicks in; with a
    canned bull/bear chat_fn we should see cost_usd accumulate per call."""
    canned_calls: list[tuple[str, str]] = []

    def chat_fn(system: str, user: str, max_tokens: int) -> tuple[str, int, float]:
        canned_calls.append((system, user))
        # Both agents recommend the same directive so majority vote is
        # decisive without complicated bull/bear logic in the test.
        return ("RECOMMEND: urgent_take_profit", 50, 0.012)

    # Set up a regime/exposure pair that the hard rule does NOT halt
    # but where exposure is non-trivial — so decide_exit drops into the
    # debate path. With CONSOLIDATION the directional bias is 0 so the
    # halt rule (rule 1) is skipped, news_classification=None skips
    # rules 2/3, and the function falls into debate.
    obs = (
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=2),
            features=_features(regime="CONSOLIDATION"),
            prev_regime="CONSOLIDATION",
            news_classification=None,
            current_exposure_lots=0.5,
        ),
    )
    tracker = CostTracker(daily_budget_usd=10.0, burst_budget_usd=5.0)
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=3),
        enable_debate=True,
        cost_tracker=tracker,
    )
    result = run_decision_replay(config, _ListSource(obs), chat_fn=chat_fn)

    # Debate ran (build_envelope + re-derive both consult chat_fn)
    assert len(canned_calls) >= 2
    # Cost tracker shows non-zero spend.
    assert tracker.daily_spend > 0
    # Each window's decision was the agreed urgent_take_profit.
    assert result.raw_decisions[0].directive == "urgent_take_profit"
    assert result.raw_decisions[0].source == "micro_debate"
    # Aggregated cost is at least one chat call per window.
    assert result.total_cost_usd >= 0.012


# ---------------------------------------------------------------------------
# Latency, table formatting, helpers, validation
# ---------------------------------------------------------------------------


def test_format_directive_distribution_table_lists_every_directive() -> None:
    obs = (
        ReplayObservation(
            ts=_BASE_TS,
            features=_features(),
            prev_regime="TREND_UP",
        ),
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=2),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))
    rendered = format_directive_distribution_table(result)
    for directive in EXIT_DIRECTIVES:
        assert directive in rendered
    assert "windows: 1" in rendered
    assert "XAUUSD" in rendered


def test_run_decision_replay_rejects_inverted_range() -> None:
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS + timedelta(days=2),
        end=_BASE_TS,  # earlier than start → invalid
    )
    with pytest.raises(ValueError, match="must be earlier"):
        run_decision_replay(config, _ListSource(()))


def test_run_decision_replay_rejects_non_positive_grains() -> None:
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=10),
        train_grains=0,
    )
    with pytest.raises(ValueError):
        run_decision_replay(config, _ListSource(()))


def test_empty_source_produces_zeroed_result() -> None:
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=10),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(()))
    assert result.total_decisions == 0
    assert result.windows_processed == 0
    assert result.avg_latency_ms == 0.0
    assert result.total_cost_usd == 0.0
    for directive in EXIT_DIRECTIVES:
        assert result.directive_distribution[directive] == 0
        assert result.directive_pct[directive] == 0.0


def test_expected_window_count_matches_walk_forward_loop() -> None:
    # 30-day window with 7-train/1-test/1-step → 23 windows
    # (cursor 0..22 → train_end at 7..29, test_end at 8..30 ≤ 30).
    count = _expected_window_count(
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=30),
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
    )
    assert count == 23


def test_empty_distribution_and_pct_helpers_match_directive_set() -> None:
    dist = _empty_distribution()
    pct = _empty_pct()
    assert set(dist.keys()) == set(EXIT_DIRECTIVES)
    assert set(pct.keys()) == set(EXIT_DIRECTIVES)
    assert all(v == 0 for v in dist.values())
    assert all(v == 0.0 for v in pct.values())


def test_30day_replay_under_perf_budget() -> None:
    """30-day, 23-window run < 60s — the [GO] perf budget."""
    import time

    obs = tuple(
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=i),
            features=_features(),
            prev_regime="TREND_UP",
        )
        for i in range(30)
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=30),
        enable_debate=False,
    )
    t0 = time.perf_counter()
    result = run_decision_replay(config, _ListSource(obs))
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0
    # 23 windows from _expected_window_count are not enforced — the
    # source provides 30 observations and decision_replay consumes them
    # all. The perf budget is the contract, not the window count.
    assert result.total_decisions == 30


def test_replay_aggregates_match_per_observation_source() -> None:
    """Mixed observations: 2 none + 1 urgent_TP + 1 halt = correct distribution."""
    nfp_ts = _BASE_TS + timedelta(days=2)
    obs = (
        ReplayObservation(
            ts=_BASE_TS,
            features=_features(),
            prev_regime="TREND_UP",
        ),
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=1),
            features=_features(),
            prev_regime="TREND_UP",
        ),
        ReplayObservation(
            ts=nfp_ts,
            features=_features(regime="TREND_UP"),
            prev_regime="TREND_UP",
            news_classification=_news_classification(
                direction="against",
                scheduled_at=nfp_ts,
            ),
            current_exposure_lots=1.0,
        ),
        ReplayObservation(
            ts=_BASE_TS + timedelta(days=3),
            features=_features(regime="TREND_DOWN"),
            prev_regime="ATH_BREAKOUT",
            current_exposure_lots=2.0,
        ),
    )
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=4),
        enable_debate=False,
    )
    result = run_decision_replay(config, _ListSource(obs))
    assert result.total_decisions == 4
    assert result.directive_distribution["none"] == 2
    assert result.directive_distribution["urgent_take_profit"] == 1
    assert result.directive_distribution["halt_and_close_all"] == 1
    # Latency aggregate is non-trivial.
    assert result.avg_latency_ms > 0.0


def test_replay_data_source_protocol_is_structural() -> None:
    """Sanity: any object with iter_observations works as a source.

    ``ReplayDataSource`` is a structural Protocol (not runtime_checkable)
    so we duck-type-check the method exists rather than isinstance.
    """
    source = _ListSource(())
    assert hasattr(source, "iter_observations")
    assert callable(source.iter_observations)


def test_decision_replay_result_is_immutable() -> None:
    """Frozen dataclass invariant — the result must reject post-construction edits."""
    config = DecisionReplayConfig(
        instrument=_INSTRUMENT,
        start=_BASE_TS,
        end=_BASE_TS + timedelta(days=2),
    )
    result: DecisionReplayResult = run_decision_replay(config, _ListSource(()))
    with pytest.raises(Exception):
        result.total_decisions = 999  # type: ignore[misc]
