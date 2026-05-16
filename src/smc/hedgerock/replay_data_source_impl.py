"""ForexDataLake-backed :class:`ReplayDataSource` for Phase 4 + Phase 5 backtests.

Phase 5.2 Stage C — lets ``decision_replay`` consume real XAUUSD
historical bars rather than the synthetic generator. The Phase 4.1
replay harness was designed Protocol-first exactly so we could plug
this in without changing the harness.

Design rules:
- This adapter **owns the lake query + window iteration**. It does
  *not* compute features — that belongs to the caller's domain
  (which features matter, what regime classifier, etc).
- Callers inject a ``feature_extractor`` callable that turns a
  per-window OHLCV ``polars.DataFrame`` into a :class:`MarketFeatures`.
  That keeps this module free of regime-classifier coupling and lets
  tests stub features deterministically.
- Window edges come from :func:`smc.backtest.walk_forward._advance` —
  same arithmetic as ``walk_forward_oos`` and ``decision_replay`` so
  three modules walk the timeline identically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

import polars as pl

from smc.ai.models import MarketRegimeAI
from smc.backtest.walk_forward import (
    Grain,
    _advance,
    _grain_default_timeframe,
)
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.hedgerock.decision_replay import ReplayObservation
from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.news_classifier import NewsClassification


__all__ = ["ForexDataLakeReplaySource", "FeatureExtractor"]


FeatureExtractor = Callable[[pl.DataFrame], MarketFeatures]
"""Build a ``MarketFeatures`` from one window's OHLCV bars.

Production: real regime classifier + volatility/swing computation.
Tests: a closure returning a fixed MarketFeatures so the test focuses
on the windowing logic alone.
"""


class ForexDataLakeReplaySource:
    """Stream :class:`ReplayObservation` from a ``ForexDataLake``.

    For each test window in ``[start, end)`` (per the
    ``train_grains`` / ``test_grains`` / ``step_grains`` rhythm):

    1. Query the lake for the train + test slice.
    2. Skip windows where either slice is empty (data gap).
    3. Run ``feature_extractor`` on the test slice → ``MarketFeatures``.
    4. Carry ``prev_regime`` forward across windows.
    5. Optionally pull a ``NewsClassification`` from
       ``news_lookup`` (caller-supplied) and an exposure value
       from ``exposure_lookup``.
    6. Emit ``ReplayObservation`` keyed at the test-end timestamp.
    """

    def __init__(
        self,
        lake: ForexDataLake,
        instrument: str,
        feature_extractor: FeatureExtractor,
        *,
        timeframe: Timeframe | None = None,
        news_lookup: Callable[[datetime], NewsClassification | None] | None = None,
        exposure_lookup: Callable[[datetime], float] | None = None,
    ) -> None:
        self._lake = lake
        self._instrument = instrument
        self._feature_extractor = feature_extractor
        self._timeframe = timeframe
        self._news_lookup = news_lookup
        self._exposure_lookup = exposure_lookup

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
        bar_tf: Timeframe = (
            self._timeframe
            if self._timeframe is not None
            else _grain_default_timeframe(grain)
        )

        observations: list[ReplayObservation] = []
        prev_regime: MarketRegimeAI | None = None
        cursor = start
        while True:
            train_end = _advance(cursor, grain, train_grains)
            test_end = _advance(train_end, grain, test_grains)
            if test_end > end:
                break

            test_bars = self._lake.query(
                self._instrument, bar_tf, train_end, test_end
            )
            if test_bars.is_empty():
                cursor = _advance(cursor, grain, step_grains)
                continue

            features = self._feature_extractor(test_bars)
            ts = test_end

            news = (
                self._news_lookup(ts) if self._news_lookup is not None else None
            )
            exposure = (
                self._exposure_lookup(ts)
                if self._exposure_lookup is not None
                else 0.0
            )

            observations.append(
                ReplayObservation(
                    ts=ts,
                    features=features,
                    prev_regime=prev_regime,
                    news_classification=news,
                    current_exposure_lots=exposure,
                )
            )
            prev_regime = features.regime
            cursor = _advance(cursor, grain, step_grains)
        return tuple(observations)
