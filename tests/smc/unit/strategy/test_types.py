"""Unit tests for smc.strategy.types — frozen Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.data.schemas import Timeframe
from smc.strategy.types import BiasDirection, EntrySignal, TradeSetup, TradeZone


class TestBiasDirection:
    def test_frozen(self) -> None:
        bias = BiasDirection(
            direction="bullish",
            confidence=0.8,
            key_levels=(2350.0, 2375.0),
            rationale="test",
        )
        with pytest.raises(Exception):
            bias.confidence = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        bias = BiasDirection(
            direction="bearish",
            confidence=0.65,
            key_levels=(2340.0,),
            rationale="D1 bearish trend",
        )
        assert bias.direction == "bearish"
        assert bias.confidence == 0.65
        assert bias.key_levels == (2340.0,)
        assert bias.rationale == "D1 bearish trend"

    def test_neutral(self) -> None:
        bias = BiasDirection(
            direction="neutral",
            confidence=0.0,
            key_levels=(),
            rationale="no trend",
        )
        assert bias.direction == "neutral"
        assert bias.confidence == 0.0


class TestTradeZone:
    def test_frozen(self) -> None:
        zone = TradeZone(
            zone_high=2355.0,
            zone_low=2350.0,
            zone_type="ob",
            direction="long",
            timeframe=Timeframe.H1,
            confidence=0.8,
        )
        with pytest.raises(Exception):
            zone.confidence = 0.5  # type: ignore[misc]

    def test_all_zone_types(self) -> None:
        for zt in ("ob", "fvg", "ob_fvg_overlap"):
            zone = TradeZone(
                zone_high=2360.0,
                zone_low=2355.0,
                zone_type=zt,  # type: ignore[arg-type]
                direction="short",
                timeframe=Timeframe.H1,
                confidence=0.7,
            )
            assert zone.zone_type == zt


class TestEntrySignal:
    def test_frozen(self) -> None:
        sig = EntrySignal(
            entry_price=2350.0,
            stop_loss=2347.0,
            take_profit_1=2356.0,
            take_profit_2=2362.0,
            risk_points=300.0,
            reward_points=600.0,
            rr_ratio=2.0,
            trigger_type="choch_in_zone",
            direction="long",
            grade="A",
        )
        with pytest.raises(Exception):
            sig.entry_price = 2351.0  # type: ignore[misc]

    def test_all_trigger_types(self) -> None:
        for tt in ("choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection"):
            sig = EntrySignal(
                entry_price=2350.0,
                stop_loss=2347.0,
                take_profit_1=2356.0,
                take_profit_2=2362.0,
                risk_points=300.0,
                reward_points=600.0,
                rr_ratio=2.0,
                trigger_type=tt,  # type: ignore[arg-type]
                direction="long",
                grade="B",
            )
            assert sig.trigger_type == tt

    def test_all_grades(self) -> None:
        for g in ("A", "B", "C"):
            sig = EntrySignal(
                entry_price=2350.0,
                stop_loss=2347.0,
                take_profit_1=2356.0,
                take_profit_2=2362.0,
                risk_points=300.0,
                reward_points=600.0,
                rr_ratio=2.0,
                trigger_type="choch_in_zone",
                direction="short",
                grade=g,  # type: ignore[arg-type]
            )
            assert sig.grade == g


class TestTradeSetup:
    def test_frozen(self, sample_entry_signal: EntrySignal, bullish_bias: BiasDirection, bullish_ob_zone: TradeZone) -> None:
        setup = TradeSetup(
            entry_signal=sample_entry_signal,
            bias=bullish_bias,
            zone=bullish_ob_zone,
            confluence_score=0.75,
            generated_at=datetime.now(tz=timezone.utc),
        )
        with pytest.raises(Exception):
            setup.confluence_score = 0.5  # type: ignore[misc]

    def test_nested_immutability(self, sample_entry_signal: EntrySignal, bullish_bias: BiasDirection, bullish_ob_zone: TradeZone) -> None:
        setup = TradeSetup(
            entry_signal=sample_entry_signal,
            bias=bullish_bias,
            zone=bullish_ob_zone,
            confluence_score=0.75,
            generated_at=datetime.now(tz=timezone.utc),
        )
        # Nested models should also be frozen
        with pytest.raises(Exception):
            setup.entry_signal.entry_price = 9999.0  # type: ignore[misc]
