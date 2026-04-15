"""Unit tests for the fill model.

Validates spread/slippage math, entry fill simulation, and the
pessimistic exit rule (SL triggers before TP on ambiguous bars).
"""

from __future__ import annotations

import pytest

from smc.backtest.fills import BarOHLC, ExitResult, FillModel
from smc.smc_core.constants import XAUUSD_POINT_SIZE


@pytest.fixture()
def fill_model() -> FillModel:
    """Default fill model: 3pt spread, 0.5pt slippage, $7 commission."""
    return FillModel(spread_points=3.0, slippage_points=0.5, commission_per_lot=7.0)


# ---------------------------------------------------------------------------
# Entry Fill Tests
# ---------------------------------------------------------------------------


class TestSimulateFill:
    def test_long_fill_price(self, fill_model: FillModel) -> None:
        """Long entry: fill = bar.open + spread + slippage (in price terms)."""
        bar = BarOHLC(open=2350.00, high=2355.00, low=2348.00, close=2352.00)
        fill = fill_model.simulate_fill("long", 2350.00, bar)

        expected = 2350.00 + (3.0 * XAUUSD_POINT_SIZE) + (0.5 * XAUUSD_POINT_SIZE)
        # spread = 3 * 0.01 = 0.03, slippage = 0.5 * 0.01 = 0.005
        # fill = 2350.00 + 0.03 + 0.005 = 2350.035
        assert fill is not None
        assert abs(fill - expected) < 1e-10

    def test_short_fill_price(self, fill_model: FillModel) -> None:
        """Short entry: fill = bar.open - slippage."""
        bar = BarOHLC(open=2350.00, high=2355.00, low=2345.00, close=2348.00)
        fill = fill_model.simulate_fill("short", 2350.00, bar)

        expected = 2350.00 - (0.5 * XAUUSD_POINT_SIZE)
        # slippage = 0.5 * 0.01 = 0.005
        # fill = 2350.00 - 0.005 = 2349.995
        assert fill is not None
        assert abs(fill - expected) < 1e-10

    def test_long_fill_rejected_when_above_high(self, fill_model: FillModel) -> None:
        """Long fill is rejected when fill price exceeds bar high."""
        # Very narrow bar — fill price would exceed high
        bar = BarOHLC(open=2350.00, high=2350.01, low=2349.99, close=2350.00)
        fill = fill_model.simulate_fill("long", 2350.00, bar)
        # fill would be 2350.035 which is > high (2350.01)
        assert fill is None

    def test_short_fill_rejected_when_below_low(self, fill_model: FillModel) -> None:
        """Short fill is rejected when fill price is below bar low."""
        bar = BarOHLC(open=2350.00, high=2350.01, low=2350.00, close=2350.00)
        fill = fill_model.simulate_fill("short", 2350.00, bar)
        # fill would be 2349.995 which is < low (2350.00)
        assert fill is None

    def test_spread_math_in_points(self, fill_model: FillModel) -> None:
        """Verify spread is applied in points (* 0.01), not pips."""
        bar = BarOHLC(open=2350.00, high=2360.00, low=2340.00, close=2355.00)
        fill = fill_model.simulate_fill("long", 2350.00, bar)

        assert fill is not None
        # Spread of 3 points = 0.03 USD (not 0.30 pips)
        spread_in_price = 3.0 * XAUUSD_POINT_SIZE
        assert spread_in_price == pytest.approx(0.03)
        assert fill > bar.open  # Long fill is always worse than open

    def test_zero_spread_zero_slippage(self) -> None:
        """With no costs, long fills exactly at bar.open."""
        fm = FillModel(spread_points=0.0, slippage_points=0.0, commission_per_lot=0.0)
        bar = BarOHLC(open=2350.00, high=2360.00, low=2340.00, close=2355.00)
        fill = fm.simulate_fill("long", 2350.00, bar)
        assert fill == 2350.00


# ---------------------------------------------------------------------------
# Exit Check Tests
# ---------------------------------------------------------------------------


class TestCheckExit:
    def test_long_sl_hit(self, fill_model: FillModel) -> None:
        """Long SL triggers when bar low touches stop level."""
        bar = BarOHLC(open=2350.00, high=2352.00, low=2345.00, close=2346.00)
        result = fill_model.check_exit("long", 2350.00, sl=2346.00, tp1=2360.00, tp2=None, bar=bar)

        assert result is not None
        assert result.reason == "sl"
        assert result.exit_price == 2346.00

    def test_long_tp1_hit(self, fill_model: FillModel) -> None:
        """Long TP1 triggers when bar high reaches target."""
        bar = BarOHLC(open=2350.00, high=2365.00, low=2349.00, close=2362.00)
        result = fill_model.check_exit("long", 2350.00, sl=2340.00, tp1=2360.00, tp2=None, bar=bar)

        assert result is not None
        assert result.reason == "tp1"
        assert result.exit_price == 2360.00

    def test_long_tp2_hit(self, fill_model: FillModel) -> None:
        """Long TP2 triggers when bar high reaches second target."""
        bar = BarOHLC(open=2350.00, high=2375.00, low=2349.00, close=2370.00)
        result = fill_model.check_exit(
            "long", 2350.00, sl=2340.00, tp1=2360.00, tp2=2370.00, bar=bar
        )

        assert result is not None
        assert result.reason == "tp2"
        assert result.exit_price == 2370.00

    def test_short_sl_hit(self, fill_model: FillModel) -> None:
        """Short SL triggers when bar high reaches stop level."""
        bar = BarOHLC(open=2350.00, high=2362.00, low=2348.00, close=2355.00)
        result = fill_model.check_exit("short", 2350.00, sl=2360.00, tp1=2340.00, tp2=None, bar=bar)

        assert result is not None
        assert result.reason == "sl"
        assert result.exit_price == 2360.00

    def test_short_tp1_hit(self, fill_model: FillModel) -> None:
        """Short TP1 triggers when bar low reaches target."""
        bar = BarOHLC(open=2350.00, high=2352.00, low=2338.00, close=2340.00)
        result = fill_model.check_exit("short", 2350.00, sl=2360.00, tp1=2340.00, tp2=None, bar=bar)

        assert result is not None
        assert result.reason == "tp1"
        assert result.exit_price == 2340.00

    def test_no_exit(self, fill_model: FillModel) -> None:
        """Neither SL nor TP hit -> None."""
        bar = BarOHLC(open=2350.00, high=2355.00, low=2347.00, close=2352.00)
        result = fill_model.check_exit("long", 2350.00, sl=2340.00, tp1=2365.00, tp2=None, bar=bar)

        assert result is None

    def test_pessimistic_sl_wins_on_ambiguous_bar(self, fill_model: FillModel) -> None:
        """CRITICAL: When both SL and TP could trigger on same bar, SL wins.

        This is the pessimistic fill rule that prevents optimistic backtest bias.
        """
        # Long: bar range covers both SL and TP1
        bar = BarOHLC(open=2350.00, high=2365.00, low=2338.00, close=2355.00)
        result = fill_model.check_exit(
            "long", 2350.00, sl=2340.00, tp1=2360.00, tp2=None, bar=bar
        )

        assert result is not None
        assert result.reason == "sl", "Pessimistic rule: SL must trigger before TP"
        assert result.exit_price == 2340.00

    def test_pessimistic_short_sl_wins(self, fill_model: FillModel) -> None:
        """Short pessimistic: SL wins when both SL and TP reachable."""
        bar = BarOHLC(open=2350.00, high=2365.00, low=2335.00, close=2345.00)
        result = fill_model.check_exit(
            "short", 2350.00, sl=2360.00, tp1=2340.00, tp2=None, bar=bar
        )

        assert result is not None
        assert result.reason == "sl"
