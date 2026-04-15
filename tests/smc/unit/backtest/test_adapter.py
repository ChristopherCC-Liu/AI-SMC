"""Unit tests for the SMCStrategyAdapter.

Validates that the adapter correctly bridges MultiTimeframeAggregator
to the StrategyLike protocol expected by the walk-forward engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from smc.backtest.adapter import SMCStrategyAdapter
from smc.backtest.engine import TradeSetupLike
from smc.backtest.walk_forward import StrategyLike
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.strategy.aggregator import MultiTimeframeAggregator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_m15_bars(n: int = 10, base_price: float = 2350.0) -> pl.DataFrame:
    """Create n synthetic M15 bars."""
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    delta = timedelta(minutes=15)
    return pl.DataFrame(
        {
            "ts": [start + delta * i for i in range(n)],
            "open": [base_price + i * 0.5 for i in range(n)],
            "high": [base_price + i * 0.5 + 2.0 for i in range(n)],
            "low": [base_price + i * 0.5 - 2.0 for i in range(n)],
            "close": [base_price + i * 0.5 + 0.5 for i in range(n)],
            "volume": [1000.0] * n,
            "spread": [3.0] * n,
            "timeframe": ["M15"] * n,
            "source": ["test"] * n,
            "schema_version": [1] * n,
        },
        schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "spread": pl.Float64,
            "timeframe": pl.String,
            "source": pl.String,
            "schema_version": pl.Int32,
        },
    )


def _empty_df() -> pl.DataFrame:
    """Empty DataFrame matching OHLCV schema."""
    return pl.DataFrame(
        {
            "ts": pl.Series([], dtype=pl.Datetime("ns", "UTC")),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSMCStrategyAdapter:
    def test_satisfies_strategy_like_protocol(self) -> None:
        """Adapter must satisfy the StrategyLike protocol."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        lake = MagicMock(spec=ForexDataLake)
        adapter = SMCStrategyAdapter(agg, lake)

        assert isinstance(adapter, StrategyLike)

    def test_train_is_noop(self) -> None:
        """train() should do nothing and not raise."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        lake = MagicMock(spec=ForexDataLake)
        adapter = SMCStrategyAdapter(agg, lake)

        bars = _make_m15_bars(5)
        adapter.train(bars)  # Should not raise

    def test_empty_bars_returns_empty(self) -> None:
        """Empty input bars should produce empty output."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        lake = MagicMock(spec=ForexDataLake)
        adapter = SMCStrategyAdapter(agg, lake)

        result = adapter.generate_setups(_empty_df())
        assert result == {}

    def test_generate_setups_calls_aggregator_per_bar(self) -> None:
        """Aggregator should be called for each M15 bar."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        agg.generate_setups.return_value = ()  # No setups

        lake = MagicMock(spec=ForexDataLake)
        lake.query.return_value = _empty_df()

        adapter = SMCStrategyAdapter(agg, lake)
        bars = _make_m15_bars(5)
        adapter.generate_setups(bars)

        # Aggregator should be called 5 times (once per bar)
        assert agg.generate_setups.call_count == 5

    def test_generate_setups_groups_by_timestamp(self) -> None:
        """When Aggregator returns setups, they should be keyed by bar ts."""
        # Create a mock setup that satisfies TradeSetupLike
        mock_setup = MagicMock(spec=["entry_signal", "confluence_score"])
        mock_setup.confluence_score = 0.8
        mock_entry = MagicMock(
            spec=["entry_price", "stop_loss", "take_profit_1", "take_profit_2", "direction", "trigger_type"]
        )
        mock_entry.entry_price = 2350.0
        mock_entry.stop_loss = 2347.0
        mock_entry.take_profit_1 = 2356.0
        mock_entry.take_profit_2 = 2362.0
        mock_entry.direction = "long"
        mock_entry.trigger_type = "choch_in_zone"
        mock_setup.entry_signal = mock_entry

        agg = MagicMock(spec=MultiTimeframeAggregator)
        # Return setups only on the 3rd call (bar index 2)
        agg.generate_setups.side_effect = [
            (),
            (),
            (mock_setup,),
            (),
            (),
        ]

        lake = MagicMock(spec=ForexDataLake)
        lake.query.return_value = _empty_df()

        adapter = SMCStrategyAdapter(agg, lake)
        bars = _make_m15_bars(5)
        result = adapter.generate_setups(bars)

        # Only bar index 2 should have setups
        assert len(result) == 1
        ts_list = bars["ts"].to_list()
        assert ts_list[2] in result
        assert len(result[ts_list[2]]) == 1

    def test_queries_htf_data_from_lake(self) -> None:
        """Adapter should query D1, H4, H1 from the lake."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        agg.generate_setups.return_value = ()

        lake = MagicMock(spec=ForexDataLake)
        lake.query.return_value = _empty_df()

        adapter = SMCStrategyAdapter(agg, lake, instrument="XAUUSD")
        bars = _make_m15_bars(3)
        adapter.generate_setups(bars)

        # Lake should be queried for D1, H4, H1
        queried_tfs = [call.args[1] for call in lake.query.call_args_list]
        assert Timeframe.D1 in queried_tfs
        assert Timeframe.H4 in queried_tfs
        assert Timeframe.H1 in queried_tfs
        # M15 should NOT be queried from lake (passed directly)
        assert Timeframe.M15 not in queried_tfs

    def test_no_lookahead_m15_filtering(self) -> None:
        """M15 data passed to Aggregator must only include bars up to current."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        agg.generate_setups.return_value = ()

        lake = MagicMock(spec=ForexDataLake)
        lake.query.return_value = _empty_df()

        adapter = SMCStrategyAdapter(agg, lake)
        bars = _make_m15_bars(5)

        adapter.generate_setups(bars)

        # Check the M15 data passed to each Aggregator call
        for i, call in enumerate(agg.generate_setups.call_args_list):
            data_arg = call.args[0]
            m15_df = data_arg[Timeframe.M15]
            # M15 data should have at most i+1 rows (no future bars)
            assert len(m15_df) <= i + 1

    def test_custom_instrument(self) -> None:
        """Adapter should pass custom instrument to lake queries."""
        agg = MagicMock(spec=MultiTimeframeAggregator)
        agg.generate_setups.return_value = ()

        lake = MagicMock(spec=ForexDataLake)
        lake.query.return_value = _empty_df()

        adapter = SMCStrategyAdapter(agg, lake, instrument="EURUSD")
        bars = _make_m15_bars(2)
        adapter.generate_setups(bars)

        for call in lake.query.call_args_list:
            assert call.args[0] == "EURUSD"
