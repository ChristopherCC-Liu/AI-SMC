"""Run a single Gate 1 v3 OOS window. Designed for chunked execution.

Usage:
    python scripts/run_v3_single_window.py <window_num>

    window_num: 1-based window index (1 = first OOS window)

Results are appended to data/gate1_v3_windows.jsonl (one JSON per line).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smc.backtest.adapter_fast import FastSMCStrategyAdapter
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig
from smc.backtest.walk_forward import _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator


def run_single_window(window_num: int) -> dict:
    """Run a single OOS window and return results dict."""
    config = BacktestConfig(
        initial_balance=10_000.0,
        instrument="XAUUSD",
        spread_points=3.0,
        slippage_points=0.5,
        commission_per_lot=7.0,
        max_concurrent_trades=3,
    )
    fill_model = FillModel(
        spread_points=config.spread_points,
        slippage_points=config.slippage_points,
        commission_per_lot=config.commission_per_lot,
    )
    engine = BarBacktestEngine(config=config, fill_model=fill_model)

    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(detector=detector)
    lake = ForexDataLake(PROJECT_ROOT / "data" / "parquet")

    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator, lake=lake, instrument="XAUUSD",
    )

    # Calculate window boundaries
    available = lake.available_range("XAUUSD", Timeframe.M15)
    if available is None:
        raise RuntimeError("No M15 data")

    data_start = available[0]
    train_months, test_months, step_months = 12, 3, 3

    # Advance to the requested window
    window_start = data_start
    for _ in range(window_num - 1):
        window_start = _add_months(window_start, step_months)

    train_end = _add_months(window_start, train_months)
    test_end = _add_months(train_end, test_months)

    print(f"Window {window_num}: train {window_start.date()}-{train_end.date()}, "
          f"test {train_end.date()}-{test_end.date()}")

    # Train (no-op for SMC strategy)
    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    # Test
    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "error": "no test data"}

    t0 = time.time()
    setups = strategy.generate_setups(test_bars)
    result = engine.run(setups, test_bars)
    elapsed = time.time() - t0

    out = {
        "window": window_num,
        "train_start": str(window_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(train_end.date()),
        "test_end": str(test_end.date()),
        "trades": result.total_trades,
        "sharpe": round(result.sharpe, 4),
        "max_dd_pct": round(result.max_drawdown_pct, 4),
        "profit_factor": round(result.profit_factor, 4),
        "win_rate": round(result.win_rate, 4),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"  Trades: {result.total_trades}, Sharpe: {result.sharpe:.4f}, "
          f"PF: {result.profit_factor:.4f}, WR: {result.win_rate:.1%}, "
          f"DD: {result.max_drawdown_pct:.2f}%, Time: {elapsed:.1f}s")
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_v3_single_window.py <window_num>")
        sys.exit(1)

    window_num = int(sys.argv[1])
    result = run_single_window(window_num)

    # Append to JSONL
    out_path = PROJECT_ROOT / "data" / "gate1_v3_windows.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
