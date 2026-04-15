"""Gate 1 OOS Walk-Forward Runner.

Executes the full walk-forward out-of-sample validation for the
AI-SMC multi-timeframe trading system with production parameters.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smc.backtest.adapter_fast import FastSMCStrategyAdapter
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig, BacktestResult
from smc.backtest.walk_forward import aggregate_oos_results, _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator


def main() -> None:
    """Run Gate 1 OOS walk-forward validation."""
    print("=" * 70)
    print("  AI-SMC Gate 1: Walk-Forward OOS Validation")
    print("=" * 70)
    print()

    # --- Configuration ---
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

    # --- Strategy ---
    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(detector=detector)

    # --- Data Lake ---
    lake_path = PROJECT_ROOT / "data" / "parquet"
    lake = ForexDataLake(lake_path)

    # Verify data availability
    m15_range = lake.available_range("XAUUSD", Timeframe.M15)
    if m15_range is None:
        print("ERROR: No M15 data found in data lake.")
        sys.exit(1)

    print(f"Data range: {m15_range[0].date()} to {m15_range[1].date()}")
    print(f"M15 bars: {lake.row_count('XAUUSD', Timeframe.M15):,}")
    print()

    # --- Strategy Adapter (fast chunked version) ---
    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator,
        lake=lake,
        instrument="XAUUSD",
    )

    # --- Walk-Forward Parameters ---
    train_months = 12
    test_months = 3
    step_months = 3

    print(f"Walk-forward config:")
    print(f"  Train window: {train_months} months")
    print(f"  Test window:  {test_months} months")
    print(f"  Step size:    {step_months} months")
    print(f"  Spread:       {config.spread_points} points")
    print(f"  Slippage:     {config.slippage_points} points")
    print(f"  Commission:   ${config.commission_per_lot}/lot")
    print(f"  Initial bal:  ${config.initial_balance:,.0f}")
    print()
    print("Running walk-forward OOS validation...")
    print("-" * 70)

    start_time = time.time()

    # --- Execute Walk-Forward (inline with progress logging) ---
    instrument = config.instrument
    timeframe = Timeframe.M15
    available = lake.available_range(instrument, timeframe)
    if available is None:
        print("ERROR: No data available.")
        sys.exit(1)

    data_start, data_end = available
    results: list[BacktestResult] = []
    window_start = data_start
    window_num = 0

    while True:
        train_end = _add_months(window_start, train_months)
        test_end = _add_months(train_end, test_months)

        if test_end > data_end:
            break

        window_num += 1
        w_start = time.time()
        print(
            f"  Window {window_num:>2}: train {window_start.strftime('%Y-%m')}-"
            f"{train_end.strftime('%Y-%m')}, test {train_end.strftime('%Y-%m')}-"
            f"{test_end.strftime('%Y-%m')} ...",
            end="",
            flush=True,
        )

        train_bars = lake.query(instrument, timeframe, window_start, train_end)
        test_bars = lake.query(instrument, timeframe, train_end, test_end)

        if train_bars.is_empty() or test_bars.is_empty():
            print(" SKIP (empty data)")
            window_start = _add_months(window_start, step_months)
            continue

        strategy.train(train_bars)
        setups = strategy.generate_setups(test_bars)
        result = engine.run(setups, test_bars)
        results.append(result)

        w_elapsed = time.time() - w_start
        print(
            f" {w_elapsed:.0f}s | {result.total_trades} trades | "
            f"Sharpe {result.sharpe:.3f} | DD {result.max_drawdown_pct*100:.1f}%",
            flush=True,
        )

        window_start = _add_months(window_start, step_months)

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"OOS windows produced: {len(results)}")
    print()

    # --- Per-Window Breakdown ---
    print("=" * 70)
    print("  PER-WINDOW BREAKDOWN")
    print("=" * 70)
    print(
        f"{'Win':>4} | {'Start':>12} | {'End':>12} | "
        f"{'Trades':>6} | {'Sharpe':>7} | {'MaxDD%':>7} | "
        f"{'PF':>6} | {'WinR%':>6} | {'PnL$':>10}"
    )
    print("-" * 90)

    for i, r in enumerate(results):
        # Compute PnL from equity curve
        pnl = r.equity_curve.equity[-1] - r.equity_curve.equity[0] if r.equity_curve.equity else 0.0
        print(
            f"  {i+1:>2} | {r.start_date.strftime('%Y-%m-%d'):>12} | "
            f"{r.end_date.strftime('%Y-%m-%d'):>12} | "
            f"{r.total_trades:>6} | {r.sharpe:>7.3f} | "
            f"{r.max_drawdown_pct * 100:>6.2f}% | "
            f"{r.profit_factor:>6.2f} | {r.win_rate * 100:>5.1f}% | "
            f"${pnl:>9.2f}"
        )

    print()

    # --- Aggregate Summary ---
    summary = aggregate_oos_results(results)

    print("=" * 70)
    print("  AGGREGATE OOS SUMMARY")
    print("=" * 70)
    print(f"  Total OOS windows:     {summary.windows}")
    print(f"  Pooled Sharpe:         {summary.pooled_sharpe:.3f}")
    print(f"  Consistency ratio:     {summary.consistency_ratio:.1%}")
    print(f"  Total OOS trades:      {summary.total_oos_trades}")

    # Compute max drawdown across all windows
    max_dd_across = max((r.max_drawdown_pct for r in results), default=0.0)
    print(f"  Max DD (worst window): {max_dd_across * 100:.2f}%")

    # Compute aggregate profit factor
    all_trades = []
    for r in results:
        all_trades.extend(r.trades)
    gross_profit = sum(t.pnl_usd for t in all_trades if t.pnl_usd > 0.0)
    gross_loss = abs(sum(t.pnl_usd for t in all_trades if t.pnl_usd < 0.0))
    agg_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Aggregate PF:          {agg_pf:.3f}")

    # Aggregate win rate
    agg_wr = sum(1 for t in all_trades if t.pnl_usd > 0) / len(all_trades) if all_trades else 0.0
    print(f"  Aggregate Win Rate:    {agg_wr:.1%}")
    print()

    # --- Gate 1 Pass/Fail Criteria ---
    print("=" * 70)
    print("  GATE 1 CRITERIA CHECK")
    print("=" * 70)

    criteria = [
        ("Pooled Sharpe > 0.5", summary.pooled_sharpe > 0.5, f"{summary.pooled_sharpe:.3f}"),
        ("Consistency > 60%", summary.consistency_ratio > 0.60, f"{summary.consistency_ratio:.1%}"),
        ("Total OOS trades >= 100", summary.total_oos_trades >= 100, f"{summary.total_oos_trades}"),
        ("Max DD < 20%", max_dd_across < 0.20, f"{max_dd_across * 100:.2f}%"),
        ("Profit Factor > 1.2", agg_pf > 1.2, f"{agg_pf:.3f}"),
    ]

    all_pass = True
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        marker = "[+]" if passed else "[-]"
        all_pass = all_pass and passed
        print(f"  {marker} {name:<30} => {value:>10}  {status}")

    print()
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  ** GATE 1 VERDICT: {verdict} **")
    print()

    # --- Save results to JSON ---
    output_path = PROJECT_ROOT / "data" / "gate1_oos_results.json"
    result_data = {
        "gate1_verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
        "windows": summary.windows,
        "pooled_sharpe": round(summary.pooled_sharpe, 4),
        "consistency_ratio": round(summary.consistency_ratio, 4),
        "total_oos_trades": summary.total_oos_trades,
        "max_drawdown_pct": round(max_dd_across, 4),
        "aggregate_profit_factor": round(agg_pf, 4),
        "aggregate_win_rate": round(agg_wr, 4),
        "per_window": [
            {
                "window": i + 1,
                "start": r.start_date.isoformat(),
                "end": r.end_date.isoformat(),
                "trades": r.total_trades,
                "sharpe": round(r.sharpe, 4),
                "max_dd_pct": round(r.max_drawdown_pct, 4),
                "profit_factor": round(r.profit_factor, 4),
                "win_rate": round(r.win_rate, 4),
                "pnl_usd": round(
                    r.equity_curve.equity[-1] - r.equity_curve.equity[0]
                    if r.equity_curve.equity else 0.0,
                    2
                ),
                "trade_details": [
                    {
                        "open_ts": t.open_ts.isoformat(),
                        "close_ts": t.close_ts.isoformat(),
                        "direction": t.direction,
                        "trigger_type": t.trigger_type,
                        "open_price": round(t.open_price, 2),
                        "close_price": round(t.close_price, 2),
                        "pnl_usd": round(t.pnl_usd, 2),
                        "close_reason": t.close_reason,
                        "setup_confluence": round(t.setup_confluence, 3),
                    }
                    for t in r.trades
                ],
            }
            for i, r in enumerate(results)
        ],
        "criteria": [
            {"criterion": name, "passed": passed, "value": value}
            for name, passed, value in criteria
        ],
    }

    output_path.write_text(json.dumps(result_data, indent=2))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
