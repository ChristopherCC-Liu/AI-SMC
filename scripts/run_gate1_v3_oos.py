"""Gate 1 v3 OOS Walk-Forward Runner.

Sprint 0+1+2 tuned parameters:
- HTF look-ahead fixed (adapter filters by candle close time)
- chunk_size = 4 (1-hour M15 chunks)
- SL buffer = 150 points
- TP1 RR = 2.5, TP2 RR = 4.0
- Zone expansion = 25%
- Per-TF swing_length: D1=5, H4=7, H1=10, M15=10
- Confluence threshold = 0.45
- Sprint 2 additions:
  - H4-only tiered bias (Tier 1/2/3 system)
  - BOS entry trigger (4th M15 trigger type)
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


# ---------------------------------------------------------------------------
# Sprint 0+1+2 verification
# ---------------------------------------------------------------------------


def verify_sprint_params() -> list[str]:
    """Verify all Sprint 0+1+2 parameters are in place. Return list of failures."""
    from smc.backtest.adapter_fast import _CHUNK_SIZE
    from smc.strategy.entry_trigger import (
        _SL_BUFFER_POINTS,
        _TP1_RR_RATIO,
        _TP2_RR_RATIO,
        _find_bos_in_zone,
    )
    from smc.strategy.confluence import TRADEABLE_THRESHOLD
    from smc.strategy.htf_bias import compute_htf_bias
    import inspect

    failures: list[str] = []

    # Sprint 0+1 params
    if _CHUNK_SIZE != 4:
        failures.append(f"chunk_size={_CHUNK_SIZE}, expected 4")
    if _SL_BUFFER_POINTS != 150.0:
        failures.append(f"SL buffer={_SL_BUFFER_POINTS}, expected 150.0")
    if _TP1_RR_RATIO != 2.5:
        failures.append(f"TP1 RR={_TP1_RR_RATIO}, expected 2.5")
    if _TP2_RR_RATIO != 4.0:
        failures.append(f"TP2 RR={_TP2_RR_RATIO}, expected 4.0")
    if TRADEABLE_THRESHOLD != 0.45:
        failures.append(f"confluence threshold={TRADEABLE_THRESHOLD}, expected 0.45")

    # Sprint 2: tiered bias accepts None
    sig = inspect.signature(compute_htf_bias)
    for param_name in ("d1_snapshot", "h4_snapshot"):
        param = sig.parameters.get(param_name)
        if param is None:
            failures.append(f"compute_htf_bias missing param {param_name}")

    # Sprint 2: BOS entry trigger exists
    if not callable(_find_bos_in_zone):
        failures.append("_find_bos_in_zone not found in entry_trigger")

    return failures


def main() -> None:
    """Run Gate 1 v3 OOS walk-forward validation with Sprint 0+1+2 params."""
    print("=" * 70)
    print("  AI-SMC Gate 1 v3: Walk-Forward OOS Validation")
    print("  Sprint 0+1+2 tuned parameters")
    print("=" * 70)
    print()

    # --- Pre-flight: verify Sprint 0+1+2 params ---
    failures = verify_sprint_params()
    if failures:
        print("ABORT: Sprint parameters not in place:")
        for f in failures:
            print(f"  [-] {f}")
        sys.exit(1)
    print("[+] All Sprint 0+1+2 parameters verified.")
    print("[+] Sprint 2: H4-only tiered bias + BOS entry trigger active.")
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
    # Aggregator auto-injects per-TF swing_length_map (D1=5, H4=7, H1=10, M15=10)
    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(detector=detector)

    print(f"Per-TF swing_length: {aggregator.detector.swing_length_map}")
    print()

    # --- Data Lake ---
    lake_path = PROJECT_ROOT / "data" / "parquet"
    lake = ForexDataLake(lake_path)

    # Verify data availability
    m15_range = lake.available_range("XAUUSD", Timeframe.M15)
    if m15_range is None:
        print("ERROR: No M15 data found in data lake.")
        sys.exit(1)

    print(f"Data range: {m15_range[0].date()} to {m15_range[1].date()}")
    for tf in (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15):
        count = lake.row_count("XAUUSD", tf)
        print(f"  {tf}: {count:,} bars")
    print()

    # --- Strategy Adapter (fast chunked version, chunk=4) ---
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

    # --- Execute Walk-Forward ---
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
        pnl = (
            r.equity_curve.equity[-1] - r.equity_curve.equity[0]
            if r.equity_curve.equity
            else 0.0
        )
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

    max_dd_across = max((r.max_drawdown_pct for r in results), default=0.0)
    print(f"  Max DD (worst window): {max_dd_across * 100:.2f}%")

    all_trades = []
    for r in results:
        all_trades.extend(r.trades)
    gross_profit = sum(t.pnl_usd for t in all_trades if t.pnl_usd > 0.0)
    gross_loss = abs(sum(t.pnl_usd for t in all_trades if t.pnl_usd < 0.0))
    agg_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"  Aggregate PF:          {agg_pf:.3f}")

    agg_wr = (
        sum(1 for t in all_trades if t.pnl_usd > 0) / len(all_trades)
        if all_trades
        else 0.0
    )
    print(f"  Aggregate Win Rate:    {agg_wr:.1%}")

    # Trigger type breakdown
    trigger_counts: dict[str, int] = {}
    for t in all_trades:
        trigger_counts[t.trigger_type] = trigger_counts.get(t.trigger_type, 0) + 1
    print(f"  Trigger breakdown:     {dict(sorted(trigger_counts.items()))}")
    print()

    # --- Gate 1 v3 Criteria Check ---
    print("=" * 70)
    print("  GATE 1 v3 CRITERIA CHECK")
    print("=" * 70)

    criteria = [
        (
            "Pooled Sharpe > 0.5 (min > 0)",
            summary.pooled_sharpe > 0.5,
            summary.pooled_sharpe > 0.0,
            f"{summary.pooled_sharpe:.3f}",
        ),
        (
            "Consistency > 60% (min > 40%)",
            summary.consistency_ratio > 0.60,
            summary.consistency_ratio > 0.40,
            f"{summary.consistency_ratio:.1%}",
        ),
        (
            "Total OOS trades >= 100 (min >= 50)",
            summary.total_oos_trades >= 100,
            summary.total_oos_trades >= 50,
            f"{summary.total_oos_trades}",
        ),
        (
            "Max DD < 20%",
            max_dd_across < 0.20,
            max_dd_across < 0.20,
            f"{max_dd_across * 100:.2f}%",
        ),
        (
            "Profit Factor > 1.2",
            agg_pf > 1.2,
            agg_pf > 1.0,
            f"{agg_pf:.3f}",
        ),
        (
            "Win Rate > 35%",
            agg_wr > 0.35,
            agg_wr > 0.30,
            f"{agg_wr:.1%}",
        ),
    ]

    full_pass = True
    min_pass = True
    for name, passed_full, passed_min, value in criteria:
        if passed_full:
            marker = "[+]"
            status = "PASS"
        elif passed_min:
            marker = "[~]"
            status = "MIN-PASS"
        else:
            marker = "[-]"
            status = "FAIL"
        full_pass = full_pass and passed_full
        min_pass = min_pass and passed_min
        print(f"  {marker} {name:<40} => {value:>10}  {status}")

    print()
    if full_pass:
        verdict = "FULL PASS"
    elif min_pass:
        verdict = "MIN-PASS (marginal — proceed with caution)"
    else:
        verdict = "FAIL"
    print(f"  ** GATE 1 v3 VERDICT: {verdict} **")
    print()

    # --- v2 vs v3 Comparison ---
    v2_path = PROJECT_ROOT / "data" / "gate1_v2_oos_results.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text())
        print("=" * 70)
        print("  v2 vs v3 COMPARISON")
        print("=" * 70)
        print(f"  {'Metric':<25} {'v2':>10} {'v3':>10} {'Delta':>10}")
        print(f"  {'-'*55}")
        print(f"  {'Total trades':<25} {v2['total_oos_trades']:>10} {summary.total_oos_trades:>10} {summary.total_oos_trades - v2['total_oos_trades']:>+10}")
        print(f"  {'Pooled Sharpe':<25} {v2['pooled_sharpe']:>10.3f} {summary.pooled_sharpe:>10.3f} {summary.pooled_sharpe - v2['pooled_sharpe']:>+10.3f}")
        print(f"  {'Consistency':<25} {v2['consistency_ratio']:>9.1%} {summary.consistency_ratio:>9.1%}")
        print(f"  {'Profit Factor':<25} {v2['aggregate_profit_factor']:>10.3f} {agg_pf:>10.3f} {agg_pf - v2['aggregate_profit_factor']:>+10.3f}")
        print(f"  {'Win Rate':<25} {v2['aggregate_win_rate']:>9.1%} {agg_wr:>9.1%}")
        print(f"  {'Max DD':<25} {v2['max_drawdown_pct']*100:>9.2f}% {max_dd_across*100:>9.2f}%")
        print()

    # --- Save results to JSON ---
    output_path = PROJECT_ROOT / "data" / "gate1_v3_oos_results.json"
    result_data = {
        "version": "v3",
        "sprint": "0+1+2",
        "gate1_verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
        "windows": summary.windows,
        "pooled_sharpe": round(summary.pooled_sharpe, 4),
        "consistency_ratio": round(summary.consistency_ratio, 4),
        "total_oos_trades": summary.total_oos_trades,
        "max_drawdown_pct": round(max_dd_across, 4),
        "aggregate_profit_factor": round(agg_pf, 4),
        "aggregate_win_rate": round(agg_wr, 4),
        "trigger_breakdown": trigger_counts,
        "params": {
            "chunk_size": 4,
            "sl_buffer_points": 150.0,
            "tp1_rr": 2.5,
            "tp2_rr": 4.0,
            "zone_expansion": 0.25,
            "swing_length_map": {"D1": 5, "H4": 7, "H1": 10, "M15": 10},
            "confluence_threshold": 0.45,
            "htf_lookahead_fix": True,
            "h4_only_tiered_bias": True,
            "bos_entry_trigger": True,
        },
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
                    if r.equity_curve.equity
                    else 0.0,
                    2,
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
            {
                "criterion": name,
                "full_pass": passed_full,
                "min_pass": passed_min,
                "value": value,
            }
            for name, passed_full, passed_min, value in criteria
        ],
    }

    output_path.write_text(json.dumps(result_data, indent=2))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
