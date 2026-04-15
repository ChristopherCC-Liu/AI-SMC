"""Gate 1 v4 OOS Walk-Forward Runner.

Sprint 0+1+2+3 parameters:
- All Sprint 0-2 params (HTF fix, chunk=4, SL=150, RR 2.5/4.0, zone 25%,
  per-TF swing, threshold 0.45, tiered bias, BOS trigger)
- Sprint 3 additions:
  - ATR regime filter (blocks Tier 2/3 in ranging markets)
  - Zone cooldown (24h after SL hit, wired via engine callback)
  - Tier-gated confluence (T2/T3 need 0.55 vs T1 0.45)
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
from smc.backtest.types import BacktestConfig, BacktestResult
from smc.backtest.walk_forward import aggregate_oos_results, _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator


def main() -> None:
    print("=" * 70)
    print("  AI-SMC Gate 1 v4: Walk-Forward OOS Validation")
    print("  Sprint 0+1+2+3 (regime filter + zone cooldown + tier-gated)")
    print("=" * 70)
    print()

    # --- Verify Sprint 3 features exist ---
    from smc.strategy.regime import classify_regime
    from smc.strategy.confluence import effective_threshold, TIER2_CONFLUENCE_FLOOR
    print(f"[+] Regime filter: classify_regime available")
    print(f"[+] Tier-gated confluence: T2/T3 floor = {TIER2_CONFLUENCE_FLOOR}")
    print(f"[+] Zone cooldown: aggregator.record_zone_loss available")
    print()

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

    # Aggregator auto-injects swing_length_map
    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(detector=detector)

    lake_path = PROJECT_ROOT / "data" / "parquet"
    lake = ForexDataLake(lake_path)

    m15_range = lake.available_range("XAUUSD", Timeframe.M15)
    if m15_range is None:
        print("ERROR: No M15 data found.")
        sys.exit(1)

    print(f"Data range: {m15_range[0].date()} to {m15_range[1].date()}")
    print()

    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator, lake=lake, instrument="XAUUSD",
    )

    train_months, test_months, step_months = 12, 3, 3

    print(f"Walk-forward: train={train_months}mo, test={test_months}mo, step={step_months}mo")
    print(f"Spread={config.spread_points}pts, Slip={config.slippage_points}pts, Comm=${config.commission_per_lot}/lot")
    print()
    print("Running walk-forward OOS validation...")
    print("-" * 70)

    start_time = time.time()
    instrument = config.instrument
    timeframe = Timeframe.M15
    available = lake.available_range(instrument, timeframe)
    if available is None:
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
            f"  Window {window_num:>2}: "
            f"test {train_end.strftime('%Y-%m')}-{test_end.strftime('%Y-%m')} ...",
            end="", flush=True,
        )

        train_bars = lake.query(instrument, timeframe, window_start, train_end)
        test_bars = lake.query(instrument, timeframe, train_end, test_end)

        if train_bars.is_empty() or test_bars.is_empty():
            print(" SKIP")
            window_start = _add_months(window_start, step_months)
            continue

        # Clear cooldowns between windows
        aggregator.clear_cooldowns()

        strategy.train(train_bars)
        setups = strategy.generate_setups(test_bars)

        # Run with zone cooldown callback
        result = engine.run(
            setups, test_bars,
            on_sl_hit=aggregator.record_zone_loss,
        )
        results.append(result)

        w_elapsed = time.time() - w_start
        print(
            f" {w_elapsed:.0f}s | {result.total_trades} trades | "
            f"Sharpe {result.sharpe:.3f} | "
            f"WR {result.win_rate*100:.0f}% | "
            f"DD {result.max_drawdown_pct*100:.1f}%",
            flush=True,
        )

        window_start = _add_months(window_start, step_months)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(results)} windows)")
    print()

    # --- Per-Window Breakdown ---
    print("=" * 70)
    print("  PER-WINDOW BREAKDOWN")
    print("=" * 70)
    print(f"{'W':>3} | {'Start':>10} | {'End':>10} | {'Tr':>4} | {'Sharpe':>7} | {'DD%':>5} | {'PF':>5} | {'WR%':>4} | {'PnL$':>9}")
    print("-" * 80)
    for i, r in enumerate(results):
        pnl = r.equity_curve.equity[-1] - r.equity_curve.equity[0] if r.equity_curve.equity else 0.0
        print(
            f" {i+1:>2} | {r.start_date.strftime('%Y-%m-%d'):>10} | "
            f"{r.end_date.strftime('%Y-%m-%d'):>10} | "
            f"{r.total_trades:>4} | {r.sharpe:>7.3f} | "
            f"{r.max_drawdown_pct*100:>4.1f}% | "
            f"{r.profit_factor:>5.2f} | {r.win_rate*100:>3.0f}% | "
            f"${pnl:>8.2f}"
        )
    print()

    # --- Aggregate ---
    summary = aggregate_oos_results(results)
    max_dd = max((r.max_drawdown_pct for r in results), default=0.0)
    all_trades = [t for r in results for t in r.trades]
    gross_profit = sum(t.pnl_usd for t in all_trades if t.pnl_usd > 0)
    gross_loss = abs(sum(t.pnl_usd for t in all_trades if t.pnl_usd < 0))
    agg_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    agg_wr = sum(1 for t in all_trades if t.pnl_usd > 0) / len(all_trades) if all_trades else 0.0

    # Trigger breakdown
    triggers: dict[str, int] = {}
    for t in all_trades:
        triggers[t.trigger_type] = triggers.get(t.trigger_type, 0) + 1

    # Close reason breakdown
    reasons: dict[str, int] = {}
    for t in all_trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    print("=" * 70)
    print("  AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"  Windows:      {summary.windows}")
    print(f"  Total trades: {summary.total_oos_trades}")
    print(f"  Pooled Sharpe:{summary.pooled_sharpe:>8.3f}")
    print(f"  Consistency:  {summary.consistency_ratio:>7.1%}")
    print(f"  Aggregate PF: {agg_pf:>8.3f}")
    print(f"  Aggregate WR: {agg_wr:>7.1%}")
    print(f"  Max DD:       {max_dd*100:>7.2f}%")
    print(f"  Triggers:     {dict(sorted(triggers.items()))}")
    print(f"  Close reasons:{dict(sorted(reasons.items()))}")
    print()

    # --- Gate 1 v4 Criteria ---
    print("=" * 70)
    print("  GATE 1 v4 CRITERIA")
    print("=" * 70)
    criteria = [
        ("Sharpe > 0.5 (min > 0)", summary.pooled_sharpe > 0.5, summary.pooled_sharpe > 0.0, f"{summary.pooled_sharpe:.3f}"),
        ("Consistency > 60% (min > 40%)", summary.consistency_ratio > 0.60, summary.consistency_ratio > 0.40, f"{summary.consistency_ratio:.1%}"),
        ("Trades >= 100 (min >= 50)", summary.total_oos_trades >= 100, summary.total_oos_trades >= 50, f"{summary.total_oos_trades}"),
        ("Max DD < 20%", max_dd < 0.20, max_dd < 0.20, f"{max_dd*100:.2f}%"),
        ("PF > 1.2 (min > 1.0)", agg_pf > 1.2, agg_pf > 1.0, f"{agg_pf:.3f}"),
        ("WR > 35% (min > 30%)", agg_wr > 0.35, agg_wr > 0.30, f"{agg_wr:.1%}"),
    ]
    full_pass = min_pass = True
    for name, pf, pm, val in criteria:
        m = "[+]" if pf else ("[~]" if pm else "[-]")
        s = "PASS" if pf else ("MIN" if pm else "FAIL")
        full_pass &= pf
        min_pass &= pm
        print(f"  {m} {name:<40} => {val:>10}  {s}")
    print()
    verdict = "FULL PASS" if full_pass else ("MIN-PASS" if min_pass else "FAIL")
    print(f"  ** GATE 1 v4 VERDICT: {verdict} **")
    print()

    # --- v3 vs v4 Comparison ---
    v3_path = PROJECT_ROOT / "data" / "gate1_v3_oos_results.json"
    if v3_path.exists():
        v3 = json.loads(v3_path.read_text())
        print("=" * 70)
        print("  v3 vs v4 COMPARISON")
        print("=" * 70)
        print(f"  {'Metric':<20} {'v3':>10} {'v4':>10} {'Delta':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Trades':<20} {v3['total_oos_trades']:>10} {summary.total_oos_trades:>10} {summary.total_oos_trades - v3['total_oos_trades']:>+10}")
        print(f"  {'Sharpe':<20} {v3['pooled_sharpe']:>10.3f} {summary.pooled_sharpe:>10.3f} {summary.pooled_sharpe - v3['pooled_sharpe']:>+10.3f}")
        print(f"  {'Consistency':<20} {v3['consistency_ratio']:>9.1%} {summary.consistency_ratio:>9.1%}")
        print(f"  {'PF':<20} {v3['aggregate_profit_factor']:>10.3f} {agg_pf:>10.3f} {agg_pf - v3['aggregate_profit_factor']:>+10.3f}")
        print(f"  {'WR':<20} {v3['aggregate_win_rate']:>9.1%} {agg_wr:>9.1%}")
        print(f"  {'MaxDD':<20} {v3['max_drawdown_pct']*100:>9.2f}% {max_dd*100:>9.2f}%")
        print()

    # --- Save ---
    out = PROJECT_ROOT / "data" / "gate1_v4_oos_results.json"
    out.write_text(json.dumps({
        "version": "v4", "sprint": "0+1+2+3", "gate1_verdict": verdict,
        "elapsed_seconds": round(elapsed, 1),
        "windows": summary.windows,
        "pooled_sharpe": round(summary.pooled_sharpe, 4),
        "consistency_ratio": round(summary.consistency_ratio, 4),
        "total_oos_trades": summary.total_oos_trades,
        "max_drawdown_pct": round(max_dd, 4),
        "aggregate_profit_factor": round(agg_pf, 4),
        "aggregate_win_rate": round(agg_wr, 4),
        "trigger_breakdown": triggers,
        "close_reason_breakdown": reasons,
        "params": {
            "chunk_size": 4, "sl_buffer": 150, "tp1_rr": 2.5, "tp2_rr": 4.0,
            "zone_expansion": 0.25, "confluence_threshold": 0.45,
            "tier2_floor": 0.55, "tier3_floor": 0.55,
            "regime_filter": True, "zone_cooldown_hours": 24,
        },
        "per_window": [
            {
                "window": i+1,
                "start": r.start_date.isoformat(), "end": r.end_date.isoformat(),
                "trades": r.total_trades, "sharpe": round(r.sharpe, 4),
                "max_dd_pct": round(r.max_drawdown_pct, 4),
                "profit_factor": round(r.profit_factor, 4),
                "win_rate": round(r.win_rate, 4),
                "pnl_usd": round(r.equity_curve.equity[-1] - r.equity_curve.equity[0] if r.equity_curve.equity else 0.0, 2),
            }
            for i, r in enumerate(results)
        ],
        "criteria": [{"criterion": n, "full_pass": pf, "min_pass": pm, "value": v} for n, pf, pm, v in criteria],
    }, indent=2))
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
