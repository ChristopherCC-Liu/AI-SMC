"""Run a single Gate 1 v5 OOS window. Designed for chunked execution.

Usage:
    python scripts/run_v5_single_window.py <window_num>

    window_num: 1-based window index (1 = first OOS window)

Key v5 differences from v4 (Sprint 4):
- ATR-adaptive SL buffer (replaces fixed 150-point buffer)
- Tuned regime thresholds based on analyst's ATR distribution data
- Confluence scoring fixes
- Trade-level diagnostics restored for cross-review verification

Results are appended to data/gate1_v5_windows.jsonl (one JSON per line).
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
from smc.strategy.regime import classify_regime


# ---------------------------------------------------------------------------
# Sprint 4 verification helpers
# ---------------------------------------------------------------------------


def _compute_atr_pct(d1_df, atr_period: int = 14) -> float | None:
    """Compute ATR(14) as % of price from D1 data. Returns None if insufficient data."""
    if d1_df is None or len(d1_df) < atr_period + 1:
        return None

    high = d1_df["high"].to_list()
    low = d1_df["low"].to_list()
    close = d1_df["close"].to_list()

    tr_values: list[float] = []
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_values.append(max(hl, hc, lc))

    if len(tr_values) < atr_period:
        return None

    atr = sum(tr_values[-atr_period:]) / atr_period
    latest_close = close[-1]
    if latest_close <= 0:
        return None

    return round((atr / latest_close) * 100.0, 4)


def _verify_sprint4_features() -> dict[str, bool]:
    """Check that Sprint 4 code changes are present in the loaded modules.

    Returns a dict of feature -> present flag. If any feature is missing,
    the runner still proceeds but flags it in the output for review.
    """
    checks: dict[str, bool] = {}

    # Check 1: ATR-adaptive SL — entry_trigger should have _compute_sl_buffer
    # and NOT the old fixed _SL_BUFFER_POINTS constant
    from smc.strategy import entry_trigger as et_mod

    has_old_fixed_sl = hasattr(et_mod, "_SL_BUFFER_POINTS")
    has_adaptive_sl = hasattr(et_mod, "_compute_sl_buffer")
    has_atr_multiplier = hasattr(et_mod, "_SL_ATR_MULTIPLIER")
    checks["atr_adaptive_sl"] = (
        has_adaptive_sl and has_atr_multiplier and not has_old_fixed_sl
    )

    # Check 2: Tuned regime thresholds — regime.py constants changed from v4
    from smc.strategy import regime as reg_mod

    trending_thresh = getattr(reg_mod, "_TRENDING_THRESHOLD", None)
    ranging_thresh = getattr(reg_mod, "_RANGING_THRESHOLD", None)
    # v4 defaults: trending=1.2, ranging=0.8
    # Sprint 4 target: trending=1.4, ranging=1.0
    checks["regime_thresholds_tuned"] = (
        trending_thresh is not None
        and ranging_thresh is not None
        and not (trending_thresh == 1.2 and ranging_thresh == 0.8)
    )

    # Check 3: Confluence weight rebalance — trigger weight raised, liquidity lowered
    from smc.strategy import confluence as conf_mod

    trigger_weight = getattr(conf_mod, "_W_ENTRY_TRIGGER", None)
    liquidity_weight = getattr(conf_mod, "_W_LIQUIDITY", None)
    # v4: trigger=0.20, liquidity=0.15
    # Sprint 4: trigger=0.25, liquidity=0.10
    checks["confluence_rebalanced"] = (
        trigger_weight is not None
        and liquidity_weight is not None
        and trigger_weight == 0.25
        and liquidity_weight == 0.10
    )

    # Check 4: H1 ATR computation in aggregator
    from smc.strategy.aggregator import MultiTimeframeAggregator

    checks["aggregator_h1_atr"] = hasattr(MultiTimeframeAggregator, "_compute_h1_atr")

    return checks


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single_window(window_num: int) -> dict:
    """Run a single OOS window with Sprint 4 features active."""
    # Verify Sprint 4 features before running
    sprint4_checks = _verify_sprint4_features()

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

    window_start = data_start
    for _ in range(window_num - 1):
        window_start = _add_months(window_start, step_months)

    train_end = _add_months(window_start, train_months)
    test_end = _add_months(train_end, test_months)

    print(f"Window {window_num}: train {window_start.date()}-{train_end.date()}, "
          f"test {train_end.date()}-{test_end.date()}")
    print(f"  Sprint 4 checks: {sprint4_checks}")

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v5", "error": "no test data"}

    # Compute regime diagnostic from D1 data at test start
    d1_data = lake.query("XAUUSD", Timeframe.D1, window_start, test_end)
    d1_at_test_start = d1_data.filter(
        d1_data["ts"] < train_end
    ) if not d1_data.is_empty() else None

    regime = classify_regime(d1_at_test_start)
    atr_pct = _compute_atr_pct(d1_at_test_start)

    print(f"  Regime: {regime}, ATR%: {atr_pct}")

    # Clear cooldowns at window start
    aggregator.clear_cooldowns()

    t0 = time.time()
    setups = strategy.generate_setups(test_bars)

    # Run with zone cooldown callback
    result = engine.run(
        setups,
        test_bars,
        on_sl_hit=aggregator.record_zone_loss,
    )
    elapsed = time.time() - t0

    # Trigger breakdown
    triggers: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for t in result.trades:
        triggers[t.trigger_type] = triggers.get(t.trigger_type, 0) + 1
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    pnl = (
        result.equity_curve.equity[-1] - result.equity_curve.equity[0]
        if result.equity_curve.equity else 0.0
    )

    # Build trade-level details for cross-review
    trade_details = []
    for t in result.trades:
        sl_distance_pts = abs(t.open_price - t.close_price) / 0.01 if t.close_reason == "sl" else None
        trade_details.append({
            "open_ts": str(t.open_ts),
            "close_ts": str(t.close_ts),
            "direction": t.direction,
            "trigger_type": t.trigger_type,
            "open_price": t.open_price,
            "close_price": t.close_price,
            "pnl_usd": round(t.pnl_usd, 2),
            "close_reason": t.close_reason,
            "setup_confluence": round(t.setup_confluence, 3),
            "sl_distance_pts": round(sl_distance_pts, 1) if sl_distance_pts is not None else None,
        })

    out = {
        "window": window_num,
        "version": "v5",
        "train_start": str(window_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(train_end.date()),
        "test_end": str(test_end.date()),
        "trades": result.total_trades,
        "sharpe": round(result.sharpe, 4),
        "max_dd_pct": round(result.max_drawdown_pct, 4),
        "profit_factor": round(result.profit_factor, 4),
        "win_rate": round(result.win_rate, 4),
        "pnl_usd": round(pnl, 2),
        "expectancy": round(result.expectancy, 4),
        "triggers": triggers,
        "close_reasons": reasons,
        "regime_at_start": regime,
        "atr_pct_at_start": atr_pct,
        "sprint4_checks": sprint4_checks,
        "trade_details": trade_details,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"  Trades: {result.total_trades}, Sharpe: {result.sharpe:.4f}, "
          f"PF: {result.profit_factor:.4f}, WR: {result.win_rate:.1%}, "
          f"DD: {result.max_drawdown_pct:.2%}, PnL: ${pnl:.2f}, "
          f"Expectancy: {result.expectancy:.2f}, Time: {elapsed:.1f}s")
    print(f"  Triggers: {triggers}, Close: {reasons}")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_v5_single_window.py <window_num>")
        print("  window_num: 1-15 (1-based)")
        sys.exit(1)

    window_num = int(sys.argv[1])
    if not 1 <= window_num <= 15:
        print(f"Error: window_num must be 1-15, got {window_num}")
        sys.exit(1)

    result = run_single_window(window_num)

    out_path = PROJECT_ROOT / "data" / "gate1_v5_windows.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
