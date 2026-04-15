"""Run a single Gate 1 v7 OOS window with Sprint 6 AI regime routing.

Usage:
    python scripts/run_v7_single_window.py <window_num> [--ai-regime]

    window_num: 1-based window index (1 = first OOS window)
    --ai-regime: Enable AI regime classification (default: ATR fallback only)

Key v7 differences from v6 (Sprint 6):
- AI regime classification with RegimeParams routing
- allowed_triggers per regime (TRANSITION: fvg_fill+bos only, no choch)
- allowed_directions per regime (TREND_UP: long only, TREND_DOWN: short only)
- Regime-aware SL/TP params (sl_atr_multiplier, tp1_rr)
- max_concurrent cap per regime
- Pre-computed regime cache for backtest performance

Results are appended to data/gate1_v7_windows.jsonl (one JSON per line).
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smc.ai.models import RegimeParams
from smc.ai.param_router import PRESETS, route
from smc.ai.regime_cache import RegimeCacheLookup, build_regime_cache
from smc.ai.regime_classifier import classify_regime_ai
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
# Sprint 6 verification
# ---------------------------------------------------------------------------

def _verify_sprint6_features() -> dict[str, bool]:
    """Check Sprint 6 code changes are active."""
    checks: dict[str, bool] = {}

    # Check 1: RegimeParams has allowed_triggers field
    checks["regime_params_allowed_triggers"] = "allowed_triggers" in RegimeParams.model_fields

    # Check 2: TRANSITION preset excludes choch_in_zone
    transition_preset = PRESETS.get("TRANSITION")
    checks["transition_no_choch"] = (
        transition_preset is not None
        and "choch_in_zone" not in transition_preset.allowed_triggers
    )

    # Check 3: TREND_UP only allows long direction
    trend_up_preset = PRESETS.get("TREND_UP")
    checks["trend_up_long_only"] = (
        trend_up_preset is not None
        and trend_up_preset.allowed_directions == ("long",)
    )

    # Check 4: classify_regime_ai has cache parameter
    sig = inspect.signature(classify_regime_ai)
    checks["classify_has_cache"] = "cache" in sig.parameters

    # Check 5: aggregator accepts regime_cache
    agg_sig = inspect.signature(MultiTimeframeAggregator.__init__)
    checks["aggregator_has_cache"] = "regime_cache" in agg_sig.parameters

    # Check 6: generate_setups accepts bar_ts
    gs_sig = inspect.signature(MultiTimeframeAggregator.generate_setups)
    checks["generate_setups_has_bar_ts"] = "bar_ts" in gs_sig.parameters

    # Check 7: check_entry accepts sl_atr_multiplier
    from smc.strategy.entry_trigger import check_entry
    ce_sig = inspect.signature(check_entry)
    checks["check_entry_sl_atr"] = "sl_atr_multiplier" in ce_sig.parameters

    # Carry forward Sprint 4+5 checks
    from smc.strategy import entry_trigger as et_mod
    checks["atr_adaptive_sl"] = hasattr(et_mod, "_compute_sl_buffer")

    return checks


def _compute_atr_pct(d1_df, atr_period: int = 14) -> float | None:
    """Compute ATR(14) as % of price from D1 data."""
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


# ---------------------------------------------------------------------------
# Regime cache management
# ---------------------------------------------------------------------------

_CACHE_PATH = PROJECT_ROOT / "data" / "regime_cache.parquet"


def _ensure_regime_cache(lake: ForexDataLake) -> RegimeCacheLookup:
    """Build regime cache if not exists, then load it."""
    if not _CACHE_PATH.exists():
        print("  Building regime cache (first run)...")
        build_regime_cache(lake, _CACHE_PATH, frequency_hours=4)
        print(f"  Cache built: {_CACHE_PATH}")
    return RegimeCacheLookup(_CACHE_PATH)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single_window(window_num: int, *, ai_regime: bool = False) -> dict:
    """Run a single OOS window with Sprint 6 AI regime routing."""
    sprint6_checks = _verify_sprint6_features()

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
    lake = ForexDataLake(PROJECT_ROOT / "data" / "parquet")

    # Load or build regime cache
    regime_cache = _ensure_regime_cache(lake)

    # Create aggregator with Sprint 6 regime routing
    aggregator = MultiTimeframeAggregator(
        detector=detector,
        ai_regime_enabled=ai_regime,
        regime_cache=regime_cache,
    )

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
    print(f"  Sprint 6 checks: {sprint6_checks}")
    print(f"  AI regime: {ai_regime}")

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v7", "error": "no test data"}

    # Regime diagnostic at test start
    d1_data = lake.query("XAUUSD", Timeframe.D1, window_start, test_end)
    h4_data = lake.query("XAUUSD", Timeframe.H4, window_start, test_end)
    d1_at_test_start = d1_data.filter(d1_data["ts"] < train_end) if not d1_data.is_empty() else None

    atr_regime = classify_regime(d1_at_test_start)
    atr_pct = _compute_atr_pct(d1_at_test_start)

    # AI regime at test start (from cache)
    ai_assessment = regime_cache.lookup(train_end)
    ai_regime_label = ai_assessment.regime if ai_assessment else "N/A"
    ai_trend_dir = ai_assessment.trend_direction if ai_assessment else "N/A"
    ai_confidence = ai_assessment.confidence if ai_assessment else 0.0
    ai_triggers = list(ai_assessment.param_preset.allowed_triggers) if ai_assessment else []
    ai_directions = list(ai_assessment.param_preset.allowed_directions) if ai_assessment else []

    print(f"  ATR regime: {atr_regime}, ATR%: {atr_pct}")
    print(f"  AI regime: {ai_regime_label} ({ai_trend_dir}, conf={ai_confidence:.2f})")
    print(f"  Allowed triggers: {ai_triggers}")
    print(f"  Allowed directions: {ai_directions}")

    # Clear cooldowns/active zones
    aggregator.clear_cooldowns()
    aggregator.clear_active_zones()

    t0 = time.time()
    setups = strategy.generate_setups(test_bars)

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

    # Trade-level details
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
        "version": "v7",
        "ai_regime_enabled": ai_regime,
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
        "atr_regime_at_start": atr_regime,
        "atr_pct_at_start": atr_pct,
        "ai_regime_at_start": ai_regime_label,
        "ai_trend_direction": ai_trend_dir,
        "ai_confidence": round(ai_confidence, 4),
        "ai_allowed_triggers": ai_triggers,
        "ai_allowed_directions": ai_directions,
        "sprint6_checks": sprint6_checks,
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
        print("Usage: python run_v7_single_window.py <window_num> [--ai-regime]")
        print("  window_num: 1-15 (1-based)")
        print("  --ai-regime: Enable AI debate pipeline (requires LLM)")
        sys.exit(1)

    window_num = int(sys.argv[1])
    if not 1 <= window_num <= 15:
        print(f"Error: window_num must be 1-15, got {window_num}")
        sys.exit(1)

    ai_regime = "--ai-regime" in sys.argv

    result = run_single_window(window_num, ai_regime=ai_regime)

    out_path = PROJECT_ROOT / "data" / "gate1_v7_windows.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
