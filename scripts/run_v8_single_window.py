"""Run a single Gate v8 OOS window — v1 vs v2 A/B comparison.

Usage:
    python scripts/run_v8_single_window.py <window_num> --version v1|v2

    window_num: 1-based window index (1 = first OOS window)
    --version:  v1 (Sprint 6 pipeline) or v2 (AI direction + dual entry)

Key v8 differences from v7:
- Supports --version flag for A/B comparison (v1 vs v2)
- v1 mode: uses FastSMCStrategyAdapter with Sprint 6 regime routing
- v2 mode: uses FastSMCStrategyAdapterV2 with AI direction + dual entry
- Output: data/gate_v8_v1.jsonl or data/gate_v8_v2.jsonl
- Per-window: trades, PF, WR, PnL, trigger type breakdown, entry_mode breakdown

Results are appended to data/gate_v8_{version}.jsonl (one JSON per line).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smc.ai.direction_cache import DirectionCacheLookup, build_direction_cache
from smc.ai.regime_cache import RegimeCacheLookup, build_regime_cache
from smc.backtest.adapter_fast import FastSMCStrategyAdapter
from smc.backtest.adapter_v2 import FastSMCStrategyAdapterV2
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig
from smc.backtest.walk_forward import _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.aggregator_v2 import AggregatorV2
from smc.ai.direction_engine import DirectionEngine


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

_REGIME_CACHE_PATH = PROJECT_ROOT / "data" / "regime_cache.parquet"
_DIRECTION_CACHE_PATH = PROJECT_ROOT / "data" / "direction_cache_v2.parquet"


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def _ensure_regime_cache(lake: ForexDataLake) -> RegimeCacheLookup:
    """Build regime cache if not exists, then load it."""
    if not _REGIME_CACHE_PATH.exists():
        print("  Building regime cache (first run)...")
        build_regime_cache(lake, _REGIME_CACHE_PATH, frequency_hours=4)
        print(f"  Cache built: {_REGIME_CACHE_PATH}")
    return RegimeCacheLookup(_REGIME_CACHE_PATH)


def _ensure_direction_cache(lake: ForexDataLake) -> DirectionCacheLookup:
    """Build direction cache if not exists, then load it."""
    if not _DIRECTION_CACHE_PATH.exists():
        print("  Building direction cache (first run)...")
        build_direction_cache(
            lake, _DIRECTION_CACHE_PATH, frequency_hours=4,
        )
        print(f"  Cache built: {_DIRECTION_CACHE_PATH}")
    return DirectionCacheLookup(_DIRECTION_CACHE_PATH)


# ---------------------------------------------------------------------------
# V1 runner (Sprint 6 pipeline)
# ---------------------------------------------------------------------------


def _run_v1_window(
    window_num: int,
    lake: ForexDataLake,
    engine: BarBacktestEngine,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
) -> dict:
    """Run v1 pipeline: FastSMCStrategyAdapter + regime routing."""
    regime_cache = _ensure_regime_cache(lake)
    detector = SMCDetector(swing_length=10)

    aggregator = MultiTimeframeAggregator(
        detector=detector,
        ai_regime_enabled=False,
        regime_cache=regime_cache,
    )

    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator, lake=lake, instrument="XAUUSD",
    )

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v1", "error": "no test data"}

    # Clear cooldowns between windows
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

    return _format_result(
        result, window_num, "v1", window_start, train_end, test_end, elapsed,
    )


# ---------------------------------------------------------------------------
# V2 runner (AI direction + dual entry)
# ---------------------------------------------------------------------------


def _run_v2_window(
    window_num: int,
    lake: ForexDataLake,
    engine: BarBacktestEngine,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
) -> dict:
    """Run v2 pipeline: FastSMCStrategyAdapterV2 + AI direction + dual entry."""
    direction_cache = _ensure_direction_cache(lake)

    detector = SMCDetector(swing_length=10)
    direction_engine = DirectionEngine(
        cache_path=_DIRECTION_CACHE_PATH,
    )

    aggregator_v2 = AggregatorV2(
        detector=detector,
        direction_engine=direction_engine,
        enable_inverted=True,
        enable_fvg_sweep=True,
    )

    strategy = FastSMCStrategyAdapterV2(
        aggregator_v2=aggregator_v2, lake=lake, instrument="XAUUSD",
    )

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v2", "error": "no test data"}

    # Clear cooldowns between windows
    aggregator_v2.clear_cooldowns()
    aggregator_v2.clear_active_zones()

    t0 = time.time()
    setups = strategy.generate_setups(test_bars)
    result = engine.run(
        setups,
        test_bars,
        on_sl_hit=aggregator_v2.record_zone_loss,
        on_trade_open=aggregator_v2.mark_zone_active,
        on_trade_close=aggregator_v2.clear_zone_active,
    )
    elapsed = time.time() - t0

    return _format_result(
        result, window_num, "v2", window_start, train_end, test_end, elapsed,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_result(
    result,
    window_num: int,
    version: str,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
    elapsed: float,
) -> dict:
    """Format a BacktestResult into a JSON-serializable dict."""
    # Trigger breakdown
    triggers: dict[str, int] = {}
    reasons: dict[str, int] = {}
    entry_modes: dict[str, int] = {}
    for t in result.trades:
        triggers[t.trigger_type] = triggers.get(t.trigger_type, 0) + 1
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    # Entry mode breakdown: classify from trigger_type
    # V2 inverted triggers: ob_breakout, choch_continuation, fvg_sweep_continuation
    _INVERTED_TRIGGERS = {"ob_breakout", "choch_continuation", "fvg_sweep_continuation"}
    for t in result.trades:
        mode = "inverted" if t.trigger_type in _INVERTED_TRIGGERS else "normal"
        entry_modes[mode] = entry_modes.get(mode, 0) + 1

    pnl = (
        result.equity_curve.equity[-1] - result.equity_curve.equity[0]
        if result.equity_curve.equity
        else 0.0
    )

    # Trade-level details
    trade_details = []
    for t in result.trades:
        sl_distance_pts = (
            abs(t.open_price - t.close_price) / 0.01
            if t.close_reason == "sl"
            else None
        )
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
            "sl_distance_pts": (
                round(sl_distance_pts, 1) if sl_distance_pts is not None else None
            ),
        })

    out = {
        "window": window_num,
        "version": version,
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
        "entry_modes": entry_modes,
        "close_reasons": reasons,
        "trade_details": trade_details,
        "elapsed_s": round(elapsed, 1),
    }

    print(
        f"  [{version}] Trades: {result.total_trades}, "
        f"Sharpe: {result.sharpe:.4f}, "
        f"PF: {result.profit_factor:.4f}, "
        f"WR: {result.win_rate:.1%}, "
        f"DD: {result.max_drawdown_pct:.2%}, "
        f"PnL: ${pnl:.2f}, "
        f"Expectancy: {result.expectancy:.2f}, "
        f"Time: {elapsed:.1f}s"
    )
    print(f"  Triggers: {triggers}, Entry modes: {entry_modes}, Close: {reasons}")

    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single_window(window_num: int, *, version: str = "v2") -> dict:
    """Run a single OOS window for the specified version."""
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

    lake = ForexDataLake(PROJECT_ROOT / "data" / "parquet")

    # Calculate window boundaries
    available = lake.available_range("XAUUSD", Timeframe.M15)
    if available is None:
        raise RuntimeError("No M15 data in data lake")

    data_start = available[0]
    train_months, test_months, step_months = 12, 3, 3

    window_start = data_start
    for _ in range(window_num - 1):
        window_start = _add_months(window_start, step_months)

    train_end = _add_months(window_start, train_months)
    test_end = _add_months(train_end, test_months)

    print(
        f"Window {window_num} [{version}]: "
        f"train {window_start.date()}-{train_end.date()}, "
        f"test {train_end.date()}-{test_end.date()}"
    )

    if version == "v1":
        return _run_v1_window(
            window_num, lake, engine, window_start, train_end, test_end,
        )
    elif version == "v2":
        return _run_v2_window(
            window_num, lake, engine, window_start, train_end, test_end,
        )
    else:
        raise ValueError(f"Unknown version: {version!r} (expected 'v1' or 'v2')")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_v8_single_window.py <window_num> --version v1|v2")
        print("  window_num: 1-15 (1-based)")
        print("  --version: v1 (Sprint 6) or v2 (AI direction + dual entry)")
        sys.exit(1)

    window_num = int(sys.argv[1])
    if not 1 <= window_num <= 15:
        print(f"Error: window_num must be 1-15, got {window_num}")
        sys.exit(1)

    version = "v2"
    if "--version" in sys.argv:
        vi = sys.argv.index("--version")
        if vi + 1 < len(sys.argv):
            version = sys.argv[vi + 1]

    result = run_single_window(window_num, version=version)

    out_path = PROJECT_ROOT / "data" / f"gate_v8_{version}.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
