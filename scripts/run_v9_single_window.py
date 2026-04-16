"""Run a single Gate v9 OOS window — v1 vs v3 (Hybrid) comparison.

Usage:
    python scripts/run_v9_single_window.py <window_num> --version v1|v3

    window_num: 1-based window index (1 = first OOS window)
    --version:  v1 (Sprint 6 pipeline) or v3 (Hybrid: v1 base + fvg_sweep)

Key v9 differences from v8:
- Compares v1 vs v3 (not v1 vs v2)
- v1 mode: identical to v8 v1 (FastSMCStrategyAdapter + regime routing)
- v3 mode: uses AggregatorV3 (v1 base + fvg_sweep_continuation trigger)
- Output: data/gate_v9_v1.jsonl or data/gate_v9_v3.jsonl
- Per-window: trades, PF, WR, PnL, trigger type breakdown
- fvg_sweep_continuation trades tracked separately for Gate criteria

Results are appended to data/gate_v9_{version}.jsonl (one JSON per line).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smc.ai.regime_cache import RegimeCacheLookup, build_regime_cache
from smc.backtest.adapter_fast import FastSMCStrategyAdapter
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig
from smc.backtest.walk_forward import _add_months
from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

_REGIME_CACHE_PATH = PROJECT_ROOT / "data" / "regime_cache.parquet"


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


# ---------------------------------------------------------------------------
# V1 runner (Sprint 6 pipeline — identical to v8 v1)
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
# V3 runner (Hybrid: v1 base + fvg_sweep_continuation)
# ---------------------------------------------------------------------------


def _run_v3_window(
    window_num: int,
    lake: ForexDataLake,
    engine: BarBacktestEngine,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
) -> dict:
    """Run v3 pipeline: AggregatorV3 (v1 base + fvg_sweep) via FastSMCStrategyAdapter."""
    # Lazy import — aggregator_v3 may not exist yet during pre-build
    from smc.strategy.aggregator_v3 import AggregatorV3

    regime_cache = _ensure_regime_cache(lake)
    detector = SMCDetector(swing_length=10)

    aggregator = AggregatorV3(
        detector=detector,
        ai_regime_enabled=False,
        regime_cache=regime_cache,
    )

    # V3 inherits from v1 aggregator, so FastSMCStrategyAdapter works directly
    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator, lake=lake, instrument="XAUUSD",
    )

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v3", "error": "no test data"}

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
        result, window_num, "v3", window_start, train_end, test_end, elapsed,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

# fvg_sweep_continuation is a new signal in v3 — tracked separately
_FVG_SWEEP_TRIGGER = "fvg_sweep_continuation"


def _format_result(
    result,
    window_num: int,
    version: str,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
    elapsed: float,
) -> dict:
    """Format a BacktestResult into a JSON-serializable dict.

    Includes fvg_sweep-specific stats for Gate v9 criteria analysis.
    """
    # Trigger breakdown
    triggers: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for t in result.trades:
        triggers[t.trigger_type] = triggers.get(t.trigger_type, 0) + 1
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    # Entry mode breakdown for v3: fvg_sweep is "continuation", others are "normal"
    entry_modes: dict[str, int] = {}
    for t in result.trades:
        mode = "fvg_sweep" if t.trigger_type == _FVG_SWEEP_TRIGGER else "normal"
        entry_modes[mode] = entry_modes.get(mode, 0) + 1

    pnl = (
        result.equity_curve.equity[-1] - result.equity_curve.equity[0]
        if result.equity_curve.equity
        else 0.0
    )

    # fvg_sweep-specific stats for Gate criteria
    fvg_sweep_trades = [t for t in result.trades if t.trigger_type == _FVG_SWEEP_TRIGGER]
    fvg_sweep_wins = sum(1 for t in fvg_sweep_trades if t.pnl_usd > 0)
    fvg_sweep_pnl = sum(t.pnl_usd for t in fvg_sweep_trades)
    fvg_sweep_count = len(fvg_sweep_trades)

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
        # v9: fvg_sweep breakdown for Gate criteria
        "fvg_sweep_stats": {
            "count": fvg_sweep_count,
            "wins": fvg_sweep_wins,
            "wr": round(fvg_sweep_wins / fvg_sweep_count, 4) if fvg_sweep_count > 0 else None,
            "pnl_usd": round(fvg_sweep_pnl, 2),
        },
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
    if fvg_sweep_count > 0:
        wr_str = f"{fvg_sweep_wins / fvg_sweep_count:.1%}" if fvg_sweep_count > 0 else "N/A"
        print(
            f"  fvg_sweep: {fvg_sweep_count} trades, "
            f"WR: {wr_str}, PnL: ${fvg_sweep_pnl:.2f}"
        )

    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single_window(window_num: int, *, version: str = "v3") -> dict:
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
    elif version == "v3":
        return _run_v3_window(
            window_num, lake, engine, window_start, train_end, test_end,
        )
    else:
        raise ValueError(f"Unknown version: {version!r} (expected 'v1' or 'v3')")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_v9_single_window.py <window_num> --version v1|v3")
        print("  window_num: 1-15 (1-based)")
        print("  --version: v1 (Sprint 6) or v3 (Hybrid: v1 + fvg_sweep)")
        sys.exit(1)

    window_num = int(sys.argv[1])
    if not 1 <= window_num <= 15:
        print(f"Error: window_num must be 1-15, got {window_num}")
        sys.exit(1)

    version = "v3"
    if "--version" in sys.argv:
        vi = sys.argv.index("--version")
        if vi + 1 < len(sys.argv):
            version = sys.argv[vi + 1]

    result = run_single_window(window_num, version=version)

    out_path = PROJECT_ROOT / "data" / f"gate_v9_{version}.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
