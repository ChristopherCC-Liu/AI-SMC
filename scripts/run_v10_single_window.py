"""Run a single Gate v10 OOS window — v1 vs v1.1 comparison.

Usage:
    python scripts/run_v10_single_window.py <window_num> --version v1|v1.1

    window_num: 1-based window index (1 = first OOS window)
    --version:  v1 (Sprint 6 pipeline) or v1.1 (v1 + fvg_fill SHORT flip)

Key v10 differences from v9:
- Compares v1 vs v1.1 (not v1 vs v3)
- v1 mode: identical to v9 v1 (FastSMCStrategyAdapter + regime routing)
- v1.1 mode: uses AggregatorV1_1 (v1 + fvg_fill SHORT flip in non-trending)
- Output: data/gate_v10_v1.jsonl or data/gate_v10_v1_1.jsonl
- Per-window: trades, PF, WR, PnL, flipped trade breakdown
- Flipped trades (fvg_fill direction=SHORT in non-trending) tracked separately

Gate v10 criteria:
- v1.1 PF > v1 PF (inversion adds value)
- v1.1 trades = v1 trades (same count, just some flipped)
- Flipped trades: WR > 50% (from theoretical 66.6%)
- Non-flipped trades: identical to v1 (no regression)

Results are appended to data/gate_v10_{version}.jsonl (one JSON per line).
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
# Non-trending regime labels (where fvg_fill SHORT flip activates)
# ---------------------------------------------------------------------------

_NON_TRENDING_REGIMES = frozenset({
    "CONSOLIDATION",
    "TRANSITION",
})


# ---------------------------------------------------------------------------
# V1 runner (Sprint 6 pipeline — identical to v9 v1)
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
# V1.1 runner (v1 + fvg_fill SHORT flip in non-trending)
# ---------------------------------------------------------------------------


def _run_v1_1_window(
    window_num: int,
    lake: ForexDataLake,
    engine: BarBacktestEngine,
    window_start: datetime,
    train_end: datetime,
    test_end: datetime,
) -> dict:
    """Run v1.1 pipeline: AggregatorV1_1 (v1 + fvg_fill SHORT flip)."""
    # Lazy import — aggregator_v1_1 may not exist yet during pre-build
    from smc.strategy.aggregator_v1_1 import AggregatorV1_1

    regime_cache = _ensure_regime_cache(lake)
    detector = SMCDetector(swing_length=10)

    aggregator = AggregatorV1_1(
        detector=detector,
        ai_regime_enabled=False,
        regime_cache=regime_cache,
    )

    # V1.1 inherits from v1 aggregator, so FastSMCStrategyAdapter works directly
    strategy = FastSMCStrategyAdapter(
        aggregator=aggregator, lake=lake, instrument="XAUUSD",
    )

    train_bars = lake.query("XAUUSD", Timeframe.M15, window_start, train_end)
    strategy.train(train_bars)

    test_bars = lake.query("XAUUSD", Timeframe.M15, train_end, test_end)
    if len(test_bars) == 0:
        return {"window": window_num, "version": "v1.1", "error": "no test data"}

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
        result, window_num, "v1.1", window_start, train_end, test_end, elapsed,
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

# Trigger type for flipped trades in v1.1
_FVG_FILL_TRIGGER = "fvg_fill_in_zone"


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

    Includes flipped-trade-specific stats for Gate v10 criteria analysis.
    Flipped trades: fvg_fill_in_zone with direction=SHORT (these are the
    trades that v1.1 flips from LONG to SHORT in non-trending regimes).
    """
    # Trigger breakdown
    triggers: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for t in result.trades:
        triggers[t.trigger_type] = triggers.get(t.trigger_type, 0) + 1
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1

    # Direction breakdown
    direction_counts: dict[str, int] = {}
    for t in result.trades:
        direction_counts[t.direction] = direction_counts.get(t.direction, 0) + 1

    pnl = (
        result.equity_curve.equity[-1] - result.equity_curve.equity[0]
        if result.equity_curve.equity
        else 0.0
    )

    # Flipped trade stats: fvg_fill_in_zone trades with direction=short
    # In v1, these would have been LONG (zone direction). In v1.1 non-trending,
    # they get flipped to SHORT.
    flipped_trades = [
        t for t in result.trades
        if t.trigger_type == _FVG_FILL_TRIGGER and t.direction == "short"
    ]
    flipped_wins = sum(1 for t in flipped_trades if t.pnl_usd > 0)
    flipped_pnl = sum(t.pnl_usd for t in flipped_trades)
    flipped_count = len(flipped_trades)

    # Non-flipped trade stats (everything else)
    non_flipped_trades = [
        t for t in result.trades
        if not (t.trigger_type == _FVG_FILL_TRIGGER and t.direction == "short")
    ]
    non_flipped_wins = sum(1 for t in non_flipped_trades if t.pnl_usd > 0)
    non_flipped_pnl = sum(t.pnl_usd for t in non_flipped_trades)
    non_flipped_count = len(non_flipped_trades)

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
        "directions": direction_counts,
        "close_reasons": reasons,
        # v10: flipped trade breakdown for Gate criteria
        "flipped_stats": {
            "count": flipped_count,
            "wins": flipped_wins,
            "wr": round(flipped_wins / flipped_count, 4) if flipped_count > 0 else None,
            "pnl_usd": round(flipped_pnl, 2),
        },
        # v10: non-flipped trade breakdown for regression check
        "non_flipped_stats": {
            "count": non_flipped_count,
            "wins": non_flipped_wins,
            "wr": round(non_flipped_wins / non_flipped_count, 4) if non_flipped_count > 0 else None,
            "pnl_usd": round(non_flipped_pnl, 2),
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
    print(f"  Triggers: {triggers}, Directions: {direction_counts}, Close: {reasons}")
    if flipped_count > 0:
        wr_str = f"{flipped_wins / flipped_count:.1%}"
        print(
            f"  Flipped (fvg_fill SHORT): {flipped_count} trades, "
            f"WR: {wr_str}, PnL: ${flipped_pnl:.2f}"
        )
    if non_flipped_count > 0:
        nf_wr = f"{non_flipped_wins / non_flipped_count:.1%}"
        print(
            f"  Non-flipped: {non_flipped_count} trades, "
            f"WR: {nf_wr}, PnL: ${non_flipped_pnl:.2f}"
        )

    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_single_window(window_num: int, *, version: str = "v1.1") -> dict:
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
    elif version == "v1.1":
        return _run_v1_1_window(
            window_num, lake, engine, window_start, train_end, test_end,
        )
    else:
        raise ValueError(f"Unknown version: {version!r} (expected 'v1' or 'v1.1')")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_v10_single_window.py <window_num> --version v1|v1.1")
        print("  window_num: 1-15 (1-based)")
        print("  --version: v1 (Sprint 6) or v1.1 (v1 + fvg_fill SHORT flip)")
        sys.exit(1)

    window_num = int(sys.argv[1])
    if not 1 <= window_num <= 15:
        print(f"Error: window_num must be 1-15, got {window_num}")
        sys.exit(1)

    version = "v1.1"
    if "--version" in sys.argv:
        vi = sys.argv.index("--version")
        if vi + 1 < len(sys.argv):
            version = sys.argv[vi + 1]

    result = run_single_window(window_num, version=version)

    # Normalize version for filename: v1.1 -> v1_1
    file_version = version.replace(".", "_")
    out_path = PROJECT_ROOT / "data" / f"gate_v10_{file_version}.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
