"""HedgeRock Simulator v1 — minimal but faithful Python simulation.

Models the core HedgeRock mechanics on H1 bars:
- Symmetric hedge: open initial buy + sell at session start
- Dynamic grid: spacing = max(GRID_BASE_USD, ATR_MULTIPLIER * ATR_D1)
- TP per position: TP_USD profit target closes individual positions
- Martingale add: when price moves grid_spacing against entry, add new position
  with lot * GearRH (gear=2.0)
- Lot cap: MaxNextLot=0.5 (lot caps), MaxLotMultiply=0.667 (alternate gear logic)
- Stop conditions: equity DD > MaxEquityDrawDown (80%) → halt
- Recovery: positions reach TP_USD → close + reset

This is ~80% faithful — doesn't simulate every HedgeRock subtlety
(SMLO trailing, ADX confirm, news filter, etc.) but captures the core
hedge-martingale dynamics relevant to "where does it blow up?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


DATA_ROOT = _ai_smc_home() / "data"
REGIME_CACHE = DATA_ROOT / "regime_cache.parquet"
H1_DIR = DATA_ROOT / "parquet" / "XAUUSD" / "H1"
D1_DIR = DATA_ROOT / "parquet" / "XAUUSD" / "D1"


# HedgeRock real-XAUUSD.set
POINT = 0.1
TP_USD = 250 * POINT  # $25
GRID_BASE_USD = 250 * POINT  # $25
ATR_MULTIPLIER = 4.0
GEAR_RH = 2.0  # martingale gear
START_LOTS = 0.1
MAX_NEXT_LOT = 0.5  # lot per position cap
MAX_LOT_MULTIPLY = 0.667  # alternate gear in deep recovery (use simpler model)
INIT_EQUITY = 10000.0
MAX_DD_PCT = 0.80  # equity DD halt threshold from .set
PIP_VALUE_PER_LOT = 100.0  # XAUUSD: 1 standard lot = 100 oz, $1 per pip(0.1) per lot


@dataclass
class Position:
    direction: int  # +1 buy, -1 sell
    entry_price: float
    lots: float
    open_ts: pd.Timestamp


@dataclass
class SimState:
    equity: float = INIT_EQUITY
    cash: float = INIT_EQUITY
    positions: list[Position] = field(default_factory=list)
    last_buy_entry: float = np.nan
    last_sell_entry: float = np.nan
    closed_pnl_total: float = 0.0
    blowup: bool = False
    high_watermark: float = INIT_EQUITY
    max_dd_pct: float = 0.0
    n_trades: int = 0
    n_winning: int = 0


def open_position(state: SimState, ts, price: float, direction: int, lots: float) -> None:
    pos = Position(direction=direction, entry_price=price, lots=lots, open_ts=ts)
    state.positions.append(pos)
    if direction == 1:
        state.last_buy_entry = price
    else:
        state.last_sell_entry = price


def close_position(state: SimState, pos: Position, exit_price: float) -> None:
    pnl = (exit_price - pos.entry_price) * pos.direction * pos.lots * PIP_VALUE_PER_LOT * 10
    # 10x because PIP_VALUE_PER_LOT is per pip (0.1), not per dollar
    # XAUUSD 1 lot 100 oz: 1 dollar move = $100 PnL per lot
    pnl = (exit_price - pos.entry_price) * pos.direction * pos.lots * 100.0
    state.cash += pnl
    state.closed_pnl_total += pnl
    state.n_trades += 1
    if pnl > 0:
        state.n_winning += 1


def step(state: SimState, ts: pd.Timestamp, bar: pd.Series, grid_spacing: float) -> None:
    """Process one H1 bar."""
    if state.blowup:
        return

    high = bar["high"]
    low = bar["low"]
    close = bar["close"]

    # 1) Close TP positions (any position with profit >= TP_USD)
    new_positions = []
    for pos in state.positions:
        if pos.direction == 1:
            # Buy: TP if high >= entry + TP_USD
            if high >= pos.entry_price + TP_USD:
                close_position(state, pos, pos.entry_price + TP_USD)
                continue
        else:
            # Sell: TP if low <= entry - TP_USD
            if low <= pos.entry_price - TP_USD:
                close_position(state, pos, pos.entry_price - TP_USD)
                continue
        new_positions.append(pos)
    state.positions = new_positions

    # 2) Reset last_buy / last_sell anchor if no buys/sells open
    has_buy = any(p.direction == 1 for p in state.positions)
    has_sell = any(p.direction == -1 for p in state.positions)
    if not has_buy:
        state.last_buy_entry = np.nan
    if not has_sell:
        state.last_sell_entry = np.nan

    # 3) Open initial hedge if no positions (at start or after full close)
    if len(state.positions) == 0:
        open_position(state, ts, close, +1, START_LOTS)
        open_position(state, ts, close, -1, START_LOTS)
        return

    # 4) Martingale add — if price moved grid_spacing against last buy/sell
    # Buy add: price drops grid_spacing below last_buy_entry → add buy with lots * gear
    if has_buy and not np.isnan(state.last_buy_entry):
        if low <= state.last_buy_entry - grid_spacing:
            last_lots = max(p.lots for p in state.positions if p.direction == 1)
            new_lots = min(last_lots * GEAR_RH, MAX_NEXT_LOT)
            if new_lots > 0.001:
                add_price = state.last_buy_entry - grid_spacing
                open_position(state, ts, add_price, +1, new_lots)

    if has_sell and not np.isnan(state.last_sell_entry):
        if high >= state.last_sell_entry + grid_spacing:
            last_lots = max(p.lots for p in state.positions if p.direction == -1)
            new_lots = min(last_lots * GEAR_RH, MAX_NEXT_LOT)
            if new_lots > 0.001:
                add_price = state.last_sell_entry + grid_spacing
                open_position(state, ts, add_price, -1, new_lots)

    # 5) Mark-to-market equity & check blowup
    floating_pnl = 0.0
    for pos in state.positions:
        floating_pnl += (close - pos.entry_price) * pos.direction * pos.lots * 100.0
    state.equity = state.cash + floating_pnl
    state.high_watermark = max(state.high_watermark, state.equity)
    dd = (state.high_watermark - state.equity) / state.high_watermark
    state.max_dd_pct = max(state.max_dd_pct, dd)
    if dd >= MAX_DD_PCT:
        # Blowup: force-close all positions at current price
        for pos in state.positions:
            close_position(state, pos, close)
        state.positions = []
        state.blowup = True


def run_simulation(df: pd.DataFrame, regime_filter: str | None = None) -> dict:
    state = SimState()
    bar_count = 0
    for _, bar in df.iterrows():
        if regime_filter is not None and bar.get("regime") != regime_filter:
            continue
        ts = bar["ts"]
        atr_d1 = bar["atr_d1"]
        if not np.isfinite(atr_d1):
            continue
        grid_spacing = max(GRID_BASE_USD, ATR_MULTIPLIER * atr_d1)
        step(state, ts, bar, grid_spacing)
        bar_count += 1

    return {
        "regime": regime_filter or "ALL",
        "n_bars": bar_count,
        "final_equity": state.equity,
        "total_return_pct": (state.equity - INIT_EQUITY) / INIT_EQUITY * 100,
        "max_dd_pct": state.max_dd_pct * 100,
        "n_trades": state.n_trades,
        "win_rate": state.n_winning / max(state.n_trades, 1),
        "blowup": state.blowup,
        "closed_pnl_total": state.closed_pnl_total,
    }


def main() -> None:
    print("Loading data...")
    h1_files = sorted(H1_DIR.glob("*/*.parquet"))
    d1_files = sorted(D1_DIR.glob("*/*.parquet"))
    h1 = pd.concat([pd.read_parquet(f) for f in h1_files], ignore_index=True).sort_values("ts").reset_index(drop=True)
    d1 = pd.concat([pd.read_parquet(f) for f in d1_files], ignore_index=True).sort_values("ts").reset_index(drop=True)
    h1["ts"] = pd.to_datetime(h1["ts"], utc=True)
    d1["ts"] = pd.to_datetime(d1["ts"], utc=True)
    regime = pd.read_parquet(REGIME_CACHE)
    regime["ts"] = pd.to_datetime(regime["ts"], utc=True)
    regime = regime[["ts", "regime"]]

    # ATR_D1 → forward-fill to H1
    d1["atr_d1"] = (
        pd.concat(
            [
                (d1["high"] - d1["low"]),
                (d1["high"] - d1["close"].shift()).abs(),
                (d1["low"] - d1["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1).rolling(14).mean()
    )
    h1_with = h1.merge(regime, on="ts", how="left")
    h1_with["date"] = h1_with["ts"].dt.floor("D")
    d1_lookup = d1.set_index("ts")[["atr_d1"]]
    d1_lookup.index = d1_lookup.index.floor("D")
    h1_with["atr_d1"] = h1_with["date"].map(d1_lookup["atr_d1"]).ffill()

    # Drop bars without regime label or ATR
    df = h1_with.dropna(subset=["regime", "atr_d1"]).reset_index(drop=True)
    print(f"Simulating on {len(df)} bars with regime + ATR")

    print("\n" + "=" * 80)
    print("HEDGEROCK SIMULATOR v1 — XAUUSD H1 (regime-conditional)")
    print("=" * 80)
    print(f"Initial equity: ${INIT_EQUITY:,.0f}")
    print(f"Params: TP=${TP_USD}, grid_base=${GRID_BASE_USD}, ATR_mult={ATR_MULTIPLIER}, gear={GEAR_RH}")
    print()

    # First: full continuous simulation (all regimes)
    full_result = run_simulation(df)
    print("Full continuous simulation (all regimes mixed):")
    print(f"  Final equity: ${full_result['final_equity']:,.2f} (return {full_result['total_return_pct']:+.2f}%)")
    print(f"  Max DD: {full_result['max_dd_pct']:.2f}%")
    print(f"  Trades: {full_result['n_trades']} | Win rate: {full_result['win_rate']:.3f}")
    print(f"  Blowup: {full_result['blowup']}")
    print()

    # Per-regime continuous (only when in given regime)
    print("Per-regime simulation (only run on bars matching regime label):")
    print(f"{'regime':<15} {'bars':>8} {'final_eq':>12} {'return%':>10} {'maxDD%':>10} {'trades':>8} {'WR':>6} {'blowup':>8}")
    print("-" * 90)
    for regime_name in sorted(df["regime"].dropna().unique()):
        result = run_simulation(df, regime_filter=regime_name)
        print(
            f"{result['regime']:<15} {result['n_bars']:>8} "
            f"${result['final_equity']:>10,.0f} "
            f"{result['total_return_pct']:>+9.2f}% "
            f"{result['max_dd_pct']:>9.2f}% "
            f"{result['n_trades']:>8} "
            f"{result['win_rate']:>5.2f} "
            f"{str(result['blowup']):>8}"
        )


if __name__ == "__main__":
    main()
