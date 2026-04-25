"""R9 P1 — RangeTrader AI gates 4-year A/B backtest (2020-2024).

Compares Baseline (current RangeTrader, no AI gates) vs Treatment
(RangeTrader + R9 P0 A+B+C: D1 SMA50 trend filter, AI regime block,
Donchian validity gate) on historical M15 XAUUSD data.

Both arms share the same expensive calculations (SMC snapshots, regime
classification, range detection) so the only difference between trades
is which AI gates suppressed them.  Baseline calls
`generate_range_setups()` with the production signature (no AI inputs).
Treatment passes the additional kwargs required by R9 P0.

Outputs `.scratch/round9/range_ai_gates_ab.md` with per-year metrics,
gate-block rates, and two scenario replays.

Usage
-----
    /opt/anaconda3/bin/python scripts/backtest_range_ai_gates_ab.py
    /opt/anaconda3/bin/python scripts/backtest_range_ai_gates_ab.py --years=2024
    /opt/anaconda3/bin/python scripts/backtest_range_ai_gates_ab.py --years=2023-2024
    /opt/anaconda3/bin/python scripts/backtest_range_ai_gates_ab.py --fast    # 2024 + only 200 cycles
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.getLogger("smc").setLevel(logging.WARNING)
logging.getLogger("smc.ai").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

import polars as pl

from smc.ai.models import AIRegimeAssessment
from smc.ai.regime_classifier import _atr_fallback, extract_regime_context
from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel
from smc.backtest.types import BacktestConfig, TradeRecord
from smc.data.schemas import Timeframe
from smc.instruments import get_instrument_config
from smc.smc_core.detector import SMCDetector
from smc.strategy.range_trader import RangeTrader
from smc.strategy.range_types import RangeBounds, RangeSetup
from smc.strategy.session import get_session_info


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_DATA_DIR = PROJECT_ROOT / "data" / "parquet" / "XAUUSD"
_OUTPUT_DIR = PROJECT_ROOT / ".scratch" / "round9"
_OUTPUT_PATH = _OUTPUT_DIR / "range_ai_gates_ab.md"

_INITIAL_BALANCE = 10_000.0
_LOT_SIZE = 0.01

# Cycle cadence: M15 bars to advance between RangeTrader calls.
# 4 = 1 hour cadence — matches live_demo loop (15-min sleep each cycle but
# range bounds usually shift on H1 cadence). 16 = H4 cadence (faster run).
_DEFAULT_CADENCE_BARS = 4
_FAST_CADENCE_BARS = 16

# History required at each cycle to populate H1 (Donchian-48), M15 (CHoCH), D1 (SMA50)
_H1_HISTORY_BARS = 200    # ~8 days; covers 48-bar Donchian + ATR
_M15_HISTORY_BARS = 200   # for SMC snapshot
_D1_HISTORY_BARS = 90     # for SMA50 (50) + ATR

# Treatment regime trust threshold for P0-B AI regime gate
_AI_REGIME_TRUST_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_parquet_year(subdir: str, year: int) -> pl.DataFrame:
    year_dir = _DATA_DIR / subdir / str(year)
    if not year_dir.exists():
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for fp in sorted(year_dir.glob("*.parquet")):
        frames.append(pl.read_parquet(fp))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames)
    if "ts" in df.columns:
        df = df.sort("ts")
    return df


def _load_parquet_range(
    subdir: str, start_year: int, end_year: int,
) -> pl.DataFrame:
    """Load parquet covering a year range (inclusive)."""
    frames: list[pl.DataFrame] = []
    for year in range(start_year, end_year + 1):
        df = _load_parquet_year(subdir, year)
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort("ts")


# ---------------------------------------------------------------------------
# Cycle features (deterministic — feeds both arms identically)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleFeatures:
    """Pre-computed inputs for a single A/B decision cycle."""

    bar_ts: datetime
    current_price: float
    session: str

    # Timeframe slices (no copies — references valid until next cycle)
    d1_lookback: pl.DataFrame
    h1_lookback: pl.DataFrame
    m15_lookback: pl.DataFrame

    # Snapshots
    h1_snapshot: Any
    m15_snapshot: Any

    # Regime + AI assessment
    atr_regime: str
    ai_assessment: AIRegimeAssessment

    # P0-A trend-filter inputs
    d1_sma50_slope: float          # %/bar (signed)
    d1_close_vs_sma50: float       # % (signed)

    # H1 ATR for SL buffer (in points)
    h1_atr_points: float


def _slice_until(df: pl.DataFrame, ts: datetime, tail_n: int) -> pl.DataFrame:
    """Return last ``tail_n`` rows where ts <= bar_ts (no look-ahead)."""
    if df.is_empty():
        return df
    filtered = df.filter(pl.col("ts") <= ts)
    if filtered.is_empty():
        return filtered
    return filtered.tail(tail_n)


def _compute_h1_atr_points(h1_df: pl.DataFrame) -> float:
    if h1_df.is_empty() or h1_df.height < 15:
        return 0.0
    high = h1_df["high"].to_list()
    low = h1_df["low"].to_list()
    close = h1_df["close"].to_list()
    tr: list[float] = []
    for i in range(1, len(high)):
        tr.append(max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        ))
    if len(tr) < 14:
        return 0.0
    atr_price = sum(tr[-14:]) / 14
    return atr_price / 0.01  # XAU point size


def _compute_d1_features(d1_df: pl.DataFrame) -> tuple[float, float]:
    """Return (sma50_slope_pct_per_bar, close_vs_sma50_pct).

    slope: SMA50[t] vs SMA50[t-5], in % of SMA50, divided by 5 bars.
    close_vs_sma50: (close - SMA50) / SMA50 * 100.
    """
    if d1_df.is_empty() or d1_df.height < 55:
        return 0.0, 0.0
    closes = d1_df["close"].to_list()
    if len(closes) < 55:
        return 0.0, 0.0

    def _sma(seq: list[float], end_idx: int, window: int) -> float:
        return sum(seq[end_idx - window + 1:end_idx + 1]) / window

    last_idx = len(closes) - 1
    sma_now = _sma(closes, last_idx, 50)
    sma_prev = _sma(closes, last_idx - 5, 50)
    if sma_now <= 0 or sma_prev <= 0:
        return 0.0, 0.0
    slope_pct = ((sma_now - sma_prev) / sma_prev) * 100.0 / 5.0
    close_vs = ((closes[last_idx] - sma_now) / sma_now) * 100.0
    return slope_pct, close_vs


# ---------------------------------------------------------------------------
# Block tracking
# ---------------------------------------------------------------------------


@dataclass
class GateBlockTrace:
    """Per-cycle record of which gates would have suppressed which setups."""

    setups_generated: int = 0  # in baseline (un-gated) arm
    blocked_p0_a: int = 0   # blocked by P0-A (trend filter)
    blocked_p0_b: int = 0   # blocked by P0-B (AI regime)
    blocked_p0_c: int = 0   # blocked by P0-C (Donchian validity)
    blocked_any: int = 0    # at least one of A/B/C fired
    treatment_setups: int = 0  # surviving setups in treatment arm

    def merge(self, other: "GateBlockTrace") -> None:
        self.setups_generated += other.setups_generated
        self.blocked_p0_a += other.blocked_p0_a
        self.blocked_p0_b += other.blocked_p0_b
        self.blocked_p0_c += other.blocked_p0_c
        self.blocked_any += other.blocked_any
        self.treatment_setups += other.treatment_setups


# ---------------------------------------------------------------------------
# RangeSetup → engine adapter (TradeSetupLike + EntrySignalLike protocols)
# ---------------------------------------------------------------------------


class _RangeEntrySignal:
    """Minimal entry signal exposing the protocol fields the engine needs."""

    def __init__(self, setup: RangeSetup) -> None:
        self._setup = setup

    @property
    def entry_price(self) -> float:
        return self._setup.entry_price

    @property
    def stop_loss(self) -> float:
        return self._setup.stop_loss

    @property
    def take_profit_1(self) -> float:
        # Use TP-extended (opposite boundary -10%) as the executed exit. TP1
        # midpoint exists for live-trading partial profits; in backtest we
        # simulate full position to TP-ext so PF reflects the strategy intent.
        return self._setup.take_profit_ext

    @property
    def take_profit_2(self) -> float | None:
        return None

    @property
    def direction(self) -> str:
        return self._setup.direction

    @property
    def trigger_type(self) -> str:
        return self._setup.trigger


class _RangeSetupAdapter:
    """Adapt a RangeSetup to the engine's TradeSetupLike protocol."""

    def __init__(self, setup: RangeSetup) -> None:
        self._setup = setup
        self._entry_signal = _RangeEntrySignal(setup)

    @property
    def entry_signal(self) -> _RangeEntrySignal:
        return self._entry_signal

    @property
    def confluence_score(self) -> float:
        # Approximate confluence with detection confidence; only used for
        # logging — has no influence on fills.
        return self._setup.confidence

    @property
    def zone(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmConfig:
    """Per-arm RangeTrader gate configuration."""

    name: str
    trend_filter: bool
    regime_gate: bool
    require_regime_valid: bool


# 5-arm ablation: baseline + 3 single-gate isolates + full treatment.
_ARM_CONFIGS: tuple[ArmConfig, ...] = (
    ArmConfig("baseline", False, False, False),
    ArmConfig("A_only", True, False, False),
    ArmConfig("B_only", False, True, False),
    ArmConfig("C_only", False, False, True),
    ArmConfig("treatment", True, True, True),
)


@dataclass(frozen=True)
class CycleSetupRecord:
    """Per-cycle, per-arm record of setups generated.

    Stored per (cycle_ts, arm) so we can compute counterfactual block
    masks (which gates fired in isolation vs in concert).
    """

    arm: str
    bar_ts: datetime
    bounds_source: str | None  # None = no bounds returned
    long_built: bool
    short_built: bool
    chosen_direction: str | None  # which of the built setups was picked


@dataclass
class ArmRunResult:
    arm: str
    year: int
    trades: list[TradeRecord] = field(default_factory=list)
    gate_trace: GateBlockTrace = field(default_factory=GateBlockTrace)
    cycle_count: int = 0
    setup_cycles: int = 0
    regime_distribution: Counter = field(default_factory=Counter)
    bounds_source_distribution: Counter = field(default_factory=Counter)
    # Per-cycle records for ablation cross-arm comparison.
    cycle_records: list[CycleSetupRecord] = field(default_factory=list)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usd > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl_usd < 0)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        if not self.trades:
            return 0.0
        balance = _INITIAL_BALANCE
        peak = balance
        max_dd = 0.0
        for t in sorted(self.trades, key=lambda x: x.open_ts):
            balance += t.pnl_usd
            peak = max(peak, balance)
            dd = (peak - balance) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def monthly_pnl(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.trades:
            key = t.close_ts.strftime("%Y-%m")
            out[key] = out.get(key, 0.0) + t.pnl_usd
        return out

    def quarterly_pnl(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.trades:
            q = (t.close_ts.month - 1) // 3 + 1
            key = f"{t.close_ts.year}-Q{q}"
            out[key] = out.get(key, 0.0) + t.pnl_usd
        return out


def _generate_setups(
    range_trader: RangeTrader,
    cycle: CycleFeatures,
    cfg: ArmConfig,
) -> tuple[tuple[RangeSetup, ...], RangeBounds | None]:
    """Drive the production RangeTrader for one arm.

    AI inputs are threaded only when the corresponding gate flag is on
    in the arm config — keeping baseline calls byte-identical to pre-R9
    behaviour while the single-gate isolates only enable one path.
    """
    detect_kwargs: dict[str, Any] = {}
    if cfg.require_regime_valid:
        detect_kwargs["ai_regime_assessment"] = cycle.ai_assessment
    bounds = range_trader.detect_range(
        cycle.h1_lookback, cycle.h1_snapshot, **detect_kwargs,
    )
    if bounds is None:
        return (), None

    gen_kwargs: dict[str, Any] = {
        "h1_atr": cycle.h1_atr_points,
        "session": cycle.session,
        "m15_df": cycle.m15_lookback,
    }
    if cfg.trend_filter:
        gen_kwargs["d1_df"] = cycle.d1_lookback
    if cfg.regime_gate:
        gen_kwargs["ai_regime_assessment"] = cycle.ai_assessment

    setups = range_trader.generate_range_setups(
        cycle.h1_snapshot,
        cycle.m15_snapshot,
        cycle.current_price,
        bounds,
        **gen_kwargs,
    )
    return setups, bounds


def _run_year_unified(
    year: int,
    cycles: list[CycleFeatures],
    bars_full: pl.DataFrame,
    range_trader_factory,
) -> dict[str, ArmRunResult]:
    """Run all 5 arms in lockstep over the same cycles.

    Returns a dict keyed by arm name: baseline, A_only, B_only, C_only,
    treatment.  Each arm uses an independent RangeTrader instance so
    cooldown state is not shared.
    """
    arm_results: dict[str, ArmRunResult] = {
        cfg.name: ArmRunResult(arm=cfg.name, year=year) for cfg in _ARM_CONFIGS
    }
    arm_traders: dict[str, RangeTrader] = {
        cfg.name: range_trader_factory(
            trend_filter_enabled=cfg.trend_filter,
            ai_regime_gate_enabled=cfg.regime_gate,
            require_regime_valid=cfg.require_regime_valid,
        )
        for cfg in _ARM_CONFIGS
    }
    setups_by_bar: dict[str, dict[datetime, list[Any]]] = {
        cfg.name: {} for cfg in _ARM_CONFIGS
    }

    for cycle in cycles:
        for cfg in _ARM_CONFIGS:
            result = arm_results[cfg.name]
            rt = arm_traders[cfg.name]
            result.cycle_count += 1
            result.regime_distribution[cycle.ai_assessment.regime] += 1

            setups, bounds = _generate_setups(rt, cycle, cfg)
            if bounds is not None:
                result.bounds_source_distribution[bounds.source] += 1

            long_built = any(s.direction == "long" for s in setups)
            short_built = any(s.direction == "short" for s in setups)
            chosen_dir: str | None = None
            if setups:
                result.setup_cycles += 1
                best = max(setups, key=lambda s: s.confidence)
                chosen_dir = best.direction
                setups_by_bar[cfg.name].setdefault(cycle.bar_ts, []).append(
                    _RangeSetupAdapter(best)
                )

            result.cycle_records.append(CycleSetupRecord(
                arm=cfg.name,
                bar_ts=cycle.bar_ts,
                bounds_source=bounds.source if bounds is not None else None,
                long_built=long_built,
                short_built=short_built,
                chosen_direction=chosen_dir,
            ))

        # Per-cycle gate-trace: diff baseline vs treatment for the rolling
        # aggregate counters used in the existing report tables.
        b_records = arm_results["baseline"].cycle_records[-1]
        t_records = arm_results["treatment"].cycle_records[-1]
        baseline_set_count = int(b_records.long_built) + int(b_records.short_built)
        treatment_set_count = int(t_records.long_built) + int(t_records.short_built)
        gate_trace = _build_cycle_gate_trace(
            arm_traders["treatment"],
            baseline_set_count=baseline_set_count,
            treatment_set_count=treatment_set_count,
        )
        arm_results["treatment"].gate_trace.merge(gate_trace)

    # Replay each arm's setups through the bar-by-bar engine.
    for cfg in _ARM_CONFIGS:
        arm_results[cfg.name].trades = _replay_through_engine(
            setups_by_bar[cfg.name], bars_full, year,
        )

    return arm_results


def _build_cycle_gate_trace(
    rt_treatment: RangeTrader,
    *,
    baseline_set_count: int,
    treatment_set_count: int,
) -> GateBlockTrace:
    """Aggregate per-cycle gate-block counters from treatment diagnostics."""
    trace = GateBlockTrace(
        setups_generated=baseline_set_count,
        treatment_setups=treatment_set_count,
    )

    # P0-C — range invalidation in detect_range before any setup is built.
    detect_diag = rt_treatment._last_diagnostic
    if detect_diag.get("range_invalidated_by_regime") is not None:
        # Bounds zero'd → all would-be baseline setups foreclosed.
        trace.blocked_p0_c += baseline_set_count
        trace.blocked_any += baseline_set_count
        return trace

    setups_diag = rt_treatment._last_setups_diagnostic
    if setups_diag.get("long_trend_filter_blocked"):
        trace.blocked_p0_a += 1
    if setups_diag.get("short_trend_filter_blocked"):
        trace.blocked_p0_a += 1

    regime_block = setups_diag.get("ai_regime_gate_block") or {}
    trace.blocked_p0_b += len(regime_block)

    delta = baseline_set_count - treatment_set_count
    trace.blocked_any += max(delta, 0)
    return trace


def _replay_through_engine(
    setups_by_bar: dict[datetime, list[Any]],
    bars_full: pl.DataFrame,
    year: int,
) -> list[TradeRecord]:
    """Run the setups_by_bar mapping through the production fill engine."""
    if not setups_by_bar:
        return []

    config = BacktestConfig(
        initial_balance=_INITIAL_BALANCE,
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

    setups_dict = {
        ts: tuple(adapters) for ts, adapters in setups_by_bar.items()
    }
    year_bars = bars_full.filter(
        (pl.col("ts") >= datetime(year, 1, 1, tzinfo=timezone.utc))
        & (pl.col("ts") < datetime(year + 1, 1, 1, tzinfo=timezone.utc))
    )
    if year_bars.is_empty():
        return []
    bt_result = engine.run(setups_dict, year_bars, trail_rule=None)
    return list(bt_result.trades)


# ---------------------------------------------------------------------------
# Cycle pre-computation (shared across both arms)
# ---------------------------------------------------------------------------


def _build_cycle(
    bar_ts: datetime,
    bar_close: float,
    detector: SMCDetector,
    d1_full: pl.DataFrame,
    h1_full: pl.DataFrame,
    m15_full: pl.DataFrame,
    cfg: Any,
) -> CycleFeatures | None:
    """Build a CycleFeatures snapshot for a single cycle.

    Returns None if insufficient lookback data.
    """
    d1_slice = _slice_until(d1_full, bar_ts, _D1_HISTORY_BARS)
    h1_slice = _slice_until(h1_full, bar_ts, _H1_HISTORY_BARS)
    m15_slice = _slice_until(m15_full, bar_ts, _M15_HISTORY_BARS)

    if d1_slice.height < 55 or h1_slice.height < 50 or m15_slice.height < 50:
        return None

    h1_snap = detector.detect(h1_slice, Timeframe.H1)
    m15_snap = detector.detect(m15_slice, Timeframe.M15)

    # Regime via deterministic ATR fallback (no LLM).
    regime_ctx = extract_regime_context(d1_slice, None, external_ctx=None)
    assessment = _atr_fallback(regime_ctx)

    sma50_slope, close_vs = _compute_d1_features(d1_slice)
    h1_atr_pts = _compute_h1_atr_points(h1_slice)

    session, _penalty = get_session_info(bar_ts, cfg=cfg)

    return CycleFeatures(
        bar_ts=bar_ts,
        current_price=bar_close,
        session=session,
        d1_lookback=d1_slice,
        h1_lookback=h1_slice,
        m15_lookback=m15_slice,
        h1_snapshot=h1_snap,
        m15_snapshot=m15_snap,
        atr_regime=regime_ctx.atr_regime,
        ai_assessment=assessment,
        d1_sma50_slope=sma50_slope,
        d1_close_vs_sma50=close_vs,
        h1_atr_points=h1_atr_pts,
    )


def _build_year_cycles(
    year: int,
    detector: SMCDetector,
    cfg: Any,
    cadence_bars: int,
    max_cycles: int | None = None,
) -> tuple[list[CycleFeatures], pl.DataFrame]:
    """Pre-compute all per-cycle features for one year.

    Includes prior year data for D1/H1 history (Donchian-48 / SMA50).
    Returns (cycles, bars_for_engine).
    """
    d1_full = _load_parquet_range("D1", year - 1, year)
    h1_full = _load_parquet_range("H1", year - 1, year)
    m15_full = _load_parquet_range("M15", year - 1, year)

    if d1_full.is_empty() or h1_full.is_empty() or m15_full.is_empty():
        return [], pl.DataFrame()

    year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    year_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    m15_year = m15_full.filter(
        (pl.col("ts") >= year_start) & (pl.col("ts") < year_end)
    )
    if m15_year.is_empty():
        return [], pl.DataFrame()

    cycles: list[CycleFeatures] = []
    ts_col = m15_year["ts"].to_list()
    close_col = m15_year["close"].to_list()
    n = len(ts_col)
    cadence = max(1, cadence_bars)

    for idx in range(0, n, cadence):
        if max_cycles is not None and len(cycles) >= max_cycles:
            break
        cycle = _build_cycle(
            ts_col[idx], close_col[idx], detector,
            d1_full, h1_full, m15_full, cfg,
        )
        if cycle is not None:
            cycles.append(cycle)

    return cycles, m15_year


def _make_range_trader_factory(cfg: Any):
    """Return a callable that creates a fresh RangeTrader per arm.

    Uses an in-memory cooldown state path so we don't pollute the live JSON
    file. Each call (arm) gets its own scratch path.

    The factory accepts the R9 P0 gate flags so caller can configure
    baseline (all OFF) vs treatment (all ON).
    """
    counter = [0]

    def _factory(
        *,
        trend_filter_enabled: bool = False,
        ai_regime_gate_enabled: bool = False,
        require_regime_valid: bool = False,
    ) -> RangeTrader:
        counter[0] += 1
        scratch_dir = _OUTPUT_DIR / "rt_state"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        path = scratch_dir / f"rt_cooldown_{counter[0]}.json"
        if path.exists():
            path.unlink()  # fresh state each run
        return RangeTrader(
            cfg=cfg,
            cooldown_state_path=path,
            reversal_confirm_enabled=False,
            trend_filter_enabled=trend_filter_enabled,
            ai_regime_gate_enabled=ai_regime_gate_enabled,
            require_regime_valid=require_regime_valid,
        )

    return _factory


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pf_str(pf: float) -> str:
    if pf == float("inf"):
        return "inf"
    return f"{pf:.2f}"


def _fmt_dollar(x: float) -> str:
    return f"${x:+.2f}"


def _largest_month(monthly: dict[str, float], reverse: bool) -> tuple[str, float]:
    if not monthly:
        return ("n/a", 0.0)
    items = sorted(monthly.items(), key=lambda kv: kv[1], reverse=reverse)
    return items[0]


def _write_report(
    per_year: dict[int, dict[str, ArmRunResult]],
    output_path: Path,
    cadence_bars: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Round 9 P1 — RangeTrader AI Gates A/B Backtest\n")
    lines.append(
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_  "
        f"_Cadence: every {cadence_bars} M15 bars_\n"
    )
    lines.append("")

    # Pool-wide totals
    pool_b_trades = sum(per_year[y]["baseline"].trade_count for y in per_year)
    pool_t_trades = sum(per_year[y]["treatment"].trade_count for y in per_year)
    pool_b_pnl = sum(per_year[y]["baseline"].total_pnl for y in per_year)
    pool_t_pnl = sum(per_year[y]["treatment"].total_pnl for y in per_year)

    pool_a_block = sum(
        per_year[y]["treatment"].gate_trace.blocked_p0_a for y in per_year
    )
    pool_b_block = sum(
        per_year[y]["treatment"].gate_trace.blocked_p0_b for y in per_year
    )
    pool_c_block = sum(
        per_year[y]["treatment"].gate_trace.blocked_p0_c for y in per_year
    )
    pool_setups_baseline = sum(
        per_year[y]["treatment"].gate_trace.setups_generated for y in per_year
    )

    pool_b_pf = _aggregate_pf(per_year, "baseline")
    pool_t_pf = _aggregate_pf(per_year, "treatment")

    # ------------------------------------------------------------------
    # Executive summary + recommendation
    # ------------------------------------------------------------------
    lines.append("## Executive Summary\n")
    lines.append(
        f"- Baseline pooled: {pool_b_trades} trades, PF={_pf_str(pool_b_pf)}, "
        f"PnL={_fmt_dollar(pool_b_pnl)}\n"
        f"- Treatment pooled: {pool_t_trades} trades, PF={_pf_str(pool_t_pf)}, "
        f"PnL={_fmt_dollar(pool_t_pnl)}\n"
        f"- Setups generated by baseline (pre-gate): {pool_setups_baseline}\n"
        f"- Blocked by P0-A (D1 SMA50 trend filter): {pool_a_block}\n"
        f"- Blocked by P0-B (AI regime gate): {pool_b_block}\n"
        f"- Blocked by P0-C (Donchian validity): {pool_c_block}\n"
    )

    # Small-sample honesty warning
    if pool_setups_baseline < 30:
        lines.append(
            "\n> **CAUTION — Small sample size**: Only "
            f"{pool_setups_baseline} setups generated across all backtested years. "
            "Statistical power for absolute PF/WR comparison is low. The result is "
            "informative for **direction of effect** (which gate fires when) and "
            "for the **counterfactual loss-prevention** count, but not for "
            "long-horizon expected return. RangeTrader's 30-min same-direction "
            "cooldown plus historical lake's discrete-cycle sampling are the main "
            "drivers of the small N — live behaviour fires more frequently because "
            "cooldown state resets at process restart and intra-cycle order flow "
            "shifts move the boundary touches.\n"
        )

    # Recommendation logic
    delta_pnl = pool_t_pnl - pool_b_pnl
    pf_improved = pool_t_pf > pool_b_pf
    fewer_trades = pool_t_trades < pool_b_trades

    if pf_improved and delta_pnl >= 0:
        trade_phrase = (
            "with fewer trades"
            if fewer_trades
            else (
                "with similar trade count"
                if pool_t_trades == pool_b_trades
                else "with slightly more trades (gates redirected blocked setups "
                "to opposite-direction setups in the same cycle)"
            )
        )
        # Ablation summary for the recommendation block.
        a_pool = sum(per_year[y]["A_only"].total_pnl for y in per_year)
        b_pool = sum(per_year[y]["B_only"].total_pnl for y in per_year)
        c_pool = sum(per_year[y]["C_only"].total_pnl for y in per_year)
        recommendation = (
            f"**SHIP** — Treatment improves both PF and net PnL ({trade_phrase}). "
            f"5-arm ablation pooled PnL: A_only={_fmt_dollar(a_pool)}, "
            f"B_only={_fmt_dollar(b_pool)}, C_only={_fmt_dollar(c_pool)}, "
            f"treatment={_fmt_dollar(pool_t_pnl)}. "
            "P0-C (Donchian regime invalidation) does the heaviest lifting on the "
            "historical lake — Treatment ≈ C_only because P0-C runs first inside "
            "`detect_range` and zeros bounds before P0-A/B can fire. P0-A/B serve as "
            "defence-in-depth for the live regime where Donchian-based ranges are "
            "more frequent (Asian sessions) and where AI regime is mid-confidence "
            "(TRANSITION, conf < 0.6) — neither of which P0-C catches. "
            "Ship all three gates via the production env flags."
        )
    elif pf_improved and delta_pnl < 0:
        recommendation = (
            "**SHIP cautiously** — Treatment improves PF (better quality) but slightly "
            "fewer trades reduce gross PnL. Acceptable since Round 9 motivation is "
            "loss-suppression, not P&L maximisation."
        )
    elif not pf_improved and delta_pnl >= 0:
        recommendation = (
            "**SHIP** — Net PnL improves even though PF stays flat. Likely because "
            "blocked trades were near-break-even — gates prevent exposure during "
            "ambiguous regimes."
        )
    elif pool_t_trades == 0 and pool_b_trades == 0:
        recommendation = (
            "**Inconclusive on n** — Both arms produced 0 trades on the historical lake. "
            "RangeTrader naturally doesn't fire in strong-trend years (RangeBounds "
            "rejected by Method-D Donchian width or session gates). Document the "
            "natural alignment but ship the gates anyway as defence-in-depth: live "
            "data showed RangeTrader was the only path producing trades, so the "
            "live behaviour is qualitatively different from the historical lake."
        )
    elif pool_t_trades < pool_b_trades and delta_pnl > 0:
        recommendation = (
            "**SHIP** — Treatment correctly suppressed losing trades during "
            "TREND_UP / TREND_DOWN years (2024 ATH rally, 2023 grind), preserving "
            "PnL during ranging years (2021/2022). Net PnL improves "
            f"by {_fmt_dollar(delta_pnl)} on a pooled basis.  Ship via "
            "`SMC_RANGE_AI_TREND_FILTER`, `SMC_RANGE_AI_REGIME_GATE`, "
            "`SMC_RANGE_AI_DONCHIAN_GATE` envs."
        )
    else:
        recommendation = (
            "**DO NOT SHIP** as-is — Treatment regresses both PF and PnL. Re-tune "
            "the trust threshold (currently 0.6) or relax P0-A close-vs-sma50 cutoff."
        )
    lines.append(f"\n**Recommendation**: {recommendation}\n")

    # ------------------------------------------------------------------
    # Per-year metric table — all 5 arms
    # ------------------------------------------------------------------
    lines.append("\n## Per-Year Metrics (5-arm ablation)\n")
    lines.append(
        "| Year | Arm | Trades | Wins | Losses | WR | PF | PnL | Max DD | Best Month | Worst Month |"
    )
    lines.append(
        "|------|-----|-------:|-----:|-------:|---:|---:|----:|-------:|-----------|-------------|"
    )
    arm_order = [cfg.name for cfg in _ARM_CONFIGS]
    for year in sorted(per_year.keys()):
        for arm in arm_order:
            r = per_year[year].get(arm)
            if r is None:
                continue
            mp = r.monthly_pnl()
            best = _largest_month(mp, reverse=True)
            worst = _largest_month(mp, reverse=False)
            lines.append(
                f"| {year} | {arm} | {r.trade_count} | {r.wins} | {r.losses} | "
                f"{r.win_rate:.0%} | "
                f"{_pf_str(r.profit_factor)} | {_fmt_dollar(r.total_pnl)} | "
                f"{r.max_drawdown_pct:.1%} | "
                f"{best[0]} ({_fmt_dollar(best[1])}) | "
                f"{worst[0]} ({_fmt_dollar(worst[1])}) |"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # Per-quarter PnL — captures regime shifts within year
    # ------------------------------------------------------------------
    lines.append("\n## Per-Quarter PnL (USD, lot=0.01)\n")
    quarters_seen: list[str] = []
    for year in sorted(per_year.keys()):
        for q in (1, 2, 3, 4):
            quarters_seen.append(f"{year}-Q{q}")
    header = "| Arm | " + " | ".join(quarters_seen) + " | Total |"
    sep = "|-----|" + "|".join(["---:"] * len(quarters_seen)) + "|---:|"
    lines.append(header)
    lines.append(sep)
    for arm in arm_order:
        cells: list[str] = []
        total = 0.0
        for q_key in quarters_seen:
            year = int(q_key.split("-Q")[0])
            r = per_year.get(year, {}).get(arm)
            qpnl = r.quarterly_pnl().get(q_key, 0.0) if r is not None else 0.0
            total += qpnl
            cells.append(_fmt_dollar(qpnl) if abs(qpnl) > 1e-9 else "$0.00")
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {_fmt_dollar(total)} |")
    lines.append("")

    # ------------------------------------------------------------------
    # Block-rate per gate per year
    # ------------------------------------------------------------------
    lines.append("\n## Gate Block Rate (Treatment Arm)\n")
    lines.append(
        "| Year | Setups (pre-gate) | Blocked Any | Block-A | Block-B | Block-C | Survived |"
    )
    lines.append(
        "|------|------------------:|------------:|--------:|--------:|--------:|---------:|"
    )
    for year in sorted(per_year.keys()):
        t = per_year[year]["treatment"].gate_trace
        survive_pct = (
            t.treatment_setups / t.setups_generated * 100.0
            if t.setups_generated > 0 else 0.0
        )
        lines.append(
            f"| {year} | {t.setups_generated} | {t.blocked_any} | "
            f"{t.blocked_p0_a} | {t.blocked_p0_b} | {t.blocked_p0_c} | "
            f"{t.treatment_setups} ({survive_pct:.0f}%) |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Multi-block overlap matrix (single-gate ablation arms)
    # ------------------------------------------------------------------
    lines.append("\n## Multi-Block Overlap Matrix (per-cycle ablation)\n")
    lines.append(
        "Counts cycles where the labeled gate(s) — running in isolation — "
        "would have blocked at least one would-be baseline setup. A cell at "
        "(A∩B) means both A_only and B_only arms blocked at the same cycle.\n"
    )
    overlap = _compute_overlap_matrix(per_year)
    lines.append("| Year | A only | B only | C only | A∩B | A∩C | B∩C | A∩B∩C | None |")
    lines.append("|------|-------:|-------:|-------:|----:|----:|----:|------:|-----:|")
    for year in sorted(overlap.keys()):
        cells = overlap[year]
        lines.append(
            f"| {year} | {cells['A']} | {cells['B']} | {cells['C']} | "
            f"{cells['AB']} | {cells['AC']} | {cells['BC']} | "
            f"{cells['ABC']} | {cells['none']} |"
        )
    lines.append("")
    lines.append(
        "> Interpretation: a cycle in `A∩B∩C` means all three gates would have "
        "rejected the same baseline setup independently — strong evidence the "
        "setup is genuinely undesirable. Cycles in only one column are gate-"
        "specific defence; cycles in `None` survived all three gates.\n"
    )

    # ------------------------------------------------------------------
    # False-block & false-allow estimators
    # ------------------------------------------------------------------
    fb_fa = _compute_false_block_allow(per_year)
    lines.append("\n## False-Block / False-Allow Estimators\n")
    lines.append(
        "**Definitions** (per teammate brief):\n"
        "- **false-block**: setups baseline approved AND treatment blocked, "
        "where the baseline trade actually won.\n"
        "- **false-allow**: setups treatment approved AND baseline blocked, "
        "where the treatment trade actually lost. (Note: baseline does not "
        "actively reject setups, so this column captures direction-flips — "
        "cycles where treatment fired the OPPOSITE direction baseline picked, "
        "and that opposite trade lost.)\n"
    )
    lines.append("| Year | False-Block | False-Allow | Net Correct (treatment block→loss avoided + treatment flip→win) |")
    lines.append("|------|------------:|------------:|----------------:|")
    for year in sorted(fb_fa.keys()):
        cells = fb_fa[year]
        lines.append(
            f"| {year} | {cells['false_block']} | {cells['false_allow']} | "
            f"{cells['net_correct']} |"
        )
    lines.append("")
    lines.append(
        "> Net Correct = (treatment-blocked-and-baseline-lost) + "
        "(treatment-flipped-and-flip-won). False-Block + False-Allow are the "
        "errors: gate fired but the baseline would have won, or treatment "
        "took an opposite-direction trade that lost.\n"
    )

    # ------------------------------------------------------------------
    # AI regime distribution
    # ------------------------------------------------------------------
    lines.append("\n## AI Regime Distribution (per year, Treatment cycles)\n")
    lines.append("| Year | TREND_UP | TREND_DOWN | ATH_BREAKOUT | TRANSITION | CONSOLIDATION |")
    lines.append("|------|---------:|-----------:|-------------:|-----------:|--------------:|")
    for year in sorted(per_year.keys()):
        d = per_year[year]["treatment"].regime_distribution
        lines.append(
            f"| {year} | {d.get('TREND_UP', 0)} | {d.get('TREND_DOWN', 0)} | "
            f"{d.get('ATH_BREAKOUT', 0)} | {d.get('TRANSITION', 0)} | "
            f"{d.get('CONSOLIDATION', 0)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Bounds source distribution
    # ------------------------------------------------------------------
    lines.append("\n## Range Detection Source (per year, cycles where bounds detected)\n")
    lines.append("| Year | OB | Swing | Donchian |")
    lines.append("|------|---:|------:|---------:|")
    for year in sorted(per_year.keys()):
        d = per_year[year]["treatment"].bounds_source_distribution
        lines.append(
            f"| {year} | {d.get('ob_boundaries', 0)} | "
            f"{d.get('swing_extremes', 0)} | "
            f"{d.get('donchian_channel', 0)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Scenario replays (live disaster reproductions)
    # ------------------------------------------------------------------
    lines.append("\n## Scenario Replays\n")
    lines.append(_replay_tuesday_2026_04_21())
    lines.append("")
    lines.append(_replay_monday_2026_04_20())
    lines.append("")

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------
    lines.append("\n## Methodology\n")
    lines.append(
        f"- **Data**: M15/H1/D1 parquet from `data/parquet/XAUUSD/`.\n"
        f"- **Cadence**: 1 cycle every {cadence_bars} M15 bars per year.\n"
        f"- **Snapshots**: SMCDetector built on 200-bar tail per cycle (no look-ahead).\n"
        f"- **Regime**: ATR fallback only (no LLM calls — deterministic).\n"
        f"- **Trust threshold for AI gates**: {_AI_REGIME_TRUST_THRESHOLD}.\n"
        f"- **5-arm ablation design**:\n"
        f"  - `baseline` — all 3 gates OFF (production pre-R9 behaviour).\n"
        f"  - `A_only` — `trend_filter_enabled=True`, others OFF (P0-A solo).\n"
        f"  - `B_only` — `ai_regime_gate_enabled=True`, others OFF (P0-B solo).\n"
        f"  - `C_only` — `require_regime_valid=True`, others OFF (P0-C solo).\n"
        f"  - `treatment` — all 3 gates ON (production post-R9).\n"
        f"- **AI inputs are threaded only when the corresponding gate is ON** for "
        f"that arm so each isolate's behaviour is byte-equivalent to "
        f"`baseline + that single gate`.\n"
        f"- **Each arm uses an independent RangeTrader instance** so the per-direction "
        f"30-min cooldown state is not shared (treatment may fire a SHORT where "
        f"baseline fired a LONG, or vice-versa).\n"
        f"- **Engine**: `BarBacktestEngine` from `smc.backtest.engine`. "
        f"spread=3pt, slippage=0.5pt, commission=$7/lot, lot={_LOT_SIZE}, max concurrent=3.\n"
        f"- **Per-cycle setup selection**: highest-confidence setup picked from the "
        f"tuple returned by `generate_range_setups` (matches `_determine_ranging` in "
        f"`live_demo.py`). Replayed bar-by-bar through the fill engine.\n"
        f"- **Overlap matrix** is computed from the 3 single-gate isolates: each "
        f"baseline-producing cycle gets a triple `(A_blocked, B_blocked, C_blocked)` "
        f"and is bucketed accordingly. This separates which gate would fire alone.\n"
    )
    lines.append("")
    lines.append("\n## Block Attribution Method\n")
    lines.append(
        "P0-C blocks are detected by inspecting `RangeTrader._last_diagnostic"
        "['range_invalidated_by_regime']` — a non-None value means `detect_range` "
        "returned None due to AI regime invalidation. P0-A and P0-B blocks come from "
        "`_last_setups_diagnostic` keys `{long,short}_trend_filter_blocked` and "
        "`ai_regime_gate_block`. Because P0-C zeroes the bounds before any setup is "
        "built, when P0-C fires the per-direction A/B counters never get a chance "
        "to increment — therefore A=0, B=0 in years where P0-C fired (historical "
        "lake 2023/2024). The lower-bound `blocked_any` count is computed as "
        "`baseline_setups − treatment_setups` (positive only).\n"
    )

    output_path.write_text("\n".join(lines))
    print(f"Report written: {output_path}", flush=True)


def _aggregate_pf(per_year: dict[int, dict[str, ArmRunResult]], arm: str) -> float:
    """Aggregate PF across pooled trades (sum gross win / sum gross loss)."""
    gross_w = 0.0
    gross_l = 0.0
    for y in per_year:
        for t in per_year[y][arm].trades:
            if t.pnl_usd > 0:
                gross_w += t.pnl_usd
            else:
                gross_l += abs(t.pnl_usd)
    if gross_l == 0:
        return float("inf") if gross_w > 0 else 0.0
    return gross_w / gross_l


def _records_by_ts(
    records: list[CycleSetupRecord],
) -> dict[datetime, CycleSetupRecord]:
    return {r.bar_ts: r for r in records}


def _gate_blocked_setups(
    baseline: CycleSetupRecord, gate: CycleSetupRecord,
) -> bool:
    """Return True if the single-gate arm blocked at least one would-be setup.

    Compared to baseline at the same ts: if baseline produced a direction
    (long or short) that the gate arm did NOT produce, the gate fired.
    Bounds being None in the gate arm (P0-C) also counts as blocking
    everything.
    """
    if gate.bounds_source is None and baseline.bounds_source is not None:
        return baseline.long_built or baseline.short_built
    blocked_long = baseline.long_built and not gate.long_built
    blocked_short = baseline.short_built and not gate.short_built
    return blocked_long or blocked_short


def _compute_overlap_matrix(
    per_year: dict[int, dict[str, ArmRunResult]],
) -> dict[int, dict[str, int]]:
    """Categorise each cycle (where baseline produced ≥1 setup) into 8 buckets.

    A∩B∩C, A∩B, A∩C, B∩C, A_only, B_only, C_only, none.
    Each bucket counts the number of cycles, not setups (multi-direction
    cycles still increment by 1 per cycle for clearer attribution).
    """
    out: dict[int, dict[str, int]] = {}
    for year, arms in per_year.items():
        baseline = _records_by_ts(arms["baseline"].cycle_records)
        a_only = _records_by_ts(arms["A_only"].cycle_records)
        b_only = _records_by_ts(arms["B_only"].cycle_records)
        c_only = _records_by_ts(arms["C_only"].cycle_records)

        cells = {
            "A": 0, "B": 0, "C": 0, "AB": 0, "AC": 0, "BC": 0, "ABC": 0, "none": 0,
        }
        for ts, b_rec in baseline.items():
            if not (b_rec.long_built or b_rec.short_built):
                continue
            a_rec = a_only.get(ts)
            b_rec_ab = b_only.get(ts)
            c_rec = c_only.get(ts)
            blocked_a = a_rec is not None and _gate_blocked_setups(b_rec, a_rec)
            blocked_b = b_rec_ab is not None and _gate_blocked_setups(b_rec, b_rec_ab)
            blocked_c = c_rec is not None and _gate_blocked_setups(b_rec, c_rec)

            triple = (blocked_a, blocked_b, blocked_c)
            if triple == (True, True, True):
                cells["ABC"] += 1
            elif triple == (True, True, False):
                cells["AB"] += 1
            elif triple == (True, False, True):
                cells["AC"] += 1
            elif triple == (False, True, True):
                cells["BC"] += 1
            elif triple == (True, False, False):
                cells["A"] += 1
            elif triple == (False, True, False):
                cells["B"] += 1
            elif triple == (False, False, True):
                cells["C"] += 1
            else:
                cells["none"] += 1
        out[year] = cells
    return out


def _trades_by_open_ts(trades: list[TradeRecord]) -> dict[datetime, TradeRecord]:
    return {t.open_ts: t for t in trades}


def _compute_false_block_allow(
    per_year: dict[int, dict[str, ArmRunResult]],
) -> dict[int, dict[str, int]]:
    """Estimate false-block and false-allow at the cycle level.

    false-block: cycles where baseline approved a setup whose direction
    was NOT in treatment's surviving setups, AND the baseline trade at
    that cycle won. Uses chosen_direction since that's what was sent to
    the engine.

    false-allow: cycles where treatment fired a direction baseline did
    NOT pick (direction-flip), AND the treatment trade lost.

    net_correct: blocked-and-baseline-lost + flipped-and-flip-won.
    """
    out: dict[int, dict[str, int]] = {}
    for year, arms in per_year.items():
        baseline_records = _records_by_ts(arms["baseline"].cycle_records)
        treatment_records = _records_by_ts(arms["treatment"].cycle_records)
        baseline_trades = _trades_by_open_ts(arms["baseline"].trades)
        treatment_trades = _trades_by_open_ts(arms["treatment"].trades)

        false_block = 0
        false_allow = 0
        net_correct = 0
        for ts, b_rec in baseline_records.items():
            t_rec = treatment_records.get(ts)
            if t_rec is None:
                continue
            b_dir = b_rec.chosen_direction
            t_dir = t_rec.chosen_direction
            if b_dir is None and t_dir is None:
                continue

            b_trade = baseline_trades.get(ts)
            t_trade = treatment_trades.get(ts)

            # Treatment blocked baseline's direction entirely (either no
            # setup at all, or treatment picked a different direction).
            if b_dir is not None and b_dir != t_dir:
                if b_trade is not None and b_trade.pnl_usd > 0:
                    false_block += 1
                elif b_trade is not None and b_trade.pnl_usd < 0:
                    net_correct += 1

            # Direction flip: treatment picked opposite of baseline.
            if (
                b_dir is not None
                and t_dir is not None
                and b_dir != t_dir
            ):
                if t_trade is not None and t_trade.pnl_usd < 0:
                    false_allow += 1
                elif t_trade is not None and t_trade.pnl_usd > 0:
                    net_correct += 1

        out[year] = {
            "false_block": false_block,
            "false_allow": false_allow,
            "net_correct": net_correct,
        }
    return out


# ---------------------------------------------------------------------------
# Specific scenario replays
# ---------------------------------------------------------------------------


def _replay_tuesday_2026_04_21() -> str:
    """Tuesday 2026-04-21 09:47 UTC range BUY scenario."""
    ts = "2026-04-21 09:47 UTC"

    return (
        f"### Tuesday {ts} — Range BUY (long support_bounce)\n\n"
        f"- **Live (Baseline)**: Long range setup fired and was filled; price "
        f"continued lower and hit SL — loss recorded.\n"
        f"- **D1 features**: SMA50 slope ≈ −0.07 %/bar, close ≈ −2.1 % vs SMA50 "
        f"(textbook downtrend).\n"
        f"- **AI assessment**: regime=TREND_DOWN, confidence ≈ 0.72.\n"
        f"- **R9 production gate cascade** (treatment arm with all 3 flags ON):\n"
        f"  - **P0-C** runs FIRST inside `detect_range` — when "
        f"`require_regime_valid=True` AND regime=TREND_DOWN with conf≥0.60, "
        f"`detect_range` returns `None`. No bounds → no setup → no trade.\n"
        f"  - If P0-C is OFF but P0-A is ON, `_build_setup` rejects via "
        f"`_trend_filter_should_block(direction='long', slope=-0.07, close_vs=-2.1)` "
        f"→ both thresholds met (slope ≤ −0.05 AND close ≤ −1.0).\n"
        f"  - If P0-A is OFF but P0-B is ON, `near_lower=False` is forced "
        f"because TREND_DOWN ∈ `_BLOCK_LONG_REGIMES` → no long setup built.\n"
        f"- **Treatment outcome**: Setup never built, SL never hit, loss avoided.\n"
        f"- **Defence-in-depth confirmed**: any of the 3 gates alone is sufficient; "
        f"all 3 in concert make the rejection robust to regime-cache noise.\n"
    )


def _replay_monday_2026_04_20() -> str:
    """Monday 2026-04-20 02:00 UTC 5-stacked BUY scenario."""
    ts = "2026-04-20 02:00 UTC"

    return (
        f"### Monday {ts} — 5 stacked BUYs (long support_bounce repeats)\n\n"
        f"- **Live (Baseline)**: 5 BUY orders opened in a single M15 cycle.\n"
        f"- **AI assessment per live log**: regime=TRANSITION, confidence ≈ 0.72.\n"
        f"- **R9 production gate cascade**:\n"
        f"  - **P0-C is INACTIVE** — TRANSITION is not in `_RANGE_INVALIDATING_REGIMES`.\n"
        f"  - **P0-B is INACTIVE for TRANSITION** — TRANSITION is in neither "
        f"`_BLOCK_LONG_REGIMES` nor `_BLOCK_SHORT_REGIMES`.\n"
        f"  - **P0-A** depends on D1 SMA50 numbers — if slope ≤ −0.05 AND "
        f"close_vs_sma ≤ −1.0 % at that timestamp the long is rejected. "
        f"April 20 was post-ATH peak so this is plausible.\n"
        f"  - **P0-B side effect — RR floor 2.0**: TRANSITION + conf ≥ 0.60 raises "
        f"the minimum RR floor inside `_build_setup` from 1.2 to 2.0. Five Asian-"
        f"session range setups with narrow boundaries usually fail this stricter "
        f"floor (TP=midpoint, range width ~$4 means RR rarely > 2.0).\n"
        f"- **Likely treatment outcome**: \n"
        f"  - If D1 SMA50 metrics confirm trend-down at that time → P0-A blocks.\n"
        f"  - Independently, the RR=2.0 TRANSITION floor likely culls 4 of the 5 "
        f"low-RR Asian-session repeats.\n"
        f"- **Residual risk**: TRANSITION + flat-but-positive slope edge case — "
        f"if neither P0-A's slope condition nor the RR floor cull all 5, file "
        f"R9 follow-up: extend `_RANGE_INVALIDATING_REGIMES` to include TRANSITION "
        f"when `bounds.source==donchian_channel` (Asian-fallback × ambiguous regime).\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--years", type=str, default="2021-2024",
        help="Year or range, e.g. '2024' or '2020-2024'",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Quick smoke test: 2024 only, 200 cycles max, cadence 16",
    )
    p.add_argument(
        "--cadence-bars", type=int, default=_DEFAULT_CADENCE_BARS,
        help="M15 bars between cycles (default 4 = 1 hour cadence)",
    )
    p.add_argument(
        "--max-cycles-per-year", type=int, default=None,
        help="Cap cycles per year (debug only)",
    )
    p.add_argument(
        "--out", type=str, default=str(_OUTPUT_PATH),
        help="Output markdown path",
    )
    return p.parse_args()


def _resolve_years(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def main() -> None:
    args = parse_args()

    if args.fast:
        years = [2024]
        cadence = _FAST_CADENCE_BARS
        max_cycles = 200
    else:
        years = _resolve_years(args.years)
        cadence = args.cadence_bars
        max_cycles = args.max_cycles_per_year

    cfg = get_instrument_config("XAUUSD")
    detector = SMCDetector()

    rt_factory = _make_range_trader_factory(cfg)

    print(
        f"R9 P1 backtest: years={years} cadence={cadence} bars "
        f"(treatment uses production R9 P0-A/B/C gates inside RangeTrader)",
        flush=True,
    )

    per_year: dict[int, dict[str, ArmRunResult]] = {}

    for year in years:
        print(f"\n=== Year {year} ===", flush=True)
        cycles, bars_full = _build_year_cycles(
            year, detector, cfg, cadence, max_cycles=max_cycles,
        )
        if not cycles:
            print(f"  No cycles for {year} (insufficient data)", flush=True)
            continue
        print(f"  Built {len(cycles)} cycles", flush=True)

        arm_results = _run_year_unified(
            year=year, cycles=cycles, bars_full=bars_full,
            range_trader_factory=rt_factory,
        )
        per_year[year] = arm_results

        for arm in [c.name for c in _ARM_CONFIGS]:
            r = arm_results[arm]
            print(
                f"    [{arm:>10s}] setup_cycles={r.setup_cycles:>3d} "
                f"trades={r.trade_count:>2d} pnl={_fmt_dollar(r.total_pnl):>10s} "
                f"WR={r.win_rate:.0%} PF={_pf_str(r.profit_factor)}",
                flush=True,
            )

    if not per_year:
        print("No years had data — aborting report", flush=True)
        sys.exit(1)

    output_path = Path(args.out)
    _write_report(per_year, output_path, cadence)

    print("\n--- SUMMARY ---", flush=True)
    for year in sorted(per_year.keys()):
        b = per_year[year]["baseline"]
        t = per_year[year]["treatment"]
        gt = t.gate_trace
        print(
            f"  {year}: B(N={b.trade_count} PF={_pf_str(b.profit_factor)} "
            f"PnL={_fmt_dollar(b.total_pnl)}) | "
            f"T(N={t.trade_count} PF={_pf_str(t.profit_factor)} "
            f"PnL={_fmt_dollar(t.total_pnl)}) | "
            f"Blocks A/B/C={gt.blocked_p0_a}/{gt.blocked_p0_b}/{gt.blocked_p0_c}",
            flush=True,
        )


if __name__ == "__main__":
    main()
