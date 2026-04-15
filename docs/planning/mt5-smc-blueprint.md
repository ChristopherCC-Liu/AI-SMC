---

# Implementation Plan: MT5 Smart Money Concepts (SMC) Trading System

## Overview

Build an XAUUSD-only Smart Money Concepts trading system within the existing alphalens-v2 repository. The system detects institutional footprints (Order Blocks, Fair Value Gaps, Break of Structure, Change of Character, liquidity sweeps) across D1/H4/H1/M15 timeframes, generates rules-based trade signals, executes backtests with realistic forex spread/slippage modeling, and paper-trades on MT5 Demo. AI debate integration follows only after rules prove profitable.

## Requirements

- **Single instrument**: XAUUSD (gold vs USD) only. No multi-instrument complexity.
- **Multi-timeframe HTF-to-LTF cascade**: D1/H4 for directional bias, H1 for zone identification (OB/FVG), M15 for precise entry timing.
- **Rules-only MVP first**: No AI/ML until rules demonstrate positive expectancy in walk-forward OOS.
- **Dual runtime**: Backtest (historical bar data) and Demo paper trading (live MT5 terminal) run in parallel.
- **macOS development**: MT5 Python SDK unavailable on macOS; use a mock adapter for local development. Windows VPS for real MT5 deployment.
- **Immutable data pipeline**: Follow existing alphalens-v2 patterns -- frozen dataclasses, Polars primary, Pydantic v2 validation, Parquet data lake.
- **Reuse existing infrastructure**: SourceAdapter Protocol, manifest system, IC research, debate models, CLI framework, retry helpers, test patterns.

## Project Structure

The MT5-SMC system lives as a new top-level package `smc` alongside the existing `alphalens` package, sharing the same repo and build system. This avoids polluting the alphalens A-share namespace while reusing its infrastructure via imports.

```
~/claudeworkplace/alphalens-v2/
  pyproject.toml                  # add [project.scripts] smc = "smc.cli.main:app"
  src/
    alphalens/                    # existing -- unchanged
    smc/                          # NEW: Smart Money Concepts package
      __init__.py
      _version.py                 # "0.1.0"
      
      # --- Data Layer ---
      data/
        __init__.py
        schemas.py                # OHLCV_SCHEMA for forex bars (ts, open, high, low, close, volume, spread, timeframe, source)
        adapters/
          __init__.py
          base.py                 # ForexAdapter Protocol (extends SourceAdapter for multi-TF)
          mt5_adapter.py          # Real MT5 terminal adapter (Windows-only)
          mt5_mock.py             # macOS mock: reads Parquet fixtures, simulates MT5 API
          csv_adapter.py          # Bootstrap: load historical CSV from MetaTrader export
        writers.py                # write_ohlcv_partitioned() -- partition by timeframe/year/month
        manifest.py               # Thin wrapper around alphalens manifest for forex sources
        lake.py                   # ForexDataLake: query interface over partitioned Parquet
      
      # --- SMC Detection ---
      smc_core/
        __init__.py
        types.py                  # Frozen dataclasses: SwingPoint, OrderBlock, FVG, BOS, CHoCH, LiquidityLevel
        swing.py                  # Swing high/low detection (wraps smartmoneyconcepts lib)
        order_block.py            # OB detection + validation (mitigated/unmitigated tracking)
        fvg.py                    # Fair Value Gap detection + fill tracking
        structure.py              # BOS/CHoCH detection (trend direction state machine)
        liquidity.py              # Equal highs/lows, trendline liquidity, sweep detection
        detector.py               # SMCDetector: orchestrates all sub-detectors on a single TF
      
      # --- Multi-Timeframe Strategy ---
      strategy/
        __init__.py
        types.py                  # TradeSetup, EntrySignal, BiasDirection, SetupGrade
        htf_bias.py               # D1/H4 directional bias (BOS/CHoCH trend + OB context)
        zone_scanner.py           # H1 OB/FVG zone identification within HTF bias
        entry_trigger.py          # M15 entry triggers (CHoCH into zone, FVG fill, OB test)
        confluence.py             # Multi-factor confluence scorer
        aggregator.py             # MultiTimeframeAggregator: HTF bias + zone + trigger -> TradeSetup
      
      # --- Backtest Engine ---
      backtest/
        __init__.py
        types.py                  # BacktestConfig, TradeRecord, EquityCurve, BacktestResult
        engine.py                 # BarBacktestEngine: event-driven bar-by-bar simulation
        fills.py                  # Fill model: spread, slippage, partial fills
        metrics.py                # Performance metrics: Sharpe, Sortino, Calmar, profit factor, etc.
        walk_forward.py           # Walk-forward OOS validation (rolling train/test windows)
      
      # --- Risk Management ---
      risk/
        __init__.py
        types.py                  # RiskConfig, PositionSize, RiskBudget
        position_sizer.py         # ATR-based lot sizing, fixed-fractional, Kelly criterion
        drawdown_guard.py         # Max drawdown circuit breaker, daily loss limit
        exposure.py               # Leverage check, margin requirement validation
      
      # --- MT5 Execution ---
      execution/
        __init__.py
        types.py                  # OrderRequest, OrderResult, PositionState
        executor.py               # MT5Executor Protocol + real implementation
        executor_mock.py          # Paper executor for macOS / backtest
        order_manager.py          # Open/modify/close lifecycle, SL/TP management
        reconciler.py             # Position reconciliation (local state vs MT5 terminal)
      
      # --- Monitoring & Reporting ---
      monitor/
        __init__.py
        types.py                  # HealthStatus, TradeAlert, DailyReport
        health.py                 # System health checks (MT5 connection, data freshness)
        alerter.py                # Trade alerts (telegram/log based)
        journal.py                # Trade journal: append-only Parquet log
      
      # --- CLI ---
      cli/
        __init__.py
        main.py                   # Typer app: smc backtest, smc live, smc zones, smc health
        panels.py                 # Rich panels for dashboard display
      
      # --- AI Debate (Phase 3) ---
      debate/
        __init__.py
        smc_analysts.py           # SMC-specific analyst prompts (structure, zones, liquidity)
        adapter.py                # Bridge: alphalens debate models -> SMC trade proposals

  tests/
    smc/                          # NEW: all SMC tests
      __init__.py
      conftest.py                 # Shared fixtures: sample OHLCV bars, known SMC patterns
      unit/
        __init__.py
        data/
          __init__.py
          test_schemas.py
          test_mt5_adapter.py
          test_mt5_mock.py
          test_csv_adapter.py
          test_lake.py
        smc_core/
          __init__.py
          test_swing.py
          test_order_block.py
          test_fvg.py
          test_structure.py
          test_liquidity.py
          test_detector.py
        strategy/
          __init__.py
          test_htf_bias.py
          test_zone_scanner.py
          test_entry_trigger.py
          test_confluence.py
          test_aggregator.py
        backtest/
          __init__.py
          test_engine.py
          test_fills.py
          test_metrics.py
          test_walk_forward.py
        risk/
          __init__.py
          test_position_sizer.py
          test_drawdown_guard.py
          test_exposure.py
        execution/
          __init__.py
          test_executor.py
          test_order_manager.py
          test_reconciler.py
      integration/
        __init__.py
        test_full_pipeline.py     # Data -> detect -> strategy -> backtest end-to-end
        test_mt5_connection.py    # Windows-only: real MT5 terminal smoke test
      fixtures/
        xauusd_d1_2023.parquet    # Known gold data with labeled SMC patterns
        xauusd_h4_2023.parquet
        xauusd_h1_2023.parquet
        xauusd_m15_2023.parquet
        known_obs.json            # Hand-labeled order blocks for validation
        known_fvgs.json           # Hand-labeled FVGs for validation
```

## Architecture Changes to pyproject.toml

Add to `/Users/christopher/claudeworkplace/alphalens-v2/pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/alphalens", "src/smc"]

[project.scripts]
alphalens = "alphalens.cli.main:app"
smc = "smc.cli.main:app"

# New dependencies to add:
# MetaTrader5>=5.0.45  (Windows-only, optional)
# smartmoneyconcepts>=0.0.9
```

New optional dependency group:

```toml
[project.optional-dependencies]
mt5 = ["MetaTrader5>=5.0.45"]
smc = ["smartmoneyconcepts>=0.0.9"]
```

## Implementation Steps

### Phase 0: Foundation (Data + SMC Detection + Basic CLI)

**Goal**: Ingest XAUUSD multi-TF OHLCV data, detect all SMC patterns, store in data lake, display via CLI.

#### Step 0.1: Forex Data Schema

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/schemas.py`

- Action: Define `FOREX_OHLCV_SCHEMA` as a PyArrow schema. Columns: `ts` (timestamp ns, UTC), `open`, `high`, `low`, `close`, `volume` (all float64), `spread` (float64, in points), `timeframe` (string, e.g. "D1", "H4", "H1", "M15"), `source` (string), `schema_version` (int32). Add `Timeframe` StrEnum with values D1, H4, H1, M15. Add `validate_forex_frame()` function that checks schema conformance and ensures no future timestamps.
- Why: The existing alphalens schema is equity-focused (Asia/Shanghai TZ, no spread field, no timeframe partitioning). Forex needs UTC timestamps, spread tracking, and multi-timeframe partitioning.
- Dependencies: None
- Risk: Low

#### Step 0.2: ForexAdapter Protocol

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/adapters/base.py`

- Action: Define `ForexAdapterSpec` frozen dataclass (source: str, instrument: str, timeframes: tuple[Timeframe, ...], description: str). Define `ForexAdapter` Protocol with method `fetch(*, instrument: str, timeframe: Timeframe, start: datetime, end: datetime) -> pl.DataFrame`. Define `ForexAdapterError(RuntimeError)`. The protocol deliberately mirrors the existing `SourceAdapter` but adds `timeframe` and `instrument` parameters.
- Why: MT5 data is per-instrument per-timeframe, unlike the equity adapters which are per-source.
- Dependencies: Step 0.1
- Risk: Low

#### Step 0.3: CSV Bootstrap Adapter

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/adapters/csv_adapter.py`

- Action: Implement `CSVAdapter(ForexAdapter)` that reads MetaTrader-exported CSV files. Handle the standard MT5 CSV format (Date, Time, Open, High, Low, Close, Tickvol, Vol, Spread). Parse timestamps to UTC. Convert spread from points to pips (divide by 10 for XAUUSD 5-digit pricing). Support batch loading from a directory of CSVs organized by timeframe.
- Why: Before MT5 SDK integration, CSV export is the simplest way to bootstrap historical data. Every MT5 user can export CSV from the terminal.
- Dependencies: Steps 0.1, 0.2
- Risk: Low

#### Step 0.4: MT5 Mock Adapter

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/adapters/mt5_mock.py`

- Action: Implement `MT5MockAdapter(ForexAdapter)` that reads from pre-built Parquet fixtures to simulate the MT5 API responses. Support `copy_rates_range()` equivalent (returns OHLCV bars for a date range). Include a `MockMT5Terminal` class that simulates `initialize()`, `shutdown()`, `copy_rates_range()`, `symbol_info_tick()`, `order_send()`, `positions_get()` -- the minimal MT5 API surface. Controlled by environment variable `SMC_MT5_MOCK=1`.
- Why: MT5 Python SDK only works on Windows. macOS development requires a faithful mock that returns realistic data shapes. The mock also enables deterministic testing.
- Dependencies: Steps 0.1, 0.2
- Risk: Medium -- must accurately model MT5 API return types (numpy structured arrays)

#### Step 0.5: Real MT5 Adapter

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/adapters/mt5_adapter.py`

- Action: Implement `MT5Adapter(ForexAdapter)` wrapping the real MetaTrader5 Python SDK. Method `fetch()` calls `mt5.copy_rates_range(symbol, timeframe, start, end)`. Map `Timeframe` enum to MT5 timeframe constants (mt5.TIMEFRAME_D1, etc.). Handle MT5 initialization/shutdown lifecycle via context manager. Use `fetch_with_retry` from alphalens retry helper for transient connection failures. Conditional import: `try: import MetaTrader5 as mt5 except ImportError: raise ImportError("MT5 SDK requires Windows")`.
- Why: Production data ingestion from MT5 terminal on Windows VPS.
- Dependencies: Steps 0.1, 0.2, reuses `alphalens.data.adapters._retry.fetch_with_retry`
- Risk: Medium -- Windows-only, cannot test on macOS

#### Step 0.6: Forex Parquet Writer

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/writers.py`

- Action: Implement `write_forex_partitioned(df, *, instrument: str, timeframe: Timeframe, root: Path) -> list[Path]`. Partitions to `root/{instrument}/{timeframe}/{yyyy}/{mm}.parquet`. Validates schema before writing. Reuses the pattern from `alphalens.data.writers.write_partitioned` but adapted for the forex schema.
- Why: Multi-timeframe data needs an extra partition level (timeframe) beyond the equity year/month scheme.
- Dependencies: Step 0.1
- Risk: Low

#### Step 0.7: Forex Data Lake Query Interface

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/data/lake.py`

- Action: Implement `ForexDataLake` class with constructor `__init__(self, root: Path)`. Methods: `query(instrument: str, timeframe: Timeframe, start: datetime, end: datetime) -> pl.DataFrame` (reads partitioned Parquet via DuckDB glob), `available_range(instrument: str, timeframe: Timeframe) -> tuple[datetime, datetime]` (min/max ts), `row_count(instrument: str, timeframe: Timeframe) -> int`. The lake is read-only; writes go through `writers.py`.
- Why: Strategy and backtest modules need a clean query interface over the data lake without knowing about partition layout.
- Dependencies: Steps 0.1, 0.6
- Risk: Low

#### Step 0.8: SMC Type Definitions

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/types.py`

- Action: Define all SMC pattern types as frozen Pydantic BaseModel (not dataclass, because we need validation):

  `SwingPoint(frozen=True)`: ts: datetime, price: float, swing_type: Literal["high", "low"], strength: int (number of bars on each side confirming the swing).

  `OrderBlock(frozen=True)`: ts_start: datetime, ts_end: datetime, high: float, low: float, ob_type: Literal["bullish", "bearish"], timeframe: Timeframe, mitigated: bool, mitigated_at: datetime | None.

  `FairValueGap(frozen=True)`: ts: datetime, high: float, low: float, fvg_type: Literal["bullish", "bearish"], timeframe: Timeframe, filled_pct: float (0.0 to 1.0), fully_filled: bool.

  `StructureBreak(frozen=True)`: ts: datetime, price: float, break_type: Literal["bos", "choch"], direction: Literal["bullish", "bearish"], timeframe: Timeframe.

  `LiquidityLevel(frozen=True)`: price: float, level_type: Literal["equal_highs", "equal_lows", "trendline"], touches: int, swept: bool, swept_at: datetime | None.

  `SMCSnapshot(frozen=True)`: ts: datetime, timeframe: Timeframe, swing_points: tuple[SwingPoint, ...], order_blocks: tuple[OrderBlock, ...], fvgs: tuple[FairValueGap, ...], structure_breaks: tuple[StructureBreak, ...], liquidity_levels: tuple[LiquidityLevel, ...], trend_direction: Literal["bullish", "bearish", "ranging"].

- Why: Immutable types ensure no hidden mutation as data flows through the detection-to-strategy pipeline. Pydantic v2 gives us validation + serialization for free.
- Dependencies: Step 0.1 (for Timeframe enum)
- Risk: Low

#### Step 0.9: Swing Point Detection

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/swing.py`

- Action: Implement `detect_swings(df: pl.DataFrame, *, swing_length: int = 10) -> tuple[SwingPoint, ...]`. Wraps `smartmoneyconcepts.smc.swing_highs_lows()` from the joshyattridge library. The wrapper converts Polars to Pandas for the lib call, extracts swing points, converts back to frozen SwingPoint tuples. Add `filter_significant_swings(swings: tuple[SwingPoint, ...], *, min_distance_pips: float) -> tuple[SwingPoint, ...]` to remove noise swings too close together.
- Why: Swing highs/lows are the foundation for all other SMC patterns (structure, OBs, liquidity).
- Dependencies: Step 0.8, smartmoneyconcepts library
- Risk: Low -- well-tested upstream library

#### Step 0.10: Order Block Detection

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/order_block.py`

- Action: Implement `detect_order_blocks(df: pl.DataFrame, *, swing_length: int = 10) -> tuple[OrderBlock, ...]`. Wraps `smc.ob()` from the library. A bullish OB is the last bearish candle before a bullish BOS; bearish OB is the last bullish candle before a bearish BOS. Add `update_mitigation(obs: tuple[OrderBlock, ...], current_bars: pl.DataFrame) -> tuple[OrderBlock, ...]` that returns new tuples with `mitigated=True` and `mitigated_at` set when price trades through the OB zone. Only unmitigated OBs are tradeable.
- Why: Order blocks are the primary trade zones in SMC -- they represent institutional supply/demand.
- Dependencies: Steps 0.8, 0.9
- Risk: Low

#### Step 0.11: Fair Value Gap Detection

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/fvg.py`

- Action: Implement `detect_fvgs(df: pl.DataFrame, *, join_consecutive: bool = True) -> tuple[FairValueGap, ...]`. Wraps `smc.fvg()`. Bullish FVG: bar[i-1].high < bar[i+1].low (gap up not filled by bar i). Add `update_fill_status(fvgs: tuple[FairValueGap, ...], current_bars: pl.DataFrame) -> tuple[FairValueGap, ...]` tracking how much of each FVG has been filled by subsequent price action. FVGs are valid entry zones until fully filled.
- Why: FVGs represent imbalance in price delivery -- SMC traders enter at FVG fills.
- Dependencies: Step 0.8
- Risk: Low

#### Step 0.12: Structure Break Detection (BOS/CHoCH)

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/structure.py`

- Action: Implement `detect_structure(df: pl.DataFrame, *, swing_length: int = 10) -> tuple[StructureBreak, ...]`. Wraps `smc.bos_choch()`. BOS = Break of Structure (trend continuation: higher high in uptrend, lower low in downtrend). CHoCH = Change of Character (trend reversal: lower low in uptrend, higher high in downtrend). Add `current_trend(breaks: tuple[StructureBreak, ...]) -> Literal["bullish", "bearish", "ranging"]` that derives the active trend direction from the most recent breaks.
- Why: BOS/CHoCH are the core trend-following and reversal signals in SMC.
- Dependencies: Steps 0.8, 0.9
- Risk: Low

#### Step 0.13: Liquidity Level Detection

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/liquidity.py`

- Action: Implement `detect_liquidity_levels(df: pl.DataFrame, swings: tuple[SwingPoint, ...], *, tolerance_pips: float = 5.0) -> tuple[LiquidityLevel, ...]`. Identify equal highs (multiple swing highs within tolerance) and equal lows (multiple swing lows within tolerance). Add `detect_liquidity_sweep(levels: tuple[LiquidityLevel, ...], current_bar: dict) -> tuple[LiquidityLevel, ...]` that returns updated levels with `swept=True` when price briefly pierces and then returns. Liquidity sweeps before OB tests are high-probability entries.
- Why: Institutional traders hunt liquidity (stop losses) at equal highs/lows before reversals.
- Dependencies: Steps 0.8, 0.9
- Risk: Medium -- liquidity detection is less standardized than other SMC patterns

#### Step 0.14: SMC Detector Orchestrator

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/smc_core/detector.py`

- Action: Implement `SMCDetector` class. Constructor takes `swing_length: int = 10`. Method `detect(df: pl.DataFrame, timeframe: Timeframe) -> SMCSnapshot` runs all sub-detectors (swing, OB, FVG, structure, liquidity) on a single timeframe's OHLCV data and returns a frozen `SMCSnapshot`. Method `detect_multi_tf(lake: ForexDataLake, instrument: str, timeframes: tuple[Timeframe, ...], as_of: datetime) -> dict[Timeframe, SMCSnapshot]` fetches data from lake and detects across all requested timeframes.
- Why: Single orchestrator prevents callers from needing to coordinate 5 sub-detectors manually.
- Dependencies: Steps 0.7, 0.9 - 0.13
- Risk: Low

#### Step 0.15: Basic CLI

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/cli/main.py`

- Action: Create Typer app with commands: `smc ingest --csv-dir PATH --instrument XAUUSD --timeframe D1` (loads CSV into data lake), `smc detect --instrument XAUUSD --timeframe H1 --date 2024-01-15` (runs SMC detection, prints patterns as Rich table), `smc lake-info` (shows data lake coverage per instrument/timeframe), `smc health` (system health check). Follow the exact pattern from `alphalens.cli.main` -- Rich panels, read-only dashboard.
- Why: CLI is the primary interface for development and monitoring.
- Dependencies: Steps 0.3, 0.6, 0.7, 0.14
- Risk: Low

#### Phase 0 Gate Criteria

- [ ] G0.1: `ForexAdapter` protocol has a passing `isinstance()` test for CSV, Mock, and MT5 adapters
- [ ] G0.2: CSV adapter ingests a 1-year XAUUSD D1 export and writes to data lake with valid schema
- [ ] G0.3: SMC detector produces non-empty results for all 5 pattern types on a 500-bar XAUUSD H1 sample
- [ ] G0.4: SMC detection matches at least 80% of hand-labeled patterns on a 100-bar test fixture (known_obs.json, known_fvgs.json)
- [ ] G0.5: `smc detect` CLI command runs end-to-end and displays formatted output
- [ ] G0.6: Data lake manifest SHA-256 is reproducible across two independent writes of identical data
- [ ] G0.7: All tests pass, 80%+ coverage on `smc.data` and `smc.smc_core` packages
- [ ] G0.8: MyPy strict passes with zero errors on all new modules
- [ ] G0.9: MT5 mock adapter correctly simulates `copy_rates_range`, `symbol_info_tick`, `order_send` return types

---

### Phase 1: Rules-Only MVP (Strategy + Backtest)

**Goal**: Multi-timeframe strategy generates trade setups. Backtest engine validates with realistic forex modeling. Walk-forward OOS shows positive expectancy.

#### Step 1.1: HTF Bias Module

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/htf_bias.py`

- Action: Implement `compute_htf_bias(d1_snapshot: SMCSnapshot, h4_snapshot: SMCSnapshot) -> BiasDirection`. BiasDirection is a frozen dataclass with `direction: Literal["bullish", "bearish", "neutral"]`, `confidence: float` (0-1), `key_levels: tuple[float, ...]` (support/resistance from HTF OBs), `rationale: str`. Logic: (1) D1 trend from latest BOS/CHoCH. (2) H4 must confirm D1 direction (same-side BOS). (3) If D1 and H4 disagree, bias is neutral. (4) Confidence scales with number of confirming structure breaks and proximity to unmitigated HTF OBs.
- Why: HTF bias is the first filter -- no trades against the higher timeframe direction.
- Dependencies: Steps 0.8, 0.12, 0.14
- Risk: Medium -- requires careful trend definition rules

#### Step 1.2: Zone Scanner

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/zone_scanner.py`

- Action: Implement `scan_zones(h1_snapshot: SMCSnapshot, bias: BiasDirection) -> tuple[TradeZone, ...]`. `TradeZone(frozen=True)`: zone_high: float, zone_low: float, zone_type: Literal["ob", "fvg", "ob_fvg_overlap"], direction: Literal["long", "short"], timeframe: Timeframe, confidence: float. Filter logic: only return zones aligned with HTF bias direction. Rank by: unmitigated OBs > FVGs > partially-filled FVGs. OBs with FVG overlap get bonus confidence. Maximum 3 active zones at a time (nearest to current price).
- Why: Zones are where the strategy expects institutional reaction -- the "where" of a trade.
- Dependencies: Steps 0.8, 1.1
- Risk: Low

#### Step 1.3: Entry Trigger Module

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/entry_trigger.py`

- Action: Implement `check_entry(m15_snapshot: SMCSnapshot, zone: TradeZone, current_price: float) -> EntrySignal | None`. `EntrySignal(frozen=True)`: entry_price: float, stop_loss: float, take_profit_1: float, take_profit_2: float, risk_pips: float, reward_pips: float, rr_ratio: float, trigger_type: Literal["choch_in_zone", "fvg_fill_in_zone", "ob_test_rejection"], grade: Literal["A", "B", "C"]. Trigger logic: (1) Price enters a zone from step 1.2. (2) M15 shows CHoCH (reversal into bias direction) inside the zone, OR M15 FVG fill inside an HTF OB. (3) Stop loss placed beyond the zone + buffer (swing high/low beyond zone). (4) TP1 at 1:2 RR, TP2 at the next HTF liquidity level or opposing OB.
- Why: Entry precision on M15 is what separates profitable SMC from random zone trading.
- Dependencies: Steps 0.8, 1.2
- Risk: High -- entry timing is the hardest part; needs extensive backtesting to calibrate

#### Step 1.4: Confluence Scorer

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/confluence.py`

- Action: Implement `score_confluence(setup: TradeSetup) -> float` returning a score in [0, 1]. Scoring factors (each 0-1, weighted sum): HTF trend alignment (0.25), zone quality (0.25: OB+FVG overlap > OB alone > FVG alone), entry trigger quality (0.20: CHoCH > FVG fill > OB test), RR ratio quality (0.15: >3:1 = 1.0, 2:1 = 0.7, <1.5:1 = 0.0), liquidity context (0.15: recent sweep before entry = bonus). Only setups scoring >= 0.6 are tradeable.
- Why: Confluence filtering reduces false signals by requiring multiple confirming factors.
- Dependencies: Steps 0.8, 1.1, 1.2, 1.3
- Risk: Medium -- weight tuning is subjective; must be validated by backtest results

#### Step 1.5: Multi-Timeframe Aggregator

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/aggregator.py`

- Action: Implement `MultiTimeframeAggregator` class. Constructor takes `SMCDetector`, `ForexDataLake`, instrument="XAUUSD". Method `generate_setups(as_of: datetime) -> tuple[TradeSetup, ...]` orchestrates the full pipeline: (1) detect SMC on D1, H4, H1, M15, (2) compute HTF bias, (3) scan H1 zones, (4) check M15 entry triggers for each zone, (5) score confluence, (6) return sorted setups (highest confluence first). `TradeSetup(frozen=True)`: entry_signal: EntrySignal, bias: BiasDirection, zone: TradeZone, confluence_score: float, generated_at: datetime.
- Why: This is the main orchestrator that produces actionable trade setups from raw data.
- Dependencies: Steps 0.14, 1.1, 1.2, 1.3, 1.4
- Risk: Medium

#### Step 1.6: Backtest Fill Model

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/backtest/fills.py`

- Action: Implement `FillModel` class. Constructor takes `spread_pips: float = 3.0` (XAUUSD typical), `slippage_pips: float = 0.5`, `commission_per_lot: float = 7.0` (USD). Method `compute_fill(order: OrderRequest, bar: dict) -> OrderResult | None`. For limit orders: fill if bar low <= limit_price (long) or bar high >= limit_price (short), adjusted by spread. For market orders: fill at bar open + slippage. For stop orders: fill at stop_price + slippage if triggered. Also model: SL/TP execution within a bar using high/low (pessimistic: if both SL and TP could trigger on same bar, SL triggers first).
- Why: Realistic fill modeling is essential for gold trading -- XAUUSD has 2-5 pip spreads and can gap significantly.
- Dependencies: Step 0.8
- Risk: High -- pessimistic intra-bar fill assumptions are critical for honest backtest results

#### Step 1.7: Backtest Engine

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/backtest/engine.py`

- Action: Implement `BarBacktestEngine` class. Constructor takes `BacktestConfig(frozen=True)`: initial_balance: float = 10_000.0, instrument: str = "XAUUSD", lot_size: float = 100_000 (standard lot for gold), max_concurrent_trades: int = 3, fill_model: FillModel. Method `run(lake: ForexDataLake, strategy: MultiTimeframeAggregator, start: datetime, end: datetime) -> BacktestResult`. Walk bar-by-bar on M15 (entry timeframe): (1) update SMC detection on each bar, (2) check open trade SL/TP against current bar, (3) generate new setups via aggregator, (4) execute fills via fill model, (5) record equity curve. Emit events for each action (entry, exit, SL, TP) to an append-only trade log.
- Why: Bar-level backtest with proper event ordering (check exits before entries) prevents look-ahead.
- Dependencies: Steps 0.7, 1.5, 1.6
- Risk: High -- correct event ordering and no look-ahead requires careful implementation

#### Step 1.8: Performance Metrics

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/backtest/metrics.py`

- Action: Implement pure functions: `sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252*24*4) -> float` (annualized, M15 bars), `sortino_ratio(returns, ...)`, `calmar_ratio(returns, ...)`, `max_drawdown(equity: np.ndarray) -> float`, `profit_factor(trades: list[TradeRecord]) -> float`, `win_rate(trades) -> float`, `avg_rr_realized(trades) -> float`, `expectancy(trades) -> float` (avg_win * win_rate - avg_loss * loss_rate). Assemble into `BacktestResult(frozen=True)` dataclass with all metrics + equity curve + trade log.
- Why: Comprehensive metrics beyond just Sharpe are needed to evaluate a forex strategy (profit factor and expectancy are more meaningful for low-frequency SMC trading).
- Dependencies: None (pure math)
- Risk: Low

#### Step 1.9: Walk-Forward Validation

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/backtest/walk_forward.py`

- Action: Implement `walk_forward_oos(lake, strategy, *, train_months: int = 12, test_months: int = 3, step_months: int = 3) -> list[BacktestResult]`. Rolling window: train on 12 months, test on next 3 months, slide by 3 months. For each window, run backtest on test period using strategy calibrated on train period. Return list of OOS BacktestResults. Add `aggregate_oos_results(results: list[BacktestResult]) -> WalkForwardSummary` with pooled Sharpe, pooled profit factor, consistency ratio (% of windows profitable).
- Why: In-sample backtest results are meaningless. Walk-forward OOS is the only honest way to estimate live performance.
- Dependencies: Step 1.7
- Risk: Medium -- strategy must have calibratable parameters for train/test to be meaningful

#### Phase 1 Gate Criteria

- [ ] G1.1: Aggregator produces at least 50 trade setups over a 6-month XAUUSD backtest window
- [ ] G1.2: Backtest fill model correctly applies spread (verified: long entry = ask = bid + spread, short entry = bid)
- [ ] G1.3: No look-ahead: all signals at bar[t] use only data from bars [0..t-1] (verified by property test)
- [ ] G1.4: In-sample backtest Sharpe > 0.5 on 2022-2024 XAUUSD (not a hard requirement, but a sanity check)
- [ ] G1.5: Walk-forward OOS: at least 60% of test windows have positive P&L (consistency ratio)
- [ ] G1.6: Walk-forward OOS pooled Sharpe > 0.0 (positive expectancy exists, even if modest)
- [ ] G1.7: Walk-forward OOS profit factor > 1.0
- [ ] G1.8: All tests pass, 80%+ coverage on `smc.strategy` and `smc.backtest` packages
- [ ] G1.9: `smc backtest --start 2022-01-01 --end 2024-12-31` CLI command runs end-to-end and generates Rich report

---

### Phase 2: Paper Trading (MT5 Demo Execution)

**Goal**: Run the strategy live on MT5 Demo account, manage orders, handle disconnections, produce daily reports.

#### Step 2.1: Risk Manager Types

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/risk/types.py`

- Action: Define `RiskConfig(frozen=True)`: max_risk_per_trade_pct: float = 1.0, max_daily_loss_pct: float = 3.0, max_total_drawdown_pct: float = 10.0, max_leverage: float = 20.0, max_concurrent_positions: int = 3, max_lot_size: float = 1.0. Define `PositionSize(frozen=True)`: lots: float, risk_usd: float, risk_pips: float, margin_required_usd: float. Define `RiskBudget(frozen=True)`: available_risk_pct: float, used_risk_pct: float, daily_loss_pct: float, total_drawdown_pct: float, can_trade: bool, rejection_reason: str | None.
- Why: Forex risk management is fundamentally different from equity (lots, leverage, pip-based sizing).
- Dependencies: None
- Risk: Low

#### Step 2.2: Position Sizer

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/risk/position_sizer.py`

- Action: Implement `compute_position_size(*, balance_usd: float, risk_pct: float, stop_loss_pips: float, pip_value_per_lot: float = 10.0) -> PositionSize`. For XAUUSD: pip_value_per_lot = $10 per pip per standard lot (0.01 move = 1 pip). Formula: risk_usd = balance * risk_pct / 100, lots = risk_usd / (stop_loss_pips * pip_value_per_lot). Clamp to min 0.01 lots, max from RiskConfig. Add `kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float` for reference sizing.
- Why: ATR-based or fixed-fractional position sizing ensures each trade risks exactly 1% (or configured percentage) of account.
- Dependencies: Step 2.1
- Risk: Low -- straightforward math, but pip value must be correct for XAUUSD

#### Step 2.3: Drawdown Guard

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/risk/drawdown_guard.py`

- Action: Implement `DrawdownGuard` class. Constructor takes `RiskConfig`. Method `check_budget(balance: float, peak_balance: float, daily_pnl: float) -> RiskBudget`. Logic: (1) daily_loss_pct = abs(daily_pnl) / balance * 100 if daily_pnl < 0; if >= max_daily_loss_pct, can_trade=False, reason="daily loss limit". (2) total_drawdown_pct = (peak_balance - balance) / peak_balance * 100; if >= max_total_drawdown_pct, can_trade=False, reason="max drawdown breaker". (3) Otherwise, compute available_risk_pct from remaining daily budget. The guard is stateless -- caller tracks peak_balance and daily_pnl.
- Why: Circuit breakers prevent catastrophic losses. The daily limit prevents revenge trading; the total drawdown limit preserves capital for strategy review.
- Dependencies: Step 2.1
- Risk: Low

#### Step 2.4: MT5 Executor Protocol and Implementation

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/execution/executor.py`

- Action: Define `MT5Executor` Protocol with methods: `send_order(request: OrderRequest) -> OrderResult`, `modify_order(ticket: int, *, sl: float | None, tp: float | None) -> OrderResult`, `close_position(ticket: int) -> OrderResult`, `get_positions() -> tuple[PositionState, ...]`, `get_account_info() -> AccountInfo`. Implement `RealMT5Executor` wrapping `mt5.order_send()`, `mt5.positions_get()`, etc. OrderRequest/OrderResult as frozen Pydantic models.
- Why: Protocol allows swapping real executor for mock in tests and macOS development.
- Dependencies: Step 0.5
- Risk: Medium -- MT5 order API has many edge cases (partial fills, requotes, off-quotes)

#### Step 2.5: Mock Executor

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/execution/executor_mock.py`

- Action: Implement `PaperExecutor(MT5Executor)`. Maintains an in-memory dict of open positions. `send_order()` fills immediately at requested price +/- spread. `get_positions()` returns current open positions. Tracks P&L per position. Writes all actions to an append-only trade journal (Parquet). Used for macOS development and backtest-integrated testing.
- Why: Full execution loop testing without MT5 terminal.
- Dependencies: Step 2.4
- Risk: Low

#### Step 2.6: Order Manager

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/execution/order_manager.py`

- Action: Implement `OrderManager` class. Constructor takes `executor: MT5Executor`, `risk_config: RiskConfig`, `drawdown_guard: DrawdownGuard`, `position_sizer`. Method `execute_setup(setup: TradeSetup, account: AccountInfo) -> OrderResult | None`. Pipeline: (1) check drawdown guard, (2) compute position size, (3) check margin requirements, (4) send order via executor, (5) set SL/TP, (6) log to trade journal. Method `manage_open_trades(current_price: float) -> list[OrderResult]` handles trailing SL, partial TP (close 50% at TP1, trail remainder to TP2).
- Why: Order lifecycle management with integrated risk checks.
- Dependencies: Steps 2.2, 2.3, 2.4
- Risk: High -- partial close and trailing SL are complex with MT5 API

#### Step 2.7: Position Reconciler

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/execution/reconciler.py`

- Action: Implement `reconcile(local_positions: dict[int, PositionState], mt5_positions: tuple[PositionState, ...]) -> ReconciliationResult`. Compare local state with MT5 terminal state. Detect discrepancies: positions closed by broker (margin call, stop-out), positions not reflected locally (manual trades). Log all discrepancies. ReconciliationResult: matched: int, local_only: list (phantom positions), mt5_only: list (untracked positions), discrepancies: list[str].
- Why: MT5 terminal is the source of truth. Network disconnections can cause local state drift.
- Dependencies: Step 2.4
- Risk: Medium

#### Step 2.8: Health Monitor

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/monitor/health.py`

- Action: Implement `HealthMonitor` class. Checks: MT5 connection alive, data freshness (last bar < 2 minutes old for M15), account margin level > 200%, no unreconciled positions, disk space for data lake. Returns `HealthStatus(frozen=True)`: all_ok: bool, checks: tuple[HealthCheck, ...]. Each `HealthCheck`: name: str, status: Literal["ok", "warning", "critical"], message: str.
- Why: Paper trading must be monitored for connection drops and data staleness.
- Dependencies: Steps 2.4, 2.7
- Risk: Low

#### Step 2.9: Trade Journal

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/monitor/journal.py`

- Action: Implement `TradeJournal` class. Append-only Parquet file at `data/journal/trades.parquet`. Columns: timestamp, action (entry/exit/modify), ticket, instrument, direction, lots, price, sl, tp, pnl, balance_after, setup_confluence_score, trigger_type. Methods: `log_action(...)`, `query(start, end) -> pl.DataFrame`, `daily_summary(date) -> DailySummary`. `DailySummary(frozen=True)`: trades_opened, trades_closed, gross_pnl, net_pnl, win_count, loss_count.
- Why: Every action must be logged for post-analysis and debugging.
- Dependencies: None
- Risk: Low

#### Step 2.10: Paper Trading Loop

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/cli/main.py` (extend with `smc live` command)

- Action: Add `smc live --mode demo` command that runs the paper trading loop. Loop: (1) every M15 bar close, fetch latest data via MT5 adapter, (2) run SMC detection + strategy aggregator, (3) check for new setups, (4) execute via order manager, (5) manage open trades (trailing SL, partial TP), (6) reconcile positions, (7) check health, (8) log to journal. Use `asyncio` for non-blocking MT5 polling. Ctrl+C gracefully shuts down (closes all positions option, or leaves open).
- Why: The main runtime for paper trading validation.
- Dependencies: Steps 1.5, 2.6, 2.7, 2.8, 2.9
- Risk: High -- live loop timing, error recovery, graceful shutdown

#### Phase 2 Gate Criteria

- [ ] G2.1: Paper executor correctly tracks P&L across 100 simulated trades (verified by unit test)
- [ ] G2.2: Position sizer produces correct lot sizes for XAUUSD at various stop-loss distances (verified: 1% risk of $10,000 with 20-pip SL = 0.50 lots)
- [ ] G2.3: Drawdown guard correctly blocks trading after daily limit hit (unit test)
- [ ] G2.4: Reconciler detects phantom positions and untracked positions (unit test with mock)
- [ ] G2.5: `smc live --mode demo` runs for 24 hours on MT5 Demo without crashes
- [ ] G2.6: Trade journal contains all 24h of actions with no gaps
- [ ] G2.7: Health monitor correctly detects and alerts on simulated MT5 disconnection
- [ ] G2.8: All tests pass, 80%+ coverage on `smc.risk`, `smc.execution`, `smc.monitor`
- [ ] G2.9: Paper trading results over 1 month of Demo align within 20% of backtest expectations (Sharpe, win rate)

---

### Phase 3: AI Debate Integration

**Goal**: Reuse the alphalens debate pipeline to add an AI layer on top of rules-based signals. AI debates whether a trade setup should be taken, providing a second opinion before execution.

#### Step 3.1: SMC Analyst Prompts

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/debate/smc_analysts.py`

- Action: Define four domain analyst system prompts for XAUUSD SMC: (1) STRUCTURE_ANALYST: analyzes D1/H4 BOS/CHoCH trend context, (2) ZONE_ANALYST: evaluates OB/FVG zone quality and freshness, (3) LIQUIDITY_ANALYST: assesses liquidity sweep context and stop-hunt probability, (4) MACRO_ANALYST: incorporates USD strength (DXY), gold-specific fundamentals (real yields, central bank buying). Each prompt includes SMC-specific terminology and evaluation criteria.
- Why: Domain-specific prompts ensure the AI debate is grounded in SMC concepts rather than generic trading.
- Dependencies: Reuses `alphalens.debate.models.AnalystView`
- Risk: Low

#### Step 3.2: Debate-to-Trade Bridge

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/debate/adapter.py`

- Action: Implement `run_smc_debate(setup: TradeSetup, smc_snapshots: dict[Timeframe, SMCSnapshot]) -> DebateResult`. Adapts the alphalens `debate.pipeline` to XAUUSD context: (1) format SMC snapshots into analyst context strings, (2) run 4 SMC analysts, (3) run bull/bear debate (2 rounds), (4) run trader agent that produces a `TradeProposal` (reused from alphalens). The proposal's `direction` maps to "take_trade" / "skip" / "reduce_size" instead of "buy"/"sell"/"hold".
- Why: Reusing the proven 7-agent debate architecture from alphalens avoids building a new AI pipeline from scratch.
- Dependencies: alphalens debate pipeline, Step 3.1
- Risk: Medium -- prompt engineering for XAUUSD SMC context is iterative

#### Step 3.3: Gatekeeper Integration

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/strategy/aggregator.py` (extend)

- Action: Add optional `debate_gatekeeper: bool = False` parameter to `MultiTimeframeAggregator`. When enabled, every setup that passes confluence threshold (>= 0.6) is also run through `run_smc_debate()`. The debate result's confidence modulates the final position size: confidence >= 0.7 = full size, 0.5-0.7 = half size, < 0.5 = skip. This is additive to the rules-only system, not a replacement.
- Why: AI debate acts as a second opinion, not a primary signal. Rules first, AI filters.
- Dependencies: Steps 3.2, 1.5
- Risk: Low -- additive feature, does not break rules-only path

#### Phase 3 Gate Criteria

- [ ] G3.1: SMC analyst prompts produce domain-relevant analysis (manual review of 10 debate transcripts)
- [ ] G3.2: Debate pipeline runs in < 30 seconds per setup (4 analysts + 2 debate rounds + trader)
- [ ] G3.3: A/B backtest: rules-only vs rules+debate over same 6-month OOS window
- [ ] G3.4: Rules+debate OOS Sharpe >= rules-only OOS Sharpe (AI does not degrade performance)
- [ ] G3.5: Rules+debate reduces trade count by 10-30% (AI filters low-quality setups)
- [ ] G3.6: All tests pass, 80%+ coverage on `smc.debate`

---

### Phase 4: Live Trading

**Goal**: Deploy to Windows VPS, start with minimum position size, scale up based on performance.

#### Step 4.1: Deployment Configuration

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/config.py`

- Action: Implement `SMCConfig` using Pydantic Settings (env vars + .env file). Fields: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, INSTRUMENT, RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT, MAX_LOT_SIZE (start at 0.01), DATA_LAKE_ROOT, JOURNAL_ROOT, LOG_LEVEL, ALERTING_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
- Why: All deployment configuration via environment variables, no hardcoded secrets.
- Dependencies: None
- Risk: Low

#### Step 4.2: Telegram Alerter

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/monitor/alerter.py`

- Action: Implement `TelegramAlerter` that sends trade alerts (entry, exit, daily summary, health warnings) to a Telegram chat. Async HTTP via httpx. Rate-limited to 20 messages/minute. Formatted messages with trade details, P&L, and emoji indicators.
- Why: Real-time monitoring of live trades when not at the computer.
- Dependencies: Step 2.9 (journal for daily summaries)
- Risk: Low

#### Step 4.3: Scaling Rules

**File**: `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/risk/scaling.py`

- Action: Implement `compute_scaling_tier(track_record: list[BacktestResult]) -> ScalingTier`. `ScalingTier(frozen=True)`: max_lot_size: float, max_concurrent: int, tier_name: str. Tiers: (1) Probation (0.01 lots, 1 position, first 30 trades), (2) Junior (0.05 lots, 2 positions, after 30+ trades with profit factor > 1.2), (3) Standard (0.10 lots, 3 positions, after 100+ trades with Sharpe > 0.5), (4) Full (configured max, after 200+ trades with consistent performance). Each tier requires clearing the previous tier's metrics.
- Why: Systematic scaling prevents premature risk increase before the strategy proves itself live.
- Dependencies: Step 1.8
- Risk: Low

#### Phase 4 Gate Criteria

- [ ] G4.1: VPS deployment runs `smc live --mode live` for 72 hours without crash
- [ ] G4.2: First 30 live trades complete (Probation tier), all logged to journal
- [ ] G4.3: Probation tier profit factor > 1.0 (break-even or better)
- [ ] G4.4: Live results within 30% of paper trading expectations (slippage/spread adjusted)
- [ ] G4.5: Telegram alerts received for all entries, exits, and daily summaries
- [ ] G4.6: Reconciler detects zero discrepancies over 72-hour test period
- [ ] G4.7: Health monitor has < 5 minutes cumulative downtime over 72 hours

---

## Dependency Graph

```
Phase 0 (Foundation):
  0.1 schemas ─────────────────┐
  0.2 adapter protocol ────────┤
  0.3 csv adapter ─────────────┤
  0.4 mt5 mock ────────────────┤ 
  0.5 mt5 real ────────────────┤
  0.6 writer ──────────────────┤───> 0.7 data lake ───> 0.14 detector ───> 0.15 CLI
  0.8 smc types ───────────────┤         ^                     ^
  0.9 swing ───────────────────┤         |                     |
  0.10 order block ────────────┤         |                     |
  0.11 fvg ────────────────────┤         |                     |
  0.12 structure ──────────────┤         |                     |
  0.13 liquidity ──────────────┘         |                     |
                                         |                     |
Phase 1 (Strategy + Backtest):           |                     |
  1.1 htf bias ──────────────────────────|─────────────────────┤
  1.2 zone scanner ──────────────────────|─────────────────────┤
  1.3 entry trigger ─────────────────────|─────────────────────┤
  1.4 confluence ────────────────────────|─────────────────────┤
  1.5 aggregator ────────────────────────┘                     |
  1.6 fills ─────────────────────────────────────────> 1.7 engine ───> 1.9 walk-forward
  1.8 metrics ───────────────────────────────────────────────────────────────┘
                                                                            |
Phase 2 (Paper Trading):                                                    |
  2.1 risk types ───> 2.2 sizer ─────────> 2.6 order manager               |
                      2.3 drawdown guard ──────┘      |                     |
  2.4 executor protocol ──────────────────────────────┤                     |
  2.5 mock executor ──────────────────────────────────┤                     |
  2.7 reconciler ─────────────────────────────────────┤                     |
  2.8 health monitor ─────────────────────────────────┤                     |
  2.9 journal ────────────────────────────────────────┤                     |
  2.10 live loop ─────────────────────────────────────┘                     |
                                                                            |
Phase 3 (AI Debate):                                                        |
  3.1 smc prompts ───> 3.2 debate bridge ───> 3.3 gatekeeper ──────────────┘
                                                                            
Phase 4 (Live):
  4.1 config ───> 4.2 telegram ───> 4.3 scaling
```

**Critical path**: 0.1 -> 0.8 -> 0.9 -> 0.12 -> 0.14 -> 1.1 -> 1.5 -> 1.7 -> 1.9 (must prove alpha before anything else matters).

**Parallelizable within Phase 0**: Steps 0.3/0.4/0.5 (three adapter implementations, independent). Steps 0.9/0.10/0.11/0.12/0.13 (five SMC sub-detectors, independent).

**Parallelizable within Phase 1**: Steps 1.1/1.2/1.3 (strategy modules, loosely coupled). Steps 1.6/1.8 (fills and metrics, no dependency on each other).

**Parallelizable across phases**: Phase 2 risk types (2.1/2.2/2.3) can start in parallel with Phase 1 backtest work since risk types are independent.

---

## Testing Strategy

### Unit Tests (per module, TDD)

- **Data layer**: Schema validation edge cases (null spread, future timestamps, timezone handling). Adapter protocol conformance. CSV parsing of various MT5 export formats. Mock adapter return type fidelity vs real MT5 structured arrays. Writer partition correctness.
- **SMC core**: Each detector against hand-labeled fixtures. Property tests with Hypothesis: (1) detected swing points are always local extrema, (2) OB high > OB low, (3) FVG gap size > 0, (4) structure breaks reference valid swing points. Known-pattern fixtures with golden file comparison.
- **Strategy**: HTF bias direction consistency (D1 bullish + H4 bullish = bullish, never neutral). Zone filtering only returns bias-aligned zones. Entry trigger RR ratio computation. Confluence score bounds (always in [0, 1]).
- **Backtest**: Fill model spread application (verified against manual calculation). No look-ahead property test (signal at t uses only data up to t-1). Equity curve monotonicity check (cash + unrealized P&L = total equity). Metrics against known portfolios with precomputed answers.
- **Risk**: Position size computation against manual examples. Drawdown guard correctly blocks at limits. Leverage calculation.
- **Execution**: Mock executor state management. Order manager lifecycle (open, modify SL, partial close, full close). Reconciler discrepancy detection.

### Integration Tests

- **Full pipeline**: Data ingest -> SMC detect -> strategy -> backtest on a small (100-bar) fixture. Verify no exceptions, positive trade count, reasonable P&L.
- **MT5 connection** (Windows-only, marker `@pytest.mark.mt5`): Real MT5 terminal connection, fetch 10 bars, send a demo order, verify position.

### E2E Tests

- **CLI smoke**: `smc ingest`, `smc detect`, `smc backtest`, `smc health` all execute without error on fixture data.
- **Paper trading loop**: Run for 5 simulated bars using mock executor, verify journal contains entries.

### Test Markers

```python
# conftest.py
markers = [
    "unit: fast, no IO",
    "integration: may touch disk or network",
    "mt5: requires Windows MT5 terminal",
    "slow: backtest runs > 10 seconds",
]
```

---

## Risks and Mitigations

### Risk 1: smart-money-concepts library accuracy

**Severity**: High
**Description**: The joshyattridge library may have bugs in OB/FVG/BOS detection. The library has not been updated in 6 months and has some open issues about incorrect pattern detection.
**Mitigation**: (1) Create a hand-labeled validation set of 50+ known patterns from actual XAUUSD charts. (2) Gate G0.4 requires 80% match against labels. (3) If the library is unreliable, rewrite the critical detectors (OB, BOS/CHoCH) from scratch using the library as reference -- the math is straightforward (swing detection + comparison).

### Risk 2: MT5 Python SDK macOS limitation

**Severity**: Medium
**Description**: MetaTrader5 Python package only works on Windows. All macOS development must use mocks.
**Mitigation**: (1) MT5 mock adapter faithfully replicates all return types (numpy structured arrays with specific dtypes). (2) Integration tests with the real MT5 terminal run on Windows VPS via CI. (3) CSV adapter provides an MT5-independent data path for development.

### Risk 3: Backtest overfitting

**Severity**: High
**Description**: SMC rules have many parameters (swing_length, OB lookback, FVG join logic, confluence weights). Easy to overfit to historical gold data.
**Mitigation**: (1) Walk-forward OOS validation is mandatory (Gate G1.5, G1.6, G1.7). (2) Parameters are kept deliberately coarse (swing_length = 10, not 7.3). (3) Phase 2 paper trading provides true OOS validation. (4) All parameter choices documented with rationale in docstrings.

### Risk 4: XAUUSD spread and slippage modeling

**Severity**: Medium
**Description**: Gold spreads vary wildly (2 pips during London session, 10+ pips during rollover/news). Fixed spread backtest may be overly optimistic.
**Mitigation**: (1) Backtest uses pessimistic fixed spread (3 pips, above typical London session) plus slippage (0.5 pips). (2) Sensitivity analysis: re-run backtest at 5, 7, 10 pip spreads. If strategy is only profitable at < 3 pip spread, it fails. (3) Paper trading on Demo will reveal real spread distribution.

### Risk 5: Live execution failures

**Severity**: High (Phase 4 only)
**Description**: MT5 requotes, order rejection, network timeouts, broker-side stop-outs.
**Mitigation**: (1) Retry with exponential backoff on transient failures (reuse alphalens `fetch_with_retry`). (2) Position reconciler runs every 60 seconds. (3) Drawdown guard as hard circuit breaker. (4) Start at 0.01 lots (Probation tier) so maximum loss per trade is negligible while proving stability. (5) Telegram alerts for any execution failure.

### Risk 6: Strategy does not produce alpha

**Severity**: High
**Description**: SMC patterns may not have edge in XAUUSD, or the edge may be too small after spread/slippage.
**Mitigation**: (1) Gate G1.6 (walk-forward OOS Sharpe > 0.0) is the hard pass/fail. If it fails, the system is complete and functional but the strategy needs rethinking -- possibly different confluence weights, different timeframe combinations, or adding macro filters (DXY, real yields). (2) The modular architecture means the strategy layer can be replaced without touching data/backtest/execution layers. (3) Phase 3 AI debate integration is specifically designed to improve signal quality if rules alone are marginal.

---

## Success Criteria (Project-Level)

- [ ] Phase 0 gate cleared: data ingestion, SMC detection, CLI all working
- [ ] Phase 1 gate cleared: walk-forward OOS positive expectancy demonstrated
- [ ] Phase 2 gate cleared: 1 month of stable Demo paper trading with no crashes
- [ ] Phase 3 gate cleared: AI debate does not degrade OOS performance
- [ ] Phase 4 gate cleared: 72 hours of live trading at minimum size without issues
- [ ] Total test count > 200, coverage > 80% across all `smc` packages
- [ ] MyPy strict passes, Ruff clean, zero security issues from bandit
- [ ] All frozen dataclasses, no mutable state in the pipeline
- [ ] README with installation, configuration, and usage instructions

---

## Key File Paths Reference

**Existing files to reuse (import from, do not modify)**:
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/data/adapters/base.py` -- SourceAdapter Protocol pattern
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/data/adapters/_retry.py` -- fetch_with_retry
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/data/manifest.py` -- manifest system
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/data/writers.py` -- write_partitioned pattern
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/data/schemas.py` -- schema validation pattern
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/preview/signals.py` -- SignalValues pattern
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/research/ic.py` -- IC research (for signal validation)
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/debate/models.py` -- AnalystView, DebateRound, TradeProposal
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/debate/pipeline.py` -- debate pipeline architecture
- `/Users/christopher/claudeworkplace/alphalens-v2/src/alphalens/cli/main.py` -- CLI pattern (Typer + Rich)

**File to modify**:
- `/Users/christopher/claudeworkplace/alphalens-v2/pyproject.toml` -- add smc package, scripts entry, optional deps

**New package root**:
- `/Users/christopher/claudeworkplace/alphalens-v2/src/smc/` -- all new SMC code
- `/Users/christopher/claudeworkplace/alphalens-v2/tests/smc/` -- all new SMC tests
