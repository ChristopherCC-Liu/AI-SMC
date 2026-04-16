# AI-SMC: AI-Powered Smart Money Concepts Trading System

An end-to-end algorithmic trading system for XAUUSD (Gold) on MetaTrader 5. Combines classical Smart Money Concepts (SMC) pattern detection with an AI regime classifier backed by a 7-agent Claude debate pipeline. Validated through 10 sprints of walk-forward out-of-sample backtesting, with paper trading live on a Windows VPS.

---

## Architecture

```
                        AI-SMC System
                             |
        +--------------------+--------------------+
        |                    |                    |
   Data Layer          SMC Detection        AI Regime
   ForexDataLake        SMCDetector        Classifier
   (Parquet)         6 sub-detectors       (7-agent
   D1/H4/H1/M15     swing,OB,FVG,          Claude
   CSV / MT5 /       BOS,CHoCH,            Debate)
   Mock adapters      liquidity
        |                    |                    |
        +--------------------+--------------------+
                             |
                    Strategy Pipeline
                  MultiTimeframeAggregator
                  D1→H4 HTF Bias
                  H1 Zone Scanner
                  M15 Entry Trigger
                  Confluence Scoring (5 factors)
                             |
                    Risk Manager
                  Position Sizer (1% equity)
                  DrawdownGuard (3%/10%)
                  Exposure Checks
                             |
                    MT5 Executor
                  BrokerPort Protocol
                  MT5BrokerPort (Windows)
                  SimBrokerPort (dev/test)
                             |
                   Live Trading Loop
                  M15 bar-close polling
                  Health Monitor
                  TradeJournal (JSONL)
                  Telegram Alerts
```

---

## Key Features

- **6 SMC pattern detectors**: swing high/low (per-TF swing_length), order blocks (bullish/bearish, mitigation tracking), fair value gaps (consecutive merge, fill %), BOS/CHoCH structure breaks, liquidity levels (equal-high/low sweeps)
- **Multi-timeframe pipeline**: D1+H4 for HTF bias (3-tier system), H1 for zone identification, M15 for entry triggers — full top-down confluence
- **ATR-adaptive stop loss**: H1 ATR(14) × regime multiplier, floor at 200 points, calibrated via Sprint 4 data analysis
- **AI regime classification**: 5 regimes (TREND_UP, TREND_DOWN, CONSOLIDATION, TRANSITION, ATH_BREAKOUT), 7-agent debate pipeline via Claude CLI, ATR fallback chain, regime-routed parameter presets
- **Walk-forward OOS backtesting**: 12-month train / 3-month test / 3-month step windows, no-lookahead guarantee, bar-by-bar M15 simulation
- **Position sizing + drawdown circuit breaker**: 1% equity risk per trade, 3% daily loss halt, 10% drawdown halt
- **MT5 real and simulated execution**: `BrokerPort` protocol, `MT5BrokerPort` (Windows), `SimBrokerPort` (macOS/Linux dev)
- **Live trading loop**: asyncio, M15 bar-close polling, health monitor, trade journal (JSONL), Telegram alerts, SIGINT graceful shutdown
- **Signal inversion research**: 3D worst-indicator matrix across 5 backtest versions, 447 trades analyzed (Sprint 9)

---

## Quick Start

### Prerequisites

- Python 3.11+
- For live trading: Windows VPS with MetaTrader 5 terminal installed
- For backtesting (macOS/Linux): no MT5 required

### Install

```bash
# Clone
git clone https://github.com/ChristopherCC-Liu/AI-SMC.git
cd AI-SMC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install (add [mt5] on Windows for MetaTrader5)
pip install -e ".[dev]"
# pip install -e ".[mt5,dev]"    # Windows with MT5
```

### Run Tests

```bash
pytest
pytest --cov=smc --cov-report=term-missing   # with coverage
```

### Backfill Data (macOS/Linux — HistData CSV source)

```bash
python scripts/backfill_histdata.py
```

This resamples the bundled M1 zip to D1/H4/H1/M15 Parquet files in `data/parquet/`.

### Run Backtest

```bash
# Single OOS window (window 1 = Jan 2021 – Apr 2021 test period)
python scripts/run_v7_single_window.py 1

# With AI regime debate enabled (requires Claude CLI)
python scripts/run_v7_single_window.py 1 --ai-regime
```

Results are appended to `data/gate1_v7_windows.jsonl`.

### CLI Commands

```bash
# Data management
smc ingest --instrument XAUUSD --start 2020-01-01 --end 2025-01-01
smc detect --timeframe H1

# Backtesting
smc backtest --config config/smc_config.yaml

# Health check
smc health

# Live paper trading (requires MT5 on Windows)
smc live --mode demo
```

### Live Demo (Windows VPS with MT5)

```bash
# Automated VPS deployment
.\scripts\deploy_vps.ps1

# Manual launch
python scripts/live_demo.py
```

See [docs/VPS_DEPLOYMENT.md](docs/VPS_DEPLOYMENT.md) for the complete step-by-step deployment guide including MT5 setup, environment configuration, and nssm auto-restart service installation.

---

## Project Structure

```
AI-SMC/
├── src/smc/
│   ├── data/              # Data lake, adapters (CSV, MT5, mock), Parquet writers, schemas
│   ├── smc_core/          # SMC pattern detectors: swing, OB, FVG, BOS/CHoCH, liquidity
│   ├── strategy/          # Multi-TF aggregators (v1/v2/v3/v1.1), confluence scoring, entry triggers
│   ├── ai/                # AI regime classifier, direction engine, 7-agent debate pipeline
│   ├── backtest/          # Bar-by-bar engine, fill model, walk-forward OOS, metrics
│   ├── risk/              # Position sizer, drawdown guard, exposure checks
│   ├── execution/         # BrokerPort protocol, MT5/Sim executors, order manager, reconciler
│   ├── monitor/           # Live loop, health checks, trade journal, Telegram alerter, timing
│   ├── cli/               # Typer CLI (ingest, detect, backtest, live, health)
│   ├── research/          # IC analysis utilities
│   └── config.py          # Pydantic-settings SMCConfig (all SMC_* env vars)
├── scripts/
│   ├── backfill_histdata.py        # macOS/Linux HistData M1 → multi-TF resampler
│   ├── backfill_data.py            # Windows MT5 backfill
│   ├── run_v3_single_window.py     # Gate runners v3–v10 (OOS validation per sprint)
│   ├── run_v4_single_window.py
│   ├── run_v5_single_window.py
│   ├── run_v6_single_window.py
│   ├── run_v7_single_window.py     # Current production gate runner (Sprint 6 AI routing)
│   ├── run_v8_single_window.py     # v1 vs v2 walk-forward comparison
│   ├── run_v9_single_window.py     # Hybrid v3 + signal inversion
│   ├── run_v10_single_window.py    # v1.1 inversion test
│   ├── worst_indicator_scan.py     # 3D worst-indicator matrix (Sprint 9)
│   ├── live_demo.py                # Standalone VPS launcher (paper mode)
│   ├── deploy_vps.ps1              # Automated Windows VPS deployment
│   ├── deploy_vps.sh               # Linux deployment helper
│   ├── install_service.ps1         # nssm Windows service installer
│   └── export_mt5_csv.mq5          # MQL5 script to export MT5 history to CSV
├── tests/                          # 39 test files (unit, integration, property-based)
│   ├── smc/unit/                   # Unit tests per module (ai, backtest, data, execution, monitor, risk, smc_core)
│   └── smc/integration/            # Pipeline integration tests
├── config/
│   └── smc_config.yaml             # Default YAML strategy and risk configuration
├── data/
│   ├── parquet/                    # Partitioned Parquet data lake (XAUUSD D1/H4/H1/M15)
│   ├── gate1_v{n}_windows.jsonl    # Gate 1 OOS results per sprint version
│   ├── gate_v{n}_v1.jsonl          # Sprint 8–10 gate results
│   ├── regime_cache.parquet        # Pre-computed regime classifications (backtest)
│   ├── worst_indicator_matrix.json # 3D signal inversion scan results
│   └── journal/                    # Live paper trade journal (JSONL)
└── docs/
    ├── VPS_DEPLOYMENT.md           # Step-by-step VPS deployment guide
    ├── HANDOFF.md                  # Developer handoff document
    └── planning/                   # Sprint planning documents
```

---

## Performance Evolution

All results are walk-forward out-of-sample (OOS) on XAUUSD M15. Train: 12 months, Test: 3 months, Step: 3 months. $10,000 initial balance, 0.01 lots, $7 commission per lot.

| Version | Profit Factor | Win Rate | Trades | Key Change |
|---------|--------------|----------|--------|------------|
| v1 | 0.23 | 9.1% | 11 | Baseline (Window 1) |
| v2 | 0.54 | 26.7% | 45 | ATR-SL + swing_length tuning |
| v5 | 0.83 | 38.1% | 63 | ATR-adaptive SL (Sprint 4) |
| v6 | 1.05 | 42.9% | 56 | Surgical fixes: disable ob_test, tighten transitional floor, anti-cluster |
| v7 | 1.02 | 41.1% | 56 | AI regime routing (Sprint 6) — stable |

**Production v1 performance (Gate v10, multiple windows)**: PF ranging 1.09–3.90, WR 40–71%, dominated by `fvg_fill_in_zone` trigger. PF 1.28–1.66 across standard windows.

---

## Configuration

### Environment Variables (.env)

All settings use the `SMC_` prefix. Copy `.env.production` to `.env` and fill in credentials:

```ini
# Runtime
SMC_ENV=paper                    # dev | paper | live
SMC_MT5_MOCK=0                   # 1 on macOS/Linux, 0 on Windows VPS
SMC_LOG_LEVEL=INFO

# MT5 Connection (Windows VPS only)
SMC_MT5_LOGIN=12345678
SMC_MT5_PASSWORD=yourpassword
SMC_MT5_SERVER=TMGM-Demo

# Risk Management
SMC_RISK_PER_TRADE_PCT=1.0       # % equity risked per trade
SMC_MAX_DAILY_LOSS_PCT=3.0       # daily loss halt threshold
SMC_MAX_DRAWDOWN_PCT=10.0        # total drawdown halt threshold
SMC_MAX_LOT_SIZE=0.01            # hard cap per order

# AI Regime (optional — v1.1+)
SMC_AI_REGIME_ENABLED=0          # 1 to enable AI debate pipeline
SMC_ANTHROPIC_API_KEY=sk-ant-... # Anthropic API key (if not using Claude CLI)
SMC_LLM_DAILY_BUDGET_USD=5.0

# Telegram Alerts (optional)
SMC_TELEGRAM_BOT_TOKEN=...
SMC_TELEGRAM_CHAT_ID=...
```

### config/smc_config.yaml

Provides YAML defaults for detection, strategy, risk, and backtest parameters. Individual values can be overridden via env vars at runtime.

```yaml
instrument: XAUUSD
timeframes:
  htf: [D1, H4]       # Higher timeframes for structural bias
  zone: H1            # Order block / FVG identification
  entry: M15          # Entry precision and trigger

smc:
  swing_length: 10             # Bars each side to confirm a swing
  ob_lookback: 50              # Max bars back to scan for order blocks
  fvg_join_consecutive: true   # Merge consecutive FVGs
  liquidity_tolerance_points: 5.0

strategy:
  min_confluence_score: 0.6    # Minimum weighted score to trigger a setup
  max_concurrent_setups: 3
  min_rr_ratio: 2.0

risk:
  risk_per_trade_pct: 1.0
  max_daily_loss_pct: 3.0
  max_drawdown_pct: 10.0
  max_lot_size: 0.01

backtest:
  spread_pips: 3.0
  slippage_pips: 0.5
  commission_per_lot: 7.0
  initial_balance: 10000.0
```

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Language | Python 3.11+ |
| Data | Polars, Pandas, PyArrow, DuckDB |
| Validation | Pydantic v2, pydantic-settings |
| SMC | smartmoneyconcepts |
| Math | NumPy, SciPy |
| MT5 | MetaTrader5 (Windows only) |
| CLI | Typer, Rich |
| AI | Claude CLI (`claude -p`), Anthropic API |
| Testing | pytest, pytest-cov, hypothesis |
| Linting | ruff, mypy (strict) |

---

## License

MIT License. See LICENSE file for details.
