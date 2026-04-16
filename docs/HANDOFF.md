# AI-SMC Developer Handoff

**As of Sprint 10 — April 2026**

This document is the single source of truth for a developer taking over the AI-SMC project. Read this before touching any code.

---

## Table of Contents

1. [Current State](#current-state)
2. [Architecture Decisions Made (and WHY)](#architecture-decisions-made-and-why)
3. [Sprint History](#sprint-history)
4. [Key Files Reference](#key-files-reference)
5. [Known Issues and Tech Debt](#known-issues-and-tech-debt)
6. [Future Roadmap](#future-roadmap)
7. [Team Structure Used](#team-structure-used)

---

## Current State

### What Is Running in Production

**v1 is the production strategy.** It is deployed as a paper demo on VPS `43.163.107.158`, connected to a TMGM Demo MT5 account (paper money, no real orders).

- Entry point: `scripts/live_demo.py` — polls MT5 every M15 bar close, logs setups to `data/journal/live_trades.jsonl`
- Strategy class: `src/smc/strategy/aggregator.py` (`MultiTimeframeAggregator`)
- Configuration: `ai_regime_enabled=False` (ATR fallback only), `enable_ob_test_trigger=False`
- Mode: PAPER — all signals are logged, no real order submission wired in the demo launcher

### Performance Expectations (from OOS backtesting)

v1 walk-forward OOS across Gate v10 windows:

| Window | Period | Trades | PF | WR |
|--------|--------|--------|-----|-----|
| 2 | Apr–Jul 2021 | 1 | Infinity | 100% |
| 4 | Oct 2021–Jan 2022 | 10 | 1.29 | 40% |
| 5 | Jan–Apr 2022 | 7 | 3.90 | 71% |

Typical range: PF 1.09–1.66, WR 40–57%. Primary trigger: `fvg_fill_in_zone` (~90% of trades).

### What Is NOT Running in Production

- **AI regime debate pipeline** (`ai_regime_enabled=False`) — built and tested (Sprint 6), held back pending live data collection. Code is in `src/smc/ai/debate/pipeline.py`.
- **v2/v3 strategies** (`aggregator_v2.py`, `aggregator_v3.py`) — experimental, tested, archived. NOT deployed.
- **v1.1 inversion** (`aggregator_v1_1.py`) — tested in Sprint 10, NOT yet deployed. Needs more live data first.
- **Real order execution** — `MT5BrokerPort` code exists and is tested, but the live demo script logs only; no `send_order` calls are wired in `live_demo.py`.

---

## Architecture Decisions Made (and WHY)

### 1. "AI judges direction, SMC finds entry timing"

This is the v2 design principle from Sprint 7. The AI regime classifier (and direction engine) are upstream filters: they tell the system whether the market is trending bullish, bearish, or in consolidation. The SMC pipeline (OB, FVG, BOS zones) handles the precise entry point within that context.

This separation proved important because the AI cannot reliably time M15 entries — it operates at D1/H4 timescales. The SMC detectors are better at finding the actual entry zone.

### 2. "Invert losing signals"

Sprint 9 ran a 3D scan (trigger × direction × regime) over 447 trades across 5 gate versions. The output (`data/worst_indicator_matrix.json`) identified patterns that were consistently bad (called "STABLE_BAD"). The hypothesis: a signal that is reliably wrong in a given context should be inverted rather than disabled.

Finding: `fvg_fill_in_zone SHORT` had 33.4% WR in non-trending regimes across all gate versions. Inverting to LONG theoretically captures 66.6% WR.

This is how `aggregator_v1_1.py` was born. Sprint 10 tested it but did not deploy it — needs live validation first.

### 3. "Don't relax v1 parameters"

Sprint 7/8 compared v1 (strict: 0.45 confluence threshold, 24h cooldown, max 3 concurrent) against v2 (relaxed: 0.35 threshold, 8h cooldown, max 5 concurrent, dual entries). v2 FAILED with PF 0.6–0.8 across all windows. v1 held PF 1.02–1.66.

The lesson: relaxing parameters to increase trade frequency destroys signal quality. The v1 parameters are calibrated; treat them as load-bearing.

### 4. "fvg_sweep was a parameter artifact"

Sprint 8 found `fvg_sweep_continuation` had 50% WR and PF 3.8 in v2 data. This looked like a strong signal worth porting to v3. Sprint 9 tested the hybrid v3 (v1 + fvg_sweep bolt-on) and found the 50% WR was an artifact of v2's wide SL buffer (relaxed `sl_atr_multiplier`), not the trigger itself. When run under v1's tighter SL, fvg_sweep performance collapsed. v3 was abandoned.

The `fvg_sweep_continuation` code exists in `src/smc/strategy/entry_v2.py` and is preserved for reference, but is NOT deployed.

### 5. "Deploy v1, iterate later"

After 10 sprints, the team had: v1 (PF > 1.0), AI classifier built but not production-validated, two failed experimental strategies. The decision was to deploy v1 for live data collection and validate AI regime enabling as a separate step with enough real data to judge it fairly.

This is a deliberate "ship what works, iterate with evidence" decision — not a failure of the AI approach.

### 6. Zone anti-clustering and ob_test disable (Sprint 5 surgical fixes)

Sprint 5 found three specific issues in v4 data:
- `ob_test_rejection` trigger had 18.2% WR — disabled (`enable_ob_test_trigger=False`)
- Transitional regime had 32.4% WR at low confluence — raised floor to 0.60
- Same zone hit multiple times per cycle — added `_active_zones` set anti-cluster

These three fixes pushed PF from 0.83 (v5) to 1.05 (v6). They are surgical, evidence-based, and should not be reverted without data.

### 7. ATR-adaptive stop loss (Sprint 4)

Fixed SL buffers (v1–v3) were the primary cause of poor win rates — too wide for ranging markets, too narrow for trending. Sprint 4 introduced `H1 ATR(14) × multiplier` with a floor of 200 points. The multiplier is regime-routed via `RegimeParams.sl_atr_multiplier` in Sprint 6. This is now the core SL mechanism and should not be bypassed.

### 8. Per-timeframe swing_length (Sprint 5)

The default `swing_length=10` on all TFs caused excessive swing noise on D1. Sprint 5 calibrated: D1=5, H4=7, H1/M15=10. This is baked into `MultiTimeframeAggregator._DEFAULT_SWING_LENGTH_MAP` and is injected automatically when no `swing_length_map` is passed to `SMCDetector`.

---

## Sprint History

### Phase 0 Foundation (commit `4ba3713`)
- XAUUSD data schemas, `ForexDataLake` Parquet query, `SMCDetector` with all 6 sub-detectors
- `MultiTimeframeAggregator` v1 skeleton, `BarBacktestEngine`, walk-forward OOS

### Phase 1 Strategy + CLI (commit `4351882`)
- `MultiTimeframeAggregator` full pipeline, confluence scoring (5-factor weighted)
- Typer CLI (`smc ingest`, `smc detect`, `smc backtest`, `smc live`, `smc health`)
- First Gate 1 OOS run

### Sprint 0–2 Tuning + Gate v2/v3 (commit `864ce9c`)
- Fixed SL computation bugs, swing_length sensitivity testing
- v2: ATR-SL + swing_length → PF 0.54. v3: regime filter + zone cooldown

### Sprint 3 Regime Filter + Gate v4 (commit `04bb82f`)
- ATR-based regime classifier (`classify_regime`): trending / transitional / ranging
- Zone cooldown 24h after loss

### Sprint 4 Data-Driven Calibration + Gate v5 (commit `6fed21c`)
- H1 ATR(14) adaptive stop loss (biggest individual PF lift: 0.54 → 0.83)
- Per-TF swing_length map
- Gate v5: PF 0.83, WR 38.1%

### Sprint 5 Surgical Fixes + Gate v6 (commit `c3e5914`)
- Three surgical fixes: disable ob_test, raise transitional floor, zone anti-cluster
- Gate v6: PF 1.05, WR 42.9% — first PF > 1.0 achievement

### Phase 2 Paper Trading Infrastructure (commit `b2fba05`)
- Risk manager: `PositionSizer`, `DrawdownGuard`, exposure checks
- MT5 execution: `BrokerPort` protocol, `MT5BrokerPort`, `SimBrokerPort`
- Live loop: `LiveLoop`, `HealthMonitor`, `TradeJournal`, `TelegramAlerter`

### Sprint 6 AI Regime Classification + Gate v7 (commit `7842a87`)
- 5-regime AI classifier: TREND_UP/DOWN, CONSOLIDATION, TRANSITION, ATH_BREAKOUT
- 7-agent debate pipeline (4 Analysts + Bull + Bear + Judge via Claude CLI)
- `RegimeParams` routing: per-regime SL multiplier, TP1 RR, allowed triggers, allowed directions
- Pre-computed `regime_cache.parquet` for backtest performance
- Gate v7: PF 1.02 — AI regime matches v6 performance without degrading it

### Sprint 7 v2 Strategy (AI Direction + Dual Entry) (commit `e0549c9`)
- `AggregatorV2`: DirectionEngine (AI direction), dual entry (normal + inverted), 6 triggers
- Relaxed parameters: 8h cooldown, 0.35 threshold, max 5 concurrent
- Gate v8 v2 test: PF 0.60–0.80 (FAIL)

### Sprint 8 v1 vs v2 Walk-Forward + Critical Findings (commit `f487cbf`)
- Side-by-side v1 vs v2 OOS across all windows
- Finding: v1 dominates in every comparable window
- `fvg_sweep_continuation` appears promising but needs investigation
- Critical finding: `fvg_fill_in_zone SHORT` is consistently 33% WR (STABLE_BAD)

### Sprint 9 Hybrid v3 + 3D Worst-Indicator Scan (commit `c8e7709`)
- `AggregatorV3`: v1 base + fvg_sweep bolt-on
- 3D worst-indicator matrix (`worst_indicator_scan.py`): 447 trades × 88 windows
- Finding: fvg_sweep WR collapses under v1 tight SL (v2 artifact confirmed)
- Finding: `fvg_fill_in_zone SHORT` STABLE_BAD across 82 trades, 5 versions

### Sprint 10 v1.1 Inversion Test + VPS Deployment (commits `41ad087`, `0d8ae03`)
- `AggregatorV1_1`: v1 + fvg_fill SHORT → LONG inversion in non-trending
- Gate v10: v1 baseline confirmed PF 1.09–1.66, v1.1 mixed (insufficient windows)
- VPS deployment: Windows Server, TMGM Demo MT5, `scripts/live_demo.py` running
- `docs/VPS_DEPLOYMENT.md` complete deployment guide

---

## Key Files Reference

Files in priority order for a new developer:

### 1. `src/smc/strategy/aggregator.py`
The production strategy orchestrator. `MultiTimeframeAggregator.generate_setups()` is the single entry point for all signal generation. Read this first to understand the full D1→H4→H1→M15 cascade, zone filtering, anti-clustering, and regime-gating logic.

### 2. `src/smc/smc_core/detector.py`
The `SMCDetector` class. Shows how the 6 sub-detectors are composed into a single `detect()` call that returns a frozen `SMCSnapshot`. Understand this before reading the sub-detectors individually.

### 3. `src/smc/ai/regime_classifier.py`
The `classify_regime_ai()` function and the three-tier fallback chain: cache → AI debate → ATR fallback → default. The `RegimeContext` dataclass is the feature snapshot fed to both paths. The `_atr_fallback()` function is what actually runs in production (v1, `ai_enabled=False`).

### 4. `src/smc/ai/debate/pipeline.py`
The 7-agent Claude debate pipeline. Three phases: (1) 4 domain-isolated Analysts (Sonnet), (2) Bull + Bear Researchers with N rounds (Opus), (3) Judge arbiter (Opus). Currently inactive in production but fully tested. Supports both Claude CLI and Anthropic API backends.

### 5. `src/smc/backtest/engine.py`
The `BarBacktestEngine`. Bar-by-bar M15 simulation with strict no-lookahead semantics. Processing order per bar: (1) check exits, (2) process new setups, (3) apply fills, (4) update equity curve. The `on_sl_hit` callback wires zone cooldown into the backtest engine.

### 6. `src/smc/monitor/live_loop.py`
The `LiveLoop` class. Asyncio-based M15 bar-close polling. The `BrokerPort`, `DataFetcher`, and `StrategyRunner` are injected via constructor — this is the dependency injection boundary between the strategy layer and the live trading infrastructure.

### 7. `scripts/live_demo.py`
The actual VPS launcher. Simplified standalone script that bypasses the full `LiveLoop` class. Reads from MT5 directly, calls `aggregator.generate_setups()`, writes to JSONL journal. This is what is running on the VPS.

### 8. `scripts/run_v7_single_window.py`
The gate runner that validates a single OOS window. Shows how to wire together `ForexDataLake`, `MultiTimeframeAggregator`, `FastSMCStrategyAdapter`, `BarBacktestEngine`, and `RegimeCacheLookup`. Use this as a template when writing new gate runners.

### 9. `src/smc/config.py`
`SMCConfig` — all application settings as a Pydantic BaseSettings model. Every tunable parameter lives here with description, defaults, and validators. Cross-reference with `config/smc_config.yaml` for the YAML defaults.

### 10. `src/smc/ai/models.py`
All frozen Pydantic types for the AI layer: `MarketRegimeAI` (5-value literal), `RegimeParams` (per-regime parameter preset), `AIRegimeAssessment` (primary output), `AnalystView`, `DebateRound`, `JudgeVerdict`, `ExternalContext`. These are the data contract between the AI and strategy layers.

---

## Known Issues and Tech Debt

### GBK Encoding on Windows VPS

`Rich` console output can trigger GBK encoding errors on Windows VPS when writing non-ASCII characters. The live demo script works around this by setting `os.environ["NO_COLOR"] = "1"` and `os.environ["PYTHONIOENCODING"] = "utf-8"` at startup. If you add Rich-based display code, keep these settings.

### Zone Cooldown Is a No-op in Backtest

`record_zone_loss()` uses `datetime.now(tz=timezone.utc)` internally for the cooldown calculation, but the backtest engine passes `bar_ts` (a historical timestamp) to `on_sl_hit`. This means the cooldown comparison (`now < cooldown_until`) always evaluates to False in the backtest — zone cooldown is not actually applied in historical simulation. This was identified in Sprint 5 and left as tech debt because the anti-clustering (`_active_zones`) does work correctly.

Fix: thread `bar_ts` through `record_zone_loss()` and use it as the reference time instead of `datetime.now()`.

### fvg_sweep_continuation Code Is Not Deployed

`_find_fvg_sweep_continuation` in `src/smc/strategy/entry_v2.py` and `AggregatorV3` in `aggregator_v3.py` exist in the codebase but are NOT production. The Sprint 9 investigation proved the signal was a parameter artifact from v2's relaxed SL. Do not re-enable without running a full OOS validation.

### v2 and v3 Modules Are Preserved for Reference Only

`aggregator_v2.py`, `aggregator_v3.py`, `aggregator_v1_1.py`, `entry_v2.py`, `confluence_v2.py`, `zone_scanner_v2.py` are all kept for reference and are tested, but none are production. When looking at import trees, do not assume these modules are active.

### Batch Architecture Prevents Stateful Backtest Features

`FastSMCStrategyAdapter` (`src/smc/backtest/adapter_fast.py`) pre-computes all setups in a single batch pass before the `BarBacktestEngine` processes them. This means stateful features like zone cooldown and anti-clustering are computed once at training time rather than incrementally per bar. For accurate backtest results, stateful features would require the engine to call the aggregator once per bar — a performance trade-off that has not been addressed.

### MT5 Account Credentials in .env

Ensure `.env` is in `.gitignore` (it is). Never commit real MT5 credentials. The production `.env` on the VPS contains the TMGM Demo account credentials and should be treated as sensitive even for paper money accounts.

### No Real Order Submission in live_demo.py

The standalone demo launcher logs setups to JSONL but does NOT call `broker.send_order()`. Full order execution requires wiring the `OrderManager` from `src/smc/execution/order_manager.py` into the loop. This is intentional for the paper phase — do not add live order submission until Gate 3 (live PF ≥ 1.3) criteria are met.

---

## Future Roadmap

These are the agreed-upon next steps in priority order as of Sprint 10:

### Step 1: Collect Live Demo Data (2 weeks minimum)

The VPS demo is running. Let it accumulate at least 2 weeks of M15 cycle logs before making any changes. Watch for:
- Are setups being generated at the expected frequency? (Expect 5–15 per month based on OOS)
- Are the `fvg_fill_in_zone` triggers dominating? (Expected from backtesting)
- Are there any infrastructure issues? (GBK errors, MT5 disconnects, missed bar closes)

Journal location on VPS: `C:\AI-SMC\data\journal\live_trades.jsonl`

### Step 2: Validate Live vs. Backtest Consistency

After 2 weeks, compare the live signal characteristics against backtest expectations:
- Direction distribution (was roughly 60% long in OOS)
- Trigger type distribution (expected ~90% fvg_fill_in_zone)
- Confluence score distribution (expected 0.6–0.9 range)
- Regime at signal time vs. expected

Any significant divergence needs investigation before enabling AI or expanding risk.

### Step 3: Enable AI Regime in Shadow Mode

Enable `SMC_AI_REGIME_ENABLED=1` in the live loop with the debate pipeline running but **not filtering setups** — log the AI regime assessment alongside each setup to compare AI regime vs. ATR regime judgments over real market conditions.

This requires: Claude CLI installed on VPS (`winget install Anthropic.Claude`), or `SMC_ANTHROPIC_API_KEY` set for API fallback.

Recommendation: Run in shadow mode for 2 weeks before using AI regime to gate actual signal generation.

### Step 4: v1.1 Inversion Live Validation

The Sprint 10 backtesting of v1.1 (`fvg_fill SHORT → LONG inversion in non-trending`) showed mixed results with insufficient OOS windows. With enough live data (50+ triggered inversions), validate whether the theoretical 66.6% inverted WR holds in live conditions.

Switch to `AggregatorV1_1` only if live inversion win rate exceeds 55% over statistically meaningful sample.

### Step 5: Gate 3 — Live PF ≥ 1.3 → Real Money

Once live paper trading over 60+ days shows:
- Profit factor ≥ 1.3 (conservative OOS baseline)
- Win rate ≥ 38% (lower bound of v6 OOS)
- Max drawdown < 5% (half of the configured limit)
- No infrastructure failures lasting > 1 bar

Then consider switching to a real money account with the minimum lot size (0.01 = 1 micro lot, ~$0.10 per point risk per 1% equity on $1000 account).

---

## Team Structure Used

This project was built using a T3 (Lead → Teammates → Subagents) agent orchestration:

- **Lead (Claude Opus 4.6)**: Decomposed sprints into domain tasks, defined acceptance criteria (Gate system), arbitrated cross-domain decisions, performed final integration review
- **Teammates (Claude Opus 4.6)**: Domain experts (data layer, strategy, AI, execution, testing) who coordinated via `SendMessage` for interface contracts and reported completions with Gate verification
- **Execution Subagents (Claude Sonnet 4.6)**: Wrote module implementations, test suites, refactoring tasks as dispatched by teammates

### Cross-Review Protocol

Every module was reviewed by a DIFFERENT agent from the one that wrote it. Code quality enforcement:
- No self-review: writer and reviewer were always different agents
- CRITICAL findings blocked merge; HIGH findings were addressed; MEDIUM tracked as debt

### Gate System

Quantitative pass/fail criteria before proceeding to the next sprint:
- **Gate 1**: OOS profit factor > 1.0 (achieved in Sprint 5 with v6, PF 1.05)
- **Gate 2**: PnL positive across full walk-forward (achieved in Sprint 6 with v7c)
- **Gate 3** (pending): Live trading PF ≥ 1.3 over 60+ days → real money authorization

### Debate Protocol for Major Decisions

Architectural decisions used 3+ perspective debate:
- Implementation approach debates: at least a skeptic, an advocate, and a senior engineer perspective
- Backtest result interpretation: factual reviewer + inferential analyst + risk perspective
- Feature inclusion decisions: performance impact + code complexity + maintenance cost

---

*Last updated: Sprint 10, April 2026. VPS demo running on 43.163.107.158.*
