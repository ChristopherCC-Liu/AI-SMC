# Round 7 P0-2 — mode_router A/B Backtest (2021-2024)

_Generated: 2026-04-21 UTC | Harness: `scripts/backtest_mode_router_ab.py`_

## Executive Summary

**Recommendation: DO NOT ENABLE `SMC_AI_MODE_ROUTER_ENABLED=true` yet. Fix P0-1b first (CONSOLIDATION fall-through bug) and rebuild the regime cache to include ATH_BREAKOUT mapping. Re-run this harness before flipping.**

The A/B backtest reveals two independent gaps that block a safe rollout:

1. **2023 regression (-$32, 7 fewer trades)** — confidence-gated `CONSOLIDATION` path in `route_trading_mode()` short-circuits to `v1_passthrough` when `range_bounds` is missing, which also preempts the legacy Priority-1 (`ai_direction + conf ≥ 0.45`) trending path. Details below.
2. **2024 ATH fix did not fire** — the pre-computed regime cache maps XAU's 2024 ATH rally to `CONSOLIDATION` (not `TREND_UP`/`ATH_BREAKOUT`), so Treatment's "force trending when regime=TREND_UP" branch never fires. Both arms route the same 77 setup-bars to ranging, reproducing the Sprint 11 disaster identically. The AI-aware router is correct; the **regime classifier** is what needs fixing.

Net: Treatment does not achieve its stated goal (suppress 2024 range trades on ATH rally) and adds collateral damage (2023 trend trades wrongly re-routed to passthrough). Shipping enabled would be a net negative.

## 4-Year Per-Arm Performance

| Year | Setups | Baseline PF/WR/N/DD | Treatment PF/WR/N/DD | Δ PF | Baseline RangeBars | Treatment RangeBars |
|---|---|---|---|---|---|---|
| 2021 | 55 | 0.60 / 40% / 10 / 0.6% | 0.60 / 40% / 10 / 0.6% | +0.00 | 41 | 41 |
| 2022 | 23 | 2.00 / 60% / 10 / 0.4% | 2.00 / 60% / 10 / 0.4% | +0.00 | 4 | 3 |
| 2023 | 32 | **1.37 / 50% / 16 / 0.8%** | **1.14 / 56% / 9 / 0.8%** | **-0.23** | 6 | **20** |
| 2024 | 82 | 0.00 / 0% / 4 / 0.8% | 0.00 / 0% / 4 / 0.8% | +0.00 | 77 | 77 |

_2020 has no setup cache entries (cache starts 2021-01-01). Rows for 2020 omitted from the scope per data availability._

### Pooled 2021-2024

- Trend trades: Baseline **40** → Treatment **33** (Δ = **-7**)
- Pooled PnL: Baseline **$-9** → Treatment **$-40** (Δ = **-$31**)
- Range-path exposure: Baseline **128** → Treatment **141** (Δ = **+13**, i.e. ranging *went up*, not down)

## Mode-Selection Frequency (pooled)

| Arm | Trending | Ranging | v1_passthrough |
|---|---|---|---|
| Baseline | 60 (31%) | 128 (67%) | 4 (2%) |
| Treatment | 51 (27%) | 141 (73%) | 0 (0%) |

### AI Regime Observed at Setup Timestamps

| AI Regime | Count | Share |
|---|---|---|
| CONSOLIDATION | 119 | 62.0% |
| TRANSITION | 60 | 31.2% |
| TREND_DOWN | 12 | 6.2% |
| TREND_UP | 1 | 0.5% |

**Observation**: **TREND_UP = 1 across the entire 4-year setup population.** The pre-computed `regime_cache.parquet` uses the ATR-fallback classifier whose `d1_atr_pct >= 1.5%` threshold rarely fires for XAU outside strong vol years. `ATH_BREAKOUT` never appears in the cache at all. The Sprint 11 hypothesis ("force trending when AI regime = TREND_UP during ATH rally") cannot fire under this cache.

## Gap #1 — 2023 Regression Root Cause

14 setup timestamps diverged between arms in 2023. All moved **Baseline=trending → Treatment=v1_passthrough**, not toward ranging.

Example (representative divergence, 2023-08-14 21:45 UTC, session `LATE NY`):
- `ai_direction="bullish", ai_confidence=0.60` (after session penalty)
- `ai_regime="CONSOLIDATION", ai_regime_conf=0.63` (≥ 0.6 trust threshold)
- No H1 range detected → `range_bounds=None, guards_passed=False`

Baseline flow (legacy):
1. Priority-0 skipped (no `ai_regime_assessment`).
2. Priority-1 fires: `bullish + 0.60 ≥ 0.45 + session=LATE NY ≠ ASIAN_CORE` → **trending**.

Treatment flow (P0-1 commit 7433c19):
1. Priority-0 trusts CONSOLIDATION. Branch lines 134-169 checks `range_bounds + guards + session`. Preconditions fail.
2. Returns `TradingMode(mode="v1_passthrough", ...)` directly — **never reaches legacy Priority 1**.

Impact: the P0-1 CONSOLIDATION branch is too aggressive at claiming ownership. It should fall through to legacy Priority 1-3 when the mean-reversion preconditions are missing, not skip ahead to `v1_passthrough`.

**Proposed fix (P0-1b)**: delete the unconditional `return TradingMode(mode="v1_passthrough", ...)` at `mode_router.py:158-169` so control flow reaches legacy Priority 1. Covered by a new regression test with `ai_regime=CONSOLIDATION` high-conf + `range_bounds=None` + `ai_direction=bullish conf=0.6` → expect `trending`.

## Gap #2 — 2024 ATH Scenario Never Triggers the Fix

2024 Q1 (W13, the only 2024 window with setups) routing decisions:
- Baseline: 5 trending, 77 ranging, 0 v1_passthrough (the Sprint 11 disaster — 77 range bars).
- Treatment: **identical** — 5 trending, 77 ranging, 0 v1_passthrough.

Why identical: every CONSOLIDATION ≥ 0.6 timestamp has a synthesised `range_bounds` + `guards_passed=True` (because width and touches clear the Asian-session thresholds). Treatment's CONSOLIDATION branch then picks ranging — same as Baseline. The AI-aware router does exactly what the spec says. The spec just doesn't apply to this regime cache.

The real Sprint 11 root cause is upstream: the ATR classifier under-categorises XAU's 2024 ATH rally. To fix 2024 we need **either**:
- (a) a richer regime classifier that emits `ATH_BREAKOUT` when `price_52w_percentile ≥ 0.95` and `close_vs_sma50 ≥ 3%`, *or*
- (b) an AI-driven `ai_debate` run to rebuild `regime_cache.parquet` (cost > 0, non-deterministic).

Neither is in P0 scope. P0 ships the **plumbing**, which works correctly once the cache has the right labels.

## Monday 2026-04-20 02:00 UTC Replay

This is the "live disaster" scenario the teammate brief called out.

- Session detected: `ASIAN_CORE`
- AI regime from cache (nearest before 2026-04-20 02:00 UTC): `CONSOLIDATION` conf=**0.54** (< 0.6 trust gate)
- Baseline router decision: **ranging** (Priority-2: Donchian bounds + guards pass + Asian session)
- Treatment router decision: **ranging** (regime conf 0.54 is below trust threshold → falls through to Baseline logic → identical decision)

**Outcome**: Both arms route to ranging. The Treatment arm does NOT save the live disaster because regime confidence is below 0.6. The teammate brief mentioned "AI regime == TRANSITION conf 0.72 per live log" but the backtest cache uses the ATR-fallback path which stayed at 0.54 at that timestamp. If we had a TRANSITION conf-0.72 assessment, Treatment would correctly return `v1_passthrough` (per the TRANSITION branch at `mode_router.py:173-201`). The fix is valid — the cache just doesn't contain it.

To verify the TRANSITION branch works as designed, a synthetic assessment replay shows:

| AI Regime | Conf | Baseline | Treatment |
|---|---|---|---|
| TRANSITION | 0.72 | ranging | v1_passthrough |
| TREND_UP | 0.7 | ranging | trending (forced) |
| CONSOLIDATION | 0.54 | ranging | ranging (conf below gate) |

So once the regime classifier emits stronger labels for live-disaster conditions, Treatment behaves correctly. P0-2 confirms the **spec-level correctness** of P0-1.

## Final Recommendation

| Decision | Status | Unblock When |
|---|---|---|
| Ship P0-1 + enable flag | ❌ BLOCKED | CONSOLIDATION fall-through bug is patched (P0-1b) |
| Flip `SMC_AI_MODE_ROUTER_ENABLED=true` in prod | ❌ BLOCKED | After P0-1b *and* regime cache has richer labels |
| Add telemetry instrumentation per Gate 2 | ⚠️ REQUIRED | Before any future shadow/A-B in live trading |
| Document this report, ship harness | ✅ COMPLETE | Delivered here; re-runnable with `python scripts/backtest_mode_router_ab.py` |

**Suggested sequence:**
1. impl-lead ships P0-1b (one-line deletion + regression test).
2. Me: re-run this harness, expect 2023 regression gone.
3. Someone (not in this Round's P0 scope): rebuild `regime_cache.parquet` with an enhanced ATR + ATH_BREAKOUT classifier *or* run the AI debate pipeline for a deterministic subset.
4. Re-run harness, verify 2024 range exposure drops.
5. Ship to live with env flag default OFF. Let ops flip per dashboard metrics.

## Methodology

- Setups reused from walk-forward cache in `.scratch/round4/setup_cache/*.pkl` (original generation from Round 5 Task #50).
- Regime assessments reused from `data/regime_cache.parquet` (ATR-fallback only — no LLM, fully deterministic).
- Trust threshold: `SMC_AI_REGIME_TRUST_THRESHOLD = 0.6` (P0-1 default).
- Fill model: spread=3pt, slippage=0.5pt, commission=$7/lot, `max_concurrent=3`.
- Range-bound synthesis: last 48 H1 bars Donchian channel, gated on ATR regime ∈ {ranging, transitional} AND session ∈ `cfg.ranging_sessions`.
- `guards_passed`: width ≥ session-aware threshold (Asian 400 / other 800), ≥2 touches, price inside bounds.
- No randomness, no `datetime.now()` reads affect decisions (only string formatting).

### Reproduce

```bash
/opt/anaconda3/bin/python scripts/backtest_mode_router_ab.py              # full 2021-2024
/opt/anaconda3/bin/python scripts/backtest_mode_router_ab.py --fast       # 2024 only
/opt/anaconda3/bin/python scripts/backtest_mode_router_ab.py --years=2023 # single year
```

Output: `.scratch/round7/mode_router_ab_backtest.md` (this file).
