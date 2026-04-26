# R-Round Promotion Gate (R10 P4.2)

A statistical decision rule that gates whether a backtested **treatment**
arm should be promoted from research to live (Demo → Real). Established in
R10 P4.2 to replace eyeballed-PF-difference judgements with a reproducible,
sample-size-aware criterion.

## The Rule

A treatment is **PROMOTE-eligible** if and only if all three conditions hold:

1. `n_baseline >= 30` AND `n_treatment >= 30` — minimum sample size in both arms.
2. `PF_diff_95%CI_lower_bound > 0` — the 95% bootstrap confidence interval on
   `(PF_treatment - PF_baseline)` lies entirely above zero.

If either condition fails, the verdict is **HOLD**: do not flip the live
flag, run more data, or revisit the design.

### What is **NOT** in the rule

- **Win-rate chi-square**: included in the verdict's rationale string for
  operator visibility but **does not** participate in the PROMOTE/HOLD
  decision. A treatment with materially better PF can have a non-significant
  WR shift if the edge comes from larger wins / smaller losses rather than
  win frequency. Forcing chi-square significance would over-reject.
- **Total PnL**: PnL pooled across years can hide regime-specific failures.
  PF normalises by total losses and is more robust to single-year outliers.
- **Max drawdown**: tracked in the per-arm summary but not gated here. DD
  is downstream of trade size; the gate decides whether the *strategy* is
  promotable, leaving size to live risk modules.

## How the bootstrap works

We use **paired-resample bootstrap** on PnL trade vectors:

```python
for i in 1..n_iter:
    b_resample = sample_with_replacement(pnl_baseline, len(pnl_baseline))
    t_resample = sample_with_replacement(pnl_treatment, len(pnl_treatment))
    diffs[i] = profit_factor(t_resample) - profit_factor(b_resample)

ci_lo = quantile(diffs, 0.025)
ci_hi = quantile(diffs, 0.975)
```

`n_iter=10_000` by default. Iterations that produce `PF=inf` in either arm
(zero losses on the resample) are excluded from the quantile computation.

The RNG is seeded (`rng_seed`, default 0) so verdicts are deterministic
across machines and across time. Re-running the gate on the same trade
vectors gives the same CI bounds — important for reproducibility when the
verdict is cited in commit messages.

### Edge cases

| Situation | Verdict | Rationale |
| --- | --- | --- |
| `n < min_n` in either arm | HOLD | "insufficient sample" |
| `PF=inf` in either arm (zero losses) | HOLD | "PF undefined" |
| CI crosses 0 | HOLD | "edge not significant" |
| Empty trade list in either arm | HOLD | n=0 hits min_n check |

## How to invoke

After any A/B backtest run that populates `per_year[year][arm].trades`:

```bash
python scripts/backtest_mode_router_ab.py --years=2020-2024
# stdout includes:
#   [GATE] n=B/T=NNN/MMM PF_diff=+0.XX 95%CI=[+a,+b] WR_chi2_p=Y.YY (informational) -> PROMOTE
#   [GATE] rationale: PROMOTE: PF_diff=+X.XXX 95%CI=[...] excludes 0; n_baseline=NNN, n_treatment=MMM both >= min_n=30. ...
```

The full markdown backtest report (e.g. `data/round7_p0_2/mode_router_ab_backtest.md`)
should also reference the gate verdict in any `## Recommendation` section.

For ad-hoc analysis the module is importable:

```python
from smc.eval.promotion_gate import evaluate_promotion

verdict = evaluate_promotion(baseline_trades, treatment_trades)
if verdict.promote:
    print("ship it:", verdict.rationale)
else:
    print("hold:", verdict.rationale)
```

`baseline_trades` / `treatment_trades` are any iterables of objects with a
`pnl_usd: float` attribute (Protocol duck typing) — the dataclass from
`backtest_mode_router_ab.py` and any future A/B harness can both feed in.

## Worked example — R7 mode_router

Hypothetical R7 numbers (illustrative; actual data may differ):

```
Pooled across 2020-2024:
  baseline:  n=148, PF=1.12, WR=53%
  treatment: n=152, PF=1.31, WR=55%

bootstrap_pf_diff_ci(n_iter=10_000, seed=0)
  → CI = [+0.04, +0.34]
chi_square_winrate(53%, 55%) → p=0.74 (NOT significant)

Decision tree:
  n_baseline=148 >= 30        ✓
  n_treatment=152 >= 30       ✓
  CI lower 0.04 > 0           ✓
  → PROMOTE
  (chi2_p=0.74 informational; WR shift small but PF edge is real)
```

A treatment that achieves PF improvement primarily through larger wins
(rather than higher win frequency) will pass this gate with chi-square
non-significant — and that's a feature, not a bug.

## Why these specific thresholds

- **min_n=30**: minimum sample size at which the bootstrap CI estimate is
  reasonably stable. Below 30, resamples are too sparse and CI bounds
  oscillate.
- **95% CI**: standard for hypothesis testing in finance literature; a 90%
  CI would PROMOTE on weaker signals (more false positives), 99% would
  reject too many genuine-but-modest edges.
- **lower bound > 0** (not `>= 0`): strict inequality — we want positive
  evidence, not "at least no evidence of harm".

## Maintenance

- Implementation lives at `src/smc/eval/promotion_gate.py`.
- Unit tests at `tests/smc/unit/eval/test_promotion_gate.py` (15 cases
  including PF-inf, seed determinism, ASCII-safe rationale, golden R7).
- All future A/B harnesses (`backtest_range_ai_gates_ab.py` and beyond)
  should call `evaluate_promotion()` at the end of their main loop and
  print the `[GATE]` line.
- If thresholds change, document the rationale in this file and bump
  the version note below.

**Version**: 1.0 (R10 P4.2, Apr 2026)
