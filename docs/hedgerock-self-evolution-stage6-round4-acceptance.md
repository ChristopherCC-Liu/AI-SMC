# Stage 6 Round 4 — Acceptance Addendum (Tasks 1–5)

> Continues `docs/hedgerock-self-evolution-stage6-round3-acceptance.md`.
> Hardens edge-case handling, ships a config validator + metrics
> exporter, formalises the regression guard, and writes the
> evolution-layer onboarding README.

## 1. Roll-up

| Task | Deliverable | Tests added |
|---|---|---|
| 1 | `tests/hedgerock/evolution/test_graceful_degradation.py` (probes 13 edge cases — no production change required) | 13 |
| 2 | `scripts/hedgerock_evolution_validate_config.py` + tests | 11 |
| 3 | `scripts/hedgerock_evolution_metrics_export.py` (`metrics/v0` JSON) + tests | 10 |
| 4 | `tests/hedgerock/evolution/test_regression_guard.py` (meta-scan over evolution layer) | 5 |
| 5 | `src/smc/hedgerock/evolution/README.md` (5-min onboarding) | 0 (docs) |

Test growth (evolution sub-suite): **587 → 626** (delta +39).
Test growth (full hedgerock): **1423 → 1462 passed**.

## 2. Files added

| Path | Purpose |
|---|---|
| `tests/hedgerock/evolution/test_graceful_degradation.py` | 13 edge cases (malformed JSONL, missing files, non-JSON audit log, garbled bytes, missing argv) |
| `scripts/hedgerock_evolution_validate_config.py` | Config validator CLI |
| `tests/hedgerock/evolution/test_validate_config_cli.py` | 11 validator tests |
| `scripts/hedgerock_evolution_metrics_export.py` | `metrics/v0` JSON snapshot |
| `tests/hedgerock/evolution/test_metrics_export.py` | 10 export tests |
| `tests/hedgerock/evolution/test_regression_guard.py` | Meta-scan: no live runtime imports, no `.mq5` code calls, ≥ 30 files in scope |
| `src/smc/hedgerock/evolution/README.md` | Onboarding doc |

## 3. Red-line invariants — re-checked at end of round 4

| Path | mtime epoch (start of session) | mtime epoch (end of round 4) | Touched? |
|---|---|---|---|
| `src/smc/hedgerock/rule_engine.py` | 1777604440 | 1777604440 | **No** |
| `src/smc/hedgerock/decision_server.py` | 1777607670 | 1777607670 | **No** |
| `src/smc/hedgerock/phase_d_walk_forward.py` | 1777609385 | 1777609385 | **No** |
| `mql5/AISMCReceiver.mq5` | 1776754859 | 1776754859 | **No** |
| `policy_registry/approved/` | (does not exist) | (does not exist) | **No** |
| `policy_registry/pointer.json` | (does not exist) | (does not exist) | **No** |
| `config/safety_bounds.yaml` | (does not exist) | (does not exist) | **No** |
| `policy_registry/shadow_artefacts/*.json` count | 12 | 12 | **No mutation** |

Production registry total JSON count: **21 → 21**.

## 4. Behavioural contracts pinned this round

- **Graceful degradation.** Malformed JSONL lines are skipped in
  the queue, ledger, and audit trail; valid lines still flow.
  Garbled audit log → `audit_log_present=true`, no parsed
  violation, report still renders. Missing argv → `SystemExit(2)`
  via argparse, never a stack trace.
- **Pre-deploy config validation.** `hedgerock_evolution_validate_config.py`
  catches missing targets, `lo > hi`, non-numeric ranges,
  non-list values; surfaces `SAFETY_CLAMPS` disagreement as WARN
  by default and FAIL under `--strict`. Read-only; deterministic.
- **Metrics snapshot.** `metrics/v0` schema with stable shape:
  `registry_audit / queue / paper_test / shadow_artefacts /
  candidates`. JSON-strict (no NaN), inputs unchanged after run.
- **Regression guard.** Meta-test asserts NO file under
  `evolution/` or `scripts/hedgerock_evolution_*.py` imports
  `rule_engine` / `decision_server` / `phase_d_walk_forward`.
  Floor of 30 files in scope; current scan covers 35
  (27 src + 8 scripts).
- **Onboarding doc.** `src/smc/hedgerock/evolution/README.md`
  documents architecture, file layout, how to add a new gate /
  candidate / CLI, and the cardinal rule.

## 5. Commits

```
18821db test(stage-6/round-4/task-1): graceful degradation across modules + CLIs
a366744 feat(stage-6/round-4/task-2): config validator CLI
a4e1e09 feat(stage-6/round-4/task-3): metrics dashboard JSON export (metrics/v0)
2e76a1d test(stage-6/round-4/task-4): regression guard meta-test for evolution layer
```

## 6. Final standing (after round 4)

- **626 tests** passing in `tests/hedgerock/evolution/`.
- **1462 tests** passing in `tests/hedgerock/` overall (no
  regression outside the evolution sub-suite).
- All four production-code mtimes unchanged.
- Production registry shadow-artefact JSON count unchanged (12).
- `policy_registry/approved/`, `pointer.json`,
  `config/safety_bounds.yaml` all still absent.

The evolution layer is now self-defending: a single meta-test
(`test_regression_guard.py`) catches future drift in addition to
each module's own isolation test, the config validator stops
deployment-time misconfiguration, and the metrics exporter gives
operators a structured view they can plug into any dashboard.
