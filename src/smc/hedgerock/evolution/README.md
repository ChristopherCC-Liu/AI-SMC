# `smc.hedgerock.evolution` — Self-Evolution Sidecar (READ-ONLY)

> Five-minute onboarding for new contributors. Architecture, file
> layout, how to run tests, and how to extend the loop without
> breaking the red-line invariants.

## TL;DR

This package is the **report-only** half of HedgeRock. It looks at
the live system's evidence and emits *recommendations*. It never
modifies live code, never promotes a candidate, never deploys. Every
public CLI is a sidecar that writes to operator-supplied paths.

If a change you're about to make would touch any of:

- `src/smc/hedgerock/rule_engine.py`
- `src/smc/hedgerock/decision_server.py`
- `src/smc/hedgerock/phase_d_walk_forward.py`
- `mql5/*.mq5`
- `policy_registry/approved/`, `policy_registry/pointer.json`
- `policy_registry/shadow_artefacts/<id>/*.json` (delete / rewrite)
- `config/safety_bounds.yaml`

…stop. The change does not belong here. The five-layer state
machine ([RFC §3](../../../../docs/hedgerock-self-evolution-rfc.md))
ends at `HUMAN_APPROVE`, and that step is intentionally manual.

## Architecture in one diagram

```
                        Phase D bundle (atlas, availability, walk-forward)
                                    + registry _audit.md
                                            │
                                            ▼
                 ┌────────────────────────────────────────────┐
                 │        evolution-report CLI (T4-F3)        │
                 │  scripts/hedgerock_evolution_report.py     │
                 └────────────────────────────────────────────┘
                                            │ candidates + gate results
                                            ▼
                 ┌────────────────────────────────────────────┐
                 │      candidate_generator (Stage 3)         │
                 │   evolution/candidate_generator.py         │
                 └────────────────────────────────────────────┘
                                            │ CandidateProposal
                                            ▼
                 ┌────────────────────────────────────────────┐
                 │      recommendation CLI (Stage 4)          │
                 │  scripts/hedgerock_evolution_recommend.py  │
                 └────────────────────────────────────────────┘
                                            │ RECOMMEND proposals
                                            ▼
                 ┌────────────────────────────────────────────┐
                 │     shadow-test queue (Stage 5)            │
                 │  evolution/shadow_test_queue.py            │
                 └────────────────────────────────────────────┘
                          │                              │
       (ages out → STALE) │                              │ (paper trades)
                          ▼                              ▼
   evolution/queue_aging.py            evolution/paper_test_ledger.py
                                                          │
                                                          ▼
                                        evolution/replay_validator.py
                                          ──→ heuristic projections
                                                          │
                                                          ▼
                              scripts/hedgerock_evolution_promote.py
                                  (DRY-RUN packet, manual approval)
                                                          │
                                                          ▼
                             HUMAN_APPROVE → live (manual, out of scope)
```

Cross-cutting:

- **Operation audit trail** (`evolution/operation_audit.py`)
  records every CLI run's per-stage success/failure with operator
  attribution.
- **Metrics export** (`scripts/hedgerock_evolution_metrics_export.py`)
  emits a `metrics/v0` JSON snapshot for dashboards.
- **Config validator**
  (`scripts/hedgerock_evolution_validate_config.py`) checks
  `config/safety_bounds_template.yaml` against `SAFETY_CLAMPS` /
  the menu before deployment.
- **End-to-end demo** (`scripts/hedgerock_evolution_demo.py`)
  orchestrates the entire chain into one command.

## File layout

| Path | Role |
|---|---|
| `evolution/policy_manifest.py` | Frozen dataclasses + JSON IO for `CandidateManifest`, `EvidenceBundle`, `PromotionGateResult`, `GateStatus`, `OverallResult`. |
| `evolution/candidate_menu.py` | The hand-curated `CANDIDATE_MENU_V0` (4 entries). |
| `evolution/promotion_gates.py` | G1–G8 implementations. G6 reads `SafetyBoundsConfig`; G8 enforces the registry append-only contract. |
| `evolution/registry_audit.py` | Loads `_audit.md` into a `RegistryAuditState`. |
| `evolution/policy_registry.py` | Append-only writer for candidate manifests + audit entries. |
| `evolution/evidence_bundle.py` | Loads atlas/availability/walk-forward + computes hashes. |
| `evolution/candidate_generator.py` | **Stage 3.** RECOMMEND / NO_RECOMMENDATION proposals with safety clamps. |
| `evolution/shadow_test_queue.py` | **Stage 5.** Append-only JSONL queue of RECOMMEND proposals. |
| `evolution/queue_aging.py` | STALE marker pass; idempotent. |
| `evolution/paper_test_ledger.py` | XAUUSD-only paper-trade ledger. |
| `evolution/replay_validator.py` | Read-only aggregator over `shadow_artefacts/`. |
| `evolution/ascii_visualisations.py` | Parameter table, gate matrix, heat ranking. |
| `evolution/operation_audit.py` | Cross-CLI operator action trail. |
| `evolution/shadow_runner.py`, `evolution/multi_window_report.py`, `evolution/policy_overlay.py`, … | Existing shadow-runner pipeline (write artefacts to `shadow_artefacts/`; documented in T4-F2/F3 reports). |

CLIs (all under `scripts/hedgerock_evolution_*.py`):

| CLI | Purpose |
|---|---|
| `hedgerock_evolution_report.py` | Run G1–G8 against current evidence; render the diagnostic report. |
| `hedgerock_evolution_recommend.py` | Add candidate generator → recommendation report with visualisations. |
| `hedgerock_evolution_queue_inspect.py` | Read-only snapshot of the queue. |
| `hedgerock_evolution_queue_age.py` | Append STALE markers for aged-out queue entries. |
| `hedgerock_evolution_promote.py` | Dry-run promotion packet. Optional `--checklist-mode` requires 7 ack flags. |
| `hedgerock_evolution_validate_config.py` | Pre-deploy validation of `safety_bounds_template.yaml`. |
| `hedgerock_evolution_metrics_export.py` | JSON snapshot for dashboards. |
| `hedgerock_evolution_demo.py` | End-to-end orchestrator. |

## Running the tests

```
python -m pytest tests/hedgerock/evolution/ -q
```

Expect ~626+ tests, < 3 seconds. The broader hedgerock suite
(`tests/hedgerock/ -q`) is ~1450+ tests in ~13 seconds.

Three test files act as anchors when reviewing changes:

| File | What it pins |
|---|---|
| `test_regression_guard.py` | No file under `evolution/` or `scripts/hedgerock_evolution_*.py` imports the live runtime. Failure here means a red-line violation got past review. |
| `test_artefact_registry_append_only.py` | No production module deletes shadow artefacts. |
| `test_real_registry_smoke.py` | The whole pipeline runs against the real production registry without mutating any artefact. |

## How to add a new gate

1. Add a `g9_<name>(*, candidate, bundle, ...)` function in
   `evolution/promotion_gates.py` returning a
   `PromotionGateResult(gate_id="G9", ...)`.
2. Append `"G9"` to the `GATE_IDS` tuple in the same file.
3. Wire it into `evaluate_all_gates`.
4. Decide whether `compute_overall_result` should treat G9
   FAIL as a hard block or a soft warning; update accordingly.
5. Add tests in
   `tests/hedgerock/evolution/test_promotion_gates.py`. Cover
   PASS, FAIL, ABSTAIN, and any NOT_RUN paths.
6. Update `_GATE_IDS` in `evolution/ascii_visualisations.py` so
   the gate matrix renders the new column.
7. Update the RFC + the operator runbook to mention what G9
   checks and what failure modes the operator should expect.
8. Re-run the full hedgerock suite — if anything outside
   `evolution/` regresses, your change leaked into the live
   layer; back it out.

## How to add a new candidate type

1. Add a new entry to `CANDIDATE_MENU_V0` in
   `evolution/candidate_menu.py`. Use the `_candidate(...)`
   helper to enforce shape consistency.
2. Add the dotted target to `_PARAMETER_CLASS_BY_TARGET` and a
   matching entry in `SAFETY_CLAMPS` inside
   `evolution/candidate_generator.py`.
3. Add the band to `config/safety_bounds_template.yaml`.
4. Run `scripts/hedgerock_evolution_validate_config.py`; it
   should PASS.
5. Run the recommendation CLI; the new candidate should appear
   in the report with either RECOMMEND or a stable NO_RECOMMEND
   reason.
6. Bump `CANDIDATE_MENU_V0` tests if they enforce a count (most
   don't).
7. Decide whether the candidate raises gross exposure / leverage
   / etc. If yes, the generator's RFC §10.1 veto will keep it at
   `parameter_class_unsupported` — which is the correct, safe
   default.

## How to add a new CLI

1. Drop the script under `scripts/hedgerock_evolution_<name>.py`.
2. Inherit the standard guardrails:
   - Reject `--<output>` paths under `policy_registry/approved/`
     or `policy_registry/pointer.json` (use the
     `_FORBIDDEN_PATH_FRAGMENTS` pattern).
   - No imports of `rule_engine`, `decision_server`, or
     `phase_d_walk_forward`.
   - When the CLI logs operator actions, use
     `operation_audit.append_operation`.
3. Add a test file `tests/hedgerock/evolution/test_<name>.py`
   covering: happy path, missing/malformed inputs, forbidden
   output path, source-level isolation grep.
4. The regression guard
   (`test_regression_guard.py::test_regression_guard_scans_at_least_30_files`)
   may need its threshold bumped — but only after the test runs
   green; do not lower the floor.

## What to read after this

- [`docs/hedgerock-self-evolution-rfc.md`](../../../../docs/hedgerock-self-evolution-rfc.md)
  — five-layer state machine + invariants.
- [`docs/hedgerock-evolution-report-runbook.md`](../../../../docs/hedgerock-evolution-report-runbook.md)
  — operator runbook for the report CLI.
- [`docs/hedgerock-30day-paper-trading-plan.md`](../../../../docs/hedgerock-30day-paper-trading-plan.md)
  — Day-by-day PAPER_TEST protocol.
- [`docs/hedgerock-shadow-broker-logging-design.md`](../../../../docs/hedgerock-shadow-broker-logging-design.md)
  — how the demo broker should record paper trades.
- [`docs/hedgerock-self-evolution-stage6-acceptance.md`](../../../../docs/hedgerock-self-evolution-stage6-acceptance.md)
  + the round-2 / round-3 / round-4 addenda — what each stage
  shipped and what tests pin which contract.

## Cardinal rule

> **The evolution layer never modifies the live layer.** The day a
> sidecar module imports `rule_engine`, the regression guard fails,
> the per-module isolation tests fail, and the build is red until
> the import is removed. Don't bypass the guards; replace your
> approach.
