# Stage 6 Follow-up — Acceptance Addendum (Tasks 1–5)

> Continues `docs/hedgerock-self-evolution-stage6-acceptance.md`.
> All five follow-up tasks landed; the report-only loop now runs
> end-to-end via a single demo command, with live code, EA, and
> the production registry untouched.

## 1. Roll-up

| Task | Deliverable | Tests added |
|---|---|---|
| 1 | `config/safety_bounds_template.yaml` (sidecar reference) | 6 |
| 2 | `scripts/hedgerock_evolution_queue_inspect.py` | 6 |
| 3 | `src/smc/hedgerock/evolution/paper_test_ledger.py` | 10 |
| 4 | `scripts/hedgerock_evolution_promote.py` (dry-run packet) | 10 |
| 5 | `scripts/hedgerock_evolution_demo.py` (end-to-end orchestrator) | 6 |

Test growth: **490 → 528** in `tests/hedgerock/evolution/` (delta +38).

## 2. Files added

| Path | Purpose |
|---|---|
| `config/safety_bounds_template.yaml` | XAUUSD-only safety bands template (operator copies manually to `config/safety_bounds.yaml`) |
| `tests/hedgerock/evolution/test_safety_bounds_template.py` | template integration tests |
| `scripts/hedgerock_evolution_queue_inspect.py` | read-only queue snapshot CLI |
| `tests/hedgerock/evolution/test_queue_inspect_cli.py` | queue-inspect tests |
| `src/smc/hedgerock/evolution/paper_test_ledger.py` | append-only paper-test ledger |
| `tests/hedgerock/evolution/test_paper_test_ledger.py` | ledger schema tests |
| `scripts/hedgerock_evolution_promote.py` | dry-run promotion-packet CLI |
| `tests/hedgerock/evolution/test_human_promotion_helper.py` | promotion helper tests |
| `scripts/hedgerock_evolution_demo.py` | end-to-end demo orchestrator |
| `tests/hedgerock/evolution/test_evolution_demo.py` | demo tests |

## 3. Red-line invariants — re-checked end of session

| Path | mtime epoch (start of session) | mtime epoch (end of session) | Touched? |
|---|---|---|---|
| `src/smc/hedgerock/rule_engine.py` | 1777604440 | 1777604440 | **No** |
| `src/smc/hedgerock/decision_server.py` | 1777607670 | 1777607670 | **No** |
| `src/smc/hedgerock/phase_d_walk_forward.py` | 1777609385 | 1777609385 | **No** |
| `mql5/AISMCReceiver.mq5` | 1776754859 | 1776754859 | **No** |
| `policy_registry/approved/` | (does not exist) | (does not exist) | **No** |
| `policy_registry/pointer.json` | (does not exist) | (does not exist) | **No** |
| `config/safety_bounds.yaml` | (does not exist) | (does not exist) | **No** (template stayed at sidecar path) |

Production registry JSON count: **21 → 21** (no shadow_artefact mutation).

## 4. End-to-end demo evidence

`scripts/hedgerock_evolution_demo.py` was run live with:

```bash
python scripts/hedgerock_evolution_demo.py \
    --workspace /tmp/hedgerock-demo-final \
    --seed-paper-trades --produce-packet
```

Stdout summary (excerpt):

```
== Stage: OBSERVE — load Phase D evidence + audit state ==
== Stage: DETECT + RECOMMEND — run report-only recommendation CLI ==
recommendations: 4 total (3 RECOMMEND / 1 NO_RECOMMENDATION)
  c1-lower-observe-floor-0.50: RECOMMEND
  c2-halt-expiry-observe-6h: RECOMMEND
  c3-aggressive-floor-0.78: NO_RECOMMENDATION — parameter_class_unsupported
  c4-range2-conf-0.70: RECOMMEND
== Stage: QUEUE — append RECOMMEND proposals to shadow-test queue ==
  enqueued: 3
== Stage: INSPECT — read-only queue snapshot ==
queue entries: 3
== Stage: PAPER-TEST SEED — write demo paper-test ledger ==
  wrote 25 demo paper-test entries
== Stage: DRY-RUN PROMOTION — produce manual-approval packet ==
wrote promotion packet → /private/tmp/hedgerock-demo-final/promotion/packet.md
== Demo complete. ==
  real-registry json count unchanged: 21
  status: NOT LIVE / NOT APPROVED / NOT DEPLOYED
```

Artefacts produced under `/tmp/hedgerock-demo-final/`:

```
fixture/_audit.md
fixture/atlas.md
fixture/availability.md
fixture/wf.md
ledger/paper_test_ledger.jsonl
promotion/packet.md
queue/queue_inspection.md
queue/shadow_test_queue.jsonl
registry/audit/<ts>.json
registry/candidates/c1-…json
registry/candidates/c2-…json
registry/candidates/c3-…json
registry/candidates/c4-…json
report/candidate_proposals.json
report/hedgerock-evolution-recommendation.md
report/phase-d-evolution-report.md
```

c3 (`raises_gross_exposure=True`) correctly stayed at
`NO_RECOMMENDATION/parameter_class_unsupported`; the other three
candidates flowed through the full loop and produced a dry-run
promotion packet.

## 5. Pinned safety properties (test-enforced)

- **Apply mode is forbidden.** `hedgerock_evolution_promote.py
  --apply` always exits non-zero with a manual-escalation message.
  Pinned by `test_apply_flag_is_always_rejected`.
- **Sentinel confirmation required.** Promotion packet rendering
  requires `--operator-confirmation` to match the literal
  `CONFIRMATION_SENTINEL`. Pinned by
  `test_missing_confirmation_fails` + `test_wrong_confirmation_fails`.
- **Paper-test threshold + drawdown floor.** Promotion packet
  refuses to render unless trades ≥ 20, pnl_sum > 0, drawdown ≥
  configured floor. Pinned by
  `test_insufficient_paper_trades_fails` +
  `test_negative_aggregate_pnl_fails`.
- **Registry violation aborts promotion packet.** Even with all
  other prerequisites met, an audit-log violation aborts the
  helper. Pinned by `test_registry_audit_violation_fails`.
- **Forbidden output paths everywhere.** Queue, ledger, packet,
  and demo workspace all reject `policy_registry/approved/`,
  `policy_registry/pointer.json`, and (where relevant) the live
  registry root. Pinned by per-CLI rejection tests.
- **Source-level isolation.** Five new test files each scan their
  target script for forbidden imports of `rule_engine`,
  `decision_server`, `phase_d_walk_forward`.

## 6. Commits

```
b9d2866 feat(stage-6/task-4): human promotion helper (dry-run only)
b943f09 feat(stage-6/task-3): paper-test ledger schema (append-only JSONL)
c818b6e feat(stage-6/task-2): queue-inspect CLI — read-only ledger snapshot
f054c6a feat(stage-6/task-1): safety_bounds template — sidecar reference, red-line live path untouched
f009b18 feat(stage-6/task-5): end-to-end self-evolution demo (report-only)
```

## 7. Final standing

The HedgeRock self-evolution loop is **demonstrable end-to-end with
one command**, **report-only**, **XAUUSD-only**, and
**human-promote-only**:

- 528 tests passing across `tests/hedgerock/evolution/`.
- All four production-code mtimes unchanged.
- Production registry shadow-artefact JSON count unchanged (12).
- `policy_registry/approved/` and `policy_registry/pointer.json`
  remain non-existent.
- `config/safety_bounds.yaml` remains non-existent; the operator
  has a template at `config/safety_bounds_template.yaml` to copy
  after security review.
