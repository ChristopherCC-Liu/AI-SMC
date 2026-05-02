# Stage 6 Round 3 — Acceptance Addendum (Tasks 1–5)

> Continues `docs/hedgerock-self-evolution-stage6-round2-acceptance.md`.
> Adds the 30-day paper-trading plan, the demo-broker shadow logging
> design, a checklist-mode promotion approval CLI, real-registry
> end-to-end smoke coverage, and a strict integration test for the
> demo's audit trail.

## 1. Roll-up

| Task | Deliverable | Tests added |
|---|---|---|
| 1 | `docs/hedgerock-30day-paper-trading-plan.md` | 0 (docs) |
| 2 | `docs/hedgerock-shadow-broker-logging-design.md` | 0 (docs) |
| 3 | `scripts/hedgerock_evolution_promote.py` extended with `--checklist-mode` + 7 ack flags + audit trail wiring | 6 |
| 4 | `tests/hedgerock/evolution/test_real_registry_smoke.py` (real-registry end-to-end) | 6 |
| 5 | `tests/hedgerock/evolution/test_demo_audit_trail_integration.py` (schema + ordering + timestamps + reruns) | 8 |

Test growth (evolution sub-suite): **567 → 587** (delta +20).
Test growth (full hedgerock): **1403 → 1423 passed**.

## 2. Files added or changed

| Path | Change |
|---|---|
| `docs/hedgerock-30day-paper-trading-plan.md` | new |
| `docs/hedgerock-shadow-broker-logging-design.md` | new |
| `scripts/hedgerock_evolution_promote.py` | adds `--checklist-mode`, 7 `--ack-*` flags, `--audit-trail`, `--operator` |
| `tests/hedgerock/evolution/test_promotion_checklist.py` | new (6 tests) |
| `tests/hedgerock/evolution/test_real_registry_smoke.py` | new (6 tests) |
| `tests/hedgerock/evolution/test_demo_audit_trail_integration.py` | new (8 tests) |

## 3. Red-line invariants — re-checked at end of round 3

| Path | mtime epoch (start of session) | mtime epoch (end of round 3) | Touched? |
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

The real-registry smoke (task 4) now actively asserts the
mtime/size/ctime triple for every JSON under
`policy_registry/shadow_artefacts/` is unchanged across replay-validator,
report-CLI, and recommendation-CLI runs against the live registry.

## 4. Behavioural contracts pinned this round

- **Checklist-mode promotion.** `hedgerock_evolution_promote.py` now
  supports a 7-line operator checklist; missing any ack blocks the
  packet and records `promotion_checklist_block:fail` in the audit
  trail. `--apply` is still always rejected.
- **Real-registry replay.** Replay validator runs against all 4
  candidate dirs of the production shadow registry (12 JSONs total);
  no JSON is mutated. The recommendation CLI against the real
  `_audit.md` produces 4 NO_RECOMMENDATION/evidence_chain_invalid
  entries with all visualisations rendered.
- **Audit trail integration.** Demo writes one entry per stage with
  schema `{timestamp, operation, result, operator, details}`;
  timestamps are monotonic; per-stage detail fields land where the
  contract documents (workspace, report+recommendation, enqueued,
  inspection, ledger+n_trades, packet, real_registry_json_count).
- **Trail forbidden-path safety.** `--audit-trail` pointed at a
  forbidden location (under `policy_registry/approved/`) prints an
  AUDIT WARN and continues; no entry is written and no file is
  created.

## 5. New CLI surface

| Flag | CLI | Effect |
|---|---|---|
| `--checklist-mode` | `hedgerock_evolution_promote.py` | Require all 7 `--ack-*` flags before the packet renders. |
| `--ack-paper-test-pass` | promote | Acknowledges paper-test threshold met. |
| `--ack-drawdown-within-floor` | promote | Acknowledges drawdown floor not breached. |
| `--ack-no-violation` | promote | Acknowledges registry append-only contract intact. |
| `--ack-coverage-sufficient` | promote | Acknowledges XAUUSD coverage threshold met. |
| `--ack-replay-projection-positive` | promote | Acknowledges replay projection is positive. |
| `--ack-no-multi-symbol` | promote | Acknowledges no cross-symbol contamination. |
| `--ack-production-mtimes-unchanged` | promote | Acknowledges live code mtimes intact. |
| `--audit-trail` | promote, demo | Append per-action entries to a sidecar JSONL. |
| `--operator` | promote, demo | Override `$USER` for trail attribution. |

## 6. Commits

```
e0dfd82 docs(stage-6/round-3/tasks-1-2): 30-day paper-trading plan + shadow broker design
d2d4e4f feat(stage-6/round-3/task-3): checklist-mode promotion approval
7d7cd96 test(stage-6/round-3/task-4): real-registry replay smoke
```

## 7. Final standing (after round 3)

- **587 tests** passing in `tests/hedgerock/evolution/`.
- **1423 tests** passing in `tests/hedgerock/` overall (no
  regression outside the evolution sub-suite).
- All four production-code mtimes unchanged.
- Production registry shadow-artefact JSON count unchanged (12).
- `policy_registry/approved/`, `pointer.json`,
  `config/safety_bounds.yaml` all still absent.
- Operators now have:
  - a 30-day paper-trading runbook to follow,
  - a shadow-broker logging design to implement when paper trading
    starts,
  - a checklist-mode CLI for the manual approval gate,
  - real-registry smoke evidence that the replay/recommendation
    pipeline behaves correctly on the production data,
  - a strict audit-trail integration test guarding the demo's
    operator ledger.
