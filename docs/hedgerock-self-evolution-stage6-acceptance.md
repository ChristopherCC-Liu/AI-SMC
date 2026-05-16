# HedgeRock Self-Evolution Loop — Stage 6 Final Acceptance Report

> Status: **report-only loop demonstrable**. The system can observe,
> detect, recommend, and queue parameter tweaks. It cannot self-modify
> live code, cannot self-promote, and cannot self-deploy. Every
> promotion to `policy_registry/approved/` remains a manual human step.

## 1. Stage roll-up

| Stage | Deliverable | Tests added |
|---|---|---|
| 1 | T4-F3 operator dry-run validation + runbook | 6 |
| 2 | `docs/hedgerock-self-evolution-rfc.md` (5-layer state machine) | 0 (docs only) |
| 3 | `src/smc/hedgerock/evolution/candidate_generator.py` | 11 |
| 4 | `scripts/hedgerock_evolution_recommend.py` (report-only CLI) | 6 |
| 5 | `src/smc/hedgerock/evolution/shadow_test_queue.py` | 9 |
| 6 | This acceptance report | 0 (audit only) |

Test growth: **458 → 490** in `tests/hedgerock/evolution/` (delta +32).

## 2. Files added

| Path | Purpose |
|---|---|
| `tests/hedgerock/evolution/test_evolution_report_t4f3.py` | argv-path dry-run contract |
| `docs/hedgerock-evolution-report-runbook.md` | operator runbook for the report CLI |
| `docs/hedgerock-self-evolution-rfc.md` | self-evolution RFC v0 |
| `src/smc/hedgerock/evolution/candidate_generator.py` | Stage 3 — proposal generator |
| `tests/hedgerock/evolution/test_candidate_generator_v0.py` | Stage 3 tests |
| `scripts/hedgerock_evolution_recommend.py` | Stage 4 — recommendation CLI |
| `tests/hedgerock/evolution/test_recommendation_cli.py` | Stage 4 tests |
| `src/smc/hedgerock/evolution/shadow_test_queue.py` | Stage 5 — append-only queue |
| `tests/hedgerock/evolution/test_shadow_test_queue.py` | Stage 5 tests |
| `docs/hedgerock-self-evolution-stage6-acceptance.md` | this report |

## 3. Files NOT touched (red-line invariants)

Verified against `git status` and mtime comparison after each stage:

| Path | mtime epoch (start of session) | Touched? |
|---|---|---|
| `src/smc/hedgerock/rule_engine.py` | 1777604440 | **No** |
| `src/smc/hedgerock/decision_server.py` | 1777607670 | **No** |
| `src/smc/hedgerock/phase_d_walk_forward.py` | 1777609385 | **No** |
| `mql5/AISMCReceiver.mq5` | 1776754859 | **No** |
| `policy_registry/approved/` | — | **Does not exist** (intentional) |
| `policy_registry/pointer.json` | — | **Does not exist** (intentional) |
| `config/safety_bounds.yaml` | — | **Does not exist** (per RFC §3 G6 (b); G6 returns `safety_bound_undefined` for every candidate) |

## 4. Production registry inventory (read-only audit)

| Subtree | JSON count |
|---|---|
| `policy_registry/audit/` | 5 |
| `policy_registry/candidates/` | 4 |
| `policy_registry/shadow_artefacts/` | **12** (3 per candidate × 4 candidates) |
| `policy_registry/approved/` | (does not exist) |
| `policy_registry/pointer.json` | (does not exist) |

Total `*.json` files under `/Users/christopher/HedgeRock/policy_registry/` = 21,
of which the **12 shadow-artefact JSONs are the append-only ledger**
this session must not touch. Their SHA-256 hashes are documented in
`/Users/christopher/HedgeRock/policy_registry/shadow_artefacts/_audit.md`.

The four lost-SHA entries previously logged (T4-F2 fixture) remain
recorded but are not on disk by design — the audit log is the only
record of what existed before the deletion incident.

## 5. Test results

```
$ python -m pytest tests/hedgerock/evolution/ -q
...
490 passed, 2 warnings in 2.6s
```

Per-file totals:

| Test file | Passed |
|---|---|
| `test_evolution_report_t4f2.py` | (T4-F2 baseline) |
| `test_evolution_report_t4f3.py` | 6 |
| `test_candidate_generator_v0.py` | 11 |
| `test_recommendation_cli.py` | 6 |
| `test_shadow_test_queue.py` | 9 |
| (all other evolution tests) | 458 baseline |

All 6 stages pass through `pytest -q` without regression.

## 6. Capability matrix

### What the system CAN do (report-only)

- **Observe** — load Phase D evidence (atlas, data-availability,
  walk-forward) plus the registry append-only audit log; surface the
  full audit triplet (`audit_log_present`, `lost_sha_count`,
  `registry_append_only_violation`).
- **Detect** — evaluate `CANDIDATE_MENU_V0` against G1–G8; surface
  per-candidate verdicts and blocking reasons.
- **Recommend** — propose micro-tweaks for parameters in the RFC §5
  safety table, clamped to hard bands; refuse to recommend on:
  evidence-chain breach, insufficient XAUUSD coverage, exposure-raising
  candidates, parameter classes not in §5, missing audit log.
- **Queue** — record RECOMMEND proposals as append-only JSONL queue
  entries with required windows / tests / blocking conditions.
- **Document** — emit operator-facing markdown reports with
  `NOT LIVE` / `NOT APPROVED` / `NOT DEPLOYED` banners.

### What the system CANNOT do (by design + by tests)

- **Cannot modify live code.** No production `.py` or `.mq5` file is
  written by any sidecar module. Pinned by source-level isolation
  tests (`test_*_does_not_import_live_runtime*`).
- **Cannot promote.** The candidate generator and the queue do not
  write under `policy_registry/approved/` or modify
  `policy_registry/pointer.json`. Pinned by path-rejection tests.
- **Cannot delete shadow artefacts.** No production module under
  `src/smc/hedgerock/evolution/` or `scripts/` calls `unlink`,
  `rmtree`, or `rename` against `policy_registry/shadow_artefacts/`.
  Pinned by `test_artefact_registry_append_only.py`.
- **Cannot self-deploy.** There is no automated path that writes to
  the EA, the strategy server, or the live decision-server route
  table. The Stage 4 recommendation CLI's report ends at the markdown
  document.
- **Cannot bypass human approval.** The Stage 5 queue is append-only;
  there is no `dequeue` / `remove` API. A queued candidate flows
  forward only via a manual human promotion script (out of scope
  for this RFC).
- **Cannot recommend on multi-symbol bundles.** The candidate
  generator hard-fails to `insufficient_xauusd_coverage` when XAUUSD
  is absent or `years_passing < 4`.

## 7. Suggested next steps (out of scope for this acceptance)

- **Multi-symbol expansion (XAGUSD, EURUSD).** Current loop hard-locked
  to XAUUSD. Adding a second symbol requires a Phase D bundle update,
  a generator-level coverage threshold review, and a new menu entry.
- **Paper-test ledger (`PAPER_TEST` state).** RFC §3 names this
  transition but does not define the schema. A future ticket should
  ship the paper-test record format and the human approval packet.
- **Manual promotion script.** The HUMAN_APPROVE → live edge is
  intentionally non-automated. A separate, audited shell tool that
  copies a queue entry into `policy_registry/approved/` (with a
  required dual-control prompt) is the cleanest implementation path.
- **Safety-bounds YAML.** `config/safety_bounds.yaml` is currently
  absent, so every G6 returns `safety_bound_undefined`. Once a human
  authors that file, the generator's RFC §5 clamps remain in force as
  a *second line of defence*.
- **Queue inspection report.** The queue is JSONL on disk; an
  operator-facing rendering CLI (read-only) would make the queued
  candidates easier to triage.

## 8. Commit ledger

```
84f9814 test(t4-f3): pin operator argv-path dry-run + ship runbook
5179a22 docs(stage-2): self-evolution RFC v0 — five-layer state machine + invariants
a1f8e61 feat(stage-3): candidate generator v0 — report-only proposals with safety clamp
67e9d2c feat(stage-4): hedgerock_evolution_recommend CLI — report-only proposal report
031aae1 feat(stage-5): shadow-test queue (append-only sidecar JSONL)
```

Stage 6 is committed alongside this report.

## 9. Final declaration

The HedgeRock self-evolution loop is **report-only, XAUUSD-only,
human-promote-only**, and the test suite (`490 passed`) pins every
red-line invariant named in the request:

- `src/smc/hedgerock/rule_engine.py`, `decision_server.py`,
  `phase_d_walk_forward.py` — untouched.
- EA `*.mq5` — untouched.
- `policy_registry/approved/`, `pointer.json` — never created.
- `policy_registry/shadow_artefacts/` — append-only enforcement
  carried forward from T4-F1/F2/F3.
- `config/safety_bounds.yaml` — never written.

Promoting any of the 4 candidate parameters to live remains a manual
step that takes the recommendation report, the queue entry, the shadow
artefact ledger, and the operator's signed approval — *in that order*.
