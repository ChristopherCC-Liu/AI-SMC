# Stage 6 Round 2 — Acceptance Addendum (Tasks 1–4 + broader sweep)

> Continues `docs/hedgerock-self-evolution-stage6-followup-acceptance.md`.
> Adds replay-based validation, queue aging, ASCII visualisations,
> and a cross-CLI operation audit trail. All red-line invariants
> still held; broader hedgerock test suite verified green.

## 1. Roll-up

| Task | Deliverable | Tests added |
|---|---|---|
| 1 | `src/smc/hedgerock/evolution/replay_validator.py` (read-only aggregator over `policy_registry/shadow_artefacts/`) | 9 |
| 2 | `src/smc/hedgerock/evolution/queue_aging.py` + `scripts/hedgerock_evolution_queue_age.py` (append-only STALE markers, idempotent) | 10 |
| 3 | `src/smc/hedgerock/evolution/ascii_visualisations.py` + recommendation-CLI integration (parameter table, gate matrix, heat ranking) | 8 |
| 4 | `src/smc/hedgerock/evolution/operation_audit.py` + `--audit-trail` flag in demo CLI | 12 |

Test growth (evolution): **528 → 567** (delta +39).
Test growth (full hedgerock): **1364 → 1403 passed**.

## 2. Files added or changed

| Path | Change |
|---|---|
| `src/smc/hedgerock/evolution/replay_validator.py` | new |
| `src/smc/hedgerock/evolution/queue_aging.py` | new |
| `src/smc/hedgerock/evolution/ascii_visualisations.py` | new |
| `src/smc/hedgerock/evolution/operation_audit.py` | new |
| `scripts/hedgerock_evolution_queue_age.py` | new |
| `scripts/hedgerock_evolution_recommend.py` | wires visualisations into report |
| `scripts/hedgerock_evolution_demo.py` | adds `--audit-trail` + per-stage logging |
| `tests/hedgerock/evolution/test_replay_validator.py` | new (9 tests) |
| `tests/hedgerock/evolution/test_queue_aging.py` | new (10 tests) |
| `tests/hedgerock/evolution/test_ascii_visualisations.py` | new (8 tests) |
| `tests/hedgerock/evolution/test_operation_audit.py` | new (12 tests) |
| `tests/hedgerock/evolution/test_recommendation_cli.py` | re-anchored c3 lookup on unique header |

## 3. Red-line invariants — re-checked at end of round 2

| Path | mtime epoch (start of session) | mtime epoch (end of round 2) | Touched? |
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

## 4. Behavioural contract notes

- **Replay validator never simulates.** It only aggregates per-window
  deltas already present in shadow artefact files. The renderer
  carries the literal `heuristic_projection_only / not a simulation`
  banner. Pinned by
  `tests/hedgerock/evolution/test_replay_validator.py::
  test_render_replay_report_carries_banners`.
- **Queue aging is append-only AND idempotent.** Running the CLI
  twice does not double-mark a stale entry; the original `QUEUED`
  line is preserved byte-for-byte. Pinned by
  `test_aging_is_idempotent` +
  `test_old_entry_is_marked_stale_and_original_line_is_preserved`.
- **No removal flags.** Source-grep test rejects any future
  ``--delete``/`--remove`/`--purge`/`--clear`/`--unlink` flag in
  the queue-age CLI. Pinned by `test_cli_has_no_delete_or_remove_flag`.
- **Operation trail is segregated from registry audit log.**
  Refused paths include `policy_registry/approved/`,
  `pointer.json`, and `policy_registry/shadow_artefacts/`. Pinned
  by parametrised `test_trail_path_under_forbidden_locations_is_rejected`.
- **Visualisations are pure functions.** No FS, no live runtime
  imports. The recommendation CLI threads gate results in optionally
  — when absent, the matrix is skipped and the rest of the report
  is unchanged. Pinned by
  `test_visualisation_module_has_no_live_runtime_imports` +
  `test_recommendation_report_includes_visualisations`.

## 5. Updated capability matrix

The system can now additionally:

- **Project past evidence.** Read existing shadow artefacts and
  surface aggregate per-window delta_pnl / delta_dd projections
  *labelled as heuristic, not as simulation*.
- **Age stale candidates.** Mark queue entries as STALE after a
  configurable threshold (default 14 days) using append-only
  marker lines.
- **Visualise decisions.** Rendered recommendation reports now
  carry parameter comparison tables with band markers, a
  candidate × G1..G8 gate matrix, and a deterministic heat
  ranking.
- **Audit operator actions.** Every demo stage (and any future
  CLI that opts in) appends one line to a JSONL operator trail
  separate from the registry audit log.

The system still cannot:

- Modify live code, EA, decision server, or the rule engine.
- Promote to `policy_registry/approved/` or update `pointer.json`.
- Delete, rename, or rewrite any file under
  `policy_registry/shadow_artefacts/`.
- Mutate `config/safety_bounds.yaml`.
- Recommend on multi-symbol bundles.
- Bypass `--operator-confirmation` in the dry-run promotion helper.

## 6. Commits (round 2)

```
b2e0588 feat(stage-6/round-2/task-1): replay-based candidate validator
06a453b feat(stage-6/round-2/task-2): queue aging — append STALE markers
88dab65 feat(stage-6/round-2/task-3): ASCII visualisations in recommendation report
c7be362 feat(stage-6/round-2/task-4): operation audit trail
```

## 7. Final standing (after round 2)

- **567 tests** passing in `tests/hedgerock/evolution/` (round-1
  ended at 528).
- **1403 tests** passing in `tests/hedgerock/` overall (round-1
  ended at 1364; full broader sweep verified green at the end of
  round 2).
- All four production-code mtimes unchanged.
- Production registry shadow-artefact JSON count unchanged (12).
- `policy_registry/approved/`, `pointer.json`,
  `config/safety_bounds.yaml` all still absent.
- Demo runs end-to-end with `--audit-trail`, producing a complete
  operator-action ledger alongside report / queue / ledger /
  packet artefacts.
