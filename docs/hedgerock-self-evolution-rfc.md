# HedgeRock Self-Evolution RFC (report-only, XAUUSD-only)

> Status: **Draft v0** — written by the r10-phase1 self-evolution
> stack (Stages 2–6 of the report-only loop). Supersedes nothing
> live. Pinned by tests in `tests/hedgerock/evolution/`.

## 0. One-line summary

The self-evolution loop produces *recommendations* about HedgeRock
parameters. It never modifies live code, never promotes a candidate
to `policy_registry/approved/`, and never updates
`policy_registry/pointer.json`. Promotion is reserved for a human
operator with the full evidence chain in front of them.

## 1. Non-goals (post Tier-1 unseal)

> **Update — Tier-1 unseal.** The originally-stated invariant
> "no evolution file imports `decision_server` or
> `phase_d_walk_forward`" has been **partially repealed**. As of
> the Tier-1 unseal:
>
> - **`replay_validator`** and **`candidate_generator`** MAY
>   `import` `decision_server` and `phase_d_walk_forward` for
>   their public, documented read-only surface
>   (`decision_server.get_live_parameters` and
>   `phase_d_walk_forward.run_walk_forward_backtest` plus the
>   frozen public dataclasses / constants).
> - **All other** evolution files remain forbidden from importing
>   either module — enforced by the regression guard whitelist in
>   `tests/hedgerock/evolution/test_regression_guard.py`.
> - **`rule_engine`** remains red-line. No file in the evolution
>   layer may import it.
> - The runbook, 30-day paper-trading plan, and acceptance docs
>   have been updated in lockstep.

The loop **does not**:

- write or modify `src/smc/hedgerock/rule_engine.py`,
  `src/smc/hedgerock/decision_server.py`, or
  `src/smc/hedgerock/phase_d_walk_forward.py` (read-only imports
  permitted for the two whitelisted modules above);
- write or modify any `*.mq5` file under `mql5/`;
- create files under `policy_registry/approved/`;
- update or rewrite `policy_registry/pointer.json`;
- mutate `config/safety_bounds.yaml`;
- delete, rename, or rewrite any `*.json` artefact under
  `policy_registry/shadow_artefacts/`;
- emit any signal that lands on a live trade decision path;
- handle multi-symbol coverage (XAUUSD only — XAGUSD / EURUSD
  expansion is RFC-out-of-scope).

These are tested invariants, not aspirations:

- Module-level scan: `tests/hedgerock/evolution/test_artefact_registry_append_only.py`
  (no delete primitive on the shadow-artefact registry).
- File-level scan: `tests/hedgerock/evolution/test_runtime_isolation.py`
  (no production module imports the report-only generator/CLI).

## 2. Roles

| Role | What it does | What it CANNOT do |
|---|---|---|
| **Phase D walk-forward** | Produce evidence (atlas, data-availability, walk-forward report). | Touch the registry, change live config. |
| **Shadow runner** | Append per-window shadow artefacts to `policy_registry/shadow_artefacts/<candidate_id>/`. | Delete or rewrite any artefact; promote anything. |
| **Evolution report CLI** | Evaluate `CANDIDATE_MENU_V0` against gates G1–G8 + registry-audit; render report. | Promote, rewrite, delete artefacts; touch `approved/` or `pointer.json`. |
| **Candidate generator (Stage 3)** | Read evidence + gate results + audit state; emit a `CandidateProposal` describing a parameter tweak. | Write to live code, EA, registry approved/, or pointer.json. |
| **Recommendation CLI (Stage 4)** | Render proposals as a markdown report with explicit `NOT LIVE` / `NOT APPROVED` / `NOT DEPLOYED` banners. | Write outside `tmp/` / report-only output paths. |
| **Shadow-test queue (Stage 5)** | Persist queued candidates with required windows + tests + blocking conditions. | Auto-approve, write to `approved/`, write to `pointer.json`, trigger live behaviour. |
| **Human approver** | Review the recommendation + queue + audit chain; manually promote a candidate to `approved/`. | (this is the only role allowed to promote.) |

## 3. State machine — five layers

```
   ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌────────────┐    ┌────────────────┐
   │ OBSERVE  │ →  │ RECOMMEND    │ →  │  SHADOW_TEST    │ →  │ PAPER_TEST │ →  │ HUMAN_APPROVE  │
   │  (read)  │    │ (report-only)│    │ (queued + run)  │    │  (sidecar) │    │ (manual)       │
   └──────────┘    └──────────────┘    └─────────────────┘    └────────────┘    └────────────────┘
```

### Transitions

| Edge | Trigger | Output |
|---|---|---|
| OBSERVE → RECOMMEND | A regular evolution-report run with **clean registry-audit** AND **XAUUSD coverage sufficient** AND at least one parameter triggers a recommendation rule. | A `CandidateProposal` written under `tmp/` or report-only path. |
| RECOMMEND → SHADOW_TEST | Operator (or the Stage 5 queue runner) decides the proposal is worth running through the shadow runner. | A queue entry under the report-only queue file (Stage 5). |
| SHADOW_TEST → PAPER_TEST | Shadow runner produced enough windows passing the existing G1–G8 + window-coverage gates. **(External to this RFC; described by future tickets.)** | An entry in the paper-test ledger (sidecar). |
| PAPER_TEST → HUMAN_APPROVE | Paper-test results show statistically meaningful improvement, with no safety-bound or audit violation. | A human-readable approval packet. |
| HUMAN_APPROVE → live | A human runs a manual promotion script, copies the manifest into `policy_registry/approved/`, and updates `pointer.json` themselves. | (This step is intentionally NOT automated.) |

### Required gates at each transition

- **OBSERVE → RECOMMEND**:
  - `registry_audit.registry_append_only_violation == False` —
    otherwise emit `NO_RECOMMENDATION/evidence_chain_invalid`.
  - XAUUSD coverage sufficient — otherwise emit
    `NO_RECOMMENDATION/insufficient_xauusd_coverage`.
  - Proposed parameter inside the safety clamp.
- **RECOMMEND → SHADOW_TEST**:
  - All Stage 4 invariants: no `approved/` or `pointer.json` write,
    no live code touched.
  - Queue entry passes Stage 5 schema validation.
- **SHADOW_TEST → PAPER_TEST**: covered by existing G1–G8 +
  window_coverage; out of scope for this RFC.
- **PAPER_TEST → HUMAN_APPROVE**: covered by future tickets.
- **HUMAN_APPROVE → live**: human-only.

## 4. Hard architectural constraints

These are **invariants**, not preferences. Any code change that
breaks one of these is a release blocker.

1. **No write path to live code.** No module under
   `src/smc/hedgerock/evolution/` may modify the production
   `rule_engine.py`, `decision_server.py`, or
   `phase_d_walk_forward.py`. Read-only imports of the two latter
   modules are permitted **only** from `replay_validator` and
   `candidate_generator`, and only against their public surface
   (`decision_server.get_live_parameters`,
   `phase_d_walk_forward.run_walk_forward_backtest`, and the
   frozen dataclasses / constants documented in those files).
   `rule_engine` remains fully red-line.
2. **No write path to `approved/`.** Every code path writing under
   `policy_registry/` is restricted to `candidates/`, `audit/`,
   `shadow_artefacts/<candidate_id>/` (append-only), or a
   sidecar `queue/` directory (Stage 5).
3. **No write path to `pointer.json`.** Stage 5 queue entries are
   written under their own directory, not by mutating
   `policy_registry/pointer.json`.
4. **Append-only registry contract.** No production module may
   `unlink`, `rmtree`, `chmod` followed by `unlink`, or `rename` a
   file under `policy_registry/shadow_artefacts/`. The only
   permitted file-system operation is *append a new artefact*.
   Pinned by `test_artefact_registry_append_only.py`.
5. **XAUUSD only.** The candidate generator refuses to recommend
   when the evidence bundle does not show XAUUSD as the primary
   replicated symbol (`bundle.year_replication["XAUUSD"]`).
6. **Safety clamp.** Every parameter recommendation is clamped to
   a hard-coded safety band before it leaves the generator.
   Out-of-band proposals are dropped with reason
   `proposal_outside_safety_clamp`.
7. **Report-only outputs.** Every Stage-3/4/5 output path is
   either `tmp/`, a markdown report, or a sidecar queue file.
   No code in this RFC writes to live config or production
   policy registry beyond the existing T4-F2/T4-F3 wiring.

## 5. Allowed parameter recommendations (v0 menu)

The candidate generator only recommends micro-tweaks to the
parameters already in `CANDIDATE_MENU_V0`. New parameter classes
require an RFC update + a new menu entry.

| Parameter class | Examples | Safety clamp |
|---|---|---|
| Confidence threshold | `_CONFIDENCE_OBSERVE_FLOOR`, `_CONFIDENCE_AGGRESSIVE_FLOOR`, `_RANGE_2_CONFIDENCE_BASELINE` | ±10% relative; never below 0.30 absolute. |
| Observe floor | `_CONFIDENCE_OBSERVE_FLOOR` | 0.40 ≤ value ≤ 0.65. |
| Cooldown | TBD on rule_engine cooldown constants | ±20% relative; never below 30 minutes absolute. |
| Aggressive cap | `_CONFIDENCE_AGGRESSIVE_FLOOR` | 0.70 ≤ value ≤ 0.85. |
| Halt expiry | `_HALT_AUTO_EXPIRY_HOURS_OBSERVE` | 4 ≤ hours ≤ 48. |

Anything outside these classes returns `NO_RECOMMENDATION/parameter_class_unsupported`.

## 6. Inputs to the generator

The candidate generator (Stage 3) consumes:

- The `EvidenceBundle` (atlas, data-availability, walk-forward,
  registry-audit state) used by the report CLI.
- Per-window metrics from the shadow runner (read-only, joined via
  the R5 double-key).
- The blocking-reasons list from the most recent evolution-report
  run.
- The gate-result map (G1–G8) from the same run.
- The registry-audit state surfaced by T4-F2 wiring.

It does **not** consume:

- Live trade journals.
- The strategy server's runtime state.
- Any tick or tickdata feed.
- Anything outside `policy_registry/` and the Phase D evidence
  artefacts.

## 7. Output schema

`CandidateProposal` (frozen dataclass — see Stage 3 implementation).

Required fields:

- `candidate_id: str` — must match an existing
  `CANDIDATE_MENU_V0` entry id (no net-new candidate ids in v0).
- `parameter_target: str` — `<module>.<symbol>` qualifier (e.g.
  `smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR`).
- `baseline_value: float` — current value read from the menu.
- `proposed_value: float` — clamped to the safety band.
- `parameter_class: Literal[...]` — one of the classes in §5.
- `triggered_by: tuple[str, ...]` — gate result ids / blocking
  reason ids that motivated the proposal.
- `expected_improvement: str` — short prose, ≤ 200 chars.
- `risks: tuple[str, ...]` — short prose, ≤ 5 items.
- `next_validation: tuple[str, ...]` — required windows / tests
  before paper-test.
- `decision: Literal["RECOMMEND", "NO_RECOMMENDATION"]`.
- `decision_reason: str` — a stable id (e.g.
  `evidence_chain_invalid`, `insufficient_xauusd_coverage`,
  `proposal_outside_safety_clamp`, `parameter_class_unsupported`,
  `no_trigger`).
- `report_only: Literal[True]` — invariant; pinned by tests.

## 8. Failure modes & their stable reason ids

| Reason id | When | Effect |
|---|---|---|
| `evidence_chain_invalid` | `registry_audit.registry_append_only_violation = True`. | `NO_RECOMMENDATION`. |
| `insufficient_xauusd_coverage` | XAUUSD year_replication missing or `years_passing < 4`. | `NO_RECOMMENDATION`. |
| `proposal_outside_safety_clamp` | Proposed value falls outside the §5 band after clamping logic. | `NO_RECOMMENDATION`. Generator is the safety net of last resort. |
| `parameter_class_unsupported` | Trigger maps to a parameter class not in §5. | `NO_RECOMMENDATION`. |
| `no_trigger` | Gate / blocking-reason set does not match any v0 trigger. | `NO_RECOMMENDATION`. |

## 9. Test invariants

- **Generator never writes outside report-only paths.** Pinned by
  Stage 3 tests; sentinel checks `policy_registry/approved/` and
  `policy_registry/pointer.json` non-existence.
- **Generator returns `NO_RECOMMENDATION` whenever
  `registry_append_only_violation` is True.** Pinned by Stage 3.
- **Recommendation CLI's report explicitly carries `NOT LIVE`,
  `NOT APPROVED`, `NOT DEPLOYED` banners.** Pinned by Stage 4.
- **Queue entry is append-only (a queued candidate cannot move to
  approved without a human).** Pinned by Stage 5.
- **Total test count strictly grows** through Stages 1–5
  (T4-F2/F3 + Stage 3/4/5).

## 10. Out-of-scope work for this RFC

- Multi-symbol coverage (XAGUSD, EURUSD).
- Automated promotion of a candidate to `approved/`.
- Automated `pointer.json` updates.
- Live decision-path modifications.
- New parameter classes beyond §5.
- Anything that runs inside the EA `*.mq5`.

## 11. References

- `tests/hedgerock/evolution/test_evolution_report_t4f2.py` — T4-F2
  registry-audit wiring contract (library `run()`).
- `tests/hedgerock/evolution/test_evolution_report_t4f3.py` — T4-F3
  argv-path dry-run contract.
- `docs/hedgerock-evolution-report-runbook.md` — operator runbook for
  the report CLI.
- `src/smc/hedgerock/evolution/promotion_gates.py` — G1–G8 gate
  implementations including `g8_shadow_comparison`'s
  registry-audit branch.
- `src/smc/hedgerock/evolution/registry_audit.py` — registry-audit
  state loader.
