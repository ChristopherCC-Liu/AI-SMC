# HedgeRock Evolution — Changelog

> Append-only chronological record of the report-only self-evolution
> loop. Versions follow semver: MAJOR for breaking schema changes,
> MINOR for new public surface, PATCH for fixes / hardening.

The starting point of this changelog is the T4-F2 baseline that
was already on disk at session start (458 evolution tests passing,
T4-F2 registry-audit wiring already shipped). Every entry below
shipped during the current self-evolution work session.

---

## v0.6.0 — 2026-05-02 (Round 4: self-defence layer)

**Added**

- Graceful-degradation test suite probing 13 edge cases across all
  CLIs and core modules (malformed JSONL, garbled bytes audit log,
  missing argv, missing ledger, non-list config values).
  *(commit `18821db`)*
- `scripts/hedgerock_evolution_validate_config.py` — pre-deploy
  config validator. Catches missing menu targets, `lo > hi`,
  non-numeric ranges, non-list values; surfaces `SAFETY_CLAMPS`
  disagreement as WARN by default and FAIL under `--strict`.
  *(commit `a366744`)*
- `scripts/hedgerock_evolution_metrics_export.py` — `metrics/v0`
  JSON snapshot with stable shape (`registry_audit / queue /
  paper_test / shadow_artefacts / candidates`). JSON-strict (no
  NaN). Inputs unchanged after run.
  *(commit `a4e1e09`)*
- `tests/hedgerock/evolution/test_regression_guard.py` — meta-scan
  asserting NO file under `evolution/` or
  `scripts/hedgerock_evolution_*.py` imports the live trading
  runtime; ≥ 30 files in scope (current 35).
  *(commit `2e76a1d`)*
- `src/smc/hedgerock/evolution/README.md` — onboarding doc with
  architecture diagram, file role table, how-to for adding new
  gates / candidates / CLIs.
  *(commit `bfbb46e`)*

**Tests**: evolution sub-suite 587 → 626 (+39); broader hedgerock
1423 → 1462 (+39).

---

## v0.5.0 — 2026-05-02 (Round 3: paper-test plan + checklist + smoke)

**Added**

- `docs/hedgerock-30day-paper-trading-plan.md` — Day -1 pre-flight,
  Days 1–30 five-step cycle, Day-30 close-out, 9 abort triggers,
  pass/fail thresholds (≥ 20 trades, pnl_sum > 0, max_dd ≥ -50, ≥ 12
  active days, replay sign-agreement). *(commit `e0dfd82`)*
- `docs/hedgerock-shadow-broker-logging-design.md` — bus-snoop
  architecture for paper-fill recording; frozen `ShadowFill` spec;
  6 failure modes; XAUUSD-only invariant. *(commit `e0dfd82`)*
- `--checklist-mode` + 7 `--ack-*` flags on
  `hedgerock_evolution_promote.py`. Missing any ack → exit 1, audit
  trail records `promotion_checklist_block:fail`. All acks present
  → packet rendered, audit trail records 7 acks +
  `promotion_checklist_complete:ok`. `--apply` still always
  rejected. *(commit `d2d4e4f`)*
- `tests/hedgerock/evolution/test_real_registry_smoke.py` —
  end-to-end smoke against the real production registry
  (12 JSONs in 4 dirs); replay validator runs without mutation;
  recommendation CLI produces `4 NO_RECOMMENDATION /
  evidence_chain_invalid` with full visualisations.
  *(commit `7d7cd96`)*
- `tests/hedgerock/evolution/test_demo_audit_trail_integration.py` —
  schema, ordering, timestamp monotonicity, per-stage detail
  fields, `--operator` override + `$USER` fallback, append-only
  across reruns, forbidden-path graceful warn. *(commit `7a057b6`)*

**Tests**: evolution sub-suite 567 → 587 (+20); broader hedgerock
1403 → 1423 (+20).

---

## v0.4.0 — 2026-05-02 (Round 2: replay + aging + visualisations + audit trail)

**Added**

- `src/smc/hedgerock/evolution/replay_validator.py` — read-only
  aggregator over `policy_registry/shadow_artefacts/<id>/*.json`;
  emits `ReplayValidationReport` with delta_pnl_pp_mean / p25 / p75,
  delta_dd_pp_worst, windows_passing / regressing; carries
  `heuristic_projection_only / not a simulation` banner.
  *(commit `b2e0588`)*
- `src/smc/hedgerock/evolution/queue_aging.py` +
  `scripts/hedgerock_evolution_queue_age.py` — appends STALE marker
  lines for QUEUED entries older than `--stale-after-days`
  (default 14). Idempotent; original lines preserved byte-for-byte;
  no removal flags. *(commit `06a453b`)*
- `src/smc/hedgerock/evolution/ascii_visualisations.py` +
  recommendation-CLI integration — parameter comparison table with
  band visual marker, candidate × G1..G8 gate matrix, deterministic
  heat ranking. *(commit `88dab65`)*
- `src/smc/hedgerock/evolution/operation_audit.py` +
  `--audit-trail` / `--operator` flags on the demo CLI — append-only
  cross-CLI operator action trail; rejects paths under
  `policy_registry/approved/`, `pointer.json`, or
  `shadow_artefacts/`. *(commit `c7be362`)*

**Tests**: evolution sub-suite 528 → 567 (+39); broader hedgerock
1364 → 1403 (+39).

---

## v0.3.0 — 2026-05-02 (Round 1 follow-up: bounds template + queue inspect + paper ledger + promotion + demo)

**Added**

- `config/safety_bounds_template.yaml` — XAUUSD-only conservative
  bands matching Stage-3 `SAFETY_CLAMPS`. The live red-line path
  `config/safety_bounds.yaml` stays absent until the operator
  copies it manually. *(commit `f054c6a`)*
- `scripts/hedgerock_evolution_queue_inspect.py` — read-only queue
  snapshot CLI; graceful empty / missing-queue handling.
  *(commit `c818b6e`)*
- `src/smc/hedgerock/evolution/paper_test_ledger.py` — append-only
  JSONL `PaperTestEntry` ledger; XAUUSD-only invariant;
  `summarise()` aggregates trades / pnl_sum / max_drawdown per
  candidate. *(commit `b943f09`)*
- `scripts/hedgerock_evolution_promote.py` — dry-run promotion
  packet helper. `--apply` always rejected; requires
  `--operator-confirmation` matching `CONFIRMATION_SENTINEL`;
  validates queue presence, paper-test thresholds (≥ 20 trades,
  pnl_sum > 0, drawdown floor), registry-audit clean state.
  *(commit `b9d2866`)*
- `scripts/hedgerock_evolution_demo.py` — end-to-end orchestrator:
  observe → recommend → queue → inspect → optional paper-test seed
  → optional dry-run packet. Verifies real-registry JSON count
  unchanged at end. *(commit `f009b18`)*

**Tests**: evolution sub-suite 458 → 528 (+70).

---

## v0.2.0 — 2026-05-02 (Stages 3–5: generator + recommendation + queue)

**Added**

- `src/smc/hedgerock/evolution/candidate_generator.py` — frozen
  `CandidateProposal`; `SAFETY_CLAMPS` table mirrors RFC §5; stable
  reason ids (`evidence_chain_invalid`,
  `insufficient_xauusd_coverage`, `proposal_outside_safety_clamp`,
  `parameter_class_unsupported`, `no_trigger`); exposure-raising
  candidates auto-veto per RFC §10.1. *(commit `a1f8e61`)*
- `scripts/hedgerock_evolution_recommend.py` — feeds report-CLI
  candidates + bundle into the generator; renders markdown
  recommendation with `NOT LIVE / NOT APPROVED / NOT DEPLOYED`
  banners; demotes RECOMMEND to
  `audit_log_absent_re_point_flag` when audit log is missing.
  *(commit `67e9d2c`)*
- `src/smc/hedgerock/evolution/shadow_test_queue.py` — append-only
  JSONL queue; `enqueue_proposals` only (no dequeue / pop / clear);
  rejects paths under `policy_registry/approved/` + `pointer.json`.
  *(commit `031aae1`)*
- `docs/hedgerock-self-evolution-rfc.md` — five-layer state machine
  (OBSERVE → RECOMMEND → SHADOW_TEST → PAPER_TEST →
  HUMAN_APPROVE) + hard architectural constraints + safety-clamp
  table (RFC §5). *(commit `5179a22`)*
- `docs/hedgerock-self-evolution-stage6-acceptance.md` — first
  acceptance report; mtime-pinned invariants table.
  *(commit `c4de463`)*

---

## v0.1.0 — 2026-05-02 (T4-F3: operator argv-path dry-run + runbook)

**Added**

- `tests/hedgerock/evolution/test_evolution_report_t4f3.py` — pins
  the operator-facing CLI contract: argv-path dry-run with real /
  tmp-copy / no-log / default-resolution / bad-argv variants.
- `docs/hedgerock-evolution-report-runbook.md` — operator runbook
  for the report CLI's argv surface.
*(commit `84f9814`)*

**Tests**: evolution sub-suite 458 → 464 (+6 over T4-F2 baseline).

---

## v0.0.x — pre-session baseline (already on disk)

The T4-F2 follow-on contracts pinned by
`tests/hedgerock/evolution/test_evolution_report_t4f2.py` (registry
audit state plumbed into the library `run()` entry point). Predecessor
work: T4-F1 (registry-audit gate), Tickets 1-3 (policy registry,
shadow runner, multi-window report), and the Phase D pipeline
(walk-forward, atlas, data-availability). The current changelog
starts here because nothing predates the session was modified.

---

## Pinned invariants across every release

These have been verified at every round end:

| Invariant | Status |
|---|---|
| `src/smc/hedgerock/rule_engine.py` mtime unchanged | ✅ |
| `src/smc/hedgerock/decision_server.py` mtime unchanged | ✅ |
| `src/smc/hedgerock/phase_d_walk_forward.py` mtime unchanged | ✅ |
| `mql5/AISMCReceiver.mq5` mtime unchanged | ✅ |
| `policy_registry/approved/` non-existent | ✅ |
| `policy_registry/pointer.json` non-existent | ✅ |
| `config/safety_bounds.yaml` non-existent | ✅ |
| `policy_registry/shadow_artefacts/*.json` count unchanged (12) | ✅ |
| Production registry total JSON count unchanged (21) | ✅ |
| No file under `evolution/` imports `rule_engine` / `decision_server` / `phase_d_walk_forward` | ✅ (regression guard) |
