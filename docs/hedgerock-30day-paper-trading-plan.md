# HedgeRock 30-Day Paper-Trading Plan (PAPER_TEST stage, REPORT-ONLY)

> **NOT LIVE.** **NOT APPROVED.** **NOT DEPLOYED.** This document
> describes the protocol for the PAPER_TEST stage of the
> self-evolution loop (RFC §3). Paper trading produces simulated
> fills against the demo broker; no orders are routed to live
> markets. The `paper_test_ledger.jsonl` schema in
> `src/smc/hedgerock/evolution/paper_test_ledger.py` is the
> single source of truth for trade records.

## 0. Pre-flight (Day -1)

Before Day 1 begins, verify the following — every line must be
ticked. If any line fails, restart the pre-flight; do not begin
paper trading.

- [ ] Stage-1 through Stage-6 acceptance reports are still green
      (`tests/hedgerock/evolution/` ≥ 567 passed).
- [ ] `policy_registry/shadow_artefacts/_audit.md` records no
      append-only violation in the active session.
- [ ] Recommendation report renders the candidate as
      `decision: \`RECOMMEND\``, with G1–G7 PASS and G8 not blocked
      by `registry_append_only_violation`.
- [ ] Candidate is in the shadow-test queue with `status: QUEUED`
      and is **not** STALE.
- [ ] Replay validator report shows `n_windows_replayed >= 8` and
      `delta_pnl_pp_mean > 0` over the historical shadow corpus.
- [ ] Operator confirms the candidate's diff against the live
      `rule_engine` constants (read-only diff; no edit) and the
      proposed value sits inside the safety band.
- [ ] Demo broker is reachable; symbol `XAUUSD` quotes streaming.
- [ ] `paper_test_ledger.jsonl` path is set under a sidecar
      directory (NOT under `policy_registry/approved/` or
      `policy_registry/pointer.json`).
- [ ] Operation audit trail path is configured; first entry is the
      pre-flight `paper_test_pre_flight: ok`.

## 1. Daily structure (Days 1–30)

Each trading day repeats the same five-step cycle. The whole loop
should fit in ≤ 30 minutes if no exceptions trigger.

### 1.1 Morning (open of XAUUSD trading session)

1. **Re-validate gate matrix.** Run
   `scripts/hedgerock_evolution_recommend.py` with the latest Phase
   D bundle and confirm the candidate is still
   `decision: RECOMMEND` with no new blocking reasons.
2. **Inspect queue state.** Run
   `scripts/hedgerock_evolution_queue_inspect.py`; confirm the
   candidate is `QUEUED` (not STALE).
3. **Check audit log.** Confirm
   `registry_append_only_violation: False` in the day's report.
4. **Audit-trail entry.** Append `paper_test_morning_ok` (or
   `paper_test_morning_block` with reason).

### 1.2 Live window (XAUUSD H1 / H4 candles closing within 24h UTC)

1. Demo broker runs the candidate parameters in shadow mode (see
   `docs/hedgerock-shadow-broker-logging-design.md`).
2. Each filled paper trade appends one row to
   `paper_test_ledger.jsonl` via `PaperTestLedger.append`.
3. Operator does NOT manually open or close orders. The protocol
   is **observe-only at the human layer**; only the demo broker
   records fills.

### 1.3 End-of-day cycle (after the H4 close UTC)

1. Compute the day's `summarise()` over the ledger; record the
   per-candidate row in the day's review.
2. Append `paper_test_eod` audit-trail entry with: `trades_today`,
   `pnl_today`, `cumulative_pnl`, `cumulative_max_dd`.
3. Verify red-line invariants — no production code mtime drift,
   no shadow-artefact JSON count change.

### 1.4 Weekly review (every Day 7, 14, 21, 28)

1. Run `replay_validator.summarise_replay` against the *current*
   shadow artefact directory and compare its
   `delta_pnl_pp_mean` against the *paper-test cumulative pnl_sum*
   for the candidate. Flag if signs diverge.
2. Run the full `tests/hedgerock/` suite. If any test fails,
   ABORT (see §3).
3. Append `paper_test_weekly_review` audit-trail entry with
   summary numbers.

### 1.5 Day-30 close-out

1. Final ledger summary: total trades, total pnl_sum, max drawdown.
2. Compare against the §2 pass/fail criteria.
3. If pass: produce a promotion packet via
   `scripts/hedgerock_evolution_promote.py` (dry-run; manual
   approval required afterward).
4. Append `paper_test_complete` audit-trail entry.

## 2. Pass / fail criteria (Day 30)

The candidate **passes** PAPER_TEST iff ALL of the following hold:

| Metric | Threshold |
|---|---|
| Total trades | ≥ 20 (XAUUSD-only; no multi-symbol) |
| Cumulative pnl_sum | > 0 (account-currency units the operator agreed on) |
| Max drawdown over the 30 days | ≥ -50 units (no breach of the floor) |
| Days with at least one trade | ≥ 12 (avoid degenerate "1 lucky trade" outcomes) |
| Sign-agreement with replay projection | the replay validator's `delta_pnl_pp_mean` and the paper-test pnl trajectory both positive |
| `registry_append_only_violation` | False on all 30 daily reports |
| Production code mtimes (rule_engine, AISMCReceiver.mq5) | unchanged across all 30 days |
| Production code mtimes (decision_server, phase_d_walk_forward) | unchanged across all 30 days; **read-only imports from `replay_validator` / `candidate_generator` are permitted (Tier-1 unseal v0.7.0)** |
| `policy_registry/approved/` | does not exist at Day 30 |
| `policy_registry/pointer.json` | does not exist at Day 30 |

The candidate **fails** PAPER_TEST iff ANY of the above does not
hold. A failed candidate flows to STALE in the queue (Day 30 + 1
or earlier if §3 abort fires).

## 3. Abort conditions

Abort the paper-test immediately if ANY of these triggers fire.
"Abort" means: stop fills on the demo broker, append the
`paper_test_abort` audit-trail entry with the reason, and do NOT
produce a promotion packet.

| Trigger | Detection |
|---|---|
| Cumulative drawdown breaches -50 floor | end-of-day `summarise()` shows `max_drawdown < -50` |
| Single-day pnl drop > 25% of cumulative pnl | end-of-day comparison vs prior day |
| Registry append-only violation observed | morning audit-log read shows `registry_append_only_violation: True` |
| Production code mtime drift on any of the four red-line files | morning mtime sweep |
| `policy_registry/approved/` or `pointer.json` appears | morning sweep |
| `tests/hedgerock/` regression (any test starts failing) | weekly review |
| Phase D bundle changes invalidate the bundle hash | morning recommend run shows `bundle_hash_sha256` mismatch with the recorded run |
| Operator notices a divergence between paper-test pnl and replay-validator projection (sign or magnitude order) | weekly review |
| Cross-symbol contamination (any non-XAUUSD ledger entry recorded) | the ledger's `build_paper_test_entry` would have raised `ValueError`; if it didn't, treat as a hard violation |

## 4. Reporting cadence

| Cadence | Artefact | Path (sidecar; never under approved/, pointer.json, or shadow_artefacts/) |
|---|---|---|
| Daily | Morning recommendation refresh | `<workspace>/day-<NN>/morning-recommendation.md` |
| Daily | Queue inspection | `<workspace>/day-<NN>/queue-inspection.md` |
| Daily | EOD summary | `<workspace>/day-<NN>/eod-summary.md` |
| Daily | Audit trail entry | `<workspace>/operation_audit.jsonl` (append) |
| Daily | Paper-test ledger | `<workspace>/ledger/paper_test_ledger.jsonl` (append) |
| Weekly | Replay validation | `<workspace>/week-<N>/replay-validation.md` |
| Weekly | Full hedgerock test sweep | `<workspace>/week-<N>/test-sweep.txt` |
| Day 30 | Final summary | `<workspace>/day-30/final-summary.md` |
| Day 30 | Dry-run promotion packet (if pass) | `<workspace>/day-30/packet.md` |

## 5. Roles

| Role | Responsibilities |
|---|---|
| Demo broker | Executes paper fills against XAUUSD quotes; appends to ledger; never routes to live |
| Operator (human) | Runs morning + EOD CLIs; reviews weekly summaries; signs the Day-30 packet |
| `replay_validator` | Re-projects historical shadow corpus weekly |
| `recommendation CLI` | Confirms candidate is still RECOMMEND each morning |
| `queue_inspect` CLI | Confirms candidate has not aged to STALE |
| `operation_audit` | Records each role's actions in the trail |

## 6. What this plan never does

- Routes any order to live trading or to a non-demo broker.
- Modifies `rule_engine.py`, `decision_server.py`,
  `phase_d_walk_forward.py`, or any `*.mq5` file. (Read-only
  imports of `decision_server` / `phase_d_walk_forward` from
  `replay_validator` / `candidate_generator` are explicitly
  allowed under the Tier-1 unseal — no writes.)
- Writes under `policy_registry/approved/` or
  `policy_registry/pointer.json`.
- Mutates `config/safety_bounds.yaml`.
- Deletes, renames, or rewrites any file under
  `policy_registry/shadow_artefacts/`.
- Auto-promotes a passing candidate. Day 30 produces a *dry-run*
  packet that a human carries forward through the manual
  HUMAN_APPROVE step.

## 7. Cross-references

- `docs/hedgerock-self-evolution-rfc.md` — five-layer state machine
  (PAPER_TEST is layer 4).
- `docs/hedgerock-shadow-broker-logging-design.md` — how the demo
  broker records the paper trades.
- `src/smc/hedgerock/evolution/paper_test_ledger.py` — ledger
  schema enforced for every paper fill.
- `scripts/hedgerock_evolution_promote.py` — dry-run promotion
  packet rendering at Day 30.
- `tests/hedgerock/evolution/test_paper_test_ledger.py` —
  invariants the ledger enforces.

## 8. Test-enforced anchors

The 30-day plan is data-only documentation. The behavioural
contracts it depends on are pinned by the existing test suite:

- Ledger XAUUSD-only invariant
  (`test_paper_test_entry_rejects_non_xauusd_symbol`).
- Ledger append-only contract (`test_ledger_appends_without_rewriting_prior_lines`).
- Promotion-helper apply-mode rejection
  (`test_apply_flag_is_always_rejected`).
- Promotion-helper paper-test thresholds
  (`test_insufficient_paper_trades_fails`,
   `test_negative_aggregate_pnl_fails`).
- Registry-violation abort
  (`test_registry_audit_violation_fails`).
- Operation-trail rejection of forbidden paths
  (`test_trail_path_under_forbidden_locations_is_rejected`).
- No production module deletes shadow artefacts
  (`test_artefact_registry_append_only.py`).

If a future change to this plan loosens any of those guarantees,
the test suite will block it; do not proceed without restoring
the invariant first.
