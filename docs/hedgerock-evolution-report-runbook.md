# HedgeRock Evolution Report — Operator Runbook (T4-F3)

> **Read-only diagnostic.** Running `scripts/hedgerock_evolution_report.py`
> never modifies `rule_engine.py`, `decision_server.py`, EA `*.mq5`,
> `policy_registry/approved/`, `policy_registry/pointer.json`, or
> `config/safety_bounds.yaml`. The CLI evaluates `CANDIDATE_MENU_V0`
> against the current Phase D evidence + the registry append-only audit
> log, writes a markdown report, and appends candidate manifests to a
> registry directory of your choosing.

This runbook anchors the contracts pinned by
`tests/hedgerock/evolution/test_evolution_report_t4f3.py`. Any flag
rename or default change that breaks this runbook breaks that suite.

## When to run

- After a new Phase D walk-forward run, to refresh the gate verdicts.
- To verify the registry append-only audit log produces the expected
  G8 ABSTAIN block on every candidate when the log records a violation.
- During a sandboxed dry-run before touching the real
  `/Users/christopher/HedgeRock/policy_registry`.

## Pre-flight

1. Confirm you are on the intended branch (`r10-phase1` for
   T4-series follow-ons).
2. Confirm the production registry exists and is read-only to
   anything except an appender:
   ```
   ls /Users/christopher/HedgeRock/policy_registry/shadow_artefacts/
   ```
3. Confirm the audit log is present at the canonical path:
   ```
   /Users/christopher/HedgeRock/policy_registry/shadow_artefacts/_audit.md
   ```

## Sandboxed dry-run (recommended)

Use a tmp registry root and a tmp report path. The audit log is
copied — never moved — so the original record stays append-only.

```bash
TMP=$(mktemp -d)
cp /Users/christopher/HedgeRock/policy_registry/shadow_artefacts/_audit.md \
   $TMP/_audit.md

python scripts/hedgerock_evolution_report.py \
  --registry-root        $TMP/registry \
  --report-path          $TMP/phase-d-evolution-report.md \
  --registry-audit-log   $TMP/_audit.md
```

Expected outcome (matches T4-F3 test assertions):

- Stdout summary prints `evaluated 4 candidates` followed by 4
  `PROMOTION_BLOCKED` lines.
- `$TMP/phase-d-evolution-report.md` header carries
  `` `registry_append_only_violation`: **True** ``,
  `` `lost_sha_count`: **4** ``, and the path you passed via
  `--registry-audit-log`.
- Each per-candidate `G8` line reads:
  ```
  - G8: **ABSTAIN** — registry_append_only_violation: shadow-artefact
    registry contract violated this session (lost_sha_count=4); see
    audit log '<your-audit-log-path>'; no candidate may PASS until the
    session restarts clean
  ```
- The real
  `/Users/christopher/HedgeRock/policy_registry/shadow_artefacts/`
  directory is NOT modified.

## Production-targeted run (default audit-log resolution)

Drop the `--registry-audit-log` flag. The CLI derives
`<registry_root>/shadow_artefacts/_audit.md` automatically. Pointing
`--registry-root` at the production directory will surface the real
log without you having to spell its path:

```bash
python scripts/hedgerock_evolution_report.py \
  --registry-root /Users/christopher/HedgeRock/policy_registry \
  --report-path   /Users/christopher/HedgeRock/docs/phase-d-evolution-report.md
```

This is the only invocation that writes into the production registry's
`candidates/` and `audit/` subdirectories. The `shadow_artefacts/`
tree is read-only to this CLI — no path under it is ever opened for
writing.

## No-audit-log scenario (red flag)

If the audit log file has been moved or renamed, the CLI will NOT
raise. Instead the report header carries:

- `` `audit_log_present`: **False** ``
- The resolved path the CLI was looking at.

**Treat this as a red flag, not a clean run.** Verify `--registry-audit-log`
points at the real path; if you genuinely intended an empty log, the
absence-surfaced output is your audit trail.

## Acceptance criteria checklist

Before you forward the report to a human approver:

- [ ] Every candidate shows `RESULT: PROMOTION_BLOCKED / *` (auto-promote
      forbidden by RFC §11).
- [ ] G8 = `ABSTAIN` for all candidates whenever the audit log records a
      violation.
- [ ] G8 reason names the audit log path you supplied.
- [ ] `lost_sha_count` in the header matches the count in each G8 reason.
- [ ] The "Canonical certification" line reads
      `safety_bounds write permission: 0`.
- [ ] `git status` for `/Users/christopher/HedgeRock/policy_registry/`
      shows no unexpected modifications under `shadow_artefacts/`.

## Out-of-scope by design

- The CLI does not write under `policy_registry/approved/` or
  `policy_registry/pointer.json`. Promotion to "approved" is a
  human-only step downstream of this report.
- The CLI does not run the EA, the strategy server, or the decision
  server. It only reads Phase D markdown artefacts and writes its own
  report + candidate manifests.
- The CLI does not invoke any code under `src/smc/hedgerock/rule_engine.py`.
  As of Tier-1 unseal v0.7.0, `decision_server.py` and
  `phase_d_walk_forward.py` are imported **read-only** from
  `replay_validator.py` and `candidate_generator.py` only — the
  whitelist is pinned by
  `tests/hedgerock/evolution/test_regression_guard.py`. The
  CLI itself routes live-parameter access through
  `candidate_generator.get_live_parameter_snapshot()` rather than
  importing `decision_server` directly. T4-F3 tests assert the
  registry-write isolation via `_real_registry_json_count()`
  invariants and the module-level boundary tests in
  `tests/hedgerock/evolution/test_artefact_registry_append_only.py`.

## Test enforcement

The contracts in this runbook are pinned by:

- `tests/hedgerock/evolution/test_evolution_report_t4f3.py` — argv-path
  dry-run (real / tmp-copy / no-log / default-resolution / bad-argv).
- `tests/hedgerock/evolution/test_evolution_report_t4f2.py` — library
  `run()` contract for the same scenarios.
- `tests/hedgerock/evolution/test_artefact_registry_append_only.py` —
  no production module under `scripts/` or `src/smc/hedgerock/evolution/`
  carries a delete primitive against the shadow-artefact registry.

If a future change needs to alter the CLI's argv surface, update this
runbook *and* the T4-F3 suite in the same commit.
