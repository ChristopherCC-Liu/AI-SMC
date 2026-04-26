# Apr 20 2026 02:46 UTC — Stacked-SL Incident Fixture

## What this fixture proves

That `smc.risk.concurrent_gates.check_concurrent_cap` and
`check_anti_stack_cooldown`, when both wired at the live pre-write gate
(R4 v5 Option B), correctly **block a 6th BUY entry** when:

- One BUY position is already open with the production magic.
- Five long-direction opening deals were observed in the prior 30
  minutes from the same magic.

Without these gates the R4-pre-fix code path would have allowed the new
setup through — that's the bug this fixture documents.

## State semantics

`state.json` records the post-incident-reconstructed inputs:

| Field                       | Meaning                                                |
| --------------------------- | ------------------------------------------------------ |
| `magic`                     | Production magic (19760418 = XAUUSD control leg).      |
| `max_concurrent`            | Cap configured at the gate (1 per leg in production).  |
| `anti_stack_cooldown_minutes` | 30 — the production cooldown.                        |
| `open_positions`            | Active positions visible to `mt5.positions_get`.       |
| `recent_long_entries`       | Recent opening deals from `mt5.history_deals_get`.     |

The fixture is **synthetic** — production journal data is private and not
checked into this repo. The shape, magic, and timing match the production
incident; the actual broker tickets do not.

## How to regenerate

`state.json` is hand-edited. To add or move entries, edit the JSON
directly. Keep `time_utc` strictly within the cooldown window relative to
`now_utc` so the test still asserts what it claims to assert.

## Assertions

`test_apr_20_stacked_sl.py` runs two complementary tests:

1. **Fix enabled** — both gates evaluated. Outcome must be: NOT
   `can_trade`. (No specific reason name asserted — the outcome is what
   matters.)
2. **Baseline without fix** — gates are bypassed. Marked
   `pytest.mark.xfail(strict=True, reason="documents the bug")`. The
   "expected failure" is the assertion that NO BUY would proceed; with
   the gates off the assertion fails (i.e., BUY would proceed) which
   pytest interprets as the expected xfail. If a future round
   accidentally re-blocks via some other path, the xfail flips to XPASS
   under `strict=True` and the suite fails — surfacing the change for
   review.
