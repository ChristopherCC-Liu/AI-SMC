# AI-SMC Architecture Notes

This document captures architectural decisions and semantics that are not
obvious from reading the code alone. Companion to `docs/HANDOFF.md` (which
is the developer on-ramp) and `docs/planning/mt5-smc-blueprint.md` (which is
the original design spec).

---

## Journal semantics (Round 6 B3)

> Round 4 v5 made the EA the sole executor. Python's `order_send` path is
> PERMANENTLY DISABLED. This note exists because "mode=PAPER" in the
> journal no longer means "no trade happened" — it means "Python did not
> itself send an order", which under Round 4 v5 is the normal case.

### What the journal is, and what it is not

The journal (`data/{SYMBOL}/journal*/live_trades.jsonl`) records **what
Python wanted to happen** each M15 cycle: the selected setup, the
regime-aware parameters, the computed SL/TP, the session gate decisions.
It is an **intent log**, not an execution truth log.

Three consequences:

1. A journal row exists whether or not the broker accepted the order.
2. `mode=PAPER` historically meant "Python did not call MT5 `order_send`",
   but under Round 4 v5 Python never calls `order_send` — so every live
   journal row is `mode=PAPER` even when the EA successfully executes.
3. Ticket attribution (`mt5_ticket`) starts `null` because the journal is
   written before the EA round-trip.  A future enhancement is EA
   write-back (see "Future work" below).

### Two sources of truth, one reconciler

| Source | What it records | Authoritative for |
|--------|-----------------|-------------------|
| Journal (`live_trades.jsonl`) | Python's intent at signal time | Parameters, SL/TP math, regime tags, why we wanted this trade |
| Broker history (`mt5.history_deals_get`) | What MT5 actually executed | Fills, PnL, close time, realised R |

The reconciler (`live_demo.py` → `fetch_closed_pnl_since` +
`build_trade_close_context`) bridges the two on close: it pulls the
broker-side deal, then looks back into the journal (via
`_find_regime_at` / entry-time match) to enrich the Telegram push and
`trade_closed` event with regime + leg metadata.

The flow, per cycle:

```
┌────────────┐   write intent   ┌─────────────────────────┐
│ live_demo  │ ───────────────▶ │ journal/live_trades.jsonl│
└─────┬──────┘                   └─────────────────────────┘
      │ POST /signal
      ▼
┌────────────┐   place order    ┌──────────────────────┐
│ EA (MT5)   │ ───────────────▶ │ broker history/deals │
└─────┬──────┘                   └─────────┬────────────┘
      │                                    │
      │                                    │ fetch_closed_pnl_since
      ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│ reconciler: trade_reconciled / trade_closed events  │
│   - pnl from broker                                 │
│   - regime / leg / rr from journal lookback        │
└─────────────────────────────────────────────────────┘
```

### The `execution_path` field (new in Round 6 B3)

Every new journal row carries `execution_path` alongside the legacy
`mode` field:

| Value | Meaning | When written |
|-------|---------|--------------|
| `"ea"` | EA is (or was) the executor | Default — Round 4 v5 production |
| `"paper"` | No execution at all | `--paper` / `--no-execute` CLI flag set (`PAPER_MODE=True`) |
| `"python"` | Python `order_send` executed | **Reserved** — never written today. Reactivated only if Round 4 v5 is rolled back. |

Legacy rows (pre-Round-6) have **no** `execution_path` field.
Consumers (measurement-lead audits, `regret_analysis.py`,
`build_daily_digest`) apply the convention: **missing field ⇒ assume
`"ea"`** (Round 4 v5 shipped before Round 6, so backfill is unnecessary).

The `mode` field is retained unchanged for reverse compatibility with
existing grep scripts, dashboards, and bunker tooling that parse
`mode=PAPER` / `LIVE_EXEC` / `MT5_FAIL_*` / `MARGIN_GATED`.

### Why both fields?

`mode` and `execution_path` answer different questions:

- `mode` → "what was the Python-side order_send outcome?" (retcode,
  retry, margin gate, paper short-circuit). Useful for diagnosing
  Python-side MT5 regressions.
- `execution_path` → "who actually puts the trade on the broker?"
  Useful for audit — the measurement team can now say "this journal
  row is an EA intent, not an orphan signal".

### Future work: EA write-back of ticket

Right now `mt5_ticket` is populated only when Python's legacy
`send_with_retry` wrapper succeeds (dormant in Round 4 v5), so it is
usually `null` in fresh journal rows. To close the attribution loop,
the EA could POST a ticket acknowledgement back to `strategy_server`
keyed by the `(magic, symbol, entry_time, direction)` tuple, which
would then patch the matching journal row. Tracked as a Round 7
candidate — not required for any current audit.

---
