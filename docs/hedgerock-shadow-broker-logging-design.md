# Demo-Broker Shadow Logging — Technical Design (PAPER_TEST stage)

> **Design document only.** No production code in this design has
> been wired yet. The contract here is what the implementation MUST
> satisfy when it lands. Until that PR ships, paper trading uses
> the existing `PaperTestLedger.append` API directly with operator-
> supplied entries.

## 0. Goal

Record every "what the candidate would have done" decision the
demo broker makes against the XAUUSD live quote stream, without
ever routing an order to a live or non-demo venue. The output
is the same `paper_test_ledger.jsonl` schema the human-promotion
helper already consumes.

## 1. Non-goals

- Sending real orders. The shadow logger never opens a TCP socket
  to anything except the demo broker the operator supplies.
- Modifying `rule_engine.py`, `decision_server.py`,
  `phase_d_walk_forward.py`, or any `*.mq5`. The shadow logger is
  a sidecar reader of decisions, not a producer.
- Writing under `policy_registry/approved/`,
  `policy_registry/pointer.json`, or
  `policy_registry/shadow_artefacts/`.
- Multi-symbol coverage. XAUUSD only.
- Replay or back-testing — the shadow logger only records what
  happens *now* on the demo broker; back-testing happens in the
  `replay_validator` against historical shadow artefacts.

## 2. Architecture

```
┌──────────────────────┐    candidate parameter set
│ recommendation CLI   │───────────────────────────────────────┐
└──────────────────────┘                                       ▼
                                                ┌─────────────────────────┐
                                                │ shadow_broker_logger    │
                                                │  (NEW MODULE — sidecar) │
                                                └────────────┬────────────┘
                              decision payload (read-only)   │
   ┌──────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────┐  XAUUSD ticks  ┌─────────────────────┐
│ live decision_server │◄──────────────│ demo broker feed     │
└──────────────────────┘                └──────────────────────┘
   │
   ▼ (read-only mirror; no mutation)
┌─────────────────────────────┐
│ shadow_broker_logger snoops │
│  the decision_server's      │
│  serialised decision bus    │
└──────────────────────────────┘
   │
   ▼
   PaperTestLedger.append(entry)        ┌─────────────────┐
   ─────────────────────────────────────►│ ledger.jsonl    │
                                         └─────────────────┘
```

Key boundary: the logger reads from a **serialised bus** the
decision_server already exposes (status broadcast, JSON SSE / file
queue / similar). It does NOT import `decision_server` symbols and
does NOT call into `rule_engine`. The bus is the public read-only
interface to the running engine.

## 3. Module surface

The implementation lands at
`src/smc/hedgerock/evolution/shadow_broker_logger.py`. Frozen
dataclass and one entry function:

```python
@dataclass(frozen=True)
class ShadowFill:
    candidate_id: str
    decision_id: str          # the bus's primary key for the decision
    decision_at: str          # ISO 8601 UTC; bus timestamp
    symbol: str               # must be "XAUUSD"
    side: str                 # "long" | "short"
    size_lots: float
    entry_price: float
    exit_price: float | None  # None means open
    exit_at: str | None       # None means open
    pnl: float | None         # None until close
    drawdown: float | None    # max adverse excursion captured by the bus
    gates_at_entry: tuple[str, ...]
    audit_log_path: str

def record_shadow_fill(*, fill: ShadowFill, ledger_path: Path,
                       audit_log_path: Path) -> PaperTestEntry: ...
```

The module:

- Validates `symbol == "XAUUSD"` (raises `ValueError` otherwise).
- Refuses ledger paths under `policy_registry/approved/`,
  `policy_registry/pointer.json`, or
  `policy_registry/shadow_artefacts/`.
- Translates `ShadowFill` into `PaperTestEntry` and delegates the
  append to `PaperTestLedger.append`. No bypass of the existing
  ledger guardrails.

## 4. Data flow constraints

1. **One ledger row per closed shadow fill.** Open fills are
   tracked in memory only; nothing lands on disk until exit.
2. **No cross-candidate aggregation in the logger.** Aggregation
   is the ledger's `summarise()` job, the recommendation CLI's
   visualisations, and the replay validator's projection.
3. **Logger is single-threaded relative to one ledger file.**
   Multiple candidates running on different demo broker streams
   each get their own ledger file (still under the same workspace
   sidecar root). They MUST NOT share a ledger file because the
   `PaperTestLedger` instance opens the file in append mode for
   each call.
4. **Bus reader is read-only.** The logger never publishes back
   to the bus. It records, period.
5. **Idempotent under crash.** On restart, the logger drops all
   in-memory open fills (operator can reconcile from the bus's
   own audit if available). It never replays a closed fill twice
   because each `decision_id` is unique on the bus.

## 5. Failure modes

| Failure | Logger behaviour |
|---|---|
| Bus unreachable | Log to operation audit (`shadow_logger_bus_unreachable: fail`); do NOT write to the ledger; operator restart required. |
| Ledger path forbidden | `ValueError` at `PaperTestLedger` construction; abort start. |
| Symbol != XAUUSD | `ValueError`; abort the fill record. Bus must surface the violation. |
| Disk full | Bubble `OSError` from `PaperTestLedger.append`; logger dies. Restart on a fresh workspace path. |
| Decision contains stale candidate_id (not in queue, or queue says STALE) | Skip fill; emit operation audit `shadow_logger_skip_stale: ok` with the decision id. The fill never lands in the ledger. |
| Registry append-only violation observed during the day | Halt the logger immediately; emit `shadow_logger_halt_violation: fail`. Operator follows §3 abort in the 30-day plan. |

## 6. Operational checklist

Before starting the logger each morning:

- [ ] Confirm the demo broker is the **demo** broker (URL contains
      a `/demo/` segment or the explicit `--demo-broker` flag).
- [ ] Confirm the bus URL/path is read-only and authenticated.
- [ ] Confirm the ledger path is under the operator's workspace
      sidecar (NOT under `policy_registry/`).
- [ ] Confirm the operation audit trail is configured.
- [ ] Confirm the candidate is still QUEUED (not STALE) via
      `hedgerock_evolution_queue_inspect`.

## 7. Test enforcement (when implemented)

The future test suite must pin:

- `record_shadow_fill` rejects non-XAUUSD symbols.
- `record_shadow_fill` rejects ledger paths under approved/,
  pointer.json, shadow_artefacts/.
- The logger never imports `rule_engine`, `decision_server`,
  `phase_d_walk_forward`, or `mql5/`.
- Skip-stale path appends an operation audit entry but no ledger
  row.
- Idempotence: feeding the same `decision_id` twice records ONE
  ledger row.

These tests will live at
`tests/hedgerock/evolution/test_shadow_broker_logger.py` once the
implementation lands.

## 8. What this design intentionally leaves out

- **The bus protocol.** Whether the decision_server exposes JSON
  SSE, a Unix domain socket, a file queue, or a Redis stream is
  out of scope here — the design only requires that the bus is a
  *read-only public boundary* of the live engine. The first
  implementation PR will pick whichever bus already exists.
- **Reconciliation with broker statements.** The logger records
  what the bus emitted; whether the demo broker's overnight
  statement matches is an operator-side reconciliation step the
  30-day plan covers in §1.4 (weekly review).
- **Multi-symbol scaling.** Multi-symbol expansion is RFC-out-of-
  scope; the logger inherits the XAUUSD invariant from the
  ledger.

## 9. Cross-references

- `docs/hedgerock-30day-paper-trading-plan.md` — the day-by-day
  protocol that consumes this logger's output.
- `docs/hedgerock-self-evolution-rfc.md` §3 — PAPER_TEST is the
  fourth state.
- `src/smc/hedgerock/evolution/paper_test_ledger.py` — the schema
  this logger writes to.
- `src/smc/hedgerock/evolution/operation_audit.py` — the trail
  this logger appends to.
