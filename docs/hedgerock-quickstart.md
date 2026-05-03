# HedgeRock Evolution Sidecar — Quick Start

Three commands from a fresh `git clone` to a green test suite.

## 1. Clone

```bash
git clone <this-repo> AI-SMC
cd AI-SMC
```

Requirements:
- Python **3.11+** (3.12 recommended)
- macOS / Linux (Windows-native untested for the evolution layer; use WSL)
- ~1 GB disk for the venv + deps

No external data, no broker account, no LLM key needed for the
evolution sidecar.

## 2. Bootstrap

```bash
bash scripts/hedgerock_bootstrap.sh
```

The script:

1. Auto-detects a Python 3.11+ interpreter (override with
   `PYTHON_BIN=/usr/bin/python3.12`).
2. Creates `.venv/` at the repo root (override with
   `VENV_DIR=/path/to/venv`).
3. Installs `ai-smc[dev]` in editable mode (`pip install -e ".[dev]"`).
4. Copies `.env.example` → `.env` (only when `.env` is absent).
5. Runs `pytest tests/hedgerock/evolution -q` to verify the install.

Expected tail:

```
... .................... [100%]
805 passed, 1 skipped, ... in 2.7s
[bootstrap] OK — evolution sidecar is installed and green.
```

The single skip is `test_audit_log_exists_and_records_2026_05_02_incident`,
which only fires when the operator-team registry is present at
`$HEDGEROCK_HOME/policy_registry/shadow_artefacts/_audit.md`. Fresh
checkouts don't have that file and the test skips cleanly.

## 3. Run the demo

```bash
source .venv/bin/activate
python scripts/hedgerock_evolution_demo.py --workspace /tmp/hedgerock-demo
```

The demo walks the full report-only loop (OBSERVE → REGIME +
ANOMALY → TIMEFRAME + STOP → STRESS-TEST → DETECT + RECOMMEND →
QUEUE → INSPECT → PAPER-TEST SEED → DRY-RUN PROMOTION) and writes
every artefact under `/tmp/hedgerock-demo`. Nothing under
`policy_registry/approved/` or `pointer.json` is ever touched.

## Configuration

All runtime locations come from environment variables with
sensible defaults — none need to be set on a fresh checkout.

| Var               | Default                  | Purpose                                  |
|-------------------|--------------------------|------------------------------------------|
| `HEDGEROCK_HOME`  | `$HOME/HedgeRock`        | Operator-team registry root              |
| `AI_SMC_HOME`     | This repo's root         | AI-SMC checkout root                     |
| `PYTHON_BIN`      | auto-detected            | Python interpreter used by bootstrap     |
| `VENV_DIR`        | `<repo>/.venv`           | Where bootstrap creates the venv         |

Override in `.env` (loaded by `python-dotenv` when the runtime
needs it) or in your shell:

```bash
export HEDGEROCK_HOME=/opt/hedgerock
bash scripts/hedgerock_bootstrap.sh
```

## What the evolution sidecar can / can't do

**Can** (report-only, plug-and-play):

- Detect market regime + anomaly + adaptive stops + multi-TF consensus.
- Generate parameter-tweak proposals with full safety-clamp + RFC §5
  validation.
- Run adversarial stress tests against six builtin shock scenarios
  (covid 2020, Russia/Ukraine 2022, SVB 2023, mid-east 2023, JPY
  intervention 2024, Fed pivot 2023).
- Render markdown recommendation reports with NOT-LIVE banners.
- Append candidates to a sidecar shadow-test queue (append-only).
- Produce dry-run promotion packets with optional 7-line operator
  checklist.

**Can't** (by design — these are the red lines):

- Modify `rule_engine.py`, `decision_server.py` (writes),
  `phase_d_walk_forward.py` (writes), or any `.mq5` file.
- Promote a candidate to `policy_registry/approved/`.
- Update `policy_registry/pointer.json`.
- Delete, rename, or rewrite a file under
  `policy_registry/shadow_artefacts/`.
- Send any signal that lands on a live trade decision path.

The Tier-1 unseal allows **read-only** imports of
`decision_server.get_live_parameters` and
`phase_d_walk_forward.run_walk_forward_backtest` from
`replay_validator.py` and `candidate_generator.py` only —
enforced by `tests/hedgerock/evolution/test_regression_guard.py`.

## Where to next

- **RFC**: `docs/hedgerock-self-evolution-rfc.md` — five-layer
  state machine + invariants.
- **Operator runbook**: `docs/hedgerock-evolution-report-runbook.md`.
- **30-day paper trading plan**:
  `docs/hedgerock-30day-paper-trading-plan.md`.
- **Final acceptance**:
  `docs/hedgerock-self-evolution-final-acceptance.md`.
- **Evolution README**: `src/smc/hedgerock/evolution/README.md`
  — onboarding for new contributors.

## Troubleshooting

| Symptom                                                   | Fix                                                                             |
|-----------------------------------------------------------|---------------------------------------------------------------------------------|
| `ERROR: Python 3.11+ not found on PATH`                   | Install Python 3.11+; or set `PYTHON_BIN=/path/to/python3.12`.                  |
| `ModuleNotFoundError: No module named 'fastapi'`          | Bootstrap didn't run cleanly. Re-run: `bash scripts/hedgerock_bootstrap.sh`.    |
| `pytest` skips `test_audit_log_exists_*`                  | Expected on fresh machines (the operator-team registry isn't present).         |
| `SyntaxWarning: invalid escape sequence`                  | Cosmetic; ignore. Comes from a bundled third-party regex literal.              |
| Evolution tests pass but `tests/smc` fails                | Pre-existing test pollution unrelated to the evolution layer; safe to ignore.  |

Got a fresh-machine breakage not listed here? Open an issue with
the failing pytest output and the value of `python --version`.
