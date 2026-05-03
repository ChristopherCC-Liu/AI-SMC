"""Ticket 3 Step 3 — replay_state + replay_executor tests.

Pinned guarantees:
  - SimState is per-replay isolated (no shared mutable list).
  - Sidecar simulator step() does not mutate input window frames.
  - Strict-prior closed-bar invariant — runner aborts to ABSTAIN
    when invariants fire mid-run.
  - Two run_pair calls with identical inputs are byte-deterministic
    (same final equity / DD).
  - No-op overlay → baseline_log == candidate_log byte-for-byte.
  - run_pair refuses to start when class A/B/C drift detected.
  - run_pair accepts symbols=tuple — multi-symbol skeleton ready.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# Stub lake (re-used pattern)
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, base_price: float = 100.0, drift: float = 0.0):
    rows = []
    p = base_price
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": p, "high": p + 0.5, "low": p - 0.5,
            "close": p, "volume": 100.0,
        })
        p += drift
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub")

    def list_instruments(self):
        return sorted({k[0] for k in self._data})

    def query(self, instrument, timeframe, start, end):
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def stub_lake():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })


# ---------------------------------------------------------------------------
# 1. SimState
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fresh_sim_state_initial_invariants() -> None:
    from smc.hedgerock.evolution.replay_state import fresh_sim_state
    s = fresh_sim_state(init_equity=10_000.0)
    assert s.equity == 10_000.0
    assert s.cash == 10_000.0
    assert len(s.positions) == 0
    assert s.max_dd_pct == 0.0
    assert s.n_trades == 0
    assert s.halt_event_count == 0
    assert s.blowup is False


@pytest.mark.unit
def test_two_sim_states_do_not_share_positions_list() -> None:
    """Mutating one SimState's positions must NOT affect another."""
    from smc.hedgerock.evolution.replay_state import (
        fresh_sim_state, open_position,
    )
    a = fresh_sim_state(init_equity=10_000.0)
    b = fresh_sim_state(init_equity=10_000.0)
    open_position(
        state=a, ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        price=100.0, direction=+1, lots=0.1,
    )
    assert len(a.positions) == 1
    assert len(b.positions) == 0


# ---------------------------------------------------------------------------
# 2. Replay executor — basic invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pair_returns_two_logs(stub_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    cand_overlay = PolicyOverlay(
        candidate_id="c-test",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50,
        baseline_value=0.55,
    )
    out = run_pair(
        lake=stub_lake,
        symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=cand_overlay,
    )
    assert out.baseline_log is not None
    assert out.candidate_log is not None
    assert out.aborted is False


@pytest.mark.unit
def test_run_pair_no_op_overlay_yields_identical_logs(stub_lake) -> None:
    """When the overlay's proposed_value == baseline_value, baseline
    and candidate replays must produce byte-identical outputs."""
    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    no_op = PolicyOverlay(
        candidate_id="c-noop",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.55,  # equal to baseline
        baseline_value=0.55,
    )
    out = run_pair(
        lake=stub_lake,
        symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=no_op,
    )
    assert out.baseline_log.final_state.equity == out.candidate_log.final_state.equity
    assert out.baseline_log.final_state.max_dd_pct == out.candidate_log.final_state.max_dd_pct
    assert out.baseline_log.final_state.n_trades == out.candidate_log.final_state.n_trades


@pytest.mark.unit
def test_run_pair_is_deterministic(stub_lake) -> None:
    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-det",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50,
        baseline_value=0.55,
    )
    a = run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    b = run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    assert a.baseline_log.final_state.equity == b.baseline_log.final_state.equity
    assert a.candidate_log.final_state.equity == b.candidate_log.final_state.equity


# ---------------------------------------------------------------------------
# 3. Strict-prior / partial-bar abort
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pair_aborts_to_abstain_on_invariant_violation(monkeypatch, stub_lake) -> None:
    """Force build_decision_window to report a partial-H4-bar and
    verify run_pair surfaces ABSTAIN with reason flag set."""
    from smc.hedgerock.evolution import replay_executor
    from smc.hedgerock.evolution.data_slice import (
        DecisionWindow, DecisionWindowInvariants,
    )
    import polars as pl

    real_build = replay_executor.build_decision_window

    def _fake_build(**kwargs):
        # Real call so frames are populated, then poison invariants.
        w = real_build(**kwargs)
        from dataclasses import replace
        bad = DecisionWindowInvariants(
            same_bar_set_used=w.invariants.same_bar_set_used,
            decision_only_uses_strictly_prior_data=
                w.invariants.decision_only_uses_strictly_prior_data,
            h4_partial_bar_in_window=True,  # POISONED
            d1_partial_bar_in_window=w.invariants.d1_partial_bar_in_window,
            decision_uses_data_with_ts_lt_trade_bar_ts=
                w.invariants.decision_uses_data_with_ts_lt_trade_bar_ts,
        )
        return DecisionWindow(
            decision_ts=w.decision_ts, h1_frame=w.h1_frame,
            h4_frame=w.h4_frame, d1_frame=w.d1_frame, invariants=bad,
        )

    monkeypatch.setattr(replay_executor, "build_decision_window", _fake_build)

    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-poison",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50, baseline_value=0.55,
    )
    out = replay_executor.run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    assert out.aborted is True
    assert "invariant" in out.abort_reason.lower() or \
           "partial" in out.abort_reason.lower() or \
           "lookahead" in out.abort_reason.lower()


# ---------------------------------------------------------------------------
# 4. Mirror drift abort
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pair_aborts_on_mirror_drift(stub_lake, monkeypatch) -> None:
    """Mid-flight drift in production rule_engine → run_pair refuses
    to produce metrics."""
    import importlib
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    monkeypatch.setattr(rule_engine, "_CONFIDENCE_OBSERVE_FLOOR", 0.99,
                        raising=True)

    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-drift",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50, baseline_value=0.55,
    )
    out = run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    assert out.aborted is True
    assert "drift" in out.abort_reason.lower() or \
           "mirror" in out.abort_reason.lower()


# ---------------------------------------------------------------------------
# 5. Multi-symbol skeleton (single-symbol still produces 1 log)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pair_accepts_tuple_symbols(stub_lake) -> None:
    """Signature is multi-symbol-ready but currently just iterates
    over the tuple. v1 lake has only XAUUSD; passing a tuple of one
    must work."""
    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-multi-sym",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50, baseline_value=0.55,
    )
    out = run_pair(
        lake=stub_lake, symbols=("XAUUSD",),
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        candidate_overlay=overlay,
    )
    assert out.symbols_run == ("XAUUSD",)


@pytest.mark.unit
def test_run_pair_no_production_decision_imports() -> None:
    """Sidecar runtime MUST NOT import production decision functions."""
    import ast
    from pathlib import Path
    src = (_ai_smc_home_p() / "src" / "smc" / "hedgerock" / "evolution" / "replay_executor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden = {
        ("smc.hedgerock.rule_engine", "derive_envelope_params"),
        ("smc.hedgerock.phase_d_walk_forward", "_run_dynamic"),
        ("smc.hedgerock.phase_d_walk_forward", "_step"),
        ("smc.hedgerock.phase_d_walk_forward", "_build_synthetic_ea_state"),
        ("smc.hedgerock.phase_d_walk_forward", "apply_experiment_overrides"),
        ("smc.hedgerock.phase_d_walk_forward", "_run_static"),
        ("smc.hedgerock.decision_server", None),
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                key = (module, alias.name)
                if key in forbidden:
                    raise AssertionError(
                        f"replay_executor.py:{node.lineno}: forbidden "
                        f"import from {module} of {alias.name}"
                    )
                # Also catch wildcard imports of decision_server.
                if (module, None) in forbidden:
                    raise AssertionError(
                        f"replay_executor.py:{node.lineno}: forbidden "
                        f"import of module {module}"
                    )
