"""Ticket 3 Step 4 — runtime isolation tests.

Two symmetric AST-based hard boundaries (per Ticket 3 plan §R3):

  - Boundary A: runtime sidecar modules MUST NOT import production
    decision functions (rule_engine.derive_envelope_params,
    decision_server.*, phase_d_walk_forward._run_dynamic, _step,
    _build_synthetic_ea_state, apply_experiment_overrides,
    _run_static).

  - Boundary B: only test files (under tests/hedgerock/evolution/)
    may import production derive_envelope_params. Any source under
    src/ or scripts/ that does so → AssertionError.

Plus:
  - replay_executor / shadow_runner / scripts must not call
    setattr / delattr / exec / eval / compile, monkeypatch.* etc.
  - replay_executor pre-flight production-constants snapshot is
    byte-equal to post-flight snapshot (proven by integration test).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)


_REPO_ROOT = (_ai_smc_home_p())
_RUNTIME_PATHS = [
    _REPO_ROOT / "src" / "smc" / "hedgerock" / "evolution",
    _REPO_ROOT / "scripts" / "hedgerock_shadow_run.py",
    _REPO_ROOT / "scripts" / "hedgerock_evolution_report.py",
]


def _runtime_python_files() -> list[Path]:
    out: list[Path] = []
    for p in _RUNTIME_PATHS:
        if p.is_dir():
            out.extend(p.glob("*.py"))
        elif p.is_file():
            out.append(p)
    # Exclude __pycache__ etc.
    return [f for f in out if f.is_file() and not f.name.startswith("__")
            or f.name == "__init__.py"]


def _all_python_files_for_b_boundary() -> list[Path]:
    """Boundary B scans BOTH runtime and tests; tests are the only
    place derive_envelope_params is allowed."""
    out: list[Path] = []
    for d in (_REPO_ROOT / "src" / "smc" / "hedgerock" / "evolution",
              _REPO_ROOT / "scripts",
              _REPO_ROOT / "tests" / "hedgerock" / "evolution"):
        if d.is_dir():
            out.extend(d.glob("**/*.py"))
    return [f for f in out if f.is_file()]


# ---------------------------------------------------------------------------
# Boundary A: runtime modules MUST NOT import production decision-side code
# ---------------------------------------------------------------------------


_FORBIDDEN_FROM_IMPORTS_RUNTIME = {
    # (module, name) — name=None means any import from the module.
    ("smc.hedgerock.rule_engine", "derive_envelope_params"),
    ("smc.hedgerock.phase_d_walk_forward", "_run_dynamic"),
    ("smc.hedgerock.phase_d_walk_forward", "_step"),
    ("smc.hedgerock.phase_d_walk_forward", "_build_synthetic_ea_state"),
    ("smc.hedgerock.phase_d_walk_forward", "apply_experiment_overrides"),
    ("smc.hedgerock.phase_d_walk_forward", "_run_static"),
}
_FORBIDDEN_MODULE_IMPORTS_RUNTIME = {
    "smc.hedgerock.decision_server",
}


@pytest.mark.unit
def test_runtime_modules_do_not_import_production_decision_path() -> None:
    failures: list[str] = []
    for f in _runtime_python_files():
        text = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            failures.append(f"{f.name}: parse error: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in _FORBIDDEN_MODULE_IMPORTS_RUNTIME:
                    failures.append(
                        f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                        f"forbidden from-import of module {module}"
                    )
                for alias in node.names:
                    if (module, alias.name) in _FORBIDDEN_FROM_IMPORTS_RUNTIME:
                        failures.append(
                            f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                            f"forbidden from-import: from {module} import {alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULE_IMPORTS_RUNTIME:
                        failures.append(
                            f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                            f"forbidden import: {alias.name}"
                        )
    if failures:
        raise AssertionError("\n".join(failures))


# ---------------------------------------------------------------------------
# Boundary B: only test files may import derive_envelope_params
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_only_test_modules_import_production_derive_envelope_params() -> None:
    """ALLOW production derive_envelope_params in test/* only."""
    violations: list[str] = []
    for f in _all_python_files_for_b_boundary():
        if "tests/" in str(f):
            continue  # tests are exempt
        text = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if (module == "smc.hedgerock.rule_engine"
                            and alias.name == "derive_envelope_params"):
                        violations.append(
                            f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                            "imports derive_envelope_params (only allowed in tests/)"
                        )
    if violations:
        raise AssertionError("\n".join(violations))


# ---------------------------------------------------------------------------
# Mutation-call scan: no setattr/delattr/exec/eval/compile/monkeypatch
# ---------------------------------------------------------------------------


_FORBIDDEN_CALL_NAMES = {"setattr", "delattr", "exec", "eval", "compile"}
_FORBIDDEN_ATTR_CALLS = {"setattr", "delattr", "monkeypatch", "patch"}


@pytest.mark.unit
def test_runtime_modules_have_no_setattr_or_monkeypatch_calls() -> None:
    """Runtime sidecar must not setattr / monkeypatch anything —
    overlay must always be applied via apply_overlay returning a
    new dict, not by writing to a module."""
    violations: list[str] = []
    for f in _runtime_python_files():
        text = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (isinstance(node.func, ast.Name)
                    and node.func.id in _FORBIDDEN_CALL_NAMES):
                violations.append(
                    f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                    f"forbidden call {node.func.id}()"
                )
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in _FORBIDDEN_ATTR_CALLS):
                violations.append(
                    f"{f.relative_to(_REPO_ROOT)}:{node.lineno}: "
                    f"forbidden attribute call .{node.func.attr}()"
                )
    if violations:
        raise AssertionError("\n".join(violations))


# ---------------------------------------------------------------------------
# Production constants byte-identical before/after a run_pair call
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pair_does_not_mutate_rule_engine_constants() -> None:
    """End-to-end: snapshot rule_engine constants, run a real
    run_pair, snapshot again — must be byte-identical."""
    import importlib
    from datetime import datetime, timedelta, timezone
    import polars as pl

    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    before = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }

    # Build a tiny stub lake.
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _bars(start, n, hours_step=1.0):
        rows = []
        for i in range(n):
            ts = start + timedelta(hours=hours_step * i)
            rows.append({"ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "volume": 100.0})
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

    lake = _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })

    from smc.hedgerock.evolution.replay_executor import run_pair
    from smc.hedgerock.evolution.policy_overlay import PolicyOverlay
    overlay = PolicyOverlay(
        candidate_id="c-leak-check",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50, baseline_value=0.55,
    )
    run_pair(
        lake=lake, symbols=("XAUUSD",),
        start=base, end=base + timedelta(days=30),
        candidate_overlay=overlay,
    )

    after = {
        k: getattr(rule_engine, k) for k in dir(rule_engine)
        if not k.startswith("__")
        and isinstance(getattr(rule_engine, k, None),
                       (int, float, str, bool, tuple))
    }
    assert before == after, (
        "run_pair leaked overlay into production rule_engine constants"
    )
