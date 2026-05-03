"""Ticket 2 Step 3 — policy_overlay immutability tests.

Pinned guarantees (per R6 step 3, R3 plan §4):
  - PolicyOverlay is frozen.
  - apply_overlay returns a new mapping; input is unmodified.
  - The overlay does NOT call setattr on the production rule_engine
    module nor monkeypatch derive_envelope_params.
  - Source AST: no setattr / monkeypatch / module-level assignment
    against rule_engine or phase_d_walk_forward.
  - After apply_overlay round-trip, production module-level constants
    snapshot is byte-identical (no global pollution).
  - candidate_id and overlay_id are deterministic + frozen.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

from smc.hedgerock.evolution.policy_overlay import (
    OVERLAY_SCHEMA_VERSION,
    PolicyOverlay,
    apply_overlay,
    compute_overlay_id,
)


from tests.hedgerock.evolution._paths import (
    ai_smc_home as _ai_smc_home_p,
    hedgerock_home as _hedgerock_home_p,
    real_audit_log as _real_audit_log_p,
    real_registry_root as _real_registry_p,
    real_shadow_artefacts_root as _real_shadow_p,
    scripts_dir as _scripts_dir_p,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_overlay() -> PolicyOverlay:
    return PolicyOverlay(
        candidate_id="c-test",
        target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        proposed_value=0.50,
        baseline_value=0.55,
    )


def _snapshot_module_constants(mod_name: str) -> dict[str, object]:
    """Snapshot module-level constants whose names start with '_' and
    that hold primitive values (catches the rule-engine knobs)."""
    mod = importlib.import_module(mod_name)
    out: dict[str, object] = {}
    for name in dir(mod):
        if not name.startswith("_"):
            continue
        if name.startswith("__"):
            continue
        try:
            val = getattr(mod, name)
        except Exception:
            continue
        if isinstance(val, (int, float, str, bool, tuple)) or val is None:
            out[name] = val
    return out


# ---------------------------------------------------------------------------
# 1. Frozen + schema version
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_overlay_schema_version_is_v1() -> None:
    assert OVERLAY_SCHEMA_VERSION == "1.0.0"


@pytest.mark.unit
def test_overlay_is_frozen_dataclass() -> None:
    o = _bare_overlay()
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        o.proposed_value = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. compute_overlay_id is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_overlay_id_deterministic() -> None:
    a = _bare_overlay()
    b = _bare_overlay()
    assert compute_overlay_id(a) == compute_overlay_id(b)


@pytest.mark.unit
def test_overlay_id_changes_with_proposed_value() -> None:
    a = _bare_overlay()
    from dataclasses import replace
    b = replace(a, proposed_value=0.49)
    assert compute_overlay_id(a) != compute_overlay_id(b)


@pytest.mark.unit
def test_overlay_id_changes_with_target() -> None:
    a = _bare_overlay()
    from dataclasses import replace
    b = replace(a, target="smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE")
    assert compute_overlay_id(a) != compute_overlay_id(b)


# ---------------------------------------------------------------------------
# 3. apply_overlay returns a NEW mapping; input untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overlay_returns_new_mapping() -> None:
    base = {"smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": 0.55}
    overlay = _bare_overlay()
    out = apply_overlay(base, overlay)
    assert out is not base
    assert base == {"smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": 0.55}, (
        "apply_overlay mutated its input mapping"
    )
    assert out["smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"] == 0.50


@pytest.mark.unit
def test_apply_overlay_preserves_other_keys() -> None:
    base = {
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": 0.55,
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR": 0.80,
    }
    overlay = _bare_overlay()
    out = apply_overlay(base, overlay)
    assert out["smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"] == 0.80


@pytest.mark.unit
def test_apply_overlay_target_missing_raises() -> None:
    """If the overlay target key isn't present in the params snapshot,
    apply_overlay refuses to add it silently (a missing key indicates
    the snapshot wasn't built from the right mirror)."""
    base = {"some_other_key": 1.0}
    overlay = _bare_overlay()
    with pytest.raises(KeyError):
        apply_overlay(base, overlay)


# ---------------------------------------------------------------------------
# 4. apply_overlay does NOT touch production rule_engine module
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_overlay_does_not_mutate_rule_engine_constants() -> None:
    snapshot_before = _snapshot_module_constants("smc.hedgerock.rule_engine")
    base = {"smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": 0.55}
    overlay = _bare_overlay()
    apply_overlay(base, overlay)
    snapshot_after = _snapshot_module_constants("smc.hedgerock.rule_engine")
    assert snapshot_before == snapshot_after, (
        "apply_overlay leaked into production rule_engine module-level state"
    )


@pytest.mark.unit
def test_apply_overlay_does_not_mutate_phase_d_walk_forward_constants() -> None:
    snapshot_before = _snapshot_module_constants(
        "smc.hedgerock.phase_d_walk_forward"
    )
    base = {
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE": 4.0,
    }
    from dataclasses import replace
    overlay = replace(
        _bare_overlay(),
        target="smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE",
        proposed_value=6.0,
        baseline_value=4.0,
    )
    apply_overlay(base, overlay)
    snapshot_after = _snapshot_module_constants(
        "smc.hedgerock.phase_d_walk_forward"
    )
    assert snapshot_before == snapshot_after


# ---------------------------------------------------------------------------
# 5. Static AST: no setattr / monkeypatch in policy_overlay source
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_overlay_source_has_no_setattr_or_monkeypatch_calls() -> None:
    src_path = _ai_smc_home_p() / "src" / "smc" / "hedgerock" / "evolution" / "policy_overlay.py"
    text = src_path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden_call_names = {"setattr", "delattr", "exec", "eval", "compile"}
    forbidden_attrs = {"setattr", "delattr", "monkeypatch", "patch"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in forbidden_call_names:
            raise AssertionError(
                f"policy_overlay.py:{node.lineno}: forbidden call {node.func.id}()"
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
            raise AssertionError(
                f"policy_overlay.py:{node.lineno}: forbidden attribute call "
                f".{node.func.attr}()"
            )


@pytest.mark.unit
def test_overlay_source_does_not_import_production_runtime() -> None:
    """policy_overlay must not import rule_engine, decision_server, or
    phase_d_walk_forward — it operates on plain dict snapshots."""
    src_path = _ai_smc_home_p() / "src" / "smc" / "hedgerock" / "evolution" / "policy_overlay.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    forbidden_modules = {
        "smc.hedgerock.rule_engine",
        "smc.hedgerock.decision_server",
        "smc.hedgerock.phase_d_walk_forward",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, (
                    f"policy_overlay imports forbidden module: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") in forbidden_modules:
                raise AssertionError(
                    f"policy_overlay from-imports forbidden module: {node.module}"
                )
