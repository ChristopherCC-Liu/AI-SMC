"""Regression guard meta-test — Tier-1 unseal aware.

Scans every ``.py`` file under:
  - ``src/smc/hedgerock/evolution/``
  - ``scripts/hedgerock_evolution_*.py``

and asserts:

1. **rule_engine** is still red-line. NO file in the evolution layer
   may import ``smc.hedgerock.rule_engine``.
2. **decision_server** and **phase_d_walk_forward** are Tier-1
   unsealed for read-only access — but ONLY for the explicit
   whitelist below. Every other evolution file remains forbidden
   from importing them.
3. No code-level reference to a ``.mq5`` file (open / Path /
   subprocess / shutil). Documentary mentions are still fine.
4. The scanned tree has at least 30 files (catches accidental
   shrinkage of the layer).

The Tier-1 whitelist is intentionally narrow: only the two modules
named in the unseal RFC update may import ``decision_server`` /
``phase_d_walk_forward``. Adding a third file to the whitelist is a
contract change that requires an RFC amendment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
_EVOLUTION_SRC = _REPO / "src" / "smc" / "hedgerock" / "evolution"
_EVOLUTION_SCRIPTS_GLOB = "hedgerock_evolution_*.py"

# The evolution layer is split across the worktree (this branch's
# committed sidecar) and the parent worktree (full tree of prod
# helpers). Both contribute files that must clear the same isolation
# checks.
_PARENT_EVOLUTION_SRC = Path(
    "/Users/christopher/claudeworkplace/AI-SMC/src/smc/hedgerock/evolution"
)
_PARENT_REPO = Path("/Users/christopher/claudeworkplace/AI-SMC")


# rule_engine remains red-line. ALL files in the evolution layer
# are forbidden from importing it.
_RULE_ENGINE_FRAGMENTS = (
    "from smc.hedgerock.rule_engine",
    "import smc.hedgerock.rule_engine",
)


# Tier-1 unsealed modules. By default still forbidden, but
# explicitly allowed for the whitelist below.
_TIER1_UNSEAL_FRAGMENTS = (
    "from smc.hedgerock.decision_server",
    "import smc.hedgerock.decision_server",
    "from smc.hedgerock.phase_d_walk_forward",
    "import smc.hedgerock.phase_d_walk_forward",
    "from smc.hedgerock import decision_server",
    "from smc.hedgerock import phase_d_walk_forward",
    # Combined-form aliases — covers `from smc.hedgerock import
    # decision_server, phase_d_walk_forward` and similar.
    "from smc.hedgerock import decision_server, phase_d_walk_forward",
)


# Files that ARE allowed to import the Tier-1 unsealed modules
# (still read-only — checked separately below).
_TIER1_UNSEAL_WHITELIST = frozenset(
    {
        "replay_validator.py",
        "candidate_generator.py",
    }
)


# Code-level ``.mq5`` references — patterns that would actually
# open, run, or write to a ``.mq5`` file. Banners + docstrings are
# allowed because they strengthen the no-touch contract.
_FORBIDDEN_MQ5_CODE_PATTERNS = (
    re.compile(r"open\([^)]*\.mq5"),
    re.compile(r"Path\([^)]*\.mq5"),
    re.compile(r"subprocess\.[A-Za-z_]+\([^)]*\.mq5"),
    re.compile(r"shutil\.[A-Za-z_]+\([^)]*\.mq5"),
    re.compile(r'"\s*\.mq5\s*"\s*[,)]?\s*$'),  # bare ".mq5" string token
)


def _scan_targets() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for d in (
        _EVOLUTION_SRC,
        _PARENT_EVOLUTION_SRC,
    ):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name not in seen:
                out.append(p)
                seen.add(p.name)
    for scripts_root in (
        _REPO / "scripts",
        _PARENT_REPO / "scripts",
    ):
        if not scripts_root.exists():
            continue
        for p in sorted(scripts_root.glob(_EVOLUTION_SCRIPTS_GLOB)):
            if p.name not in seen:
                out.append(p)
                seen.add(p.name)
    return out


# ---------------------------------------------------------------------------
# 1. rule_engine remains red-line everywhere.
# ---------------------------------------------------------------------------


def test_no_evolution_file_imports_rule_engine() -> None:
    targets = _scan_targets()
    assert targets, "regression guard found no files to scan"

    offenders: list[tuple[Path, str]] = []
    for p in targets:
        text = p.read_text(encoding="utf-8")
        for fragment in _RULE_ENGINE_FRAGMENTS:
            if fragment in text:
                offenders.append((p, fragment))
    assert not offenders, (
        "rule_engine imports detected (rule_engine remains red-line):\n"
        + "\n".join(f"  {p}: {f!r}" for p, f in offenders)
    )


# ---------------------------------------------------------------------------
# 2. Tier-1 unseal — only whitelisted files may import
#    decision_server / phase_d_walk_forward.
# ---------------------------------------------------------------------------


def test_only_whitelisted_files_import_tier1_unsealed_modules() -> None:
    targets = _scan_targets()
    offenders: list[tuple[Path, str]] = []
    for p in targets:
        if p.name in _TIER1_UNSEAL_WHITELIST:
            continue
        text = p.read_text(encoding="utf-8")
        for fragment in _TIER1_UNSEAL_FRAGMENTS:
            if fragment in text:
                offenders.append((p, fragment))
    assert not offenders, (
        "Tier-1 unsealed imports detected outside the whitelist "
        f"({sorted(_TIER1_UNSEAL_WHITELIST)}):\n"
        + "\n".join(f"  {p}: {f!r}" for p, f in offenders)
    )


def test_whitelisted_files_use_only_read_only_symbols() -> None:
    """Whitelisted files may import decision_server /
    phase_d_walk_forward, but they must use only the public
    read-only surface — no setters, no writers, no mutating calls.

    The decision_server module exposes one getter
    (``get_live_parameters``) and a frozen mapping
    (``LIVE_PARAMETER_KEYS``). The phase_d_walk_forward module
    exposes one pure function (``run_walk_forward_backtest``) and
    frozen dataclasses + a constant.

    This test scans the whitelisted files and checks that any
    attribute access against the imported modules lands on this
    public read-only surface.
    """
    allowed_decision_server = {
        "get_live_parameters",
        "LIVE_PARAMETER_KEYS",
    }
    allowed_walk_forward = {
        "run_walk_forward_backtest",
        "BacktestResult",
        "BacktestWindowResult",
        "PUBLIC_BACKTEST_PARAMETERS",
        "_HALT_AUTO_EXPIRY_HOURS_OBSERVE",
    }

    attr_pattern = re.compile(
        r"\b(decision_server|phase_d_walk_forward)\.([A-Za-z_][A-Za-z0-9_]*)"
    )
    offenders: list[tuple[Path, int, str]] = []
    for fname in _TIER1_UNSEAL_WHITELIST:
        p = _EVOLUTION_SRC / fname
        if not p.exists():
            continue
        for i, line in enumerate(
            p.read_text(encoding="utf-8").splitlines(), start=1,
        ):
            for m in attr_pattern.finditer(line):
                mod, attr = m.group(1), m.group(2)
                allowed = (
                    allowed_decision_server if mod == "decision_server"
                    else allowed_walk_forward
                )
                if attr not in allowed:
                    offenders.append((p, i, f"{mod}.{attr}"))
    assert not offenders, (
        "whitelisted file accessed a non-public symbol on a Tier-1 "
        "module:\n"
        + "\n".join(f"  {p}:L{i}: {s}" for p, i, s in offenders)
    )


# ---------------------------------------------------------------------------
# 3. No .py file mentions a `.mq5` filename in code.
# ---------------------------------------------------------------------------


def test_no_evolution_file_makes_code_calls_against_mq5() -> None:
    """Scan for *code* that would touch a `.mq5` file (open, Path,
    subprocess, shutil). Documentation that mentions `.mq5` to
    *re-state* the no-touch contract is intentionally allowed."""
    targets = _scan_targets()
    offenders: list[tuple[Path, str]] = []
    for p in targets:
        for i, line in enumerate(
            p.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith(("#", '"', "'")) or "*.mq5*" in stripped:
                continue
            for pat in _FORBIDDEN_MQ5_CODE_PATTERNS:
                if pat.search(line):
                    offenders.append((p, f"L{i}: {stripped}"))
                    break
    assert not offenders, (
        "evolution file makes code-level .mq5 references:\n"
        + "\n".join(f"  {p}: {snippet}" for p, snippet in offenders)
    )


# ---------------------------------------------------------------------------
# 4. The scan is wide enough.
# ---------------------------------------------------------------------------


def test_regression_guard_scans_at_least_30_files() -> None:
    targets = _scan_targets()
    assert len(targets) >= 30, (
        f"evolution layer shrank to {len(targets)} files; review "
        "the regression-guard scope before lowering the threshold"
    )


# ---------------------------------------------------------------------------
# 5. Helpful failure messages — synthetic offender smoke.
# ---------------------------------------------------------------------------


def test_failure_message_lists_offender_and_fragment(tmp_path: Path) -> None:
    fake_offender = tmp_path / "offender.py"
    fake_offender.write_text(
        "from smc.hedgerock.rule_engine import _CONFIDENCE_OBSERVE_FLOOR\n",
        encoding="utf-8",
    )
    text = fake_offender.read_text(encoding="utf-8")
    matched = [f for f in _RULE_ENGINE_FRAGMENTS if f in text]
    assert matched == ["from smc.hedgerock.rule_engine"]


# ---------------------------------------------------------------------------
# 6. Whitelist is intentional + minimal.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tier1_whitelist_is_exactly_replay_and_candidate_generator() -> None:
    assert _TIER1_UNSEAL_WHITELIST == frozenset(
        {"replay_validator.py", "candidate_generator.py"}
    )


@pytest.mark.unit
def test_rule_engine_remains_red_line_in_fragment_table() -> None:
    """The fragment table for rule_engine must cover both the
    ``from`` and ``import`` forms — additions to the table elsewhere
    should not silently drop these."""
    assert "from smc.hedgerock.rule_engine" in _RULE_ENGINE_FRAGMENTS
    assert "import smc.hedgerock.rule_engine" in _RULE_ENGINE_FRAGMENTS
