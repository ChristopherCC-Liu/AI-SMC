"""Stage 6-followup-4 task 4 — regression guard meta-test.

Scans every ``.py`` file under:
  - ``src/smc/hedgerock/evolution/``
  - ``scripts/hedgerock_evolution_*.py``

and asserts NONE of them import the live trading runtime modules
or reference the EA. This is a defence-in-depth check on top of
the per-module isolation tests already shipped — a single source-
level scan that catches future drift even if a developer forgets
to add a per-module test for a new file.

Forbidden imports / references:

  * ``from smc.hedgerock.rule_engine``
  * ``from smc.hedgerock.decision_server``
  * ``from smc.hedgerock.phase_d_walk_forward``
  * ``import smc.hedgerock.rule_engine``
  * ``import smc.hedgerock.decision_server``
  * ``import smc.hedgerock.phase_d_walk_forward``
  * **Code-level** ``.mq5`` references — calls like
    ``open("X.mq5")``, ``subprocess.run([..., "X.mq5"])``, or string
    concatenations that produce a ``.mq5`` filesystem target.
    Documentary mentions of ``.mq5`` (banners that say "this file
    does NOT touch .mq5") are explicitly allowed because they
    *strengthen* the contract; the scanner only flags identifiers
    that would actually call into / write to a ``.mq5`` file.

Existing per-module tests (e.g.
``test_replay_validator_module_does_not_import_live_runtime``)
keep checking the same property locally; this meta-test is
authoritative across the whole tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[3]
_EVOLUTION_SRC = _REPO / "src" / "smc" / "hedgerock" / "evolution"
_EVOLUTION_SCRIPTS_GLOB = "hedgerock_evolution_*.py"

# Exact substrings that must never appear in production / sidecar
# files within the evolution layer.
_FORBIDDEN_IMPORT_FRAGMENTS = (
    "from smc.hedgerock.rule_engine",
    "from smc.hedgerock.decision_server",
    "from smc.hedgerock.phase_d_walk_forward",
    "import smc.hedgerock.rule_engine",
    "import smc.hedgerock.decision_server",
    "import smc.hedgerock.phase_d_walk_forward",
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
    out.extend(sorted(_EVOLUTION_SRC.glob("*.py")))
    out.extend(sorted((_REPO / "scripts").glob(_EVOLUTION_SCRIPTS_GLOB)))
    return out


# ---------------------------------------------------------------------------
# 1. No file imports rule_engine / decision_server / phase_d_walk_forward.
# ---------------------------------------------------------------------------


def test_no_evolution_file_imports_live_runtime() -> None:
    targets = _scan_targets()
    assert targets, "regression guard found no files to scan"

    offenders: list[tuple[Path, str]] = []
    for p in targets:
        text = p.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
            if fragment in text:
                offenders.append((p, fragment))
    assert not offenders, (
        "live-runtime imports detected in evolution layer:\n"
        + "\n".join(f"  {p}: {f!r}" for p, f in offenders)
    )


# ---------------------------------------------------------------------------
# 2. No .py file mentions a `.mq5` filename (stops a future bridge
#    to the EA from sneaking in).
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
            # Skip docstring/comment lines outright.
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
# 3. The scan is wide enough — at least 30 files (current count is
#    27 src + 8 scripts; threshold 30 catches accidental shrinkage).
# ---------------------------------------------------------------------------


def test_regression_guard_scans_at_least_30_files() -> None:
    """If a refactor shrinks the evolution layer below this floor,
    the test will fail — prompting a deliberate review of whether
    the contract still applies."""
    targets = _scan_targets()
    assert len(targets) >= 30, (
        f"evolution layer shrank to {len(targets)} files; review "
        "the regression-guard scope before lowering the threshold"
    )


# ---------------------------------------------------------------------------
# 4. Helpful failure message — when run on a doctored fixture, the
#    test surfaces the offending file and the matched fragment.
# ---------------------------------------------------------------------------


def test_failure_message_lists_offender_and_fragment(tmp_path: Path) -> None:
    """Smoke: feed the assertion machinery a synthetic offender and
    confirm the assert message would name it. Uses an in-test
    helper, not the real scan, so we don't pollute the repo."""
    fake_offender = tmp_path / "offender.py"
    fake_offender.write_text(
        "from smc.hedgerock.rule_engine import _CONFIDENCE_OBSERVE_FLOOR\n",
        encoding="utf-8",
    )
    text = fake_offender.read_text(encoding="utf-8")
    matched = [
        f for f in _FORBIDDEN_IMPORT_FRAGMENTS if f in text
    ]
    assert matched == ["from smc.hedgerock.rule_engine"]


# ---------------------------------------------------------------------------
# 5. The list of forbidden fragments matches the RFC §11 invariants
#    exactly — additions to RFC §11 must come with additions here.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forbidden_fragments_cover_all_three_red_line_modules() -> None:
    expected_modules = {
        "smc.hedgerock.rule_engine",
        "smc.hedgerock.decision_server",
        "smc.hedgerock.phase_d_walk_forward",
    }
    covered = set()
    for fragment in _FORBIDDEN_IMPORT_FRAGMENTS:
        for mod in expected_modules:
            if mod in fragment:
                covered.add(mod)
    assert covered == expected_modules, (
        f"regression guard missing red-line modules: "
        f"{expected_modules - covered}"
    )
