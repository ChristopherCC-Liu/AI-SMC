"""Root pytest configuration.

1. Ensures ``src/smc`` takes precedence over the ``tests/smc``
   shadow package.

   Problem: pytest adds ``tests/`` to ``sys.path`` when it finds
   ``tests/smc/__init__.py``, causing ``import smc`` to resolve to
   the test package instead of the source package. This file is
   loaded before any sub-package conftest, so it patches
   ``sys.path`` and evicts the wrong ``smc`` from ``sys.modules``
   before ``tests/smc/conftest.py`` runs.

2. Tier-1 unseal worktree extension (no-op on fresh checkouts).

   When the suite runs inside a sparse Claude Code worktree under
   ``.claude/worktrees/<branch>/`` the worktree's ``src/`` only
   ships the evolution sidecar; the rest of ``smc.hedgerock`` lives
   in the parent worktree's ``src/``. We extend the package
   ``__path__`` to the location named by ``$AI_SMC_HOME`` (default:
   the repo root inferred from this file's location) so sibling
   files resolve. On a normal ``git clone`` ``AI_SMC_HOME`` and the
   conftest's own repo are identical and this section is a no-op.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).parent.parent / "src")

# Insert src/ at position 0 so it beats the tests/ entry that pytest added.
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Evict any already-imported smc.* from sys.modules so that subsequent
# imports resolve against the corrected sys.path.
_stale = [k for k in list(sys.modules) if k == "smc" or k.startswith("smc.")]
for _key in _stale:
    del sys.modules[_key]


# ---------------------------------------------------------------------------
# Tier-1 unseal — extend the smc/hedgerock package __path__ lists so the
# worktree can pull in sibling files (policy_manifest, candidate_menu, etc.)
# from the parent worktree without copying them in.
# ---------------------------------------------------------------------------


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent


_PARENT_REPO_SRC = _ai_smc_home() / "src"


def _extend_namespace(module_name: str, extra_dir: Path) -> None:
    if not extra_dir.exists():
        return
    if module_name not in sys.modules:
        return
    mod = sys.modules[module_name]
    if not hasattr(mod, "__path__"):
        return
    extra_str = str(extra_dir.resolve())
    if extra_str not in list(mod.__path__):
        mod.__path__.append(extra_str)  # type: ignore[attr-defined]


def _bootstrap_namespace_extensions() -> None:
    try:
        import smc  # noqa: F401
    except ImportError:
        return
    _extend_namespace("smc", _PARENT_REPO_SRC / "smc")
    try:
        import smc.hedgerock  # noqa: F401
    except ImportError:
        return
    _extend_namespace(
        "smc.hedgerock", _PARENT_REPO_SRC / "smc" / "hedgerock",
    )
    try:
        import smc.hedgerock.evolution  # noqa: F401
    except ImportError:
        return
    _extend_namespace(
        "smc.hedgerock.evolution",
        _PARENT_REPO_SRC / "smc" / "hedgerock" / "evolution",
    )


_bootstrap_namespace_extensions()


# CLI scripts in the parent worktree (e.g. hedgerock_evolution_report.py)
# need to be importable when the recommend CLI does
# ``sys.path.insert(0, _THIS_DIR); import hedgerock_evolution_report``
# from within the worktree. Append the parent's scripts/ to sys.path
# so the import resolves through the standard mechanism.
_PARENT_SCRIPTS = _PARENT_REPO_SRC.parent / "scripts"
if _PARENT_SCRIPTS.exists():
    _ps = str(_PARENT_SCRIPTS.resolve())
    if _ps not in sys.path:
        sys.path.append(_ps)
