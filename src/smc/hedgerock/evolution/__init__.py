"""Self-evolution sidecar — Tier-1 unseal merge layer.

> Submodules ``replay_validator.py`` and ``candidate_generator.py``
> are now allowed to read-only import ``smc.hedgerock.decision_server``
> and ``smc.hedgerock.phase_d_walk_forward``. ``rule_engine`` is
> still red-line. See the regression-guard whitelist for the full
> contract.

The worktree branch ``claude/interesting-bell-82f0fd`` only commits
the sidecar files; sibling production-side files
(``policy_manifest.py``, ``candidate_menu.py``, ...) live in the
parent worktree. We extend ``__path__`` so both directories
contribute submodules, with worktree files taking precedence.
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
