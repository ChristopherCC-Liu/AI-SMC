"""HedgeRock subpackage — Tier-1 unseal merge layer.

The branch ``claude/interesting-bell-82f0fd`` only commits the
self-evolution sidecar plus the two newly-unsealed prod modules
(``phase_d_walk_forward.py``, ``decision_server.py``). The full
HedgeRock prod tree (``rule_engine.py``, ``schemas.py``,
``data_availability.py``, etc.) lives in the parent worktree.

We use ``pkgutil.extend_path`` so the worktree's ``hedgerock`` dir
contributes its files alongside the parent's, with the worktree's
files taking precedence (because the worktree's ``smc/`` is first
on ``smc.__path__`` after pytest's pythonpath override).
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
