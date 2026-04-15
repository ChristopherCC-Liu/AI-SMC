"""Root pytest configuration.

Ensures ``src/smc`` takes precedence over the ``tests/smc`` shadow package.

Problem: pytest adds ``tests/`` to ``sys.path`` when it finds ``tests/smc/__init__.py``,
causing ``import smc`` to resolve to the test package instead of the source package.
This file is loaded before any sub-package conftest, so it patches ``sys.path`` and
evicts the wrong ``smc`` from ``sys.modules`` before ``tests/smc/conftest.py`` runs.
"""

from __future__ import annotations

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
