"""DEPRECATED THIN WRAPPER — calls scripts/serve_decision_8788.py:main.

Phase B-closeout #2: this script previously ran a partly-mocked stack
and was easy to confuse with the real production entry. To eliminate
drift between two near-identical scripts, this file is now a forwarder
to ``serve_decision_8788.main`` so both invocations share one
``--market-provider {lake,mock}`` contract.

Behaviour:
    - identical CLI to serve_decision_8788.py
    - --market-provider IS REQUIRED (never silently defaults to mock)
    - --data-lake-root / FOREX_DATA_LAKE_ROOT must point at a real
      lake when --market-provider=lake

Usage:
    python scripts/run_production_decision_server.py --market-provider lake \\
        --data-lake-root /path/to/lake

This file is kept only because legacy systemd / shell aliases reference
it by name. New scripts / docs should call ``serve_decision_8788.py``
directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load serve_decision_8788.py as a module — both scripts live next to
# each other in scripts/ and have no package init.
_HERE = Path(__file__).resolve().parent
_SERVE_PATH = _HERE / "serve_decision_8788.py"
_spec = importlib.util.spec_from_file_location("_serve_decision_8788", _SERVE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"could not import serve_decision_8788 at {_SERVE_PATH!s}")
_serve_module = importlib.util.module_from_spec(_spec)
sys.modules["_serve_decision_8788"] = _serve_module
_spec.loader.exec_module(_serve_module)


def main(argv: list[str] | None = None) -> int:
    """Forward to serve_decision_8788.main with the same argv contract."""
    return _serve_module.main(argv)


if __name__ == "__main__":
    sys.exit(main())
