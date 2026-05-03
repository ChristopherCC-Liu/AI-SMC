"""Phase B-closeout #2 — run_production_decision_server.py is a thin
wrapper around serve_decision_8788.main(argv).

Specifically tests that the legacy entry-point name no longer silently
defaults to a mocked stack:
    - missing --market-provider → SystemExit (forwarded from argparse)
    - --market-provider=lake without root → StartupError
    - --market-provider=lake with valid root → ForexDataLakeMarketFeaturesProvider
    - --market-provider=mock works (development only) → StaticMockProvider

The wrapper imports serve_decision_8788 dynamically; we exercise the
factory through that wrapper to confirm the delegation works.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load both wrapper + serve_decision_8788. The wrapper itself imports
# serve_decision_8788 at import time, so loading it tells us the
# delegation is wired.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _import_wrapper():
    spec = importlib.util.spec_from_file_location(
        "_run_production_decision_server_under_test",
        _SCRIPTS_DIR / "run_production_decision_server.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wrapper():
    return _import_wrapper()


# ---------------------------------------------------------------------------
# Wrapper has the same shape as serve_decision_8788
# ---------------------------------------------------------------------------


def test_wrapper_exposes_main(wrapper) -> None:
    assert callable(wrapper.main)


def test_wrapper_inner_serve_module_loaded(wrapper) -> None:
    """The wrapper is required to actually load serve_decision_8788
    so build_market_provider is reachable through it."""
    assert hasattr(wrapper, "_serve_module")
    inner = wrapper._serve_module
    assert hasattr(inner, "build_market_provider")
    assert hasattr(inner, "main")
    assert hasattr(inner, "_parse_args")


# ---------------------------------------------------------------------------
# Provider selection contract — must match serve_decision_8788
# ---------------------------------------------------------------------------


def test_wrapper_lake_without_root_raises_startup_error(wrapper) -> None:
    """The legacy name MUST NOT silently default to mock."""
    inner = wrapper._serve_module
    with pytest.raises(inner.StartupError):
        inner.build_market_provider(kind="lake", data_lake_root=None)


def test_wrapper_lake_with_valid_root_returns_lake_provider(
    wrapper, tmp_path: Path,
) -> None:
    inner = wrapper._serve_module
    provider, label = inner.build_market_provider(
        kind="lake", data_lake_root=str(tmp_path),
    )
    assert label == "ForexDataLakeMarketFeaturesProvider"


def test_wrapper_mock_explicit_returns_static_mock(wrapper) -> None:
    inner = wrapper._serve_module
    provider, label = inner.build_market_provider(
        kind="mock", data_lake_root=None,
    )
    assert label == "StaticMockProvider"


def test_wrapper_argparse_requires_market_provider(wrapper) -> None:
    """Same CLI contract as serve_decision_8788 — --market-provider is
    required, no implicit default."""
    inner = wrapper._serve_module
    with pytest.raises(SystemExit):
        inner._parse_args([])
