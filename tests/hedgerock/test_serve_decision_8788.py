"""Phase B-closeout #2 — serve_decision_8788 startup factory tests.

Covers the build_market_provider factory's contract:
    - explicit lake selection requires a valid root
    - missing root raises StartupError (no silent mock fallback)
    - mock selection produces StaticMockProvider with a warning
    - /status surfaces the chosen provider label
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script as a module — it lives outside the package tree.
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "serve_decision_8788.py"
)
_spec = importlib.util.spec_from_file_location("serve_decision_8788", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
serve_module = importlib.util.module_from_spec(_spec)
sys.modules["serve_decision_8788"] = serve_module
_spec.loader.exec_module(serve_module)


# ---------------------------------------------------------------------------
# build_market_provider — pure factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_provider_explicit_returns_static_mock() -> None:
    provider, label = serve_module.build_market_provider(
        kind="mock", data_lake_root=None,
    )
    assert label == "StaticMockProvider"
    # Duck-type check: has get_features
    assert hasattr(provider, "get_features")


@pytest.mark.unit
def test_lake_provider_requires_root() -> None:
    """Phase B-closeout #2: --market-provider lake without a root → fail fast."""
    with pytest.raises(serve_module.StartupError) as exc_info:
        serve_module.build_market_provider(kind="lake", data_lake_root=None)
    assert "data-lake-root" in exc_info.value.message or \
           "FOREX_DATA_LAKE_ROOT" in exc_info.value.message


@pytest.mark.unit
def test_lake_provider_rejects_nonexistent_root(tmp_path: Path) -> None:
    bad = tmp_path / "does_not_exist"
    with pytest.raises(serve_module.StartupError) as exc_info:
        serve_module.build_market_provider(kind="lake", data_lake_root=str(bad))
    assert "does not exist" in exc_info.value.message


@pytest.mark.unit
def test_lake_provider_with_valid_empty_root_constructs_ok(tmp_path: Path) -> None:
    """An EXISTING but empty lake root is acceptable at startup —
    the lake just has no bars yet. /signal will 503 on first poll
    until ingest catches up. We don't fail at startup for this."""
    provider, label = serve_module.build_market_provider(
        kind="lake", data_lake_root=str(tmp_path),
    )
    assert label == "ForexDataLakeMarketFeaturesProvider"
    assert hasattr(provider, "get_features")


@pytest.mark.unit
def test_lake_provider_picks_up_env_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOREX_DATA_LAKE_ROOT", str(tmp_path))
    provider, label = serve_module.build_market_provider(
        kind="lake", data_lake_root=None,
    )
    assert label == "ForexDataLakeMarketFeaturesProvider"


@pytest.mark.unit
def test_unknown_provider_kind_rejected() -> None:
    with pytest.raises(serve_module.StartupError):
        serve_module.build_market_provider(kind="something-else", data_lake_root=None)


# ---------------------------------------------------------------------------
# CLI parsing — argparse contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_requires_market_provider_arg() -> None:
    """argparse must reject runs that omit --market-provider."""
    with pytest.raises(SystemExit):
        serve_module._parse_args([])


@pytest.mark.unit
def test_cli_accepts_lake_with_root() -> None:
    args = serve_module._parse_args(["--market-provider", "lake",
                                     "--data-lake-root", "/tmp/lake"])
    assert args.market_provider == "lake"
    assert args.data_lake_root == "/tmp/lake"


@pytest.mark.unit
def test_cli_accepts_mock() -> None:
    args = serve_module._parse_args(["--market-provider", "mock"])
    assert args.market_provider == "mock"


@pytest.mark.unit
def test_cli_rejects_other_provider_values() -> None:
    with pytest.raises(SystemExit):
        serve_module._parse_args(["--market-provider", "lakehouse"])


# ---------------------------------------------------------------------------
# /status reports market_provider_label
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_status_reports_market_provider_label() -> None:
    """create_app accepts a market_provider_label kwarg that the
    serve script wires from build_market_provider's return value."""
    from fastapi.testclient import TestClient
    from smc.hedgerock.decision_server import MarketFeatures, create_app

    class _StubProvider:
        def get_features(self, symbol):
            return MarketFeatures(
                volatility_rank=0.5, hh_count=4, ll_count=2,
                h4_trend_bars=3, regime="TREND_UP",
            )

    app = create_app(
        _StubProvider(),
        market_provider_label="ForexDataLakeMarketFeaturesProvider",
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["market_provider_label"] == "ForexDataLakeMarketFeaturesProvider"


@pytest.mark.unit
def test_status_default_label_is_unknown_when_omitted() -> None:
    """Don't lie about the provider — the default reads 'unknown'."""
    from fastapi.testclient import TestClient
    from smc.hedgerock.decision_server import MarketFeatures, create_app

    class _StubProvider:
        def get_features(self, symbol):
            return MarketFeatures(
                volatility_rank=0.5, hh_count=4, ll_count=2,
                h4_trend_bars=3, regime="TREND_UP",
            )

    app = create_app(_StubProvider(), enable_debate=False)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["market_provider_label"] == "unknown"
