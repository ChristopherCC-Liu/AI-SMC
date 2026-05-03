"""Path A entry — production-wired HedgeRock decision server on 8788.

Phase B-closeout #2: ``--market-provider`` is now an explicit choice.
There is NO silent fallback from "lake" to "mock". A misconfigured
production deploy fails fast at startup rather than serving fabricated
features that the EA would treat as real signal.

    --market-provider lake    (production / demo) — requires --data-lake-root
                              or FOREX_DATA_LAKE_ROOT pointing at a non-empty
                              parquet store. If the lake is missing or has no
                              recent bars, startup raises SystemExit non-zero
                              and /signal would 503 anyway.

    --market-provider mock    (development only)  — explicit opt-in for the
                              StaticMockProvider. Never used in production.

The provider type is reported in /status (``market_provider_label``)
so an operator can see at a glance which mode is live.

Other providers (news, exposure, filter_inputs) keep their previous
fallback semantics — they are bonus signals, not the primary feature
source.

Binds 127.0.0.1 by default — the EA polls via WebRequest from localhost
or via SSH-forwarded port from the VPS. Never expose externally.

Usage:
    python scripts/serve_decision_8788.py --market-provider lake \\
        --data-lake-root /path/to/lake
    python scripts/serve_decision_8788.py --market-provider mock  # dev only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from smc.ai.cost_tracker import CostTracker
from smc.data.lake import ForexDataLake
from smc.hedgerock.decision_server import DEFAULT_PORT, create_app
from smc.hedgerock.exposure_provider_impl import MT5BrokerExposureProvider
from smc.hedgerock.forex_data_lake_provider import (
    ForexDataLakeMarketFeaturesProvider,
)
from smc.hedgerock.mock_provider import (
    StaticFilterInputsProvider,
    StaticMockProvider,
    default_static_features,
)
from smc.hedgerock.news_engine import NewsEngine
from smc.hedgerock.news_features_provider_impl import NewsEngineFeaturesProvider

logger = logging.getLogger("serve_decision_8788")

LAKE_ROOT_ENV = "FOREX_DATA_LAKE_ROOT"


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class StartupError(SystemExit):
    """Raised when the operator picks ``--market-provider lake`` but the
    lake is missing / empty / unreachable. Inherits from SystemExit so
    the script exits non-zero without a tracebacky stack."""

    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message
        logger.error("startup error: %s", message)


def build_market_provider(
    *,
    kind: str,
    data_lake_root: str | None,
):
    """Return ``(provider, label)`` based on the operator's selection.

    Raises :class:`StartupError` for any misconfiguration of the lake
    path. **Never silently falls back to mock.**
    """
    if kind == "mock":
        logger.warning(
            "market_provider=mock — DEVELOPMENT ONLY. Decision Center will "
            "emit fabricated features. Do not run live with this flag."
        )
        return StaticMockProvider(default_static_features()), "StaticMockProvider"

    if kind == "lake":
        root_str = data_lake_root or os.environ.get(LAKE_ROOT_ENV)
        if not root_str:
            raise StartupError(
                "--market-provider lake requires --data-lake-root or "
                f"{LAKE_ROOT_ENV} env var. Refusing to start."
            )
        root = Path(root_str)
        if not root.exists() or not root.is_dir():
            raise StartupError(
                f"data lake root {root!s} does not exist or is not a directory"
            )
        # The lake itself does not require a probe; construction is
        # cheap. We delegate "no usable bars" to the provider's first
        # /signal which raises FeaturesUnavailable → 503.
        try:
            lake = ForexDataLake(root)
        except Exception as exc:  # noqa: BLE001
            raise StartupError(f"failed to open ForexDataLake({root!s}): {exc}") from exc
        return ForexDataLakeMarketFeaturesProvider(lake), "ForexDataLakeMarketFeaturesProvider"

    raise StartupError(f"unknown --market-provider value: {kind!r}")


# ---------------------------------------------------------------------------
# Other providers — best-effort, fall back gracefully
# ---------------------------------------------------------------------------


def _build_news_engine() -> NewsEngine:
    return NewsEngine()


def _build_exposure_provider() -> MT5BrokerExposureProvider | None:
    try:
        from smc.execution.executor import MT5BrokerPort  # noqa: PLC0415

        broker = MT5BrokerPort()
        return MT5BrokerExposureProvider(broker)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MT5BrokerPort unavailable; running with flat exposure: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--market-provider",
        required=True,
        choices=["lake", "mock"],
        help=(
            "lake = ForexDataLakeMarketFeaturesProvider (production / demo). "
            "mock = StaticMockProvider (development only — never use live)."
        ),
    )
    parser.add_argument(
        "--data-lake-root",
        default=None,
        help=(
            "Path to the parquet data lake root. Required when "
            "--market-provider=lake unless FOREX_DATA_LAKE_ROOT env is set."
        ),
    )
    parser.add_argument(
        "--enable-debate",
        action="store_true",
        help="enable LLM micro-debate inside exit_decider (cost $)",
    )
    parser.add_argument(
        "--daily-budget-usd",
        type=float,
        default=7.0,
        help="CostTracker daily budget when --enable-debate set",
    )
    parser.add_argument("--disable-news", action="store_true")
    parser.add_argument("--disable-filters", action="store_true")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    market_provider, market_provider_label = build_market_provider(
        kind=args.market_provider,
        data_lake_root=args.data_lake_root,
    )

    if args.disable_news:
        news_provider = None
    else:
        engine = _build_news_engine()
        exposure_for_news = _build_exposure_provider()
        if exposure_for_news is None:
            class _FlatExposure:
                def get_exposure_lots(self, symbol: str) -> float:
                    return 0.0

            news_provider = NewsEngineFeaturesProvider(engine, _FlatExposure())
        else:
            news_provider = NewsEngineFeaturesProvider(engine, exposure_for_news)

    exposure_provider = _build_exposure_provider()
    filter_inputs_provider = None if args.disable_filters else StaticFilterInputsProvider()
    cost_tracker = (
        CostTracker(daily_budget_usd=args.daily_budget_usd) if args.enable_debate else None
    )

    app = create_app(
        market_provider,
        news_provider=news_provider,
        exposure_provider=exposure_provider,
        filter_inputs_provider=filter_inputs_provider,
        market_provider_label=market_provider_label,
        enable_rule_engine=True,  # Phase C — production wiring
        cost_tracker=cost_tracker,
        enable_debate=args.enable_debate,
    )

    print(  # noqa: T201
        f"HedgeRock decision server (v2.0.0): http://{args.host}:{args.port}\n"
        f"  endpoints:              /healthz, /signal?symbol=XAUUSD, /status\n"
        f"  market_provider:        {market_provider_label}\n"
        f"  news_provider:          {'NewsEngineFeaturesProvider' if news_provider else 'None'}\n"
        f"  exposure_provider:      {'MT5BrokerExposureProvider' if exposure_provider else 'None (flat)'}\n"
        f"  filter_inputs_provider: {'StaticFilterInputsProvider' if filter_inputs_provider else 'None'}\n"
        f"  debate_enabled:         {args.enable_debate}\n"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
