"""Run the HedgeRock decision server locally for smoke testing.

Usage:
    python scripts/run_hedgerock_decision_server.py
    python scripts/run_hedgerock_decision_server.py --port 8788 --regime TREND_UP

The server binds 127.0.0.1 only — never expose this externally; the
MQL5 EA polls it via WebRequest from the same host (or via SSH-forwarded
port from the VPS).

Phase 1 uses a static mock provider. Phase 2 will swap to the real
ForexDataLake-backed provider via --provider lake.
"""

from __future__ import annotations

import argparse
import logging
from typing import get_args

import uvicorn

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.decision_server import DEFAULT_PORT, create_app
from smc.hedgerock.mock_provider import (
    ScriptedMockProvider,
    StaticMockProvider,
    default_static_features,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host (default: 127.0.0.1; do not change unless you know why)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--regime",
        choices=list(get_args(MarketRegimeAI)),
        default="TREND_UP",
        help="initial regime served by the static mock provider",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help=(
            "use a scripted provider that cycles through TREND_UP → "
            "TRANSITION → CONSOLIDATION → TREND_DOWN. Useful for testing "
            "transition_lock end-to-end."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.scripted:
        provider = ScriptedMockProvider(
            [
                default_static_features("TREND_UP"),
                default_static_features("TRANSITION"),
                default_static_features("CONSOLIDATION"),
                default_static_features("TREND_DOWN"),
            ]
        )
    else:
        provider = StaticMockProvider(default_static_features(args.regime))

    app = create_app(provider)
    print(  # noqa: T201
        f"HedgeRock decision server: http://{args.host}:{args.port}\n"
        f"  endpoints: /healthz, /signal?symbol=XAUUSD, /status\n"
        f"  provider: {'scripted' if args.scripted else f'static ({args.regime})'}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
