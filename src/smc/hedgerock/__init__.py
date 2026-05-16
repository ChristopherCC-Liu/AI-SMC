"""HedgeRock × AI-SMC fusion subpackage.

Decision Center for HedgeRock EA strengthening:
- Timeframe routing (single TF lock)
- Regime transition lock (cooldown after regime jumps)
- Strategy / parameter selection
- News-driven exit directive
- AI exit decision (micro debate)

The MQL5 EA polls ``decision_server`` over HTTP and reads
``RegimeCache.json`` written by ``cache_writer``. All decision logic
lives here in Python; the EA only applies the result.

This file also retains the ``pkgutil.extend_path`` shim from the
Tier-1 unseal merge so the worktree's ``hedgerock`` dir can
contribute its files alongside the parent's when both are on
``sys.path`` (no-op when only one path exists, which is the
production case).
"""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]


from smc.hedgerock.schemas import (
    EXIT_DIRECTIVES,
    NEWS_DIRECTIONS,
    NEWS_INTENSITIES,
    SCHEMA_VERSION,
    SUPPORTED_TIMEFRAMES,
    ExitDirective,
    NewsDirection,
    NewsIntensity,
    SignalEnvelope,
    Timeframe,
)


__all__ = [
    "EXIT_DIRECTIVES",
    "ExitDirective",
    "NEWS_DIRECTIONS",
    "NEWS_INTENSITIES",
    "NewsDirection",
    "NewsIntensity",
    "SCHEMA_VERSION",
    "SUPPORTED_TIMEFRAMES",
    "SignalEnvelope",
    "Timeframe",
]
