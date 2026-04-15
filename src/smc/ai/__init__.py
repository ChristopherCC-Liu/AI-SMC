"""AI-powered market regime classification for the AI-SMC trading system.

Sprint 6 module: classifies XAUUSD market regime and maps it to optimal
strategy parameters.  The AI system does NOT generate trade signals — it
provides context-aware parameter routing for the existing SMC pipeline.

Submodules:
    models          Frozen Pydantic types (AIRegimeAssessment, RegimeParams, etc.)
    param_router    Maps regime enum → frozen parameter presets
    cost_tracker    Daily LLM spend tracking against configurable budget
    external_context Macro data aggregation (DXY, VIX, real rates) with caching
"""

from smc.ai.models import (
    AIRegimeAssessment,
    AnalystView,
    DebateRound,
    ExternalContext,
    JudgeVerdict,
    MarketRegimeAI,
    RegimeParams,
    RegimeState,
)
from smc.ai.param_router import route
from smc.ai.regime_classifier import (
    RegimeContext,
    classify_regime_ai,
    extract_regime_context,
)

__all__ = [
    "AIRegimeAssessment",
    "AnalystView",
    "DebateRound",
    "ExternalContext",
    "JudgeVerdict",
    "MarketRegimeAI",
    "RegimeContext",
    "RegimeParams",
    "RegimeState",
    "classify_regime_ai",
    "extract_regime_context",
    "route",
]
