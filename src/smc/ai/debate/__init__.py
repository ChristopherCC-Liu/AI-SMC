"""AI multi-agent debate pipeline for XAUUSD market regime classification.

Adapted from alphalens-v2's 7-agent debate architecture:
  1. 4 Analysts (fast brain): Trend, Zone, Macro, Risk
  2. 2 Researchers (slow brain): Bull vs Bear on regime classification
  3. 1 Judge (slow brain): Final regime verdict

The pipeline classifies the market into one of 5 regimes:
  TREND_UP, TREND_DOWN, CONSOLIDATION, TRANSITION, ATH_BREAKOUT

Each regime maps to a frozen RegimeParams preset via param_router.
"""
