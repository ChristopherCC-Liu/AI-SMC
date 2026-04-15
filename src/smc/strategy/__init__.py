"""SMC multi-timeframe strategy layer.

Public API:
- types: BiasDirection, TradeZone, EntrySignal, TradeSetup, SetupGrade
- htf_bias: compute_htf_bias
- zone_scanner: scan_zones
- entry_trigger: check_entry
- confluence: score_confluence
- aggregator: MultiTimeframeAggregator
"""
