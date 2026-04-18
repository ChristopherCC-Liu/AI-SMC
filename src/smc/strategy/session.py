"""Session utilities — resolve UTC hour to session name and confidence penalty.

Extracted from scripts/live_demo.py:get_session_info() and generalised to
accept an InstrumentConfig so it works for both XAU and BTC (and future
instruments).

XAU:  6 FX sessions (Asian core/transition/London/Overlap/NY/Late NY)
BTC:  2 sessions (HIGH_VOL UTC 12-22, LOW_VOL UTC 22-12 wraps midnight)
"""

from __future__ import annotations

from datetime import datetime, timezone

from smc.instruments.types import InstrumentConfig

__all__ = ["get_session_info"]


def get_session_info(
    now: datetime | None = None,
    *,
    cfg: InstrumentConfig | None = None,
) -> tuple[str, float]:
    """Return (session_name, session_penalty) for given UTC time.

    XAU: 6 FX sessions (Asian core/transition/London/Overlap/NY/Late NY)
    BTC: 2 sessions (HIGH_VOL UTC 12-22, LOW_VOL UTC 22-12 wraps midnight)
    """
    from smc.instruments import get_instrument_config
    if cfg is None:
        cfg = get_instrument_config("XAUUSD")
    if now is None:
        now = datetime.now(timezone.utc)
    hour = now.hour
    # iterate cfg.sessions to find matching (start, end) — handle midnight wrap
    for name, (start, end) in cfg.sessions.items():
        if start < end:
            if start <= hour < end:
                return name, _session_penalty(name, cfg)
        else:  # wraps midnight (e.g., BTC LOW_VOL 22 → 12)
            if hour >= start or hour < end:
                return name, _session_penalty(name, cfg)
    # fallback if no match (shouldn't happen with complete coverage)
    return "UNKNOWN", 0.0


def _session_penalty(name: str, cfg: InstrumentConfig) -> float:
    """Session confidence penalty — legacy XAU behavior.

    XAU: ASIAN_CORE=0.2, ASIAN_LONDON_TRANSITION=0.15, LATE NY=0.1, else 0.0
    BTC: uniformly 0.0 (24/7 no session discount)
    """
    if cfg.symbol == "XAUUSD":
        return {
            "ASIAN_CORE": 0.2,
            "ASIAN_LONDON_TRANSITION": 0.15,
            "LONDON": 0.0,
            "LONDON/NY OVERLAP": 0.0,
            "NEW YORK": 0.0,
            "LATE NY": 0.1,
        }.get(name, 0.0)
    return 0.0
