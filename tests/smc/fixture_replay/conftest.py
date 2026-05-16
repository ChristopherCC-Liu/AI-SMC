"""Shared fixtures for the incident-replay regression suite (R10 P4.1).

This conftest provides:
- Minimal MT5 IO stand-ins (PositionLike / DealLike Protocol implementers)
  so tests can drive smc.risk.concurrent_gates without the MetaTrader5
  package or a live terminal.
- A frozen UTC clock helper.
- A deterministic fixture loader for the synthetic parquet+state.json bundles
  under fixtures/<incident_slug>/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# MT5 IO stand-ins — implement the Protocols defined in
# smc.risk.concurrent_gates without pulling in the MetaTrader5 package.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakePosition:
    """Minimal PositionLike: only ``magic`` + ``symbol`` are read."""

    symbol: str
    magic: int


@dataclass(frozen=True)
class FakeDeal:
    """Minimal DealLike implementing the fields concurrent_gates inspects.

    ``entry`` follows MT5 semantics: 0 = IN (opening deal), 1 = OUT.
    ``type`` follows MT5 semantics:  0 = BUY, 1 = SELL.
    ``time`` is Unix seconds.
    """

    symbol: str
    magic: int
    entry: int
    type: int
    time: int


# ---------------------------------------------------------------------------
# Frozen clock — deterministic ``now`` so tests are reproducible.
# ---------------------------------------------------------------------------


@dataclass
class FrozenClock:
    """Mutable wrapper around an aware UTC datetime — call instance to read."""

    t: datetime

    def __call__(self) -> datetime:
        return self.t


@pytest.fixture
def frozen_utc_clock() -> FrozenClock:
    """Default clock fixed to 2026-04-20 02:46 UTC (the stacked-SL incident)."""
    return FrozenClock(datetime(2026, 4, 20, 2, 46, tzinfo=UTC))


# ---------------------------------------------------------------------------
# Fixture bundle loader — reads parquet + state.json from a slug directory.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncidentBundle:
    """In-memory representation of one fixture bundle."""

    slug: str
    state: dict
    bars: dict[str, pl.DataFrame]  # keyed by timeframe label (m15 / h1 / d1 / h4)


def _fixtures_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def load_bundle(slug: str) -> IncidentBundle:
    """Load a fixture bundle by directory slug.

    Each slug directory contains:
      - state.json          — pre-incident state (positions, deals, config)
      - {tf}.parquet        — OHLCV bars per timeframe (optional)
    Missing parquet files are silently absent in the resulting ``bars`` dict.
    """
    root = _fixtures_root() / slug
    if not root.exists():
        raise FileNotFoundError(f"fixture bundle not found: {root}")

    state_path = root / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"fixture state.json missing: {state_path}")
    with state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)

    bars: dict[str, pl.DataFrame] = {}
    for tf in ("m15", "h1", "d1", "h4"):
        p = root / f"{tf}.parquet"
        if p.exists():
            bars[tf] = pl.read_parquet(p)

    return IncidentBundle(slug=slug, state=state, bars=bars)


@pytest.fixture
def incident_bundle():
    """Return the bundle loader as a callable so tests can pick the slug."""
    return load_bundle
