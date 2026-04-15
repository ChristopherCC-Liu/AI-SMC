"""Base protocol and shared types for all Forex data adapters.

Adapters are the abstraction layer between external data sources (MT5 terminal,
CSV files, parquet fixtures) and the data lake writer.  Any class that
implements :class:`ForexAdapter` can be passed to the ingest pipeline without
modification.

Usage::

    from smc.data.adapters.base import ForexAdapter, ForexAdapterSpec
    from smc.data.schemas import Timeframe

    assert isinstance(my_adapter, ForexAdapter)  # runtime_checkable
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import polars as pl

from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Adapter specification (frozen metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForexAdapterSpec:
    """Immutable descriptor for a :class:`ForexAdapter` implementation.

    Attributes:
        source: Unique source identifier, e.g. ``"mt5"``, ``"csv"``.
        instrument: Primary instrument the adapter is configured for, e.g.
            ``"XAUUSD"``.  Adapters may support additional instruments via
            :meth:`ForexAdapter.fetch`.
        timeframes: Tuple of timeframes the source can provide.
        description: Human-readable description for logging and diagnostics.
    """

    source: str
    instrument: str
    timeframes: tuple[Timeframe, ...]
    description: str


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class ForexAdapterError(RuntimeError):
    """Single error type raised by all :class:`ForexAdapter` implementations.

    Wraps transient connection failures, parse errors, and unsupported
    instrument/timeframe combinations into a single catch-all so callers
    do not need to import provider-specific exceptions.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ForexAdapter(Protocol):
    """Protocol that every data adapter must satisfy.

    Attributes:
        spec: Immutable :class:`ForexAdapterSpec` describing this adapter.

    The :meth:`fetch` method must return a :class:`polars.DataFrame` with
    **at minimum** the following columns (additional columns are allowed):

    - ``ts``: :class:`polars.Datetime` with time zone ``"UTC"``, nanosecond
      precision.
    - ``open``, ``high``, ``low``, ``close``, ``volume``: :class:`float`.
    - ``spread``: :class:`float`, in points.

    Rows are **not** guaranteed to be sorted by ``ts`` — callers should sort
    if order matters.

    Raises:
        ForexAdapterError: on any retrieval or parsing failure.
    """

    spec: ForexAdapterSpec

    def fetch(
        self,
        *,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch OHLCV bars for *instrument* in the half-open interval [start, end).

        Args:
            instrument: Symbol to retrieve, e.g. ``"XAUUSD"``.
            timeframe: One of the :class:`~smc.data.schemas.Timeframe` variants.
            start: Inclusive start time.  Must be tz-aware (UTC).
            end: Exclusive end time.  Must be tz-aware (UTC).

        Returns:
            A :class:`polars.DataFrame` with OHLCV columns and a ``ts`` column
            in UTC.

        Raises:
            ForexAdapterError: on connection failure, unsupported instrument,
                or unsupported timeframe.
        """
        ...
