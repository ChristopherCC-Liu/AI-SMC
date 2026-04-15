"""Mock MT5 adapter for macOS/Linux development without a live terminal.

When the environment variable ``SMC_MT5_MOCK=1`` is set (or when the real
MetaTrader5 package is unavailable), this module provides:

1. :class:`MockMT5Terminal` — a drop-in simulator for the ``MetaTrader5``
   Python module API (``initialize``, ``shutdown``, ``copy_rates_range``,
   ``symbol_info_tick``, ``order_send``, ``positions_get``).

2. :class:`MT5MockAdapter` — a :class:`~smc.data.adapters.base.ForexAdapter`
   that reads from pre-written Parquet fixture files instead of a live
   terminal, making it trivially fast for unit and integration tests.

Fixture layout::

    {fixtures_dir}/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

This mirrors the data-lake partition layout written by
:func:`~smc.data.writers.write_forex_partitioned`, so any lake snapshot can
be used as a fixture directory.

Usage::

    import os
    os.environ["SMC_MT5_MOCK"] = "1"

    from smc.data.adapters.mt5_mock import MT5MockAdapter
    from smc.data.schemas import Timeframe
    from pathlib import Path
    from datetime import datetime, timezone

    adapter = MT5MockAdapter(fixtures_dir=Path("tests/fixtures"))
    df = adapter.fetch(
        instrument="XAUUSD",
        timeframe=Timeframe.H1,
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from smc.data.adapters.base import ForexAdapter, ForexAdapterError, ForexAdapterSpec
from smc.data.lake import ForexDataLake
from smc.data.schemas import SCHEMA_VERSION, Timeframe

# ---------------------------------------------------------------------------
# Environment flag
# ---------------------------------------------------------------------------

_MOCK_ENV_VAR: str = "SMC_MT5_MOCK"


def is_mock_mode() -> bool:
    """Return True if ``SMC_MT5_MOCK=1`` is set in the environment."""
    return os.environ.get(_MOCK_ENV_VAR, "0").strip() == "1"


# ---------------------------------------------------------------------------
# Simulated MT5 data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockTick:
    """Simulated ``symbol_info_tick`` result."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: int  # unix epoch seconds


@dataclass(frozen=True)
class MockOrderResult:
    """Simulated result of ``order_send``."""

    retcode: int          # 10009 = TRADE_RETCODE_DONE
    deal: int
    order: int
    volume: float
    price: float
    comment: str


@dataclass(frozen=True)
class MockPosition:
    """Simulated open position from ``positions_get``."""

    ticket: int
    symbol: str
    type: int       # 0 = BUY, 1 = SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    comment: str


# ---------------------------------------------------------------------------
# MockMT5Terminal
# ---------------------------------------------------------------------------


class MockMT5Terminal:
    """Simulates the ``MetaTrader5`` Python module for offline development.

    This class mirrors the MT5 module-level function API as instance methods.
    It reads OHLCV data from a :class:`~smc.data.lake.ForexDataLake` backed
    by parquet fixtures, so behaviour is deterministic and reproducible.

    Args:
        fixtures_dir: Root directory of the fixture parquet hierarchy.
        default_spread: Default spread (in points) injected when ticks are
            simulated.
    """

    # MT5 TIMEFRAME constants (mirrors MetaTrader5 module attribute names)
    TIMEFRAME_M1: int = 1
    TIMEFRAME_M5: int = 5
    TIMEFRAME_M15: int = 15
    TIMEFRAME_H1: int = 60
    TIMEFRAME_H4: int = 240
    TIMEFRAME_D1: int = 1440

    def __init__(
        self,
        fixtures_dir: Path,
        default_spread: float = 20.0,
    ) -> None:
        self._lake = ForexDataLake(fixtures_dir)
        self._default_spread = default_spread
        self._initialised: bool = False
        self._positions: list[MockPosition] = []
        self._next_ticket: int = 1000

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        """Simulate terminal initialisation.  Always succeeds."""
        self._initialised = True
        return True

    def shutdown(self) -> None:
        """Simulate terminal shutdown."""
        self._initialised = False

    def last_error(self) -> tuple[int, str]:
        """Return (0, 'OK') — the mock never errors."""
        return (0, "OK")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: datetime,
        date_to: datetime,
    ) -> np.ndarray | None:
        """Return structured numpy array matching MT5's ``copy_rates_range``.

        The returned dtype mirrors the real MT5 format::

            [('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
             ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
             ('spread', '<i4'), ('real_volume', '<i8')]

        Args:
            symbol: Instrument symbol, e.g. ``"XAUUSD"``.
            timeframe: MT5 timeframe constant (e.g. ``TIMEFRAME_H1 = 60``).
            date_from: Start of the range (tz-aware).
            date_to: End of the range (tz-aware).

        Returns:
            Structured numpy array, or ``None`` if no data is available.
        """
        tf_enum = _mt5_const_to_timeframe(timeframe)
        if tf_enum is None:
            return None

        try:
            df = self._lake.query(symbol, tf_enum, start=date_from, end=date_to)
        except Exception:
            return None

        if df.is_empty():
            return None

        dtype = np.dtype(
            [
                ("time", "<i8"),
                ("open", "<f8"),
                ("high", "<f8"),
                ("low", "<f8"),
                ("close", "<f8"),
                ("tick_volume", "<i8"),
                ("spread", "<i4"),
                ("real_volume", "<i8"),
            ]
        )

        n = len(df)
        arr = np.zeros(n, dtype=dtype)
        # Convert ts (datetime[ns, UTC]) to epoch seconds (int64)
        ts_ns = df["ts"].cast(pl.Int64).to_numpy()
        arr["time"] = ts_ns // 1_000_000_000
        arr["open"] = df["open"].to_numpy()
        arr["high"] = df["high"].to_numpy()
        arr["low"] = df["low"].to_numpy()
        arr["close"] = df["close"].to_numpy()
        arr["tick_volume"] = df["volume"].cast(pl.Int64).to_numpy()
        arr["spread"] = df["spread"].cast(pl.Int32).to_numpy()
        arr["real_volume"] = np.zeros(n, dtype=np.int64)
        return arr

    def symbol_info_tick(self, symbol: str) -> MockTick | None:
        """Return a simulated last tick.  Uses the latest close from fixtures."""
        now = datetime.now(tz=timezone.utc)
        rng = self._lake.available_range(symbol, Timeframe.M1)
        if rng is None:
            rng = self._lake.available_range(symbol, Timeframe.H1)
        if rng is None:
            return None

        _start, end = rng
        # Fetch the single last bar
        df = self._lake.query(symbol, Timeframe.M1, start=_start, end=end)
        if df.is_empty():
            df = self._lake.query(symbol, Timeframe.H1, start=_start, end=end)
        if df.is_empty():
            return None

        last_close = float(df["close"][-1])
        half_spread = self._default_spread / 2.0 * 0.01  # rough pip conversion
        return MockTick(
            symbol=symbol,
            bid=last_close - half_spread,
            ask=last_close + half_spread,
            last=last_close,
            volume=0.0,
            time=int(now.timestamp()),
        )

    # ------------------------------------------------------------------
    # Trading simulation
    # ------------------------------------------------------------------

    def order_send(self, request: dict[str, Any]) -> MockOrderResult:
        """Simulate an order submission.  Always fills at the requested price.

        Args:
            request: MT5 trade request dict with keys ``action``, ``symbol``,
                ``volume``, ``type``, ``price``, ``sl``, ``tp``, ``comment``.

        Returns:
            :class:`MockOrderResult` with ``retcode=10009`` (DONE).
        """
        price = float(request.get("price", 0.0))
        volume = float(request.get("volume", 0.01))
        order_type = int(request.get("type", 0))
        symbol = str(request.get("symbol", "XAUUSD"))
        comment = str(request.get("comment", "mock"))

        ticket = self._next_ticket
        self._next_ticket += 1

        pos = MockPosition(
            ticket=ticket,
            symbol=symbol,
            type=order_type,
            volume=volume,
            price_open=price,
            sl=float(request.get("sl", 0.0)),
            tp=float(request.get("tp", 0.0)),
            profit=0.0,
            comment=comment,
        )
        self._positions = [*self._positions, pos]

        return MockOrderResult(
            retcode=10009,
            deal=ticket,
            order=ticket,
            volume=volume,
            price=price,
            comment="mock fill",
        )

    def positions_get(
        self,
        symbol: str | None = None,
    ) -> tuple[MockPosition, ...]:
        """Return open positions, optionally filtered by *symbol*."""
        positions = self._positions
        if symbol is not None:
            positions = [p for p in positions if p.symbol == symbol]
        return tuple(positions)


# ---------------------------------------------------------------------------
# MT5MockAdapter (ForexAdapter protocol)
# ---------------------------------------------------------------------------


class MT5MockAdapter:
    """ForexAdapter backed by parquet fixtures — no live MT5 terminal needed.

    Reads from a fixture directory that mirrors the data-lake partition layout.
    Useful for unit tests and CI pipelines running on macOS/Linux.

    The adapter is **not** a context manager; it does not manage any external
    resource lifecycle.

    Args:
        fixtures_dir: Root of the fixture parquet hierarchy.
        instrument: Default instrument written into adapter spec.
        source_name: Source label for the ``source`` column (default:
            ``"mt5_mock"``).
    """

    def __init__(
        self,
        fixtures_dir: Path,
        instrument: str = "XAUUSD",
        source_name: str = "mt5_mock",
    ) -> None:
        self._lake = ForexDataLake(fixtures_dir)
        self._instrument = instrument
        self._source_name = source_name
        self._spec = ForexAdapterSpec(
            source=source_name,
            instrument=instrument,
            timeframes=tuple(Timeframe),
            description=(
                f"MT5 mock adapter reading parquet fixtures from {fixtures_dir}"
            ),
        )

    @property
    def spec(self) -> ForexAdapterSpec:
        """Immutable adapter specification."""
        return self._spec

    def fetch(
        self,
        *,
        instrument: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch bars from parquet fixtures simulating MT5 output.

        Args:
            instrument: Symbol to fetch, e.g. ``"XAUUSD"``.
            timeframe: Timeframe variant.
            start: Inclusive start (tz-aware).
            end: Exclusive end (tz-aware).

        Returns:
            :class:`polars.DataFrame` sorted ascending by ``ts``.

        Raises:
            ForexAdapterError: if no fixture data exists for the combination.
        """
        try:
            df = self._lake.query(instrument, timeframe, start=start, end=end)
        except Exception as exc:
            raise ForexAdapterError(
                f"MT5MockAdapter failed to query fixture data for "
                f"{instrument}/{timeframe}: {exc}"
            ) from exc

        if df.is_empty():
            raise ForexAdapterError(
                f"No fixture data found for {instrument}/{timeframe} "
                f"in range [{start.isoformat()}, {end.isoformat()})."
            )

        # Ensure source column reflects mock identity
        return df.with_columns(
            pl.lit(f"{self._source_name}:{instrument}").alias("source")
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mt5_const_to_timeframe(const: int) -> Timeframe | None:
    """Map an MT5 TIMEFRAME_* integer constant to a :class:`Timeframe` enum."""
    mapping: dict[int, Timeframe] = {
        1: Timeframe.M1,
        5: Timeframe.M5,
        15: Timeframe.M15,
        60: Timeframe.H1,
        240: Timeframe.H4,
        1440: Timeframe.D1,
    }
    return mapping.get(const)


__all__ = [
    "is_mock_mode",
    "MockTick",
    "MockOrderResult",
    "MockPosition",
    "MockMT5Terminal",
    "MT5MockAdapter",
]
