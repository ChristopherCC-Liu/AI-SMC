"""Real MT5 adapter — wraps the MetaTrader5 Python SDK (Windows only).

This module provides :class:`MT5Adapter`, which connects to a live MetaTrader 5
terminal on Windows and fetches historical OHLCV bars via
``mt5.copy_rates_range()``.

Import strategy
---------------
The ``MetaTrader5`` package is only available on Windows and requires an
active terminal installation.  The import is deferred to the point of
instantiation so that the rest of the ``smc`` package can be imported on
macOS/Linux without errors.  If the package is absent, a helpful
:class:`ForexAdapterError` is raised immediately when :class:`MT5Adapter`
is created.

Retry logic
-----------
Transient MT5 failures (network hiccups, terminal busy) are retried up to
``max_retries`` times with exponential back-off.

Context manager
---------------
:class:`MT5Adapter` implements ``__enter__``/``__exit__`` so it can be used
in a ``with`` block that automatically calls ``mt5.initialize()`` and
``mt5.shutdown()``::

    from smc.data.adapters.mt5_adapter import MT5Adapter
    from smc.data.schemas import Timeframe
    from datetime import datetime, timezone

    with MT5Adapter(login=12345, password="secret", server="Broker-Demo") as adapter:
        df = adapter.fetch(
            instrument="XAUUSD",
            timeframe=Timeframe.H1,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

Timeframe mapping
-----------------
::

    Timeframe.M1  -> mt5.TIMEFRAME_M1
    Timeframe.M5  -> mt5.TIMEFRAME_M5
    Timeframe.M15 -> mt5.TIMEFRAME_M15
    Timeframe.H1  -> mt5.TIMEFRAME_H1
    Timeframe.H4  -> mt5.TIMEFRAME_H4
    Timeframe.D1  -> mt5.TIMEFRAME_D1
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import TracebackType
from typing import TYPE_CHECKING, Any

import polars as pl

from smc.data.adapters.base import ForexAdapter, ForexAdapterError, ForexAdapterSpec
from smc.data.schemas import SCHEMA_VERSION, Timeframe

if TYPE_CHECKING:
    # Only imported for type hints — never executed at runtime on non-Windows
    import MetaTrader5 as _MT5Type  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

# Maps our enum to MT5 module attribute names (resolved at runtime)
_TIMEFRAME_ATTR: dict[Timeframe, str] = {
    Timeframe.M1: "TIMEFRAME_M1",
    Timeframe.M5: "TIMEFRAME_M5",
    Timeframe.M15: "TIMEFRAME_M15",
    Timeframe.H1: "TIMEFRAME_H1",
    Timeframe.H4: "TIMEFRAME_H4",
    Timeframe.D1: "TIMEFRAME_D1",
}

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MT5Adapter:
    """ForexAdapter that fetches data from a live MetaTrader 5 terminal.

    This adapter is **Windows-only**.  Attempting to instantiate it on other
    platforms raises :class:`ForexAdapterError` immediately.

    Args:
        login: MT5 account login number.  ``None`` for the default account
            already open in the terminal.
        password: MT5 account password.  ``None`` to skip authentication.
        server: Broker server name.  ``None`` to use the terminal default.
        instrument: Primary instrument used to populate the adapter spec.
        source_name: Source label written into the ``source`` column
            (default: ``"mt5"``).
        max_retries: Number of retry attempts for transient failures (default 3).
        retry_delay: Initial delay in seconds between retries; doubles each
            attempt (exponential back-off, default 1.0 s).
        timeout_ms: Terminal connection timeout in milliseconds (default 10 000).
    """

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        instrument: str = "XAUUSD",
        source_name: str = "mt5",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout_ms: int = 10_000,
    ) -> None:
        self._mt5 = _import_mt5()
        self._login = login
        self._password = password
        self._server = server
        self._instrument = instrument
        self._source_name = source_name
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout_ms = timeout_ms
        self._initialised: bool = False

        self._spec = ForexAdapterSpec(
            source=source_name,
            instrument=instrument,
            timeframes=tuple(Timeframe),
            description=(
                f"Live MT5 adapter for {instrument} "
                f"(server={server or 'default'}, login={login or 'default'})"
            ),
        )

    # ------------------------------------------------------------------
    # ForexAdapter protocol
    # ------------------------------------------------------------------

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
        """Fetch OHLCV bars from the live MT5 terminal.

        The adapter must be initialised (either explicitly via
        :meth:`initialize` or by using the context manager) before calling
        this method.

        Args:
            instrument: Symbol to fetch, e.g. ``"XAUUSD"``.
            timeframe: Timeframe variant.
            start: Inclusive start (tz-aware).
            end: Exclusive end (tz-aware).

        Returns:
            :class:`polars.DataFrame` sorted ascending by ``ts``.

        Raises:
            ForexAdapterError: if not initialised, if the symbol is not found,
                if the timeframe is unsupported, or on persistent terminal
                errors after retries.
        """
        if not self._initialised:
            raise ForexAdapterError(
                "MT5Adapter is not initialised.  Call initialize() first or "
                "use the adapter as a context manager."
            )
        if start.tzinfo is None or end.tzinfo is None:
            raise ForexAdapterError("'start' and 'end' must be tz-aware datetimes.")

        mt5_timeframe = self._resolve_timeframe(timeframe)
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)

        rates = self._fetch_with_retry(instrument, mt5_timeframe, start_utc, end_utc)
        return self._rates_to_frame(rates, instrument, timeframe)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> MT5Adapter:
        """Initialise the MT5 terminal connection."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Shut down the MT5 terminal connection."""
        self.shutdown()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Connect to the MT5 terminal.

        Raises:
            ForexAdapterError: if the terminal is unavailable or login fails.
        """
        kwargs: dict[str, Any] = {"timeout": self._timeout_ms}
        if self._login is not None:
            kwargs["login"] = self._login
        if self._password is not None:
            kwargs["password"] = self._password
        if self._server is not None:
            kwargs["server"] = self._server

        success = self._mt5.initialize(**kwargs)
        if not success:
            code, msg = self._mt5.last_error()
            raise ForexAdapterError(
                f"MT5 initialize() failed with error {code}: {msg}. "
                "Check that the MetaTrader 5 terminal is running and the "
                "login credentials are correct."
            )
        self._initialised = True

    def shutdown(self) -> None:
        """Disconnect from the MT5 terminal."""
        if self._initialised:
            self._mt5.shutdown()
            self._initialised = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_timeframe(self, timeframe: Timeframe) -> Any:
        """Return the MT5 module-level timeframe constant for *timeframe*."""
        attr_name = _TIMEFRAME_ATTR.get(timeframe)
        if attr_name is None:
            raise ForexAdapterError(f"Unsupported timeframe: {timeframe!r}")
        constant = getattr(self._mt5, attr_name, None)
        if constant is None:
            raise ForexAdapterError(
                f"MetaTrader5 module does not expose attribute {attr_name!r}. "
                "Ensure you have a compatible version installed."
            )
        return constant

    def _fetch_with_retry(
        self,
        symbol: str,
        mt5_timeframe: Any,
        start: datetime,
        end: datetime,
    ) -> Any:
        """Call ``mt5.copy_rates_range`` with exponential back-off retry.

        Args:
            symbol: Instrument symbol.
            mt5_timeframe: MT5 timeframe constant.
            start: UTC start datetime.
            end: UTC end datetime.

        Returns:
            Structured numpy array from ``copy_rates_range``.

        Raises:
            ForexAdapterError: if all retries are exhausted or data is ``None``.
        """
        last_error: str = ""
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            rates = self._mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
            if rates is not None and len(rates) > 0:
                return rates

            code, msg = self._mt5.last_error()
            last_error = f"MT5 error {code}: {msg}"

            if attempt < self._max_retries:
                time.sleep(delay)
                delay *= 2.0  # exponential back-off

        raise ForexAdapterError(
            f"copy_rates_range({symbol!r}) returned no data after "
            f"{self._max_retries} attempt(s).  Last error: {last_error}"
        )

    def _rates_to_frame(
        self,
        rates: Any,
        instrument: str,
        timeframe: Timeframe,
    ) -> pl.DataFrame:
        """Convert a structured numpy array from MT5 to a Polars DataFrame.

        The MT5 array dtype is::

            [('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
             ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
             ('spread', '<i4'), ('real_volume', '<i8')]

        We map ``tick_volume`` → ``volume`` and ``time`` (epoch s) → ``ts``
        (datetime[ns, UTC]).
        """
        import numpy as np

        # Build Polars DataFrame from numpy structured array
        # Each field is extracted independently to avoid mutation
        n = len(rates)
        ts_ns = rates["time"].astype(np.int64) * 1_000_000_000  # s → ns

        df = pl.DataFrame(
            {
                "ts": pl.Series(ts_ns, dtype=pl.Int64)
                    .cast(pl.Datetime("ns"))
                    .dt.replace_time_zone("UTC"),
                "open": pl.Series(rates["open"].astype(float), dtype=pl.Float64),
                "high": pl.Series(rates["high"].astype(float), dtype=pl.Float64),
                "low": pl.Series(rates["low"].astype(float), dtype=pl.Float64),
                "close": pl.Series(rates["close"].astype(float), dtype=pl.Float64),
                "volume": pl.Series(rates["tick_volume"].astype(float), dtype=pl.Float64),
                "spread": pl.Series(rates["spread"].astype(float), dtype=pl.Float64),
            }
        ).with_columns(
            [
                pl.lit(str(timeframe)).alias("timeframe"),
                pl.lit(f"{self._source_name}:{instrument}").alias("source"),
                pl.lit(SCHEMA_VERSION).cast(pl.Int32).alias("schema_version"),
            ]
        )

        return df.sort("ts")


# ---------------------------------------------------------------------------
# Module-level import helper
# ---------------------------------------------------------------------------


def _import_mt5() -> Any:
    """Import and return the ``MetaTrader5`` module.

    Raises:
        ForexAdapterError: with a helpful message if the package is missing or
            we are not running on Windows.
    """
    import sys

    if sys.platform != "win32":
        raise ForexAdapterError(
            "MetaTrader5 is only available on Windows.  "
            "On macOS/Linux, use MT5MockAdapter instead:\n\n"
            "    from smc.data.adapters.mt5_mock import MT5MockAdapter\n\n"
            "or set SMC_MT5_MOCK=1 in your environment."
        )

    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]

        return mt5
    except ImportError as exc:
        raise ForexAdapterError(
            "MetaTrader5 Python package is not installed.  "
            "Install it with:\n\n"
            "    pip install MetaTrader5\n\n"
            "or add it via:\n\n"
            "    pip install 'ai-smc[mt5]'\n\n"
            f"Original error: {exc}"
        ) from exc


__all__ = ["MT5Adapter"]
