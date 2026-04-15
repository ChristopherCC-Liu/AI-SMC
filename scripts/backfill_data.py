"""Backfill historical OHLCV data into the AI-SMC data lake.

USAGE
-----
Run from the project root (so that ``src/`` is on the Python path, or after
``pip install -e .``):

    # Full default backfill (XAUUSD, 2020-01-01 → 2025-01-01, all timeframes)
    python scripts/backfill_data.py

    # Custom instrument / date range
    python scripts/backfill_data.py \\
        --instrument XAUUSD \\
        --start 2022-01-01 \\
        --end 2024-01-01 \\
        --timeframes D1,H4

    # macOS / Linux (yfinance fallback, D1 only)
    python scripts/backfill_data.py --timeframes D1

PLATFORM NOTES
--------------
Windows (MT5 available):
    Uses the MetaTrader5 SDK via ``MT5Adapter``.  The MetaTrader 5 terminal
    must be running and logged in before executing this script.  To install the
    SDK:  pip install "ai-smc[mt5]"

macOS / Linux (no MT5):
    Falls back to ``yfinance`` for D1 data using the GC=F (Gold Futures) ticker,
    which tracks XAUUSD closely.  H4 / H1 / M15 timeframes are skipped with a
    warning because yfinance does not reliably provide intraday forex data back
    to 2020.

    Install yfinance:  pip install yfinance

OUTPUT
------
Parquet files are written to:
    data/parquet/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

Manifests are written to:
    data/manifests/{slug}.json

DEPENDENCIES
------------
  - rich            (progress bars, already in pyproject.toml)
  - polars          (already in pyproject.toml)
  - pyarrow         (already in pyproject.toml)
  - yfinance        (macOS fallback; pip install yfinance)
  - MetaTrader5     (Windows only; pip install "ai-smc[mt5]")
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Rich console — imported early so we can print coloured errors on failure
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
# When run as ``python scripts/backfill_data.py`` from the repo root, ``src``
# must be on sys.path so that ``import smc`` resolves.  We add it here so the
# script works without ``pip install -e .``.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Project imports (after sys.path fixup)
# ---------------------------------------------------------------------------
from smc.data.manifest import build_manifest, write_manifest  # noqa: E402
from smc.data.schemas import SCHEMA_VERSION, Timeframe  # noqa: E402
from smc.data.writers import write_forex_partitioned  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default data lake root — relative to project root
_DATA_ROOT = _PROJECT_ROOT / "data" / "parquet"
_MANIFESTS_ROOT = _PROJECT_ROOT / "data" / "manifests"

# yfinance ticker that proxies XAUUSD (Gold Futures continuous contract)
_YFINANCE_GOLD_TICKER = "GC=F"

# Timeframes supported by yfinance in daily interval mode
_YFINANCE_SUPPORTED: frozenset[Timeframe] = frozenset({Timeframe.D1})


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        prog="backfill_data",
        description=(
            "Download historical OHLCV data and write it to the AI-SMC data lake.\n\n"
            "On Windows, uses MetaTrader5 SDK (full timeframe support).\n"
            "On macOS/Linux, uses yfinance for D1 data only (GC=F gold futures)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instrument",
        default="XAUUSD",
        help="Instrument symbol (default: XAUUSD).  yfinance will always use GC=F regardless.",
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date in YYYY-MM-DD format, inclusive (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end",
        default="2025-01-01",
        help="End date in YYYY-MM-DD format, exclusive (default: 2025-01-01).",
    )
    parser.add_argument(
        "--timeframes",
        default="D1,H4,H1,M15",
        help="Comma-separated list of timeframes to backfill (default: D1,H4,H1,M15).",
    )
    parser.add_argument(
        "--data-root",
        default=str(_DATA_ROOT),
        help=f"Root directory for parquet output (default: {_DATA_ROOT}).",
    )
    parser.add_argument(
        "--manifests-root",
        default=str(_MANIFESTS_ROOT),
        help=f"Directory for manifest JSON files (default: {_MANIFESTS_ROOT}).",
    )
    return parser.parse_args()


def _parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a UTC-aware datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        console.print(f"[bold red]Invalid date format:[/] {date_str!r}  Expected YYYY-MM-DD")
        raise SystemExit(1) from exc


def _parse_timeframes(tf_str: str) -> list[Timeframe]:
    """Parse a comma-separated timeframe string into a list of Timeframe values."""
    valid = {tf.value: tf for tf in Timeframe}
    result: list[Timeframe] = []
    for token in tf_str.split(","):
        token = token.strip().upper()
        if token not in valid:
            console.print(
                f"[bold red]Unknown timeframe:[/] {token!r}  "
                f"Valid choices: {', '.join(valid)}"
            )
            raise SystemExit(1)
        result.append(valid[token])
    return result


# ---------------------------------------------------------------------------
# MT5 backfill path
# ---------------------------------------------------------------------------

def _backfill_mt5(
    instrument: str,
    timeframes: list[Timeframe],
    start: datetime,
    end: datetime,
    data_root: Path,
    manifests_root: Path,
) -> None:
    """Run the full backfill using the MT5Adapter (Windows only)."""
    from smc.data.adapters.mt5_adapter import MT5Adapter

    console.print("\n[bold cyan]MT5 backfill mode[/] — connecting to terminal...")

    with MT5Adapter(instrument=instrument) as adapter:
        console.print(f"[green]MT5 connected.[/]  Fetching {len(timeframes)} timeframe(s).")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall = progress.add_task(
                f"[cyan]Backfilling {instrument}...", total=len(timeframes)
            )

            for tf in timeframes:
                progress.update(overall, description=f"[cyan]{instrument} / {tf}")

                df = adapter.fetch(
                    instrument=instrument,
                    timeframe=tf,
                    start=start,
                    end=end,
                )

                if df.is_empty():
                    console.print(
                        f"  [yellow]WARNING:[/] No data returned for {instrument}/{tf} "
                        f"in the requested range.  Skipping."
                    )
                    progress.advance(overall)
                    continue

                written = write_forex_partitioned(
                    df,
                    instrument=instrument,
                    timeframe=tf,
                    root=data_root,
                )

                ts_col = df["ts"]
                ts_min = ts_col.min()
                ts_max = ts_col.max()

                manifest = build_manifest(
                    source=f"mt5:{instrument}",
                    source_url=f"mt5://{instrument}",
                    files=sorted(written),
                    row_count=len(df),
                    date_min=_polars_ts_to_utc(ts_min),
                    date_max=_polars_ts_to_utc(ts_max),
                )
                write_manifest(manifest, manifests_root)

                progress.advance(overall)
                console.print(
                    f"  [green]OK[/]  {instrument}/{tf}: "
                    f"{len(df):,} bars → {len(written)} file(s)"
                )

    console.print("\n[bold green]MT5 backfill complete.[/]")


# ---------------------------------------------------------------------------
# yfinance backfill path (macOS / Linux fallback)
# ---------------------------------------------------------------------------

def _backfill_yfinance(
    instrument: str,
    timeframes: list[Timeframe],
    start: datetime,
    end: datetime,
    data_root: Path,
    manifests_root: Path,
) -> None:
    """Run D1-only backfill using yfinance as a fallback on non-Windows systems."""
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as exc:
        console.print(
            "[bold red]yfinance is not installed.[/]  "
            "Install it with:\n\n    pip install yfinance\n"
        )
        raise SystemExit(1) from exc

    import polars as pl

    console.print(
        "\n[bold yellow]yfinance fallback mode[/] — MetaTrader5 is not available on this platform."
    )

    # Warn about unsupported timeframes and filter to D1 only
    unsupported = [tf for tf in timeframes if tf not in _YFINANCE_SUPPORTED]
    supported = [tf for tf in timeframes if tf in _YFINANCE_SUPPORTED]

    if unsupported:
        console.print(
            f"  [yellow]WARNING:[/] yfinance does not support intraday backfill to 2020. "
            f"Skipping timeframes: {', '.join(str(tf) for tf in unsupported)}"
        )

    if not supported:
        console.print(
            "[bold red]No supported timeframes remain after filtering.[/]  "
            "On macOS/Linux, only D1 is available via yfinance.  "
            "Pass [bold]--timeframes D1[/] to backfill daily data."
        )
        raise SystemExit(1)

    console.print(
        f"  Fetching [bold]{_YFINANCE_GOLD_TICKER}[/] (gold futures) "
        f"from {start.date()} to {end.date()} as D1 proxy for {instrument}..."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[yellow]Downloading {_YFINANCE_GOLD_TICKER}...", total=1)

        ticker = yf.Ticker(_YFINANCE_GOLD_TICKER)
        raw = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            actions=False,
        )

        progress.advance(task)

    if raw is None or raw.empty:
        console.print(
            f"[bold red]No data returned from yfinance for {_YFINANCE_GOLD_TICKER}.[/]  "
            "Check your internet connection and try again."
        )
        raise SystemExit(1)

    # Convert pandas DataFrame → polars DataFrame with the project OHLCV schema
    df = _yfinance_to_polars(raw, instrument=instrument, timeframe=Timeframe.D1)

    console.print(f"  [green]Downloaded {len(df):,} daily bars.[/]")

    written = write_forex_partitioned(
        df,
        instrument=instrument,
        timeframe=Timeframe.D1,
        root=data_root,
    )

    ts_col = df["ts"]
    ts_min = ts_col.min()
    ts_max = ts_col.max()

    manifest = build_manifest(
        source=f"yfinance:{instrument}",
        source_url=f"yfinance://{_YFINANCE_GOLD_TICKER}",
        files=sorted(written),
        row_count=len(df),
        date_min=_polars_ts_to_utc(ts_min),
        date_max=_polars_ts_to_utc(ts_max),
    )
    write_manifest(manifest, manifests_root)

    console.print(
        f"\n[bold green]yfinance backfill complete.[/]  "
        f"{instrument}/D1: {len(df):,} bars → {len(written)} file(s)"
    )


def _yfinance_to_polars(
    raw,  # pandas DataFrame from yfinance
    instrument: str,
    timeframe: Timeframe,
) -> "polars.DataFrame":  # noqa: F821 — polars imported inside function
    """Convert a yfinance pandas DataFrame to a project-schema Polars DataFrame.

    yfinance columns (auto_adjust=True):
        Open, High, Low, Close, Volume  (index: DatetimeTZDtype UTC)

    We map:
        Volume → volume
        spread  → 0.0 (not available from yfinance)

    Args:
        raw: pandas DataFrame returned by ``yfinance.Ticker.history()``.
        instrument: Symbol name written into the ``source`` column.
        timeframe: Timeframe enum value.

    Returns:
        Polars DataFrame conforming to the FOREX OHLCV schema.
    """
    import polars as pl

    # Reset index so the DatetimeIndex becomes a regular column
    df_pd = raw.reset_index()

    # The datetime column may be named "Date", "Datetime", or "index"
    ts_col_name = next(
        (c for c in df_pd.columns if c.lower() in ("date", "datetime", "index")),
        df_pd.columns[0],
    )

    # Convert to UTC-aware pandas timestamps → polars Int64 ns → pl.Datetime
    ts_series = df_pd[ts_col_name].dt.tz_localize(None).astype("int64")  # epoch ns

    df = pl.DataFrame(
        {
            "ts": pl.Series(ts_series.tolist(), dtype=pl.Int64)
                .cast(pl.Datetime("ns"))
                .dt.replace_time_zone("UTC"),
            "open": pl.Series(df_pd["Open"].tolist(), dtype=pl.Float64),
            "high": pl.Series(df_pd["High"].tolist(), dtype=pl.Float64),
            "low": pl.Series(df_pd["Low"].tolist(), dtype=pl.Float64),
            "close": pl.Series(df_pd["Close"].tolist(), dtype=pl.Float64),
            "volume": pl.Series(df_pd["Volume"].tolist(), dtype=pl.Float64),
            "spread": pl.Series([0.0] * len(df_pd), dtype=pl.Float64),
        }
    ).with_columns(
        [
            pl.lit(str(timeframe)).alias("timeframe"),
            pl.lit(f"yfinance:{instrument}").alias("source"),
            pl.lit(SCHEMA_VERSION).cast(pl.Int32).alias("schema_version"),
        ]
    )

    return df.sort("ts")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _polars_ts_to_utc(value: object) -> datetime:
    """Convert a Polars min/max timestamp to a tz-aware UTC datetime.

    Polars returns a ``datetime`` when the series has a time zone, or an
    ``int`` (epoch ns) when it does not.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    raise TypeError(f"Cannot convert {type(value)!r} to UTC datetime.")


def _print_summary(
    instrument: str,
    timeframes: list[Timeframe],
    start: datetime,
    end: datetime,
    data_root: Path,
    manifests_root: Path,
    backend: str,
) -> None:
    """Print a Rich table summarising the planned backfill before execution."""
    table = Table(title="Backfill Plan", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="bold")
    table.add_column("Value")

    table.add_row("Instrument", instrument)
    table.add_row("Timeframes", ", ".join(str(tf) for tf in timeframes))
    table.add_row("Start (inclusive)", start.date().isoformat())
    table.add_row("End (exclusive)", end.date().isoformat())
    table.add_row("Backend", backend)
    table.add_row("Data root", str(data_root))
    table.add_row("Manifests root", str(manifests_root))

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the backfill script."""
    args = _parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    timeframes = _parse_timeframes(args.timeframes)
    data_root = Path(args.data_root)
    manifests_root = Path(args.manifests_root)

    if start >= end:
        console.print("[bold red]--start must be earlier than --end.[/]")
        raise SystemExit(1)

    # Detect platform and choose backend
    is_windows = sys.platform == "win32"
    backend = "MT5 (MetaTrader5 SDK)" if is_windows else "yfinance (macOS/Linux fallback)"

    _print_summary(
        instrument=args.instrument,
        timeframes=timeframes,
        start=start,
        end=end,
        data_root=data_root,
        manifests_root=manifests_root,
        backend=backend,
    )

    # Ensure output directories exist
    data_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    if is_windows:
        try:
            _backfill_mt5(
                instrument=args.instrument,
                timeframes=timeframes,
                start=start,
                end=end,
                data_root=data_root,
                manifests_root=manifests_root,
            )
        except ImportError as exc:
            # MT5 package absent on Windows — surface a clear error
            console.print(
                f"[bold red]MetaTrader5 import failed:[/] {exc}\n\n"
                "Install it with:\n\n    pip install 'ai-smc[mt5]'\n"
            )
            raise SystemExit(1) from exc
    else:
        _backfill_yfinance(
            instrument=args.instrument,
            timeframes=timeframes,
            start=start,
            end=end,
            data_root=data_root,
            manifests_root=manifests_root,
        )


if __name__ == "__main__":
    main()
