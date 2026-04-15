"""Download XAUUSD M1 data from HistData.com and resample to M15/H1/H4/D1.

USAGE
-----
Run from the project root:

    python scripts/backfill_histdata.py
    python scripts/backfill_histdata.py --start-year 2020 --end-year 2024
    python scripts/backfill_histdata.py --instrument XAUUSD --data-dir ./data

OUTPUT
------
Parquet files:
    data/parquet/{instrument}/{timeframe}/{yyyy}/{mm}.parquet

Manifests:
    data/manifests/{slug}.json

CSV FORMAT (HistData MT platform)
----------------------------------
Each zip contains a .csv file with comma-separated rows, no header:
    2023.01.02,18:00,1826.837,1827.337,1826.617,1826.637,0
    Fields: Date, Time, Open, High, Low, Close, Volume
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path fixup — must happen before any smc imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd
import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ---------------------------------------------------------------------------
# Project imports (after sys.path fixup)
# ---------------------------------------------------------------------------
from smc.data.lake import ForexDataLake
from smc.data.manifest import build_manifest, write_manifest
from smc.data.schemas import SCHEMA_VERSION, Timeframe
from smc.data.writers import write_forex_partitioned

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_ROOT = _PROJECT_ROOT / "data" / "parquet"
_MANIFESTS_ROOT = _PROJECT_ROOT / "data" / "manifests"

_DEFAULT_SPREAD = 3.0  # default spread in points for histdata source

# Resample rules for each target timeframe
_RESAMPLE_RULES: dict[Timeframe, str] = {
    Timeframe.M15: "15min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1D",
}

console = Console()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_histdata",
        description=(
            "Download XAUUSD M1 data from HistData.com and resample to "
            "M15, H1, H4, D1, then write to the AI-SMC data lake."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="First year to download, inclusive (default: 2020).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last year to download, inclusive (default: 2024).",
    )
    parser.add_argument(
        "--instrument",
        default="XAUUSD",
        help="Instrument symbol (default: XAUUSD).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(_PROJECT_ROOT / "data"),
        help=f"Base data directory (default: {_PROJECT_ROOT / 'data'}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_year(year: int, instrument: str, tmp_dir: str) -> str | None:
    """Download M1 zip for *year* from HistData.com.

    Returns the path to the downloaded zip file, or None if download fails.
    """
    from histdata.api import Platform as P
    from histdata.api import TimeFrame as TF
    from histdata.api import download_hist_data

    try:
        zip_path = download_hist_data(
            year=str(year),
            pair=instrument.lower(),
            platform=P.META_TRADER,
            time_frame=TF.ONE_MINUTE,
            output_directory=tmp_dir,
            verbose=False,
        )
        return zip_path
    except AssertionError as exc:
        console.print(f"  [yellow]WARNING:[/] Year {year} not available: {exc}")
        return None
    except Exception as exc:
        console.print(f"  [yellow]WARNING:[/] Failed to download year {year}: {exc}")
        return None


def _download_year_month_by_month(
    year: int, instrument: str, tmp_dir: str
) -> list[str]:
    """Download M1 data month-by-month for years that require it (e.g. current year).

    Returns list of successfully downloaded zip paths.
    """
    from histdata.api import Platform as P
    from histdata.api import TimeFrame as TF
    from histdata.api import download_hist_data

    current_date = datetime.now(tz=timezone.utc)
    zip_paths: list[str] = []

    for month in range(1, 13):
        # Don't request future months
        if year == current_date.year and month >= current_date.month:
            break
        try:
            zip_path = download_hist_data(
                year=str(year),
                month=month,
                pair=instrument.lower(),
                platform=P.META_TRADER,
                time_frame=TF.ONE_MINUTE,
                output_directory=tmp_dir,
                verbose=False,
            )
            zip_paths.append(zip_path)
        except AssertionError as exc:
            console.print(f"  [yellow]WARNING:[/] {year}/{month:02d} not available: {exc}")
        except Exception as exc:
            console.print(f"  [yellow]WARNING:[/] Failed {year}/{month:02d}: {exc}")

    return zip_paths


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _parse_zip_to_dataframe(zip_path: str) -> pd.DataFrame:
    """Parse a HistData MT zip into a pandas DataFrame with a DatetimeIndex.

    HistData MT CSV format (comma-separated, no header):
        2023.01.02,18:00,1826.837,1827.337,1826.617,1826.637,0
        Fields: Date, Time, Open, High, Low, Close, Volume

    Returns a DataFrame with columns [open, high, low, close, volume]
    and a UTC DatetimeIndex named 'ts'.
    """
    rows: list[tuple[str, float, float, float, float, float]] = []

    with zipfile.ZipFile(zip_path) as zf:
        # Find the CSV file (ignore the .txt readme)
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file found in {zip_path}")
        csv_name = csv_names[0]

        with zf.open(csv_name) as raw_file:
            text = io.TextIOWrapper(raw_file, encoding="utf-8", errors="replace")
            for line in text:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                date_str, time_str, open_s, high_s, low_s, close_s, vol_s = (
                    parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
                )
                rows.append((
                    f"{date_str} {time_str}",
                    float(open_s),
                    float(high_s),
                    float(low_s),
                    float(close_s),
                    float(vol_s),
                ))

    if not rows:
        raise ValueError(f"No data rows parsed from {zip_path}")

    ts_strs = [r[0] for r in rows]
    timestamps = pd.to_datetime(ts_strs, format="%Y.%m.%d %H:%M", utc=True)

    df = pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        },
        index=timestamps,
    )
    df.index.name = "ts"
    return df


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------


def _resample_ohlcv(m1_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an M1 OHLCV DataFrame to a coarser timeframe.

    Args:
        m1_df: Pandas DataFrame with DatetimeIndex and OHLCV columns.
        rule: Pandas offset alias, e.g. '15min', '1h', '4h', '1D'.

    Returns:
        Resampled DataFrame (rows with any NaN dropped).
    """
    resampled = m1_df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    return resampled


# ---------------------------------------------------------------------------
# Polars conversion
# ---------------------------------------------------------------------------


def _pandas_to_polars_ohlcv(
    df: pd.DataFrame,
    *,
    timeframe: Timeframe,
    source: str,
) -> pl.DataFrame:
    """Convert a resampled pandas OHLCV DataFrame to a Polars DataFrame.

    Adds required schema columns: spread, timeframe, source, schema_version.
    The DatetimeIndex becomes the 'ts' column (Datetime[ns, UTC]).

    Args:
        df: Pandas DataFrame with a UTC DatetimeIndex and OHLCV columns.
        timeframe: Target timeframe enum value.
        source: Source string written into the 'source' column.

    Returns:
        Polars DataFrame conforming to the FOREX OHLCV schema.
    """
    # Reset index so 'ts' becomes a column
    df_reset = df.reset_index()

    # Convert ts to epoch-ns int to avoid timezone tz-aware conversion issues
    ts_ns = df_reset["ts"].astype("int64").tolist()

    polars_df = pl.DataFrame(
        {
            "ts": pl.Series(ts_ns, dtype=pl.Int64)
            .cast(pl.Datetime("ns"))
            .dt.replace_time_zone("UTC"),
            "open": pl.Series(df_reset["open"].tolist(), dtype=pl.Float64),
            "high": pl.Series(df_reset["high"].tolist(), dtype=pl.Float64),
            "low": pl.Series(df_reset["low"].tolist(), dtype=pl.Float64),
            "close": pl.Series(df_reset["close"].tolist(), dtype=pl.Float64),
            "volume": pl.Series(df_reset["volume"].tolist(), dtype=pl.Float64),
            "spread": pl.Series(
                [_DEFAULT_SPREAD] * len(df_reset), dtype=pl.Float64
            ),
        }
    ).with_columns(
        [
            pl.lit(str(timeframe)).alias("timeframe"),
            pl.lit(source).alias("source"),
            pl.lit(SCHEMA_VERSION).cast(pl.Int32).alias("schema_version"),
        ]
    )

    return polars_df.sort("ts")


# ---------------------------------------------------------------------------
# UTC datetime helper
# ---------------------------------------------------------------------------


def _polars_ts_to_utc(value: object) -> datetime:
    """Convert a Polars min/max timestamp value to a tz-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    raise TypeError(f"Cannot convert {type(value)!r} to UTC datetime.")


# ---------------------------------------------------------------------------
# Summary verification
# ---------------------------------------------------------------------------


def _print_lake_summary(
    instrument: str,
    data_root: Path,
    start_year: int,
    end_year: int,
) -> None:
    """Query the lake for each timeframe and print row counts."""
    lake = ForexDataLake(data_root)
    start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year + 1, 1, 1, tzinfo=timezone.utc)

    table = Table(
        title=f"Data Lake Summary — {instrument}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Timeframe", style="bold")
    table.add_column("Row Count", justify="right")
    table.add_column("Date Min")
    table.add_column("Date Max")

    for tf in [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]:
        df = lake.query(instrument, tf, start=start, end=end)
        if df.is_empty():
            table.add_row(str(tf), "0", "—", "—")
        else:
            ts_min = df["ts"].min()
            ts_max = df["ts"].max()
            table.add_row(
                str(tf),
                f"{len(df):,}",
                str(_polars_ts_to_utc(ts_min).date()),
                str(_polars_ts_to_utc(ts_max).date()),
            )

    console.print(table)


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------


def _process_timeframe(
    m1_df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: Timeframe,
    rule: str,
    data_root: Path,
    manifests_root: Path,
    source: str,
) -> int:
    """Resample, write, and manifest one target timeframe.

    Returns the number of bars written.
    """
    resampled = _resample_ohlcv(m1_df, rule)
    if resampled.empty:
        console.print(f"  [yellow]WARNING:[/] No bars after resample to {timeframe}")
        return 0

    polars_df = _pandas_to_polars_ohlcv(
        resampled, timeframe=timeframe, source=source
    )

    written = write_forex_partitioned(
        polars_df,
        instrument=instrument,
        timeframe=timeframe,
        root=data_root,
    )

    ts_col = polars_df["ts"]
    manifest = build_manifest(
        source=source,
        source_url=f"histdata://xauusd/{timeframe}",
        files=sorted(written),
        row_count=len(polars_df),
        date_min=_polars_ts_to_utc(ts_col.min()),
        date_max=_polars_ts_to_utc(ts_col.max()),
    )
    write_manifest(manifest, manifests_root)

    return len(polars_df)


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    start_year: int = args.start_year
    end_year: int = args.end_year
    instrument: str = args.instrument.upper()
    data_base = Path(args.data_dir)
    data_root = data_base / "parquet"
    manifests_root = data_base / "manifests"

    if start_year > end_year:
        console.print("[bold red]--start-year must be <= --end-year[/]")
        raise SystemExit(1)

    current_year = datetime.now(tz=timezone.utc).year

    # Print plan
    plan_table = Table(title="HistData Backfill Plan", show_header=True, header_style="bold magenta")
    plan_table.add_column("Parameter", style="bold")
    plan_table.add_column("Value")
    plan_table.add_row("Instrument", instrument)
    plan_table.add_row("Years", f"{start_year} – {end_year}")
    plan_table.add_row("Timeframes", "M1 (source) → M15, H1, H4, D1")
    plan_table.add_row("Data root", str(data_root))
    plan_table.add_row("Manifests root", str(manifests_root))
    console.print(plan_table)

    data_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    all_m1_frames: list[pd.DataFrame] = []

    years = list(range(start_year, end_year + 1))

    with tempfile.TemporaryDirectory() as tmp_dir:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            dl_task = progress.add_task(
                f"[cyan]Downloading {instrument} M1 data...", total=len(years)
            )

            for year in years:
                progress.update(
                    dl_task,
                    description=f"[cyan]Downloading {instrument} {year}...",
                )

                zip_paths: list[str] = []

                # Current year requires month-by-month download
                if year >= current_year:
                    console.print(
                        f"  [yellow]NOTE:[/] Year {year} is current/future — "
                        "downloading month-by-month."
                    )
                    zip_paths = _download_year_month_by_month(year, instrument, tmp_dir)
                else:
                    zip_path = _download_year(year, instrument, tmp_dir)
                    if zip_path is not None:
                        zip_paths = [zip_path]

                if not zip_paths:
                    console.print(f"  [yellow]SKIP:[/] No data for {year}")
                    progress.advance(dl_task)
                    continue

                # Parse each zip
                year_frames: list[pd.DataFrame] = []
                for zp in zip_paths:
                    try:
                        df_year = _parse_zip_to_dataframe(zp)
                        year_frames.append(df_year)
                        console.print(
                            f"  [green]Parsed[/] {Path(zp).name}: "
                            f"{len(df_year):,} M1 bars"
                        )
                    except Exception as exc:
                        console.print(f"  [red]ERROR[/] parsing {zp}: {exc}")
                    finally:
                        # Clean up zip immediately after parsing
                        try:
                            Path(zp).unlink(missing_ok=True)
                        except OSError:
                            pass

                if year_frames:
                    all_m1_frames.extend(year_frames)

                progress.advance(dl_task)

    if not all_m1_frames:
        console.print("[bold red]No M1 data was downloaded. Exiting.[/]")
        raise SystemExit(1)

    # Concatenate all years
    console.print("\n[bold cyan]Concatenating all M1 data...[/]")
    m1_combined = pd.concat(all_m1_frames).sort_index()
    # Remove duplicates (same timestamp from overlapping downloads)
    m1_combined = m1_combined[~m1_combined.index.duplicated(keep="last")]
    console.print(
        f"  Total M1 bars: [bold green]{len(m1_combined):,}[/]  "
        f"({m1_combined.index.min().date()} → {m1_combined.index.max().date()})"
    )

    source_id = f"histdata:{instrument}"

    # Resample and write each target timeframe
    console.print("\n[bold cyan]Resampling and writing to data lake...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        tf_task = progress.add_task(
            "[cyan]Processing timeframes...", total=len(_RESAMPLE_RULES)
        )

        for tf, rule in _RESAMPLE_RULES.items():
            progress.update(
                tf_task, description=f"[cyan]Resampling M1 → {tf}..."
            )
            try:
                bars_written = _process_timeframe(
                    m1_combined,
                    instrument=instrument,
                    timeframe=tf,
                    rule=rule,
                    data_root=data_root,
                    manifests_root=manifests_root,
                    source=source_id,
                )
                console.print(
                    f"  [green]OK[/]  {instrument}/{tf}: "
                    f"{bars_written:,} bars written"
                )
            except Exception as exc:
                console.print(
                    f"  [bold red]FAILED[/] {instrument}/{tf}: {exc}"
                )
            progress.advance(tf_task)

    # Final summary via ForexDataLake.query()
    console.print("\n[bold cyan]Verifying data lake...[/]")
    _print_lake_summary(instrument, data_root, start_year, end_year)

    console.print("\n[bold green]HistData backfill complete.[/]")


if __name__ == "__main__":
    main()
