"""AI-SMC command-line interface.

Entry point: ``smc`` (registered in ``pyproject.toml`` under ``[project.scripts]``).

All commands use lazy imports inside the command body so that ``smc --help``
and tab-completion are instant regardless of how many heavy dependencies the
core modules pull in.

Commands
--------
smc ingest     -- Load CSV files into the local data lake.
smc detect     -- Run SMC pattern detection for a given instrument / timeframe / date.
smc lake-info  -- Show data lake coverage per instrument and timeframe.
smc health     -- System health check (config validation, data freshness, MT5 ping).
smc version    -- Print the package version and exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="smc",
    help="AI-SMC: Smart Money Concepts trading system for XAUUSD.",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# smc version
# ---------------------------------------------------------------------------


@app.command(name="version")
def cmd_version() -> None:
    """Print the package version and exit."""
    from smc._version import __version__

    console.print(
        Panel(
            f"[bold cyan]AI-SMC[/bold cyan]  v[green]{__version__}[/green]",
            title="Version",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# smc ingest
# ---------------------------------------------------------------------------


@app.command(name="ingest")
def cmd_ingest(
    csv_dir: Path = typer.Option(
        ...,
        "--csv-dir",
        help="Directory containing CSV files to ingest.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    instrument: str = typer.Option(
        "XAUUSD",
        "--instrument",
        help="Instrument symbol (e.g. XAUUSD).",
    ),
    timeframe: str = typer.Option(
        "D1",
        "--timeframe",
        help="Timeframe string (D1, H4, H1, M15, M5, M1).",
    ),
) -> None:
    """Load CSV price data into the local data lake.

    Reads all ``*.csv`` files under CSV_DIR, validates them against the
    FOREX_OHLCV_SCHEMA, and writes partitioned Parquet files to the data
    directory defined in SMCConfig.
    """
    from smc.config import SMCConfig
    from smc.data.schemas import FOREX_OHLCV_SCHEMA, Timeframe

    cfg = SMCConfig()

    # Validate timeframe early for a clean error message.
    try:
        tf = Timeframe(timeframe.upper())
    except ValueError:
        valid = [t.value for t in Timeframe]
        err_console.print(
            f"[red]Invalid timeframe '{timeframe}'. Valid values: {valid}[/red]"
        )
        raise typer.Exit(code=1) from None

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        err_console.print(f"[yellow]No CSV files found in {csv_dir}[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Ingesting [bold]{len(csv_files)}[/bold] CSV file(s)\n"
            f"Instrument : [cyan]{instrument}[/cyan]\n"
            f"Timeframe  : [cyan]{tf.value}[/cyan]\n"
            f"Source dir : [dim]{csv_dir}[/dim]\n"
            f"Data dir   : [dim]{cfg.data_dir}[/dim]",
            title="[bold]SMC Ingest[/bold]",
        )
    )

    try:
        from smc.data.adapters.base import ingest_csv_files  # type: ignore[import]

        result = ingest_csv_files(
            csv_files=csv_files,
            instrument=instrument,
            timeframe=tf,
            data_dir=cfg.data_dir,
        )
        console.print(f"[green]Ingest complete.[/green] Rows written: {result.row_count}")
    except ImportError:
        err_console.print(
            "[red]smc.data.adapters.base.ingest_csv_files not yet implemented.[/red]"
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Ingest failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# smc detect
# ---------------------------------------------------------------------------


@app.command(name="detect")
def cmd_detect(
    instrument: str = typer.Option(
        "XAUUSD",
        "--instrument",
        help="Instrument symbol.",
    ),
    timeframe: str = typer.Option(
        "H1",
        "--timeframe",
        help="Timeframe to run detection on.",
    ),
    date: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--date",
        help="Date to detect patterns for (YYYY-MM-DD). Defaults to latest available.",
    ),
) -> None:
    """Run SMC pattern detection and print results as a Rich table.

    Loads OHLCV data for INSTRUMENT at TIMEFRAME, runs the SMC detection
    pipeline (order blocks, FVGs, swing highs/lows, liquidity levels), and
    prints a summary table of all detected patterns.
    """
    from smc.config import SMCConfig
    from smc.data.schemas import Timeframe

    cfg = SMCConfig()

    try:
        tf = Timeframe(timeframe.upper())
    except ValueError:
        valid = [t.value for t in Timeframe]
        err_console.print(
            f"[red]Invalid timeframe '{timeframe}'. Valid values: {valid}[/red]"
        )
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            f"Instrument : [cyan]{instrument}[/cyan]\n"
            f"Timeframe  : [cyan]{tf.value}[/cyan]\n"
            f"Date       : [cyan]{date or 'latest'}[/cyan]",
            title="[bold]SMC Detect[/bold]",
        )
    )

    try:
        from smc.smc_core import detect_patterns  # type: ignore[import]

        patterns = detect_patterns(
            instrument=instrument,
            timeframe=tf,
            date=date,
            data_dir=cfg.data_dir,
        )
    except ImportError:
        err_console.print(
            "[red]smc.smc_core.detect_patterns not yet implemented.[/red]"
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Detection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not patterns:
        console.print("[yellow]No patterns detected for the given parameters.[/yellow]")
        return

    table = Table(
        title=f"SMC Patterns — {instrument} {tf.value} ({date or 'latest'})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Direction", style="yellow")
    table.add_column("Price Level", justify="right")
    table.add_column("Strength", justify="right")
    table.add_column("Timestamp", style="dim")

    for pattern in patterns:
        table.add_row(
            str(getattr(pattern, "pattern_type", "?")),
            str(getattr(pattern, "direction", "?")),
            f"{getattr(pattern, 'price_level', 0.0):.2f}",
            f"{getattr(pattern, 'strength', 0.0):.3f}",
            str(getattr(pattern, "ts", "?")),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# smc lake-info
# ---------------------------------------------------------------------------


@app.command(name="lake-info")
def cmd_lake_info() -> None:
    """Show data lake coverage per instrument and timeframe.

    Reads manifest files from the data directory and prints a summary table
    of available data ranges, row counts, and schema versions.
    """
    from smc.config import SMCConfig
    from smc.data.manifest import Manifest

    cfg = SMCConfig()
    manifests_dir = cfg.data_dir / "manifests"

    if not manifests_dir.exists():
        console.print(
            Panel(
                f"[yellow]No manifests directory found at[/yellow] [dim]{manifests_dir}[/dim]\n"
                "Run [bold]smc ingest[/bold] to populate the data lake.",
                title="[bold]Data Lake Info[/bold]",
            )
        )
        return

    manifest_files = sorted(manifests_dir.glob("*.json"))
    if not manifest_files:
        console.print("[yellow]No manifest files found. Data lake is empty.[/yellow]")
        return

    table = Table(
        title="Data Lake Coverage",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Date Min", style="green")
    table.add_column("Date Max", style="green")
    table.add_column("Schema", justify="right", style="dim")
    table.add_column("Fetched At", style="dim")

    for path in manifest_files:
        try:
            manifest = Manifest.from_json(path.read_text(encoding="utf-8"))
            table.add_row(
                manifest.source,
                f"{manifest.row_count:,}",
                manifest.date_min[:10],
                manifest.date_max[:10],
                str(manifest.schema_version),
                manifest.fetched_at[:19],
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row(
                path.stem,
                "[red]ERROR[/red]",
                f"[red]{exc}[/red]",
                "",
                "",
                "",
            )

    console.print(table)


# ---------------------------------------------------------------------------
# smc health
# ---------------------------------------------------------------------------


@app.command(name="health")
def cmd_health() -> None:
    """Run a system health check.

    Validates the active configuration, checks data freshness, and (when not
    in mock mode) attempts an MT5 connection ping.
    """
    from datetime import datetime, timezone

    from smc.config import SMCConfig

    cfg = SMCConfig()

    table = Table(
        title="System Health",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    def _ok(detail: str = "") -> tuple[str, str]:
        return "[bold green]OK[/bold green]", detail

    def _warn(detail: str = "") -> tuple[str, str]:
        return "[bold yellow]WARN[/bold yellow]", detail

    def _fail(detail: str = "") -> tuple[str, str]:
        return "[bold red]FAIL[/bold red]", detail

    # Config validation
    try:
        _ = SMCConfig()
        status, detail = _ok(f"env={cfg.env}, instrument={cfg.instrument}")
    except Exception as exc:  # noqa: BLE001
        status, detail = _fail(str(exc))
    table.add_row("Config", status, detail)

    # Data dir
    if cfg.data_dir.exists():
        status, detail = _ok(str(cfg.data_dir))
    else:
        status, detail = _warn(f"Directory not found: {cfg.data_dir}")
    table.add_row("Data directory", status, detail)

    # Data freshness — check the newest manifest
    manifests_dir = cfg.data_dir / "manifests"
    if manifests_dir.exists():
        manifest_files = sorted(manifests_dir.glob("*.json"))
        if manifest_files:
            from smc.data.manifest import Manifest

            newest = manifest_files[-1]
            try:
                m = Manifest.from_json(newest.read_text(encoding="utf-8"))
                # Warn if last data is older than 2 days
                last_date = datetime.fromisoformat(m.date_max)
                age_days = (datetime.now(tz=timezone.utc) - last_date).days
                if age_days <= 2:
                    status, detail = _ok(f"{m.source} — {m.date_max[:10]} ({age_days}d ago)")
                else:
                    status, detail = _warn(
                        f"{m.source} — {m.date_max[:10]} ({age_days}d ago, stale?)"
                    )
            except Exception as exc:  # noqa: BLE001
                status, detail = _warn(f"Cannot parse manifest: {exc}")
        else:
            status, detail = _warn("No manifests found — run smc ingest")
    else:
        status, detail = _warn("No manifests dir — data lake empty")
    table.add_row("Data freshness", status, detail)

    # MT5 connectivity
    if cfg.mt5_mock:
        status, detail = _warn("mt5_mock=True — MT5 connection skipped (macOS/dev)")
    elif not cfg.has_mt5_credentials():
        status, detail = _warn("MT5 credentials not configured (SMC_MT5_LOGIN etc.)")
    else:
        try:
            import MetaTrader5 as mt5  # type: ignore[import]

            if mt5.initialize(
                server=cfg.mt5_server,
                login=cfg.mt5_login,
                password=cfg.mt5_password.get_secret_value(),
            ):
                info = mt5.terminal_info()
                mt5.shutdown()
                status, detail = _ok(f"Connected — {info.name if info else 'unknown'}")
            else:
                status, detail = _fail(f"mt5.initialize() failed: {mt5.last_error()}")
        except ImportError:
            status, detail = _fail("MetaTrader5 package not installed (pip install ai-smc[mt5])")
        except Exception as exc:  # noqa: BLE001
            status, detail = _fail(str(exc))
    table.add_row("MT5 connection", status, detail)

    # LLM key
    if cfg.has_llm():
        status, detail = _ok(f"model={cfg.anthropic_model}")
    else:
        status, detail = _warn("SMC_ANTHROPIC_API_KEY not set (Phase 3+ feature)")
    table.add_row("LLM (Anthropic)", status, detail)

    # Telegram
    if cfg.has_telegram():
        status, detail = _ok("Bot token and chat ID configured")
    else:
        status, detail = _warn("SMC_TELEGRAM_BOT_TOKEN / SMC_TELEGRAM_CHAT_ID not set")
    table.add_row("Telegram alerts", status, detail)

    console.print(table)
    console.print(
        f"\n[dim]Check time: {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}[/dim]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
