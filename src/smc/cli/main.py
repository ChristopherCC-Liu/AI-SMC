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

def _trend_color(trend: str) -> str:
    """Return a Rich color name for the given trend direction."""
    if trend == "bullish":
        return "green"
    if trend == "bearish":
        return "red"
    return "yellow"


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

    csv_files = sorted(csv_dir.rglob("*.csv"))
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
        from smc.data.ingest import ingest_csv_files

        manifest = ingest_csv_files(
            csv_dir=csv_dir,
            instrument=instrument,
            timeframe=tf,
            data_dir=cfg.data_dir,
        )
        console.print(
            Panel(
                f"[green]Ingest complete![/green]\n"
                f"Rows ingested : [bold]{manifest.row_count:,}[/bold]\n"
                f"SHA-256        : [dim]{manifest.sha256[:16]}...[/dim]\n"
                f"Date range     : [cyan]{manifest.date_min[:10]}[/cyan] → "
                f"[cyan]{manifest.date_max[:10]}[/cyan]\n"
                f"Manifest path  : [dim]{cfg.data_dir / 'manifests'}[/dim]",
                title="[bold green]Ingest Result[/bold green]",
            )
        )
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

    # Parse date string to datetime if provided
    from datetime import datetime as dt_cls
    from datetime import timezone as tz_cls

    detect_date: dt_cls | None = None
    if date is not None:
        try:
            detect_date = dt_cls.fromisoformat(date).replace(tzinfo=tz_cls.utc)
        except ValueError:
            err_console.print(
                f"[red]Invalid date format '{date}'. Use YYYY-MM-DD.[/red]"
            )
            raise typer.Exit(code=1) from None

    try:
        from smc.smc_core.pipeline import detect_patterns

        snapshot = detect_patterns(
            instrument=instrument,
            timeframe=tf,
            date=detect_date,
            data_dir=cfg.data_dir,
        )
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Detection failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    # --- Render SMCSnapshot as Rich tables ---

    # Trend direction
    console.print(
        f"\n[bold]Trend Direction:[/bold] [{_trend_color(snapshot.trend_direction)}]"
        f"{snapshot.trend_direction.upper()}[/{_trend_color(snapshot.trend_direction)}]\n"
    )

    # Swing Points
    if snapshot.swing_points:
        sw_table = Table(
            title="Swing Points",
            show_header=True,
            header_style="bold magenta",
        )
        sw_table.add_column("Type", style="cyan", no_wrap=True)
        sw_table.add_column("Price", justify="right")
        sw_table.add_column("Strength", justify="right")
        sw_table.add_column("Timestamp", style="dim")
        for sp in snapshot.swing_points:
            sw_table.add_row(
                sp.swing_type,
                f"{sp.price:.2f}",
                str(sp.strength),
                sp.ts.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(sw_table)

    # Order Blocks
    if snapshot.order_blocks:
        ob_table = Table(
            title="Order Blocks",
            show_header=True,
            header_style="bold magenta",
        )
        ob_table.add_column("Type", style="cyan", no_wrap=True)
        ob_table.add_column("High", justify="right")
        ob_table.add_column("Low", justify="right")
        ob_table.add_column("Mitigated", style="yellow")
        ob_table.add_column("Start", style="dim")
        for ob in snapshot.order_blocks:
            ob_table.add_row(
                ob.ob_type,
                f"{ob.high:.2f}",
                f"{ob.low:.2f}",
                "Yes" if ob.mitigated else "No",
                ob.ts_start.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(ob_table)

    # Fair Value Gaps
    if snapshot.fvgs:
        fvg_table = Table(
            title="Fair Value Gaps",
            show_header=True,
            header_style="bold magenta",
        )
        fvg_table.add_column("Type", style="cyan", no_wrap=True)
        fvg_table.add_column("High", justify="right")
        fvg_table.add_column("Low", justify="right")
        fvg_table.add_column("Filled %", justify="right")
        fvg_table.add_column("Timestamp", style="dim")
        for fvg in snapshot.fvgs:
            fvg_table.add_row(
                fvg.fvg_type,
                f"{fvg.high:.2f}",
                f"{fvg.low:.2f}",
                f"{fvg.filled_pct:.1f}%",
                fvg.ts.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(fvg_table)

    # Structure Breaks
    if snapshot.structure_breaks:
        sb_table = Table(
            title="Structure Breaks",
            show_header=True,
            header_style="bold magenta",
        )
        sb_table.add_column("Type", style="cyan", no_wrap=True)
        sb_table.add_column("Direction", style="yellow")
        sb_table.add_column("Price", justify="right")
        sb_table.add_column("Timestamp", style="dim")
        for sb in snapshot.structure_breaks:
            sb_table.add_row(
                sb.break_type.upper(),
                sb.direction,
                f"{sb.price:.2f}",
                sb.ts.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(sb_table)

    # Liquidity Levels
    if snapshot.liquidity_levels:
        lq_table = Table(
            title="Liquidity Levels",
            show_header=True,
            header_style="bold magenta",
        )
        lq_table.add_column("Type", style="cyan", no_wrap=True)
        lq_table.add_column("Price", justify="right")
        lq_table.add_column("Touches", justify="right")
        lq_table.add_column("Swept", style="yellow")
        for lq in snapshot.liquidity_levels:
            lq_table.add_row(
                lq.level_type,
                f"{lq.price:.2f}",
                str(lq.touches),
                "Yes" if lq.swept else "No",
            )
        console.print(lq_table)

    # Summary
    console.print(
        Panel(
            f"Snapshot ts     : [dim]{snapshot.ts.strftime('%Y-%m-%d %H:%M')}[/dim]\n"
            f"Swing points    : [bold]{len(snapshot.swing_points)}[/bold]\n"
            f"Order blocks    : [bold]{len(snapshot.order_blocks)}[/bold]\n"
            f"Fair value gaps  : [bold]{len(snapshot.fvgs)}[/bold]\n"
            f"Structure breaks : [bold]{len(snapshot.structure_breaks)}[/bold]\n"
            f"Liquidity levels : [bold]{len(snapshot.liquidity_levels)}[/bold]",
            title="[bold]Detection Summary[/bold]",
        )
    )


# ---------------------------------------------------------------------------
# smc backtest
# ---------------------------------------------------------------------------


@app.command(name="backtest")
def cmd_backtest(
    instrument: str = typer.Option(
        "XAUUSD",
        "--instrument",
        help="Instrument symbol.",
    ),
    train_months: int = typer.Option(
        12,
        "--train-months",
        help="Training window length in months.",
    ),
    test_months: int = typer.Option(
        3,
        "--test-months",
        help="Test (OOS) window length in months.",
    ),
    step_months: int = typer.Option(
        3,
        "--step-months",
        help="Window slide step in months.",
    ),
) -> None:
    """Run walk-forward out-of-sample backtest.

    Uses the SMC multi-timeframe strategy (D1+H4 bias, H1 zones, M15 entries)
    with the bar-by-bar backtest engine and pessimistic fill model.

    Requires data for D1, H4, H1, and M15 timeframes in the data lake.
    Run ``smc ingest`` first to populate the lake.
    """
    from pathlib import Path

    from smc.backtest.adapter import SMCStrategyAdapter
    from smc.backtest.engine import BarBacktestEngine
    from smc.backtest.fills import FillModel
    from smc.backtest.types import BacktestConfig
    from smc.backtest.walk_forward import aggregate_oos_results, walk_forward_oos
    from smc.config import SMCConfig
    from smc.data.lake import ForexDataLake
    from smc.smc_core.detector import SMCDetector
    from smc.strategy.aggregator import MultiTimeframeAggregator

    cfg = SMCConfig()

    console.print(
        Panel(
            f"Instrument    : [cyan]{instrument}[/cyan]\n"
            f"Train window  : [cyan]{train_months}[/cyan] months\n"
            f"Test window   : [cyan]{test_months}[/cyan] months\n"
            f"Step          : [cyan]{step_months}[/cyan] months\n"
            f"Data dir      : [dim]{cfg.data_dir}[/dim]",
            title="[bold]SMC Walk-Forward Backtest[/bold]",
        )
    )

    try:
        # Build the pipeline
        bt_config = BacktestConfig(instrument=instrument)
        fill_model = FillModel(
            bt_config.spread_points,
            bt_config.slippage_points,
            bt_config.commission_per_lot,
        )
        engine = BarBacktestEngine(bt_config, fill_model)
        detector = SMCDetector(swing_length=cfg.swing_length)
        aggregator = MultiTimeframeAggregator(detector=detector)
        lake = ForexDataLake(cfg.data_dir)

        strategy = SMCStrategyAdapter(aggregator, lake, instrument=instrument)

        # Run walk-forward OOS
        results = walk_forward_oos(
            engine,
            strategy,
            lake,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )

        if not results:
            console.print(
                "[yellow]No OOS windows could be constructed. "
                "Ensure sufficient data is ingested (need at least "
                f"{train_months + test_months} months).[/yellow]"
            )
            raise typer.Exit(code=1)

        summary = aggregate_oos_results(results)

        # --- Render results ---
        # Summary panel
        console.print(
            Panel(
                f"OOS Windows       : [bold]{summary.windows}[/bold]\n"
                f"Total OOS Trades  : [bold]{summary.total_oos_trades}[/bold]\n"
                f"Pooled Sharpe     : [bold]{summary.pooled_sharpe:.3f}[/bold]\n"
                f"Consistency Ratio : [bold]{summary.consistency_ratio:.1%}[/bold]",
                title="[bold green]Walk-Forward Summary[/bold green]",
            )
        )

        # Per-window table
        wf_table = Table(
            title="OOS Window Results",
            show_header=True,
            header_style="bold magenta",
        )
        wf_table.add_column("Window", style="cyan", no_wrap=True)
        wf_table.add_column("Period", style="dim")
        wf_table.add_column("Trades", justify="right")
        wf_table.add_column("Sharpe", justify="right")
        wf_table.add_column("Sortino", justify="right")
        wf_table.add_column("Max DD%", justify="right")
        wf_table.add_column("Win Rate", justify="right")
        wf_table.add_column("Profit Factor", justify="right")

        for idx, r in enumerate(results, 1):
            period = (
                f"{r.start_date.strftime('%Y-%m-%d')} to "
                f"{r.end_date.strftime('%Y-%m-%d')}"
            )
            wf_table.add_row(
                f"#{idx}",
                period,
                str(r.total_trades),
                f"{r.sharpe:.3f}",
                f"{r.sortino:.3f}",
                f"{r.max_drawdown_pct:.2%}",
                f"{r.win_rate:.1%}",
                f"{r.profit_factor:.2f}" if r.profit_factor < 1000 else "inf",
            )

        console.print(wf_table)

    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Backtest failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


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
# smc live
# ---------------------------------------------------------------------------


@app.command(name="live")
def cmd_live(
    mode: str = typer.Option(
        "demo",
        "--mode",
        help="Trading mode: demo or live.",
    ),
    instrument: str = typer.Option(
        "XAUUSD",
        "--instrument",
        help="Instrument to trade.",
    ),
) -> None:
    """Run the SMC strategy in live/demo mode.

    Polls MT5 every M15 bar close, runs the strategy pipeline,
    executes trades via the risk/execution layer, and monitors health.

    Use --mode=demo for paper trading (default), --mode=live for real orders.
    """
    import asyncio

    from smc.config import SMCConfig

    cfg = SMCConfig()

    # Override env based on mode argument
    if mode == "live" and not cfg.is_live():
        err_console.print(
            "[red]Mode 'live' requested but SMC_ENV is not 'live'. "
            "Set SMC_ENV=live to enable real trading.[/red]"
        )
        raise typer.Exit(code=1)

    if mode not in ("demo", "live"):
        err_console.print(f"[red]Invalid mode '{mode}'. Use 'demo' or 'live'.[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Mode       : [cyan]{mode}[/cyan]\n"
            f"Instrument : [cyan]{instrument}[/cyan]\n"
            f"MT5 Mock   : [cyan]{cfg.mt5_mock}[/cyan]\n"
            f"Env        : [cyan]{cfg.env}[/cyan]",
            title="[bold]SMC Live Trading[/bold]",
        )
    )

    try:
        from pathlib import Path

        from smc.data.adapters.mt5_mock import MT5MockAdapter, MockMT5Terminal, is_mock_mode
        from smc.monitor.live_loop import LiveLoop
        from smc.smc_core.detector import SMCDetector
        from smc.strategy.aggregator import MultiTimeframeAggregator

        # Build broker port
        if cfg.mt5_mock or is_mock_mode():
            console.print("[yellow]Running in mock mode (SimBrokerPort)[/yellow]")
            mock_terminal = MockMT5Terminal(fixtures_dir=cfg.data_dir)
            mock_terminal.initialize()

            class _MockBrokerPort:
                """Minimal broker port wrapping MockMT5Terminal."""

                def __init__(self, terminal: MockMT5Terminal) -> None:
                    self._terminal = terminal

                def get_account_info(self) -> dict:
                    return {"balance": 10_000.0, "equity": 10_000.0, "margin_level": None}

                def get_positions(self, symbol: str) -> list:
                    return [
                        {"ticket": p.ticket, "symbol": p.symbol}
                        for p in self._terminal.positions_get(symbol)
                    ]

                def get_current_price(self, symbol: str) -> float | None:
                    tick = self._terminal.symbol_info_tick(symbol)
                    return tick.last if tick else None

            broker = _MockBrokerPort(mock_terminal)
            data_fetcher = MT5MockAdapter(fixtures_dir=cfg.data_dir, instrument=instrument)
        else:
            # Real MT5 — import conditionally
            err_console.print(
                "[red]Real MT5 broker not yet wired. "
                "Set SMC_MT5_MOCK=1 for paper trading.[/red]"
            )
            raise typer.Exit(code=1)

        # Build strategy
        detector = SMCDetector(swing_length=cfg.swing_length)
        aggregator = MultiTimeframeAggregator(
            detector=detector,
            ai_regime_enabled=cfg.ai_regime_enabled,
            sl_fitness_enabled=cfg.sl_fitness_judge_enabled,
            sl_fitness_min_sl_atr_ratio=cfg.sl_fitness_min_sl_atr_ratio,
            sl_fitness_max_sl_atr_ratio=cfg.sl_fitness_max_sl_atr_ratio,
            sl_fitness_low_vol_percentile=cfg.sl_fitness_low_vol_percentile,
            sl_fitness_transition_conf_floor=cfg.sl_fitness_transition_conf_floor,
            sl_fitness_counter_trend_ai_conf=cfg.sl_fitness_counter_trend_ai_conf,
            synthetic_zones_enabled=cfg.synthetic_zones_enabled,
            synthetic_zones_min_historical=cfg.synthetic_zones_min_historical,
        )

        # Build live loop
        live_loop = LiveLoop(
            config=cfg,
            broker=broker,
            data_fetcher=data_fetcher,
            strategy=aggregator,
            instrument=instrument,
            journal_dir=Path("./logs/journal"),
        )

        console.print("[bold green]Starting live loop... Press Ctrl+C to stop.[/bold green]")
        asyncio.run(live_loop.run())

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested by user.[/yellow]")
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[red]Live loop failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc


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
