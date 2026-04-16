"""AI-SMC Live Dashboard — Rich terminal monitoring panel.

Usage on VPS:
    cd C:\AI-SMC
    .venv\Scripts\python.exe scripts/dashboard.py

Refreshes every 5 seconds. Shows account, price, signals, health, journal.
Press Ctrl+C to exit.
"""
import sys
import os
import time
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import MetaTrader5 as mt5
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich import box

from smc.data.schemas import Timeframe
from smc.strategy.regime import classify_regime
from smc.smc_core.detector import SMCDetector

console = Console()

JOURNAL_PATH = Path("data/journal/live_trades.jsonl")
LOG_PATH = Path("logs/live_stdout.log")
STATE_PATH = Path("data/live_state.json")
AI_ANALYSIS_PATH = Path("data/ai_analysis.json")


def _load_json(path: Path) -> dict | None:
    """Load a JSON file safely, return None on failure."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def make_action_banner() -> Panel:
    """Giant action banner showing current AI+SMC decision at top of screen."""
    now = datetime.now(timezone.utc)
    state = _load_json(STATE_PATH)

    if state is None:
        # No live state yet — show waiting banner
        grid = Table.grid(expand=True)
        grid.add_column(justify="center")
        grid.add_row(Text("=== WAITING FOR LIVE LOOP ===", style="bold yellow"))
        grid.add_row(Text("Run: python scripts/live_demo.py to start", style="dim"))
        grid.add_row(Text(now.strftime("%Y-%m-%d %H:%M:%S UTC"), style="dim"))
        return Panel(grid, style="bold yellow", box=box.DOUBLE)

    action_raw = state.get("action", "HOLD")
    action_word = action_raw.split()[0].upper()
    price = state.get("price", 0)
    reason = state.get("reason", "")
    ai_dir = state.get("ai_direction", "?").upper()
    ai_conf = state.get("ai_confidence", 0)
    setup = state.get("best_setup")

    # Color coding by action
    if action_word == "BUY":
        arrow = "▲▲▲"
        style = "bold green"
    elif action_word == "SELL":
        arrow = "▼▼▼"
        style = "bold red"
    else:
        arrow = "==="
        style = "bold yellow"

    grid = Table.grid(expand=True)
    grid.add_column(justify="center")

    # Line 1: giant action
    grid.add_row(Text(
        f"  {arrow}  {action_raw}  XAUUSD @ ${price:.2f}  {arrow}  ",
        style=style,
    ))

    # Line 2: AI + SMC summary
    setup_trigger = setup.get("trigger", "?") if setup else "no setup"
    setup_dir = setup.get("direction", "?").upper() if setup else "-"
    grid.add_row(Text(
        f"AI: {ai_dir} ({ai_conf:.0%}) + SMC: {setup_dir} {setup_trigger}",
        style="white",
    ))

    # Line 3: SL / TP / Confluence (or reason when HOLD)
    if setup:
        sl = setup.get("sl", 0)
        tp = setup.get("tp1", 0)
        conf = setup.get("confluence", 0)
        grid.add_row(Text(
            f"SL: ${sl:.2f} | TP: ${tp:.2f} | Confluence: {conf:.2f}",
            style="dim white",
        ))
    else:
        grid.add_row(Text(reason, style="dim white"))

    # Line 4: timestamp + cycle
    cycle = state.get("cycle", "?")
    ts = state.get("timestamp", "")[:19]
    grid.add_row(Text(f"Cycle #{cycle} | {ts} UTC", style="dim"))

    return Panel(grid, style=style, box=box.DOUBLE, title="[bold white]AI-SMC ACTION[/bold white]")


def make_account_panel() -> Panel:
    info = mt5.account_info()
    if not info:
        return Panel("[red]MT5 Disconnected[/red]", title="Account", border_style="red")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", width=16)
    table.add_column("Value", style="bold")

    balance_style = "green" if info.profit >= 0 else "red"

    table.add_row("Account", f"[cyan]{info.login}[/cyan]")
    table.add_row("Server", f"{info.server}")
    table.add_row("Balance", f"[bold]{info.currency} {info.balance:,.2f}[/bold]")
    table.add_row("Equity", f"[{balance_style}]{info.currency} {info.equity:,.2f}[/{balance_style}]")
    table.add_row("P&L", f"[{balance_style}]{info.currency} {info.profit:+,.2f}[/{balance_style}]")
    table.add_row("Margin Used", f"{info.currency} {info.margin:,.2f}")
    table.add_row("Free Margin", f"{info.currency} {info.margin_free:,.2f}")
    margin_pct = info.margin_level if info.margin_level else 0
    margin_style = "green" if margin_pct > 200 else ("yellow" if margin_pct > 100 else "red")
    table.add_row("Margin Level", f"[{margin_style}]{margin_pct:.1f}%[/{margin_style}]")
    table.add_row("Leverage", f"1:{info.leverage}")

    return Panel(table, title="[bold white]Account[/bold white]", border_style="cyan", box=box.ROUNDED)


def make_price_panel() -> Panel:
    tick = mt5.symbol_info_tick("XAUUSD")
    sym = mt5.symbol_info("XAUUSD")
    if not tick or not sym:
        return Panel("[red]No Price Data[/red]", title="XAUUSD", border_style="red")

    spread = tick.ask - tick.bid
    bar_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    age = (datetime.now(timezone.utc) - bar_time).total_seconds()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", width=12)
    table.add_column("Value", style="bold")

    table.add_row("Bid", f"[bold yellow]{tick.bid:.2f}[/bold yellow]")
    table.add_row("Ask", f"[bold yellow]{tick.ask:.2f}[/bold yellow]")
    table.add_row("Spread", f"{spread:.2f} ({spread * 100:.0f} pts)")
    table.add_row("Last Update", f"{bar_time.strftime('%H:%M:%S')} ({age:.0f}s ago)")
    table.add_row("Day High", f"{sym.session_price_limit_max:.2f}" if sym.session_price_limit_max > 0 else "N/A")
    table.add_row("Day Low", f"{sym.session_price_limit_min:.2f}" if sym.session_price_limit_min > 0 else "N/A")

    return Panel(table, title="[bold white]XAUUSD Price[/bold white]", border_style="yellow", box=box.ROUNDED)


def make_positions_panel() -> Panel:
    positions = mt5.positions_get(symbol="XAUUSD")

    if not positions or len(positions) == 0:
        return Panel(
            Align.center(Text("No open positions", style="dim")),
            title="[bold white]Open Positions[/bold white]",
            border_style="green",
            box=box.ROUNDED,
        )

    table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
    table.add_column("Ticket", style="cyan", width=10)
    table.add_column("Type", width=6)
    table.add_column("Lots", width=6)
    table.add_column("Open Price", width=10)
    table.add_column("Current", width=10)
    table.add_column("SL", width=10)
    table.add_column("TP", width=10)
    table.add_column("P&L", width=10)

    for pos in positions:
        pnl_style = "green" if pos.profit >= 0 else "red"
        dir_text = "[green]BUY[/green]" if pos.type == 0 else "[red]SELL[/red]"
        table.add_row(
            str(pos.ticket),
            dir_text,
            f"{pos.volume:.2f}",
            f"{pos.price_open:.2f}",
            f"{pos.price_current:.2f}",
            f"{pos.sl:.2f}" if pos.sl > 0 else "-",
            f"{pos.tp:.2f}" if pos.tp > 0 else "-",
            f"[{pnl_style}]{pos.profit:+.2f}[/{pnl_style}]",
        )

    return Panel(table, title=f"[bold white]Open Positions ({len(positions)})[/bold white]", border_style="green", box=box.ROUNDED)


def make_regime_panel() -> Panel:
    """Show current regime classification from latest D1 data."""
    try:
        import polars as pl
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_D1, 0, 30)
        if rates is None or len(rates) == 0:
            return Panel("[dim]No D1 data[/dim]", title="Regime", border_style="magenta")

        df = pl.DataFrame({
            "ts": [datetime.fromtimestamp(r[0], tz=timezone.utc) for r in rates],
            "open": [float(r[1]) for r in rates],
            "high": [float(r[2]) for r in rates],
            "low": [float(r[3]) for r in rates],
            "close": [float(r[4]) for r in rates],
        }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))

        regime = classify_regime(df)

        regime_colors = {
            "trending": "green",
            "transitional": "yellow",
            "ranging": "red",
        }
        color = regime_colors.get(regime, "white")

        # Compute simple SMA50 direction
        closes = [float(r[4]) for r in rates]
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            direction = "UP" if closes[-1] > sma20 else "DOWN"
            dir_color = "green" if direction == "UP" else "red"
        else:
            direction = "N/A"
            dir_color = "dim"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim", width=14)
        table.add_column("Value", style="bold")

        table.add_row("Regime", f"[bold {color}]{regime.upper()}[/bold {color}]")
        table.add_row("SMA20 Trend", f"[{dir_color}]{direction}[/{dir_color}]")
        table.add_row("D1 Close", f"${closes[-1]:.2f}")
        table.add_row("D1 Bars", f"{len(rates)}")

        return Panel(table, title="[bold white]Market Regime[/bold white]", border_style="magenta", box=box.ROUNDED)
    except Exception as e:
        return Panel(f"[red]{e}[/red]", title="Regime", border_style="red")


def make_health_panel() -> Panel:
    checks = []

    # MT5 connection
    info = mt5.account_info()
    checks.append(("MT5 Connection", bool(info), "Connected" if info else "DISCONNECTED"))

    # Data freshness
    tick = mt5.symbol_info_tick("XAUUSD")
    if tick:
        age = (datetime.now(timezone.utc) - datetime.fromtimestamp(tick.time, tz=timezone.utc)).total_seconds()
        fresh = age < 300
        checks.append(("Data Fresh", fresh, f"{age:.0f}s ago"))
    else:
        checks.append(("Data Fresh", False, "No tick"))

    # Margin
    if info:
        margin_ok = info.margin_level > 200 or info.margin_level == 0
        checks.append(("Margin Level", margin_ok, f"{info.margin_level:.0f}%"))

    # Live loop — check live_state.json freshness (<900s = running)
    state_exists = STATE_PATH.exists()
    if state_exists:
        state_age = time.time() - STATE_PATH.stat().st_mtime
        alive = state_age < 900
        checks.append(("Live Loop", alive, f"state {state_age:.0f}s ago"))
    else:
        # Fallback to log file
        log_exists = LOG_PATH.exists()
        if log_exists:
            log_age = time.time() - LOG_PATH.stat().st_mtime
            alive = log_age < 1800
            checks.append(("Live Loop", alive, f"log {log_age:.0f}s ago"))
        else:
            checks.append(("Live Loop", False, "Not running"))

    # AI Analysis freshness
    ai_exists = AI_ANALYSIS_PATH.exists()
    if ai_exists:
        ai_age = time.time() - AI_ANALYSIS_PATH.stat().st_mtime
        ai_fresh = ai_age < 14400  # 4 hours
        checks.append(("AI Analysis", ai_fresh, f"{ai_age / 60:.0f}min ago"))
    else:
        checks.append(("AI Analysis", False, "No analysis"))

    # Journal
    journal_exists = JOURNAL_PATH.exists()
    if journal_exists:
        with open(JOURNAL_PATH) as f:
            lines = f.readlines()
        checks.append(("Journal", True, f"{len(lines)} entries"))
    else:
        checks.append(("Journal", True, "0 entries (waiting)"))

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Status", width=3)
    table.add_column("Check", width=16)
    table.add_column("Detail", style="dim")

    for name, ok, detail in checks:
        icon = "[green]OK[/green]" if ok else "[red]!![/red]"
        table.add_row(icon, name, detail)

    all_ok = all(ok for _, ok, _ in checks)
    border = "green" if all_ok else "red"
    title_extra = " [green]ALL OK[/green]" if all_ok else " [red]ISSUES[/red]"

    return Panel(table, title=f"[bold white]Health{title_extra}[/bold white]", border_style=border, box=box.ROUNDED)


def make_journal_panel() -> Panel:
    """Trade journal showing AI's role in each signal — AGREED, BLOCKED, or CONFLICT."""
    if not JOURNAL_PATH.exists():
        return Panel(
            Align.center(Text("No trades yet — waiting for signals", style="dim italic")),
            title="[bold white]Trade History + AI Role[/bold white]",
            border_style="blue",
            box=box.ROUNDED,
        )

    with open(JOURNAL_PATH) as f:
        lines = f.readlines()

    if not lines:
        return Panel(
            Align.center(Text("Journal empty", style="dim")),
            title="[bold white]Trade History + AI Role[/bold white]",
            border_style="blue",
            box=box.ROUNDED,
        )

    table = Table(box=box.SIMPLE, show_lines=False)
    table.add_column("Time", style="dim", width=9)
    table.add_column("Action", width=10)
    table.add_column("Dir", width=6)
    table.add_column("Entry", width=10)
    table.add_column("SL", width=10)
    table.add_column("TP", width=10)
    table.add_column("Trigger", width=14)
    table.add_column("AI Role", width=12)
    table.add_column("Conf", width=6)

    for line in lines[-10:]:
        try:
            d = json.loads(line.strip())
            dir_style = "green" if d.get("direction") == "long" else "red"

            # Determine AI's role in this signal
            ai_dir = d.get("ai_direction", "?")
            smc_dir = d.get("direction", "?")
            action = d.get("action", "?")

            if ai_dir == "bullish" and smc_dir == "long":
                ai_role = "[bold green]AGREED[/bold green]"
            elif ai_dir == "bearish" and smc_dir == "short":
                ai_role = "[bold green]AGREED[/bold green]"
            elif "HOLD" in action.upper():
                ai_role = "[bold red]BLOCKED[/bold red]"
            elif ai_dir == "neutral":
                ai_role = "[yellow]NEUTRAL[/yellow]"
            else:
                ai_role = "[red]CONFLICT[/red]"

            time_str = d.get("time", "")
            if "T" in time_str:
                time_str = time_str.split("T")[1][:8]
            else:
                time_str = time_str[:8]

            table.add_row(
                time_str,
                f"[{dir_style}]{action[:9]}[/{dir_style}]",
                f"[{dir_style}]{smc_dir[:5].upper()}[/{dir_style}]",
                f"${d.get('entry', d.get('price', 0)):.2f}",
                f"${d.get('sl', 0):.2f}",
                f"${d.get('tp1', 0):.2f}",
                d.get("trigger", "?")[:14],
                ai_role,
                f"{d.get('confluence', 0):.2f}",
            )
        except (json.JSONDecodeError, KeyError):
            continue

    total = len(lines)
    return Panel(
        table,
        title=f"[bold white]Trade History + AI Role ({total} total)[/bold white]",
        border_style="blue",
        box=box.ROUNDED,
    )


def make_ai_panel() -> Panel:
    """Show AI market analysis with full reasoning — the WHY behind every decision."""
    a = _load_json(AI_ANALYSIS_PATH)
    if a is None:
        return Panel(
            Align.center(Text("Run: smc-analyze to generate AI analysis", style="dim italic")),
            title="[bold white]AI Analysis — WHY[/bold white]",
            border_style="bright_magenta",
            box=box.ROUNDED,
        )

    dir_colors = {"bullish": "bold green", "bearish": "bold red", "neutral": "bold yellow"}

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="dim", width=16)
    table.add_column("Value")

    # Direction + confidence
    d = a.get("direction", "unknown")
    c = a.get("confidence", 0)
    dir_style = dir_colors.get(d, "white")
    table.add_row("Direction", f"[{dir_style}]{d.upper()}[/{dir_style}]  ({c:.0%})")

    # AI debate override
    if a.get("ai_direction"):
        ai_d = a["ai_direction"]
        ai_style = dir_colors.get(ai_d, "white")
        table.add_row("AI Debate", f"[{ai_style}]{ai_d.upper()}[/{ai_style}]  (via {a.get('ai_source', '?')})")

    # --- KEY FACTORS as bullet list (the WHY) ---
    factors = a.get("key_factors", [])
    ai_drivers = a.get("ai_key_drivers", [])
    all_reasons = factors + ai_drivers
    if all_reasons:
        table.add_row("", "")
        table.add_row("[bold]WHY[/bold]", "")
        for reason in all_reasons[:6]:
            table.add_row("", f"[cyan]>[/cyan] {reason}")
    else:
        # Fall back to reasoning string
        reasoning = a.get("reasoning", a.get("ai_reasoning", ""))
        if reasoning:
            table.add_row("", "")
            table.add_row("[bold]WHY[/bold]", "")
            # Split long reasoning into readable lines
            for chunk in _wrap_text(reasoning, 50):
                table.add_row("", f"[dim]{chunk}[/dim]")

    # Technical summary row
    table.add_row("", "")
    tech_parts = []
    if a.get("sma20") and a.get("sma50"):
        cross = a.get("sma20_vs_sma50", "?")
        cross_label = "GOLDEN" if cross == "golden_cross" else "DEATH"
        tech_parts.append(f"SMA20 ${a['sma20']:.0f} / SMA50 ${a['sma50']:.0f} ({cross_label})")
    if a.get("change_5d_pct") is not None:
        chg = a["change_5d_pct"]
        chg_style = "green" if chg > 0 else "red"
        tech_parts.append(f"5D [{chg_style}]{chg:+.2f}%[/{chg_style}]")
    if a.get("h4_trend"):
        h4_style = "green" if a["h4_trend"] == "up" else "red"
        tech_parts.append(f"H4 [{h4_style}]{a['h4_trend'].upper()}[/{h4_style}]")
    if a.get("atr_pct"):
        vol = a.get("volatility", "?")
        vol_style = "red" if vol == "high" else ("green" if vol == "low" else "yellow")
        tech_parts.append(f"ATR {a['atr_pct']:.2f}% [{vol_style}]{vol.upper()}[/{vol_style}]")

    if tech_parts:
        table.add_row("Technicals", " | ".join(tech_parts[:2]))
        if len(tech_parts) > 2:
            table.add_row("", " | ".join(tech_parts[2:]))

    # Support / Resistance
    if a.get("resistance_20d"):
        table.add_row("S/R (20D)", f"${a.get('support_20d', 0):.2f} — ${a['resistance_20d']:.2f}")

    # Freshness
    assessed = a.get("assessed_at", "")
    if assessed:
        table.add_row("", "")
        table.add_row("Updated", f"[dim]{assessed[:19]}[/dim]")

    source = a.get("source", "?")
    return Panel(
        table,
        title=f"[bold white]AI Analysis — WHY[/bold white] [dim]({source})[/dim]",
        border_style="bright_magenta",
        box=box.ROUNDED,
    )


def _wrap_text(text: str, width: int) -> list[str]:
    """Split text into lines of approximately `width` characters."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


def make_log_panel() -> Panel:
    if not LOG_PATH.exists():
        return Panel("[dim]No log file[/dim]", title="Live Loop Log", border_style="dim")

    with open(LOG_PATH) as f:
        lines = f.readlines()

    recent = [l.strip() for l in lines[-8:] if l.strip()]
    text = "\n".join(recent) if recent else "[dim]Empty[/dim]"
    return Panel(text, title=f"[bold white]Live Loop Log (last 8 lines)[/bold white]", border_style="dim", box=box.ROUNDED)


def make_layout() -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="action_banner", size=6),
        Layout(name="upper", size=12),
        Layout(name="ai_row", size=18),
        Layout(name="middle", size=10),
        Layout(name="lower"),
    )

    layout["upper"].split_row(
        Layout(name="account", ratio=1),
        Layout(name="price", ratio=1),
        Layout(name="regime", ratio=1),
    )

    layout["ai_row"].split_row(
        Layout(name="ai_analysis", ratio=3),
        Layout(name="health", ratio=2),
    )

    layout["middle"].split_row(
        Layout(name="positions"),
    )

    layout["lower"].split_row(
        Layout(name="journal", ratio=3),
        Layout(name="log", ratio=2),
    )

    return layout


def render(layout: Layout) -> Layout:
    layout["action_banner"].update(make_action_banner())
    layout["account"].update(make_account_panel())
    layout["price"].update(make_price_panel())
    layout["regime"].update(make_regime_panel())
    layout["ai_analysis"].update(make_ai_panel())
    layout["health"].update(make_health_panel())
    layout["positions"].update(make_positions_panel())
    layout["journal"].update(make_journal_panel())
    layout["log"].update(make_log_panel())
    return layout


def main():
    if not mt5.initialize():
        console.print("[red bold]MT5 initialization failed![/red bold]")
        sys.exit(1)

    console.print("[bold cyan]AI-SMC Dashboard starting...[/bold cyan]")
    console.print("Press [bold]Ctrl+C[/bold] to exit.\n")

    layout = make_layout()

    try:
        with Live(render(layout), console=console, refresh_per_second=0.2, screen=True):
            while True:
                render(layout)
                time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        mt5.shutdown()
        console.print("\n[dim]Dashboard closed.[/dim]")


if __name__ == "__main__":
    main()
