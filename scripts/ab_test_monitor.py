"""AB-test monitor for HedgeRock_v2_patched vs HedgeRock_Lite.

Pulls the two CSV files produced by ``mql5/ab_dump_deals.mq5`` from the
VPS, parses them into in-memory deal lists, computes side-by-side
metrics (win rate, profit factor, net PnL, max drawdown), and prints a
plain-text comparison table.

The remote dump script is expected to live in MQL5\\Scripts on the VPS;
the operator must run it manually in MT5 (or attach it to a chart as a
periodic EA) to refresh the CSVs before each monitor invocation.  The
monitor itself never *mutates* remote state — it only ``scp`` pulls the
two CSV files into ``data/ab/``.

## Auth

Same as ``scripts/deploy_hedgerock_lite.py``: ``MT5_SSH_PASSWORD`` env
var, ``sshpass -e`` so the password never appears in argv.

## Exit codes

    0 — metrics computed and printed
    2 — credentials missing / SSH unreachable
    3 — pull failed
    4 — parse failed (no rows or malformed CSVs)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "43.163.107.158"
DEFAULT_USER = "Administrator"
DEFAULT_TERMINAL_ID = "7643C0B96C7AD5841307C9E1EB0B9252"
DEFAULT_MAGIC_A = 20222222   # HedgeRock_v2_patched
DEFAULT_MAGIC_B = 30333333   # HedgeRock_Lite


# ---------------------------------------------------------------------------
# Pure data + metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Deal:
    """One row from ``ab_deals_<magic>.csv``.

    Only the columns we use for metrics are kept as typed fields; the
    rest stay as strings if/when we need them.
    """

    ticket: int
    time_utc: str
    entry: int           # 0 = IN, 1 = OUT, 2 = INOUT, 3 = OUT_BY
    volume: float
    price: float
    profit: float
    swap: float
    commission: float
    position_id: int
    magic: int


@dataclass(frozen=True)
class Metrics:
    magic: int
    n_deals: int
    n_closes: int        # only DEAL_ENTRY_OUT counts toward win/loss stats
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float  # gross_profit / |gross_loss|; inf if no losses
    net_pnl: float
    max_drawdown: float   # peak-to-trough on the closed-deal equity curve

    def to_row(self) -> tuple[str, ...]:
        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        return (
            str(self.magic),
            str(self.n_deals),
            str(self.n_closes),
            f"{self.win_rate:.1%}",
            pf,
            f"{self.net_pnl:.2f}",
            f"{self.max_drawdown:.2f}",
        )


def parse_csv(path: Path) -> list[Deal]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    out: list[Deal] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out.append(
                    Deal(
                        ticket=int(row["ticket"]),
                        time_utc=row["time_utc"],
                        entry=int(row["entry"]),
                        volume=float(row["volume"]),
                        price=float(row["price"]),
                        profit=float(row["profit"]),
                        swap=float(row["swap"]),
                        commission=float(row["commission"]),
                        position_id=int(row["position_id"]),
                        magic=int(row["magic"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


def _max_drawdown(closed_pnls: list[float]) -> float:
    """Peak-to-trough drawdown on the cumulative closed-deal curve."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in closed_pnls:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_metrics(deals: list[Deal], magic: int) -> Metrics:
    closes = [d for d in deals if d.entry in (1, 2, 3)]  # OUT / INOUT / OUT_BY
    realized = [d.profit + d.swap + d.commission for d in closes]
    wins = sum(1 for p in realized if p > 0)
    losses = sum(1 for p in realized if p < 0)
    gross_profit = sum(p for p in realized if p > 0)
    gross_loss = sum(p for p in realized if p < 0)
    n_closed_with_pnl = wins + losses  # zeros excluded from win-rate denom
    return Metrics(
        magic=magic,
        n_deals=len(deals),
        n_closes=len(closes),
        wins=wins,
        losses=losses,
        win_rate=(wins / n_closed_with_pnl) if n_closed_with_pnl else 0.0,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=(
            float("inf") if gross_loss == 0 and gross_profit > 0
            else (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0
        ),
        net_pnl=sum(realized),
        max_drawdown=_max_drawdown(realized),
    )


def format_comparison(m_a: Metrics, m_b: Metrics, *, label_a: str, label_b: str) -> str:
    cols = ["magic", "deals", "closes", "win-rate", "PF", "net PnL", "max DD"]
    row_a = (label_a,) + m_a.to_row()[1:]
    row_b = (label_b,) + m_b.to_row()[1:]
    widths = [
        max(len(cols[i]), len(row_a[i]), len(row_b[i])) for i in range(len(cols))
    ]
    sep = " | "

    def fmt(row: tuple[str, ...]) -> str:
        return sep.join(c.ljust(widths[i]) for i, c in enumerate(row))

    header = fmt(tuple(cols))
    bar = "-+-".join("-" * w for w in widths)
    return "\n".join([header, bar, fmt(row_a), fmt(row_b)])


# ---------------------------------------------------------------------------
# SSH / SCP pull
# ---------------------------------------------------------------------------


def _password() -> str:
    pw = os.environ.get("MT5_SSH_PASSWORD")
    if not pw:
        raise RuntimeError("MT5_SSH_PASSWORD env var is not set")
    return pw


def _scp_pull(
    *,
    host: str,
    user: str,
    remote_path: str,
    local_path: Path,
    timeout: int = 60,
) -> tuple[int, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout}",
        "-q",
        f"{user}@{host}:{remote_path}",
        str(local_path),
    ]
    argv = ["sshpass", "-e", *base]
    env = os.environ.copy()
    env["SSHPASS"] = _password()
    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        return 127, f"scp/sshpass missing: {exc!r}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    return out.returncode, out.stderr


def remote_csv_path(terminal_id: str, magic: int) -> str:
    return (
        r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\\"
        + terminal_id
        + r"\MQL5\Files\ab_deals_"
        + str(magic)
        + ".csv"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--terminal-id", default=DEFAULT_TERMINAL_ID)
    p.add_argument("--magic-a", type=int, default=DEFAULT_MAGIC_A)
    p.add_argument("--magic-b", type=int, default=DEFAULT_MAGIC_B)
    p.add_argument("--label-a", default="HedgeRock_v2")
    p.add_argument("--label-b", default="HedgeRock_Lite")
    p.add_argument(
        "--local-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "ab"),
        help="local directory to write the pulled CSVs into",
    )
    p.add_argument(
        "--from-local",
        action="store_true",
        help="skip SSH pull; just parse whatever CSVs are already in --local-dir",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_a = local_dir / f"ab_deals_{args.magic_a}.csv"
    local_b = local_dir / f"ab_deals_{args.magic_b}.csv"

    if not args.from_local:
        try:
            _password()
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 2
        for magic, local_path in ((args.magic_a, local_a), (args.magic_b, local_b)):
            remote = remote_csv_path(args.terminal_id, magic)
            rc, stderr = _scp_pull(
                host=args.host,
                user=args.user,
                remote_path=remote,
                local_path=local_path,
            )
            if rc != 0:
                print(
                    f"[error] pull magic={magic} rc={rc}: {stderr}"
                    "\n  hint: run mql5/ab_dump_deals.mq5 in MT5 first.",
                    file=sys.stderr,
                )
                return 3

    deals_a = parse_csv(local_a)
    deals_b = parse_csv(local_b)
    if not deals_a and not deals_b:
        print(
            "[error] both CSVs empty or missing — has the dump script ever run?",
            file=sys.stderr,
        )
        return 4

    m_a = compute_metrics(deals_a, args.magic_a)
    m_b = compute_metrics(deals_b, args.magic_b)
    print(format_comparison(m_a, m_b, label_a=args.label_a, label_b=args.label_b))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
