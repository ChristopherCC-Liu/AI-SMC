"""Deploy HedgeRock_Lite EA + ab_dump_deals script to the live MT5 VPS.

The script is *push-only for these two files* and refuses to run any
mutating remote command outside `mkdir` / `metaeditor64.exe /compile`.

## Files pushed

    mql5/HedgeRock_Lite.mq5  →  <terminal>\\MQL5\\Experts\\HedgeRock_Lite.mq5
    mql5/ab_dump_deals.mq5   →  <terminal>\\MQL5\\Scripts\\ab_dump_deals.mq5

## Auth

The remote SSH password is read from ``MT5_SSH_PASSWORD``.  The script
never writes it to disk and never echoes it back.  ``sshpass -e`` is
used so the secret never appears on the command line.

## Exit codes

    0 — upload + compile both succeeded
    2 — credentials missing / SSH unreachable
    3 — upload failed
    4 — compile failed (file is on the VPS but no .ex5 was produced)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "43.163.107.158"
DEFAULT_USER = "Administrator"
DEFAULT_TERMINAL_ID = "7643C0B96C7AD5841307C9E1EB0B9252"
DEFAULT_METAEDITOR = r"C:\Program Files\TMGM MT5 Terminal\metaeditor64.exe"
DEFAULT_INCLUDE = r"C:\Program Files\TMGM MT5 Terminal\MQL5\Include"


@dataclass(frozen=True)
class DeployTarget:
    host: str
    user: str
    terminal_id: str
    metaeditor_path: str
    include_path: str

    @property
    def mql5_root(self) -> str:
        return (
            r"C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\\"
            + self.terminal_id
            + r"\MQL5"
        )

    @property
    def experts_dir(self) -> str:
        return self.mql5_root + r"\Experts"

    @property
    def scripts_dir(self) -> str:
        return self.mql5_root + r"\Scripts"


# ---------------------------------------------------------------------------
# SSH / SCP helpers (mirrors scripts/hedgerock_mt5_tick_pull.py conventions)
# ---------------------------------------------------------------------------


def _password() -> str:
    pw = os.environ.get("MT5_SSH_PASSWORD")
    if not pw:
        raise RuntimeError(
            "MT5_SSH_PASSWORD env var is not set. Export it before running, "
            "e.g. `export MT5_SSH_PASSWORD='...'`"
        )
    return pw


def _ssh(
    target: DeployTarget,
    command: str,
    *,
    timeout: int = 60,
) -> tuple[int, str, str]:
    """Execute a single remote command via ssh + sshpass -e."""
    base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "BatchMode=no",
        f"{target.user}@{target.host}",
        command,
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
        return 127, "", f"ssh/sshpass missing: {exc!r}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    return out.returncode, out.stdout, out.stderr


def _scp_push(
    target: DeployTarget,
    local_path: Path,
    remote_path: str,
    *,
    timeout: int = 60,
) -> tuple[int, str]:
    if not local_path.is_file():
        return 2, f"local file not found: {local_path}"
    base = [
        "scp",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={timeout}",
        "-q",
        str(local_path),
        f"{target.user}@{target.host}:{remote_path}",
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


# ---------------------------------------------------------------------------
# Deploy steps
# ---------------------------------------------------------------------------


def ensure_remote_dirs(target: DeployTarget) -> None:
    for d in (target.experts_dir, target.scripts_dir):
        cmd = f'powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path {shlex.quote(d)} | Out-Null"'
        rc, _stdout, stderr = _ssh(target, cmd)
        if rc != 0:
            raise RuntimeError(f"mkdir {d} failed rc={rc}: {stderr}")


def upload(target: DeployTarget, local_dir: Path) -> None:
    pairs = [
        (local_dir / "mql5" / "HedgeRock_Lite.mq5", target.experts_dir),
        (local_dir / "mql5" / "ab_dump_deals.mq5", target.scripts_dir),
    ]
    for local, remote_dir in pairs:
        remote_path = remote_dir + "\\" + local.name
        # scp accepts forward-slashes on the Windows side via OpenSSH
        remote_for_scp = remote_path.replace("\\", "/")
        rc, stderr = _scp_push(target, local, remote_for_scp)
        if rc != 0:
            raise RuntimeError(f"upload {local.name} failed rc={rc}: {stderr}")
        print(f"  [OK] {local.name} → {remote_path}")


def compile_remote(target: DeployTarget, remote_mq5_path: str) -> tuple[int, str]:
    """Invoke metaeditor64.exe /compile and return (rc, log_excerpt)."""
    log_path = remote_mq5_path + ".log"
    cmd = (
        "powershell -NoProfile -Command "
        + shlex.quote(
            "& "
            + f'"{target.metaeditor_path}"'
            + f' /compile:"{remote_mq5_path}"'
            + f' /include:"{target.include_path}"'
            + f' /log:"{log_path}"'
            + "; Get-Content -Tail 40 -ErrorAction SilentlyContinue "
            + f'"{log_path}"'
        )
    )
    rc, stdout, stderr = _ssh(target, cmd, timeout=180)
    log = stdout or stderr
    return rc, log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--terminal-id", default=DEFAULT_TERMINAL_ID)
    p.add_argument("--metaeditor", default=DEFAULT_METAEDITOR)
    p.add_argument("--include", default=DEFAULT_INCLUDE)
    p.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="local AI-SMC repo root (defaults to parent of this script)",
    )
    p.add_argument(
        "--skip-compile",
        action="store_true",
        help="upload only, leave compile to the operator's scheduled task",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would happen, do not contact the VPS",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = DeployTarget(
        host=args.host,
        user=args.user,
        terminal_id=args.terminal_id,
        metaeditor_path=args.metaeditor,
        include_path=args.include,
    )
    repo_root = Path(args.repo_root).expanduser().resolve()

    if args.dry_run:
        print("[dry-run] would push:")
        print(f"  {repo_root / 'mql5' / 'HedgeRock_Lite.mq5'} → {target.experts_dir}")
        print(f"  {repo_root / 'mql5' / 'ab_dump_deals.mq5'}  → {target.scripts_dir}")
        if not args.skip_compile:
            print("[dry-run] would compile via", target.metaeditor_path)
        return 0

    try:
        _password()
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    try:
        ensure_remote_dirs(target)
        print("[1/3] remote dirs ready")
        upload(target, repo_root)
        print("[2/3] files uploaded")
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 3

    if args.skip_compile:
        print("[3/3] compile skipped (per --skip-compile)")
        print("Next: attach HedgeRock_Lite to a 2nd XAUUSD M15 chart in MT5.")
        return 0

    ea_remote = target.experts_dir + r"\HedgeRock_Lite.mq5"
    rc, log = compile_remote(target, ea_remote)
    print("[3/3] compile log tail:")
    print(log.strip() or "(no log)")
    if rc != 0:
        print(f"[error] compile rc={rc}", file=sys.stderr)
        return 4

    print("Done. Next steps:")
    print("  1. In MT5, refresh Navigator → Expert Advisors → HedgeRock_Lite")
    print("  2. Attach to a second XAUUSD M15 chart (Allow live trading)")
    print("  3. Watch the Experts tab — first 'opened BUY/SELL' log line confirms it")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
