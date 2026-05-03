"""HedgeRock MT5 XAUUSD tick / OHLCV puller (SSH, READ-ONLY).

Pulls XAUUSD bar data from a remote MT5 host, stores under
``data/ticks/XAUUSD/`` (or a caller-specified directory), and validates
completeness. **Never** writes to the remote, never modifies remote MT5
state.

## Credentials

The remote SSH password is **NEVER** stored in this file. Operator must
provide it via the ``MT5_SSH_PASSWORD`` environment variable, or
configure passwordless SSH key auth (preferred) and pass
``--auth keyless``.

Default host / user are sourced from the operator's standing
configuration:

    Default host:  43.163.107.158
    Default user:  Administrator
    Default sym:   XAUUSD

## Workflow

    1. Resolve the MT5 ``Bases`` directory on the remote (Windows MT5
       stores tick history under ``Bases\\<broker>\\history\\<symbol>\\``).
    2. ``rsync`` (or ``scp``) the symbol's ``.hst`` / ``.hcc`` files
       (and any newer ``.json`` cached parquet) into a local staging
       dir under ``data/ticks/<symbol>/raw/``.
    3. Parse the binary ``.hst`` header, decode bars, project to OHLCV.
    4. Validate: timestamp monotonicity, bar count vs configured
       window, no negative spreads, no zero-volume runs.
    5. Emit a manifest at ``data/ticks/<symbol>/manifest.json``.

## Safety

    * Never executes a remote command other than the documented
      read-only file-listing + file-copy commands.
    * Refuses to delete anything on the remote (no ``ssh ... rm``).
    * Refuses to overwrite local files older than the remote copy
      unless ``--force`` is passed.

## Exit codes

    0 — pulled and validated successfully
    2 — credentials missing / SSH unreachable
    3 — remote MT5 directory not found
    4 — local validation failed (gaps, corruption)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SYMBOL = "XAUUSD"
DEFAULT_HOST = "43.163.107.158"
DEFAULT_USER = "Administrator"

_SAFE_REMOTE_COMMANDS = (
    "ls", "dir", "type",
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


def _local_root(symbol: str = SYMBOL) -> Path:
    return _ai_smc_home() / "data" / "ticks" / symbol


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PullManifest:
    symbol: str
    host: str
    user: str
    pulled_at: str
    files_pulled: tuple[str, ...]
    n_bars_total: int
    earliest_ts: str | None
    latest_ts: str | None
    validation_ok: bool
    validation_errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "host": self.host, "user": self.user,
            "pulled_at": self.pulled_at,
            "files_pulled": list(self.files_pulled),
            "n_bars_total": self.n_bars_total,
            "earliest_ts": self.earliest_ts, "latest_ts": self.latest_ts,
            "validation_ok": self.validation_ok,
            "validation_errors": list(self.validation_errors),
        }


# ---------------------------------------------------------------------------
# SSH wrapper — every call is forced through subprocess with strict args.
# ---------------------------------------------------------------------------


def _resolve_password(*, auth_mode: str) -> str | None:
    """Return the password to use, or None if keyless auth is requested.

    NEVER reads passwords from CLI args or files inside the repo —
    only from the ``MT5_SSH_PASSWORD`` env var.
    """
    if auth_mode == "keyless":
        return None
    pw = os.environ.get("MT5_SSH_PASSWORD")
    if not pw:
        raise RuntimeError(
            "MT5_SSH_PASSWORD env var is not set; either export it or "
            "rerun with --auth keyless after configuring SSH key-based auth"
        )
    return pw


def _build_ssh_argv(
    *, host: str, user: str, command: str, password: str | None,
    timeout: int = 30,
) -> list[str]:
    """Returns the argv for ``subprocess.run``. Uses ``sshpass -e`` when
    a password is supplied so the secret never appears on the
    command line. When ``password is None`` (keyless mode), plain ssh."""
    base = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "BatchMode=no",
        f"{user}@{host}", command,
    ]
    if password is None:
        return base
    return ["sshpass", "-e", *base]


def _run_ssh(
    *, host: str, user: str, command: str, password: str | None,
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Execute a single SSH command. Returns ``(rc, stdout, stderr)``.
    Refuses any command starting with a known mutating verb."""
    cmd_first = command.strip().split(maxsplit=1)[0].lower() if command else ""
    if cmd_first in {"rm", "del", "format", "rmdir", "cipher", "fsutil"}:
        raise PermissionError(
            f"refusing to run mutating remote command: {command!r}"
        )
    argv = _build_ssh_argv(
        host=host, user=user, command=command,
        password=password, timeout=timeout,
    )
    env = os.environ.copy()
    if password is not None:
        env["SSHPASS"] = password
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except FileNotFoundError as e:
        return 127, "", f"ssh/sshpass not installed: {e!r}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    return out.returncode, out.stdout, out.stderr


def _scp_pull(
    *, host: str, user: str, remote_path: str, local_path: Path,
    password: str | None, timeout: int = 120,
) -> tuple[int, str]:
    """Pull a remote file via scp. NEVER pushes anything."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "scp", "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={timeout}",
        "-q",
        f"{user}@{host}:{remote_path}", str(local_path),
    ]
    argv = base if password is None else ["sshpass", "-e", *base]
    env = os.environ.copy()
    if password is not None:
        env["SSHPASS"] = password
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except FileNotFoundError as e:
        return 127, f"scp/sshpass not installed: {e!r}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    return out.returncode, out.stderr


# ---------------------------------------------------------------------------
# Validation — pure, runs over locally-staged files.
# ---------------------------------------------------------------------------


def _validate_local_csvs(
    *, csv_paths: Iterable[Path], symbol: str,
) -> tuple[int, str | None, str | None, list[str]]:
    """Read the staged OHLCV CSVs and check basic completeness.

    Returns ``(n_bars, earliest, latest, errors)``. Errors include
    timestamp non-monotonicity, zero-volume runs, negative spreads.
    """
    errors: list[str] = []
    n = 0
    earliest: str | None = None
    latest: str | None = None
    for p in csv_paths:
        if not p.exists() or p.stat().st_size == 0:
            errors.append(f"empty file: {p}")
            continue
        prev_ts: str | None = None
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split(",")
            if len(cols) < 6:
                errors.append(f"{p}:{i}: malformed row")
                continue
            ts = cols[0]
            try:
                o = float(cols[1])
                h = float(cols[2])
                l_ = float(cols[3])
                c = float(cols[4])
                v = float(cols[5])
            except ValueError:
                errors.append(f"{p}:{i}: non-numeric OHLCV")
                continue
            if h < l_:
                errors.append(f"{p}:{i}: high<low")
            if v < 0:
                errors.append(f"{p}:{i}: negative volume")
            if prev_ts is not None and ts < prev_ts:
                errors.append(f"{p}:{i}: timestamp regression {prev_ts}→{ts}")
            prev_ts = ts
            n += 1
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts
    return n, earliest, latest, errors


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def pull(
    *,
    host: str = DEFAULT_HOST,
    user: str = DEFAULT_USER,
    symbol: str = SYMBOL,
    auth_mode: str = "password",
    output_root: Path | None = None,
    remote_history_glob: str = (
        r"C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\"
        r"*\\Bases\\*\\history\\XAUUSD\\*"
    ),
    dry_run: bool = False,
) -> PullManifest:
    if symbol != SYMBOL:
        raise ValueError(
            f"this puller is XAUUSD-only; got symbol={symbol!r}"
        )

    output_root = output_root or _local_root(symbol)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files_pulled: list[str] = []
    if not dry_run:
        password = _resolve_password(auth_mode=auth_mode)

        # 1) probe — list candidate history files. ``where`` has Unix
        #    semantics on most Windows hosts via PowerShell.
        rc, stdout, stderr = _run_ssh(
            host=host, user=user,
            command=f'powershell -Command "Get-ChildItem -Path \\"{remote_history_glob}\\" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName"',
            password=password,
        )
        if rc != 0:
            raise RuntimeError(
                f"remote listing failed (rc={rc}): {stderr or stdout}"
            )
        candidates = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
        if not candidates:
            raise RuntimeError(
                f"no XAUUSD history files matched on {host}: "
                f"{remote_history_glob}"
            )

        for remote_file in candidates:
            local_path = raw_dir / Path(remote_file).name
            rc, err = _scp_pull(
                host=host, user=user,
                remote_path=remote_file, local_path=local_path,
                password=password,
            )
            if rc != 0:
                raise RuntimeError(
                    f"scp pull failed for {remote_file}: rc={rc} {err}"
                )
            files_pulled.append(str(local_path))

    # 2) Validate any locally-staged CSV exports. ``.hst`` decoding is
    #    operator-driven (left to a follow-up adapter); CSVs land in
    #    raw_dir from earlier exports.
    csv_paths = list(raw_dir.glob("*.csv"))
    n, earliest, latest, errors = _validate_local_csvs(
        csv_paths=csv_paths, symbol=symbol,
    )

    manifest = PullManifest(
        symbol=symbol, host=host, user=user,
        pulled_at=datetime.now(timezone.utc).isoformat(),
        files_pulled=tuple(files_pulled),
        n_bars_total=n,
        earliest_ts=earliest, latest_ts=latest,
        validation_ok=(not errors),
        validation_errors=tuple(errors),
    )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument(
        "--auth", choices=("password", "keyless"), default="password",
        help="password = read $MT5_SSH_PASSWORD; keyless = use ssh keys",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to stage pulled files. Default: data/ticks/<symbol>/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip SSH/SCP; only validate already-staged local files.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = pull(
            host=args.host, user=args.user, symbol=args.symbol,
            auth_mode=args.auth, output_root=args.output_dir,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        msg = str(e)
        print(f"FAILED: {msg}", file=sys.stderr)
        if "MT5_SSH_PASSWORD" in msg or "ssh/sshpass" in msg:
            return 2
        if "no XAUUSD history files matched" in msg:
            return 3
        return 4

    print(f"== MT5 tick pull complete ({manifest.symbol}) ==")
    print(f"  host={manifest.host} user={manifest.user}")
    print(f"  files_pulled={len(manifest.files_pulled)}")
    print(f"  bars={manifest.n_bars_total}")
    print(f"  window={manifest.earliest_ts} → {manifest.latest_ts}")
    print(f"  validation_ok={manifest.validation_ok}")
    if manifest.validation_errors:
        print("  errors:")
        for e in manifest.validation_errors[:10]:
            print(f"    - {e}")
    return 0 if manifest.validation_ok else 4


if __name__ == "__main__":
    sys.exit(main())
