"""Stage 7 — verify a HedgeRock evolution fingerprint chain.

Two subcommands:

  * ``verify`` — read a JSONL fingerprint chain and assert every
    entry's ``entry_hash`` recomputes correctly AND the
    ``prev_hash`` linkage is contiguous from ``GENESIS_PREV_HASH``.
  * ``diff`` — compare two chains. Reports the first divergent
    entry index plus the operation_type at that index.

Read-only — never writes to either chain. Exits 0 on success,
non-zero on the first detected break.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smc.hedgerock.evolution.fingerprint import (
    FingerprintChain,
    verify_chain,
)


def _cmd_verify(args: argparse.Namespace) -> int:
    chain_path = Path(args.chain_path)
    result = verify_chain(chain_path)
    if result.ok:
        print(
            f"OK: {chain_path} — {result.n_entries} entries, "
            f"chain integrity verified."
        )
        return 0
    print(
        f"BREAK at index {result.first_break_index} of "
        f"{chain_path}: {result.first_break_reason}",
        file=sys.stderr,
    )
    return 2


def _cmd_diff(args: argparse.Namespace) -> int:
    a_path = Path(args.chain_a)
    b_path = Path(args.chain_b)
    chain_a = FingerprintChain(a_path).entries()
    chain_b = FingerprintChain(b_path).entries()
    print(
        f"chain A: {a_path} — {len(chain_a)} entries"
    )
    print(
        f"chain B: {b_path} — {len(chain_b)} entries"
    )
    diverged_at: int | None = None
    for i in range(min(len(chain_a), len(chain_b))):
        if chain_a[i].entry_hash != chain_b[i].entry_hash:
            diverged_at = i
            break
    if diverged_at is None and len(chain_a) == len(chain_b):
        print("MATCH: chains are byte-identical.")
        return 0
    if diverged_at is None:
        diverged_at = min(len(chain_a), len(chain_b))
    op_a = chain_a[diverged_at].operation_type if diverged_at < len(chain_a) else "(none)"
    op_b = chain_b[diverged_at].operation_type if diverged_at < len(chain_b) else "(none)"
    print(
        f"DIVERGE at index {diverged_at}: "
        f"A.operation_type={op_a!r} vs B.operation_type={op_b!r}",
        file=sys.stderr,
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    verify_p = sub.add_parser("verify", help="verify a single chain")
    verify_p.add_argument("chain_path", type=Path)
    verify_p.set_defaults(func=_cmd_verify)
    diff_p = sub.add_parser("diff", help="diff two chains")
    diff_p.add_argument("chain_a", type=Path)
    diff_p.add_argument("chain_b", type=Path)
    diff_p.set_defaults(func=_cmd_diff)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
