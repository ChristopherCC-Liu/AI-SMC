"""P0-2 — Deterministic fingerprint book.

Each fingerprint records ``(operation, inputs, outputs, params,
git_commit, algorithm_version, prev_hash)`` so an operator can
later prove that a specific recommendation came from a specific
input + code state, and that the chain of fingerprints has not
been re-ordered or replayed with edits.

Hashes are SHA-256 over canonical JSON:

  * ``sort_keys=True``
  * ``separators=(",", ":")`` (no whitespace)
  * ``ensure_ascii=False`` (UTF-8)

The chain itself is append-only: every entry's ``prev_hash`` is the
``entry_hash`` of the previous line in the JSONL log. Tampering
with any line breaks the chain at every subsequent position;
:func:`verify_chain` walks the file and reports the first break.

Isolation: this module never imports ``rule_engine`` or the Tier-1
unsealed prod modules. ``git_commit`` is read once via subprocess
when a chain is opened; missing/non-git directories report
``"unknown"`` instead of raising.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


__all__ = [
    "ALGORITHM_VERSION",
    "FingerprintChain",
    "FingerprintEntry",
    "GENESIS_PREV_HASH",
    "VerifyResult",
    "compute_fingerprint",
    "current_git_commit",
    "verify_chain",
]


ALGORITHM_VERSION: str = "fingerprint/v1"
GENESIS_PREV_HASH: str = "0" * 64  # used by the first entry in a chain


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_payload(payload: Any) -> str:
    """Public canonical-hash helper. Stable across Python versions /
    dict-insertion order / unicode encoding choices."""
    return _sha256_hex(_canonical_json_bytes(payload))


def current_git_commit(repo_root: Path | None = None) -> str:
    """Return the current git HEAD short SHA, or ``"unknown"`` when
    not in a git checkout / git is unavailable."""
    cwd = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(cwd), check=True, capture_output=True, text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerprintEntry:
    """One chain entry. ``entry_hash`` is the SHA-256 over every
    other field (canonically encoded), so it serves both as a
    self-integrity check and as the next line's ``prev_hash``."""

    operation_type: str
    timestamp: str
    input_hash: str
    output_hash: str
    params_hash: str
    git_commit: str
    algorithm_version: str
    prev_hash: str
    entry_hash: str = ""

    def to_canonical_dict(self) -> dict[str, Any]:
        """All fields except ``entry_hash`` — the substance of the
        entry that the hash signs over."""
        return {
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "params_hash": self.params_hash,
            "git_commit": self.git_commit,
            "algorithm_version": self.algorithm_version,
            "prev_hash": self.prev_hash,
        }

    def recompute_entry_hash(self) -> str:
        return _sha256_hex(_canonical_json_bytes(self.to_canonical_dict()))


def compute_fingerprint(
    *,
    operation_type: str,
    inputs: Any,
    outputs: Any,
    params: Any | None = None,
    prev_hash: str = GENESIS_PREV_HASH,
    timestamp: str | None = None,
    git_commit: str | None = None,
    algorithm_version: str = ALGORITHM_VERSION,
) -> FingerprintEntry:
    """Build a fully-populated :class:`FingerprintEntry` (including
    its ``entry_hash``) from the supplied operation context."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    gc = git_commit if git_commit is not None else current_git_commit()
    body = FingerprintEntry(
        operation_type=str(operation_type),
        timestamp=ts,
        input_hash=hash_payload(inputs),
        output_hash=hash_payload(outputs),
        params_hash=hash_payload(params if params is not None else {}),
        git_commit=gc,
        algorithm_version=algorithm_version,
        prev_hash=prev_hash,
    )
    return FingerprintEntry(
        **{**asdict(body), "entry_hash": body.recompute_entry_hash()},
    )


# ---------------------------------------------------------------------------
# Append-only chain backed by JSONL
# ---------------------------------------------------------------------------


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _assert_chain_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"fingerprint chain path would land under a forbidden "
                f"location: {text!r} (matched fragment {fragment!r})"
            )


@dataclass
class FingerprintChain:
    """Append-only JSONL fingerprint log."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        _assert_chain_path_safe(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def entries(self) -> list[FingerprintEntry]:
        out: list[FingerprintEntry] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            try:
                out.append(FingerprintEntry(**obj))
            except TypeError:
                continue
        return out

    def latest_hash(self) -> str:
        existing = self.entries()
        if not existing:
            return GENESIS_PREV_HASH
        return existing[-1].entry_hash

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        operation_type: str,
        inputs: Any,
        outputs: Any,
        params: Any | None = None,
        timestamp: str | None = None,
    ) -> FingerprintEntry:
        entry = compute_fingerprint(
            operation_type=operation_type,
            inputs=inputs,
            outputs=outputs,
            params=params,
            prev_hash=self.latest_hash(),
            timestamp=timestamp,
        )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False)
                + "\n"
            )
        return entry


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    n_entries: int
    first_break_index: int | None
    first_break_reason: str | None


def verify_chain(chain_path: Path) -> VerifyResult:
    """Walk the chain and verify per-entry hash + prev-hash linkage."""
    cp = Path(chain_path)
    if not cp.exists():
        return VerifyResult(
            ok=False, n_entries=0,
            first_break_index=None,
            first_break_reason=f"chain file not found: {cp}",
        )
    chain = FingerprintChain(cp)
    entries = chain.entries()
    expected_prev = GENESIS_PREV_HASH
    for i, e in enumerate(entries):
        if e.algorithm_version != ALGORITHM_VERSION:
            return VerifyResult(
                ok=False, n_entries=len(entries),
                first_break_index=i,
                first_break_reason=(
                    f"unknown algorithm_version {e.algorithm_version!r} "
                    f"at entry {i}"
                ),
            )
        if e.prev_hash != expected_prev:
            return VerifyResult(
                ok=False, n_entries=len(entries),
                first_break_index=i,
                first_break_reason=(
                    f"prev_hash mismatch at entry {i}: "
                    f"recorded={e.prev_hash[:12]}…, "
                    f"expected={expected_prev[:12]}…"
                ),
            )
        recomputed = e.recompute_entry_hash()
        if recomputed != e.entry_hash:
            return VerifyResult(
                ok=False, n_entries=len(entries),
                first_break_index=i,
                first_break_reason=(
                    f"entry_hash mismatch at entry {i}: "
                    f"recorded={e.entry_hash[:12]}…, "
                    f"recomputed={recomputed[:12]}…"
                ),
            )
        expected_prev = e.entry_hash
    return VerifyResult(
        ok=True, n_entries=len(entries),
        first_break_index=None, first_break_reason=None,
    )
