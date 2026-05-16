"""Phase D-cont3 / Ticket 1 — filesystem-backed policy registry.

**Sidecar layer.** Writes ONLY under ``<root>/candidates/`` and
``<root>/audit/``. Every other path raises ``RegistryWriteForbidden``
on attempted write — Plan §6.2 + RFC §11 invariants.

The registry never:
  - writes under ``<root>/approved/`` (read-only access for the
    pointer; ``set_pointer`` raises ``NotImplementedError`` because
    pointer flips are MVP+1 / human-only);
  - writes outside ``<root>`` (any path under ``src/``, ``config/``,
    EA / .mq5, etc. is forbidden — the registry has no business
    touching those locations).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import re

from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    CandidateState,
    dump_manifest,
    load_manifest,
)


__all__ = [
    "IllegalCandidateState",
    "InvalidCandidateId",
    "PolicyRegistry",
    "RegistryWriteForbidden",
    "StaleCandidateError",
    "validate_candidate_id",
]


class StaleCandidateError(ValueError):
    """Raised when a candidate manifest already on disk does not match
    what the current candidate menu (or other authoritative source)
    says it should be — e.g. ``diff.proposed_value`` drifted, the
    target knob changed, or the ``state`` is something the loader did
    not expect.

    Distinct from :class:`smc.hedgerock.evolution.policy_manifest.ManifestIntegrityError`
    (which signals byte-level tamper) — this signals semantic drift
    between a previously-persisted manifest and the current source of
    truth. Both must fail-closed: the report CLI refuses to proceed
    until the discrepancy is resolved by a human.
    """


# Candidate IDs must be safe filename components: alphanumeric start,
# then alphanumerics + dot + underscore + hyphen. This bans path
# separators ("/", "\\"), parent-traversal ("../"), whitespace,
# leading dots/hyphens, and NUL/control bytes.
_CANDIDATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CANDIDATE_ID_MAX_LEN = 128


class InvalidCandidateId(ValueError):
    """Raised when a ``candidate_id`` would resolve to a path the
    registry must not write to (path separators, parent-traversal,
    leading dot, whitespace, etc.)."""


def validate_candidate_id(candidate_id: Any) -> str:
    """Reject any candidate_id that could escape the candidates
    subtree. Returns the validated string on success.

    Allowed characters: ``[A-Za-z0-9._-]``, with the first character
    restricted to ``[A-Za-z0-9]``. Length cap: 128 bytes.

    Banned by construction: ``/``, ``\\``, ``..``, whitespace,
    leading ``.`` / ``-``, empty string, non-string types, NUL.
    """
    if not isinstance(candidate_id, str):
        raise InvalidCandidateId(
            f"candidate_id must be a string, got {type(candidate_id).__name__}"
        )
    if not candidate_id:
        raise InvalidCandidateId("candidate_id must be non-empty")
    if len(candidate_id) > _CANDIDATE_ID_MAX_LEN:
        raise InvalidCandidateId(
            f"candidate_id length {len(candidate_id)} exceeds "
            f"{_CANDIDATE_ID_MAX_LEN}"
        )
    if not _CANDIDATE_ID_PATTERN.match(candidate_id):
        raise InvalidCandidateId(
            f"candidate_id {candidate_id!r} contains forbidden characters; "
            f"must match {_CANDIDATE_ID_PATTERN.pattern!r} "
            "(no path separators, no '..', no whitespace, no leading '.' or '-')"
        )
    return candidate_id


class RegistryWriteForbidden(PermissionError):
    """Raised when a caller attempts to write outside the candidate /
    audit subtrees of the registry."""


class IllegalCandidateState(ValueError):
    """Raised when a candidate manifest has a state above the MVP
    write surface (`draft` / `tested` only)."""


_ALLOWED_WRITE_STATES = (CandidateState.DRAFT, CandidateState.TESTED)


def _audit_timestamp() -> str:
    """Filesystem-safe UTC timestamp for audit log entries. Replaced
    by tests via monkeypatch when needed."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")


class PolicyRegistry:
    """Filesystem registry. Construct with the registry root; all
    subdirectories are created lazily on first write."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    # ---- directories ----------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def candidates_dir(self) -> Path:
        return self._root / "candidates"

    @property
    def audit_dir(self) -> Path:
        return self._root / "audit"

    @property
    def approved_dir(self) -> Path:
        return self._root / "approved"

    @property
    def pointer_path(self) -> Path:
        return self._root / "pointer.json"

    # ---- write surface --------------------------------------------------

    def _assert_writable_path(self, path: Path) -> None:
        """Path-allowlist enforcement. The registry only writes under
        ``<root>/candidates/`` and ``<root>/audit/``."""
        target = Path(path).resolve()
        candidates = self.candidates_dir.resolve()
        audit = self.audit_dir.resolve()
        for allowed in (candidates, audit):
            try:
                target.relative_to(allowed)
                return
            except ValueError:
                continue
        raise RegistryWriteForbidden(
            f"write to {target} is forbidden — sidecar may only write under "
            f"{candidates} and {audit}"
        )

    def _assert_dir_not_symlink(self, path: Path, label: str) -> None:
        """Refuse to write into a symlinked registry directory.

        A symlinked ``candidates/`` or ``audit/`` could redirect the
        registry's write surface to an unrelated location. Fail-closed
        per Ticket 1 hard boundary — the registry treats only real
        directories under its own root as legitimate write targets.
        """
        if path.exists() and path.is_symlink():
            raise RegistryWriteForbidden(
                f"refusing to write under symlinked {label} directory: "
                f"{path} → {os.readlink(path)} "
                "(symlinked registry directories could redirect the write "
                "surface — fail-closed per Ticket 1 hard boundary)"
            )

    def write_to_path(
        self, path: Path, manifest: CandidateManifest,
    ) -> Path:
        """Generic write helper. Used internally by ``write_candidate``;
        exposed publicly so tests can directly assert that writes to
        forbidden paths raise."""
        self._assert_writable_path(path)
        if manifest.state not in _ALLOWED_WRITE_STATES:
            raise IllegalCandidateState(
                f"cannot write candidate in state {manifest.state.value!r}; "
                f"MVP only writes {[s.value for s in _ALLOWED_WRITE_STATES]}"
            )
        dump_manifest(manifest, path)
        return path

    def write_candidate(self, manifest: CandidateManifest) -> Path:
        """Persist a candidate manifest under ``candidates/<id>.json``.

        ``candidate_id`` is validated against
        :func:`validate_candidate_id` before any path is built;
        defence-in-depth: the resolved parent directory is then
        cross-checked against ``candidates_dir.resolve()`` to refuse
        any path that escapes the candidates subtree even if the
        validator had a bug.
        """
        validate_candidate_id(manifest.candidate_id)
        # Symlink check BEFORE mkdir: if candidates_dir already exists
        # as a symlink, mkdir(exist_ok=True) is a no-op and we would
        # otherwise write through the link. Fail-closed instead.
        self._assert_dir_not_symlink(self.candidates_dir, "candidates")
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        path = self.candidates_dir / f"{manifest.candidate_id}.json"
        # Defence-in-depth: the resolved parent must be exactly the
        # candidates directory. Symlinks / unusual filesystems cannot
        # leak the write surface.
        resolved_parent = path.resolve().parent
        candidates_resolved = self.candidates_dir.resolve()
        if resolved_parent != candidates_resolved:
            raise RegistryWriteForbidden(
                f"candidate_id {manifest.candidate_id!r} resolves to "
                f"{path.resolve()}, escaping {candidates_resolved}"
            )
        return self.write_to_path(path, manifest)

    def append_audit(self, entry: dict[str, Any]) -> Path:
        """Append-only audit log. Filename is timestamp-based; a second
        write with the same timestamp raises ``FileExistsError``."""
        self._assert_dir_not_symlink(self.audit_dir, "audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        ts = _audit_timestamp()
        path = self.audit_dir / f"{ts}.json"
        if path.exists():
            raise FileExistsError(
                f"audit entry already exists at {path}; audit log is "
                "append-only"
            )
        body = {"timestamp": ts, **entry}
        path.write_text(json.dumps(body, indent=2, sort_keys=True))
        os.chmod(path, 0o444)
        return path

    # ---- read surface ---------------------------------------------------

    def list_candidates(self) -> list[str]:
        if not self.candidates_dir.exists():
            return []
        return sorted(p.stem for p in self.candidates_dir.glob("*.json"))

    def get_candidate(self, candidate_id: str) -> CandidateManifest:
        """Read-side counterpart to :meth:`write_candidate`. Same
        validation rules apply: a candidate_id with forbidden
        characters cannot reach the disk surface, period.
        ``load_manifest`` then enforces the strict envelope + content
        hash check, so a tampered or bare-layout file raises
        :class:`smc.hedgerock.evolution.policy_manifest.ManifestIntegrityError`.
        """
        validate_candidate_id(candidate_id)
        path = self.candidates_dir / f"{candidate_id}.json"
        # Defence-in-depth (symmetric with write_candidate).
        resolved_parent = path.resolve().parent
        candidates_resolved = self.candidates_dir.resolve()
        if resolved_parent != candidates_resolved:
            raise RegistryWriteForbidden(
                f"candidate_id {candidate_id!r} resolves to "
                f"{path.resolve()}, escaping {candidates_resolved}"
            )
        if not path.exists():
            raise FileNotFoundError(f"candidate not found: {candidate_id}")
        return load_manifest(path)

    def get_pointer(self) -> str | None:
        """Return the currently-approved candidate id, or None if no
        pointer has been seeded yet. MVP starts with no pointer."""
        if not self.pointer_path.exists():
            return None
        try:
            data = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data.get("approved_id")

    def set_pointer(self, candidate_id: str) -> None:
        """**MVP+1 / human-only operation.** Always raises in the MVP."""
        raise NotImplementedError(
            "pointer flip is MVP+1 / human-only — the report-only "
            "registry does not advance the live approved policy"
        )
