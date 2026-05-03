"""Stage 6 / Round 10 phase 1 — auto purifier (append-only stale-data sweep).

Purpose
-------
Sweep three known stale-data sinks of the evolution layer and MOVE
expired rows into a sibling ``archive/`` tree under the same workspace.
Nothing is ever deleted. Each archive action is logged to a JSONL
audit file under ``<workspace>/audit/auto_purifier.jsonl``.

Targets
-------

* ``<workspace>/queue/shadow_test_queue.jsonl`` — JSONL, one row per
  shadow test enqueue. Stale rows (``now - enqueued_at > max_age_days``)
  move to ``<archive_root>/queue/<id>.json``. The original file is
  rewritten with only the kept rows.
* ``<workspace>/calibrator/state.json`` — JSON document with a
  ``priors`` list. Priors with stale ``last_updated_at`` move into a
  sibling archive file ``<archive_root>/calibrations/state-<ts>.json``.
  The original is rewritten with the kept priors.
* ``<workspace>/candidates/*.json`` — one candidate per file. Stale
  candidates (in the in-memory list passed in) move to
  ``<archive_root>/candidates/<id>.json``.

Safety constraints
------------------

* Refuses workspaces that land under ``policy_registry/approved`` —
  see :func:`_assert_workspace_safe`. The check is a copy of the shape
  used by ``run_xauusd_evolution_dry_run.py`` (do **not** import that
  script).
* Never calls ``os.remove`` / ``Path.unlink`` / ``shutil.rmtree``.
  Archive moves use :func:`shutil.move`. Original files are rewritten
  in-place with the kept rows; expired rows are never lost.
* All public dataclasses are ``frozen=True`` so reports are immutable.
* Idempotent: re-running a purification on already-purified data is a
  no-op (already-archived ids and already-evicted priors are skipped).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


__all__ = [
    "AutoPurifier",
    "FullPurgeReport",
    "PurgeAction",
    "PurgeReport",
]


# ---------------------------------------------------------------------------
# Forbidden-parent check (copy of the shape used by the dry-run
# orchestrator — intentionally NOT imported from there).
# ---------------------------------------------------------------------------


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)


def _assert_workspace_safe(path: Path) -> None:
    """Raise ``ValueError`` if *path* sits under a forbidden production
    registry parent.
    """

    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"auto-purifier refuses workspace under a forbidden "
                f"location: {text!r} (matched {fragment!r})"
            )


# ---------------------------------------------------------------------------
# Data classes (all frozen).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PurgeAction:
    """One archive action — a single row was moved to the archive."""

    item_id: str
    archived_path: str
    reason: str


@dataclass(frozen=True)
class PurgeReport:
    """Per-target report (queue / calibrations / candidates)."""

    target: str
    actions: tuple[PurgeAction, ...]
    n_archived: int
    n_kept: int
    audit_log_path: str
    generated_at: str


@dataclass(frozen=True)
class FullPurgeReport:
    """Aggregate report from :meth:`AutoPurifier.run_full_purification`."""

    queue: PurgeReport | None
    calibrations: PurgeReport | None
    candidates: PurgeReport | None
    workspace: str
    overall_archived: int
    generated_at: str


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_iso(ts: str) -> datetime | None:
    try:
        out = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out


def _age_days(now: datetime, ts: str) -> float | None:
    parsed = _parse_iso(ts)
    if parsed is None:
        return None
    delta = now - parsed
    return delta.total_seconds() / 86400.0


def _default_archive_root(reference: Path) -> Path:
    """Default archive root for an in-place target file at *reference*.

    For ``<workspace>/queue/shadow_test_queue.jsonl`` we want
    ``<workspace>/archive``. The reference is the target file; its
    parent is the bucket dir, its grandparent is the workspace.
    """

    return reference.parent.parent / "archive"


def _default_audit_log(reference: Path) -> Path:
    """Default audit log path for an in-place target file at *reference*."""

    return reference.parent.parent / "audit" / "auto_purifier.jsonl"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_audit(
    audit_log_path: Path,
    *,
    operation: str,
    item_id: str,
    src: str,
    dst: str,
    reason: str,
    max_age_days: int,
    now: datetime,
) -> None:
    _ensure_dir(audit_log_path.parent)
    record = {
        "timestamp": _iso(now),
        "operation": operation,
        "item_id": item_id,
        "from": src,
        "to": dst,
        "reason": reason,
        "max_age_days": max_age_days,
    }
    with audit_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


class AutoPurifier:
    """Append-only stale-data purifier.

    All methods are static. Each method moves stale rows to an archive
    location and writes one JSONL line per move to the audit log.
    Nothing is ever deleted.
    """

    @staticmethod
    def purge_stale_queue_items(
        queue_path: Path,
        *,
        max_age_days: int = 60,
        archive_root: Path | None = None,
        audit_log_path: Path | None = None,
        now: datetime | None = None,
    ) -> PurgeReport:
        queue_path = Path(queue_path)
        _assert_workspace_safe(queue_path)
        now_dt = now if now is not None else _utcnow()
        archive_root = (
            Path(archive_root)
            if archive_root is not None
            else _default_archive_root(queue_path)
        )
        audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else _default_audit_log(queue_path)
        )

        target_dir = archive_root / "queue"

        actions: list[PurgeAction] = []
        kept_lines: list[str] = []

        if not queue_path.exists():
            return PurgeReport(
                target="queue",
                actions=(),
                n_archived=0,
                n_kept=0,
                audit_log_path=str(audit_log_path),
                generated_at=_iso(now_dt),
            )

        with queue_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Preserve unparseable rows — never silently drop.
                    kept_lines.append(line)
                    continue
                enqueued_at = row.get("enqueued_at")
                age = _age_days(now_dt, enqueued_at) if isinstance(enqueued_at, str) else None
                if age is None or age <= max_age_days:
                    kept_lines.append(line)
                    continue
                item_id = str(row.get("id") or row.get("candidate_id") or "")
                if not item_id:
                    # No id -> can't archive safely, keep in place.
                    kept_lines.append(line)
                    continue
                _ensure_dir(target_dir)
                dst = target_dir / f"{item_id}.json"
                if dst.exists():
                    # Already archived; do not duplicate. Drop from queue.
                    continue
                dst.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
                reason = f"age={age:.1f}d > max={max_age_days}d"
                actions.append(
                    PurgeAction(
                        item_id=item_id,
                        archived_path=str(dst),
                        reason=reason,
                    )
                )
                _append_audit(
                    audit_log_path,
                    operation="purge_stale_queue_items",
                    item_id=item_id,
                    src=str(queue_path),
                    dst=str(dst),
                    reason=reason,
                    max_age_days=max_age_days,
                    now=now_dt,
                )

        # Rewrite original file with only the kept rows.
        if actions:
            with queue_path.open("w", encoding="utf-8") as fh:
                for line in kept_lines:
                    fh.write(line + "\n")

        return PurgeReport(
            target="queue",
            actions=tuple(actions),
            n_archived=len(actions),
            n_kept=len(kept_lines),
            audit_log_path=str(audit_log_path),
            generated_at=_iso(now_dt),
        )

    @staticmethod
    def purge_expired_calibrations(
        cal_path: Path,
        *,
        max_age_days: int = 45,
        archive_root: Path | None = None,
        audit_log_path: Path | None = None,
        now: datetime | None = None,
    ) -> PurgeReport:
        cal_path = Path(cal_path)
        _assert_workspace_safe(cal_path)
        now_dt = now if now is not None else _utcnow()
        archive_root = (
            Path(archive_root)
            if archive_root is not None
            else _default_archive_root(cal_path)
        )
        audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else _default_audit_log(cal_path)
        )

        target_dir = archive_root / "calibrations"

        if not cal_path.exists():
            return PurgeReport(
                target="calibrations",
                actions=(),
                n_archived=0,
                n_kept=0,
                audit_log_path=str(audit_log_path),
                generated_at=_iso(now_dt),
            )

        try:
            doc = json.loads(cal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return PurgeReport(
                target="calibrations",
                actions=(),
                n_archived=0,
                n_kept=0,
                audit_log_path=str(audit_log_path),
                generated_at=_iso(now_dt),
            )

        priors = doc.get("priors")
        if not isinstance(priors, list):
            priors = []

        kept_priors: list[dict] = []
        expired_priors: list[dict] = []
        actions: list[PurgeAction] = []

        for prior in priors:
            if not isinstance(prior, dict):
                kept_priors.append(prior)
                continue
            ts = prior.get("last_updated_at")
            age = _age_days(now_dt, ts) if isinstance(ts, str) else None
            if age is None or age <= max_age_days:
                kept_priors.append(prior)
                continue
            expired_priors.append(prior)
            item_id = str(
                prior.get("parameter_class")
                or prior.get("id")
                or f"prior-{len(actions)}"
            )
            reason = f"age={age:.1f}d > max={max_age_days}d"
            actions.append(
                PurgeAction(
                    item_id=item_id,
                    archived_path="",  # filled below once archive file is written
                    reason=reason,
                )
            )

        if not expired_priors:
            return PurgeReport(
                target="calibrations",
                actions=(),
                n_archived=0,
                n_kept=len(kept_priors),
                audit_log_path=str(audit_log_path),
                generated_at=_iso(now_dt),
            )

        _ensure_dir(target_dir)
        ts_suffix = now_dt.strftime("%Y%m%dT%H%M%S")
        archive_file = target_dir / f"state-{ts_suffix}.json"
        # Avoid clobber if multiple runs happen in the same second.
        idx = 0
        while archive_file.exists():
            idx += 1
            archive_file = target_dir / f"state-{ts_suffix}-{idx}.json"

        archived_doc = {
            "schema": doc.get("schema"),
            "saved_at": doc.get("saved_at"),
            "archived_at": _iso(now_dt),
            "priors": expired_priors,
        }
        archive_file.write_text(
            json.dumps(archived_doc, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        # Rewrite actions with the archive path now we have it.
        actions_with_paths: list[PurgeAction] = []
        for action in actions:
            actions_with_paths.append(
                PurgeAction(
                    item_id=action.item_id,
                    archived_path=str(archive_file),
                    reason=action.reason,
                )
            )
            _append_audit(
                audit_log_path,
                operation="purge_expired_calibrations",
                item_id=action.item_id,
                src=str(cal_path),
                dst=str(archive_file),
                reason=action.reason,
                max_age_days=max_age_days,
                now=now_dt,
            )

        # Rewrite original file with the kept priors.
        new_doc = dict(doc)
        new_doc["priors"] = kept_priors
        new_doc["saved_at"] = _iso(now_dt)
        cal_path.write_text(
            json.dumps(new_doc, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        return PurgeReport(
            target="calibrations",
            actions=tuple(actions_with_paths),
            n_archived=len(actions_with_paths),
            n_kept=len(kept_priors),
            audit_log_path=str(audit_log_path),
            generated_at=_iso(now_dt),
        )

    @staticmethod
    def archive_obsolete_candidates(
        candidates: list[dict],
        *,
        max_age_days: int = 90,
        archive_root: Path,
        audit_log_path: Path | None = None,
        now: datetime | None = None,
    ) -> PurgeReport:
        archive_root = Path(archive_root)
        _assert_workspace_safe(archive_root)
        now_dt = now if now is not None else _utcnow()
        audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else archive_root.parent / "audit" / "auto_purifier.jsonl"
        )

        target_dir = archive_root / "candidates"

        actions: list[PurgeAction] = []
        n_kept = 0

        for cand in candidates:
            if not isinstance(cand, dict):
                n_kept += 1
                continue
            ts = cand.get("generated_at")
            age = _age_days(now_dt, ts) if isinstance(ts, str) else None
            if age is None or age <= max_age_days:
                n_kept += 1
                continue
            item_id = str(cand.get("id") or "")
            if not item_id:
                n_kept += 1
                continue
            _ensure_dir(target_dir)
            dst = target_dir / f"{item_id}.json"
            if dst.exists():
                # Idempotent: already archived previously, do not duplicate
                # and do not log a second audit line. Also do not count as
                # "kept" — the candidate is gone from the in-memory store.
                continue
            dst.write_text(
                json.dumps(cand, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            reason = f"age={age:.1f}d > max={max_age_days}d"
            actions.append(
                PurgeAction(
                    item_id=item_id,
                    archived_path=str(dst),
                    reason=reason,
                )
            )
            _append_audit(
                audit_log_path,
                operation="archive_obsolete_candidates",
                item_id=item_id,
                src="<in-memory>",
                dst=str(dst),
                reason=reason,
                max_age_days=max_age_days,
                now=now_dt,
            )

        return PurgeReport(
            target="candidates",
            actions=tuple(actions),
            n_archived=len(actions),
            n_kept=n_kept,
            audit_log_path=str(audit_log_path),
            generated_at=_iso(now_dt),
        )

    @staticmethod
    def run_full_purification(
        workspace: Path,
        *,
        queue_max_age_days: int = 60,
        cal_max_age_days: int = 45,
        candidates_max_age_days: int = 90,
        now: datetime | None = None,
    ) -> FullPurgeReport:
        workspace = Path(workspace)
        _assert_workspace_safe(workspace)
        now_dt = now if now is not None else _utcnow()

        archive_root = workspace / "archive"
        audit_log_path = workspace / "audit" / "auto_purifier.jsonl"

        # Queue
        queue_path = workspace / "queue" / "shadow_test_queue.jsonl"
        queue_report: PurgeReport | None
        if queue_path.exists():
            queue_report = AutoPurifier.purge_stale_queue_items(
                queue_path,
                max_age_days=queue_max_age_days,
                archive_root=archive_root,
                audit_log_path=audit_log_path,
                now=now_dt,
            )
        else:
            queue_report = None

        # Calibrations
        cal_path = workspace / "calibrator" / "state.json"
        cal_report: PurgeReport | None
        if cal_path.exists():
            cal_report = AutoPurifier.purge_expired_calibrations(
                cal_path,
                max_age_days=cal_max_age_days,
                archive_root=archive_root,
                audit_log_path=audit_log_path,
                now=now_dt,
            )
        else:
            cal_report = None

        # Candidates — file-backed flow. We pass the loaded dicts through
        # the pure helper to get the report, then shutil.move the source
        # files into the archive location (replacing the freshly-written
        # copy from the helper). Nothing is unlinked.
        candidates_dir = workspace / "candidates"
        candidates_report: PurgeReport | None
        if candidates_dir.exists() and candidates_dir.is_dir():
            loaded: list[dict] = []
            source_paths: dict[str, Path] = {}
            for p in sorted(candidates_dir.glob("*.json")):
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict):
                    loaded.append(doc)
                    cid = str(doc.get("id") or "")
                    if cid:
                        source_paths[cid] = p
            candidates_report = AutoPurifier.archive_obsolete_candidates(
                loaded,
                max_age_days=candidates_max_age_days,
                archive_root=archive_root,
                audit_log_path=audit_log_path,
                now=now_dt,
            )
            # For the file-backed flow, replace the helper-written copy
            # with a true shutil.move of the source. shutil.move on POSIX
            # overwrites the destination atomically, so the original
            # bytes end up at the archive path and the source path is
            # gone — no unlink call anywhere in this module.
            for action in candidates_report.actions:
                src = source_paths.get(action.item_id)
                if src is None or not src.exists():
                    continue
                dst = Path(action.archived_path)
                _ensure_dir(dst.parent)
                try:
                    shutil.move(str(src), str(dst))
                except OSError:
                    # Source missing or already moved — leave the helper
                    # copy in place.
                    pass
        else:
            candidates_report = None

        overall = sum(
            r.n_archived
            for r in (queue_report, cal_report, candidates_report)
            if r is not None
        )

        return FullPurgeReport(
            queue=queue_report,
            calibrations=cal_report,
            candidates=candidates_report,
            workspace=str(workspace),
            overall_archived=overall,
            generated_at=_iso(now_dt),
        )
