"""Test-only path helpers for the evolution-layer suite.

The evolution test suite checks behaviour against three roots:

  * ``HEDGEROCK_HOME`` — operator-team data layout (the live
    policy registry, mql5 sources, docs).
  * ``AI_SMC_HOME``   — this repo's checkout root.
  * ``SCRIPTS_DIR``   — the evolution CLI scripts.

All three are resolvable from environment variables with sensible
defaults so a fresh ``git clone`` runs without any manual config.
The defaults intentionally point at locations that may not exist
on a fresh machine — every consumer of these helpers degrades
gracefully (skips the test, returns a clean state, etc.) when the
target is absent.
"""

from __future__ import annotations

import os
from pathlib import Path


__all__ = [
    "ai_smc_home",
    "hedgerock_home",
    "real_audit_log",
    "real_registry_root",
    "real_shadow_artefacts_root",
    "scripts_dir",
]


def hedgerock_home() -> Path:
    """``$HEDGEROCK_HOME`` with default ``$HOME/HedgeRock``."""
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def ai_smc_home() -> Path:
    """``$AI_SMC_HOME`` with default = repo root computed from this
    file's location (parents[3] = repo root)."""
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[3]


def scripts_dir() -> Path:
    return ai_smc_home() / "scripts"


def real_registry_root() -> Path:
    return hedgerock_home() / "policy_registry"


def real_shadow_artefacts_root() -> Path:
    return real_registry_root() / "shadow_artefacts"


def real_audit_log() -> Path:
    return real_shadow_artefacts_root() / "_audit.md"
