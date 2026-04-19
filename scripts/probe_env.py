"""Probe what Python actually sees for SMC_MACRO_ENABLED + related env vars.
Run via VPS start_live.bat-alike context to reproduce control leg env.
"""
import os
import sys

sys.path.insert(0, "src")

for key in (
    "SMC_MACRO_ENABLED",
    "SMC_JOURNAL_SUFFIX",
    "SMC_MACRO_MAGIC",
    "SMC_MT5_EXECUTE",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
):
    print(f"os.environ[{key!r}] = {os.environ.get(key)!r}")

print()
try:
    from smc.config import SMCConfig

    cfg = SMCConfig()
    print(f"cfg.macro_enabled = {cfg.macro_enabled!r}")
    print(f"cfg.journal_suffix = {cfg.journal_suffix!r}")
    print(f"cfg.macro_magic = {cfg.macro_magic!r}")
except Exception as e:
    print(f"SMCConfig EXCEPTION: {e}")
