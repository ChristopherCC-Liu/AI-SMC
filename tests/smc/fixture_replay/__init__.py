"""Incident-replay regression suite (R10 P4.1).

Each module under this package reproduces a known production incident as a
synthetic fixture and asserts the relevant defensive gate catches it. Every
R-round PR must keep `pytest -m fixture_replay` green; if a future round
inadvertently regresses one of these, the test fails before merge.

Two test patterns per incident:
- ``test_<incident>_blocked_with_fix_enabled`` — gate ON → assert no bad outcome.
- ``test_<incident>_baseline_without_fix_documents_bug`` — gate OFF →
  ``pytest.mark.xfail(strict=True)`` documents that the bug WOULD fire
  without the fix, locking the historical incident into the suite.
"""
