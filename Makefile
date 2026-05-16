.PHONY: xauusd-dry-run xauusd-health xauusd-report help

PYTHON ?= python
OUTPUT_DIR ?= tmp/xauusd_evolution_dry_run/$(shell date -u +%Y%m%dT%H%M%SZ)
LOOKBACK_DAYS ?= 365

help:
	@echo "XAUUSD evolution sidecar — operator targets (REPORT-ONLY, never live)"
	@echo
	@echo "  make xauusd-dry-run      End-to-end dry-run; writes report to OUTPUT_DIR"
	@echo "  make xauusd-health       Run only the health-check CLI (auto-recover ON)"
	@echo "  make xauusd-report       Run only the recommend pipeline against last run"
	@echo
	@echo "Variables:"
	@echo "  OUTPUT_DIR     where to write artefacts (default: tmp/xauusd_evolution_dry_run/<UTC>)"
	@echo "  LOOKBACK_DAYS  how many days of bars to load (default: 365)"
	@echo "  PYTHON         interpreter to use (default: python)"

xauusd-dry-run:
	$(PYTHON) scripts/run_xauusd_evolution_dry_run.py \
		--output-dir $(OUTPUT_DIR) \
		--lookback-days $(LOOKBACK_DAYS)

xauusd-health:
	$(PYTHON) scripts/hedgerock_evolution_health.py \
		--registry-root $(OUTPUT_DIR)/registry \
		--audit-log $(OUTPUT_DIR)/audit.md \
		--queue-path $(OUTPUT_DIR)/queue/shadow_test_queue.jsonl \
		--ledger-path $(OUTPUT_DIR)/ledger/paper_test_ledger.jsonl \
		--auto-recover

xauusd-report:
	@if [ -z "$$LAST_RUN" ]; then \
		LAST_RUN=$$(ls -1dt tmp/xauusd_evolution_dry_run/* 2>/dev/null | head -1); \
		if [ -z "$$LAST_RUN" ]; then \
			echo "no prior dry-run found under tmp/xauusd_evolution_dry_run/" >&2; \
			exit 1; \
		fi; \
		echo "using last run: $$LAST_RUN"; \
		cat $$LAST_RUN/xauusd_dry_run_report.md; \
	else \
		cat $$LAST_RUN/xauusd_dry_run_report.md; \
	fi
