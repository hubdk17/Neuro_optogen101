# Single entry point for the data-derived pipeline.
#   make all    : extraction + reanalysis + calibration (+ environment/run log)
#   make test   : unit tests (incl. archive isolation)
# Every step's output is appended to results/calibration/run_log/run.log.

PY      ?= python3
NWB_DIR ?= /tmp/nwb_scratch
LOG     := results/calibration/run_log/run.log
RUN      = mkdir -p results/calibration/run_log && echo "=== $$(date -u +%FT%TZ) $(1)" >> $(LOG) && \
           $(PY) $(1) 2>&1 | tee -a $(LOG); test $${PIPESTATUS[0]} -eq 0

SHELL := /bin/bash

.PHONY: all reanalysis calibration validate extract infer population figures \
        regression ccg validity flash criteria labelnoise resource calfigs env test clean-derived

all: env reanalysis calibration

reanalysis: validate extract infer population figures

calibration: regression ccg validity flash criteria labelnoise resource calfigs

env:
	$(call RUN,scripts/calibration/08_environment.py)

validate:
	$(call RUN,scripts/reanalysis/00_validate_estimators.py)

extract:
	$(call RUN,scripts/reanalysis/01_extract_cohort.py --nwb-dir $(NWB_DIR))

infer:
	$(call RUN,scripts/reanalysis/02_unit_inference.py)

population:
	$(call RUN,scripts/reanalysis/03_population_analysis.py)

figures:
	$(call RUN,scripts/reanalysis/04_figures.py)

regression:
	$(call RUN,scripts/calibration/00_regression_baseline.py)

ccg:
	$(call RUN,scripts/calibration/01_matched_ccg.py)

validity:
	$(call RUN,scripts/calibration/02_latency_validity.py)

flash:
	$(call RUN,scripts/calibration/03_flash_confound.py)

criteria:
	$(call RUN,scripts/calibration/04_criteria_calibration.py)

labelnoise:
	$(call RUN,scripts/calibration/05_label_noise.py)

resource:
	$(call RUN,scripts/calibration/06_resource_table.py)

calfigs:
	$(call RUN,scripts/calibration/07_figures.py)

test:
	$(PY) -m pytest -q tests

clean-derived:
	rm -rf data/derived
