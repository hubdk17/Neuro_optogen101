"""
scripts/reanalysis/03_population_analysis.py recomputes the original operational
heuristic from results/ml_final/master_ml_dataset_28spec.parquet. That table was
produced by the legacy src/feature_extraction.py layer, so these tests pin down
that the columns 03 reads are computed from the recorded spikes:

* first-spike latency, trial reliability and evoked spike counts agree unit by
  unit with the independent NWB re-extraction (results/reanalysis/tables/
  unit_results.parquet, same 10-ms pulses, same [1, 9) ms window);
* modulation ratio and the reference class follow from the table's own measured
  columns by the documented formulas;
* the session -> specimen / Cre line identities in results/cohort/
  full_28_specimen_manifest.csv (read by 01 and 02) match Allen's sessions table.

Columns of the master table that are NOT data-derived (optical_intensity,
protocol, *_spike_count, label_confidence, evidence/uncertainty scores) are
listed in archive/README.md and are not read by the pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.labeling import label_unit_features_table

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "results/ml_final/master_ml_dataset_28spec.parquet"
REANALYSIS = ROOT / "results/reanalysis/tables/unit_results.parquet"
MANIFEST = ROOT / "results/cohort/full_28_specimen_manifest.csv"
ALLEN_SESSIONS = ROOT / "data/metadata/sessions.csv"

# columns of the master table read by scripts/reanalysis/03_population_analysis.py
READ_BY_03 = ["session_id", "unit_id", "trial_reliability", "median_latency_ms", "modulation_ratio",
              "p_value", "effect_size", "reference_class", "response_at_each_light_level"]


@pytest.fixture(scope="module")
def joined():
    m = pd.read_parquet(MASTER)
    u = pd.read_parquet(REANALYSIS, columns=["unit_id", "session_id", "p10_n", "p10_ne",
                                              "p10_rel_raw", "p10_median_first_ms"])
    return m, m.merge(u, on="unit_id", suffixes=("", "_re"))


def test_columns_read_by_03_present(joined):
    m, _ = joined
    assert set(READ_BY_03) <= set(m.columns)
    assert m.unit_id.is_unique and m.session_id.nunique() == 28


def test_every_master_unit_is_in_reanalysis(joined):
    m, j = joined
    assert len(j) == len(m)
    assert (j.session_id == j.session_id_re).all()


def test_spike_derived_columns_match_reanalysis(joined):
    _, j = joined
    same_trials = j.n_trials == j.p10_n
    assert same_trials.mean() > 0.99
    s = j[same_trials]
    # evoked spike counts in [1, 9) ms (master stores the rate, rounded to 3 decimals)
    ne = s.evoked_rate * 0.008 * s.n_trials
    assert (np.abs(ne - s.p10_ne) < 0.5).mean() > 0.999
    # fraction of trials with >= 1 evoked spike (master rounds to 3 decimals)
    assert (np.abs(s.trial_reliability - s.p10_rel_raw) <= 0.0005 + 1e-9).mean() > 0.999
    # median first-spike latency, ms
    both = s.median_latency_ms.notna() & s.p10_median_first_ms.notna()
    assert (s.median_latency_ms.isna() == s.p10_median_first_ms.isna()).mean() > 0.999
    assert (np.abs(s.median_latency_ms[both] - s.p10_median_first_ms[both]) <= 0.0005 + 1e-9).mean() > 0.999


def test_modulation_ratio_and_label_follow_from_measured_columns(joined):
    m, _ = joined
    mod = m.evoked_rate / (m.baseline_rate + 1.0)
    # rates and ratio are stored rounded to 3 decimals
    assert np.allclose(mod, m.modulation_ratio, rtol=1e-3, atol=2e-3)
    relabelled = label_unit_features_table(m.drop(columns=["reference_class"]))
    assert (relabelled.reference_class.values == m.reference_class.values).all()


def test_manifest_identities_match_allen_sessions():
    man = pd.read_csv(MANIFEST)
    allen = pd.read_csv(ALLEN_SESSIONS).rename(columns={"id": "session_id"})
    j = man.merge(allen, on="session_id", suffixes=("", "_allen"))
    assert len(man) == 28 and len(j) == 28
    assert (j.specimen_id == j.specimen_id_allen).all()
    assert (j.genotype == j.genotype_allen).all()
    assert (j.cre_line == j.genotype_allen.str.split("/").str[0]).all()
    assert (j.sex == j.sex_allen).all() and (j.age_in_days == j.age_in_days_allen).all()
