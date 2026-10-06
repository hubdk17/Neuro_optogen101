# Calibrating optotagging labels in the Allen Visual Coding Neuropixels cohort

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reanalysis, from raw spike times, of optogenetic identification ("optotagging") in 28 Pvalb-, Sst- and Vip-IRES-Cre × Ai32 mice from the Allen Brain Observatory Visual Coding Neuropixels dataset. The project asks how well the operational criteria used to label optotagged units are calibrated, and what those labels support downstream.

> **`archive/` contains superseded material that was NOT computed from data** (random draws, hand-written rules, hard-coded constants), including the earlier manuscript draft. It is kept only for provenance; nothing in the current pipeline uses it, and a test enforces that. See [`archive/README.md`](archive/README.md).

## Authors

**To be completed by the authors.** Real names, affiliations, ORCIDs and CRediT contributions are required before any submission. They could not be filled in here and must not be invented. The previous placeholder ("Computational Neurophysiology and Bioinformatics Consortium") is not an author list.

## What the pipeline computes

Every number is computed from spike times in the Allen NWB files.

| Stage | Script | What it does |
|---|---|---|
| 0 | `scripts/reanalysis/00_validate_estimators.py` | Checks every estimator by simulation with known truth: false-positive rates, power, bias and CI coverage, FDR calibration, mixture recovery, CCG test calibration. |
| 1 | `scripts/reanalysis/01_extract_cohort.py` | Streams each of the 28 sessions from the public S3 bucket, reduces it with `src/reanalysis/extract_nwb.py` and deletes the NWB. Each derived record holds: aligned opto spikes, spontaneous-epoch rates and spike trains, flash-aligned spikes, reference ZETA and spontaneous CCGs. |
| 2 | `scripts/reanalysis/02_unit_inference.py` | Per unit, for 10-ms and 5-ms pulses: exact conditional Poisson test against a 460-ms same-trial baseline; sham-window statistics; chance-corrected reliability per light level; ML fit of a first-spike model (ρ, δ, σ); SALT (port of Kvitsiani et al. 2013); seeded reference ZETA; window counts to 500 ms; per-pulse 10-Hz train responses. |
| 3 | `scripts/reanalysis/03_population_analysis.py` | Local FDR with an empirical null from sham windows; onset-artifact exclusion; method comparison; 5-ms replication; waveform check; animal-level yield and suppression; multinomial-mixture response profiles; dose and train models; depth models; CCG connectivity; variance decomposition. |
| 4 | `scripts/reanalysis/04_figures.py` | Figures R1–R6 from the tables only. |
| C0 | `scripts/calibration/00_regression_baseline.py` | Regression check of the committed results; exits non-zero on any mismatch. |
| C1+ | `scripts/calibration/` | Calibration of the field's tagging criteria, validity of the latency/jitter discriminator, the non-opsin (visual) confound, label-noise propagation, and the per-unit resource table. See `reports/CALIBRATION_RESULTS.md`. |

Statistical choices: the animal (n = 28) is the unit of replication for every between-group comparison. Unit-pooled tests overstate precision 3–5× here (ICC 0.035 for light activation, design effect ≈ 25).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pinned; Python 3.11
make all                               # extraction (~60 GB transfer, ~1.5 h) + all analyses
make test                              # unit tests incl. archive isolation
```

`make all` runs stages 0–4, the regression check and the calibration analyses in order, and writes package versions and a run log to `results/calibration/run_log/`. Extraction needs about 4 GB of free disk at peak (one NWB plus the derived records in `data/derived/`, which are git-ignored and regenerated).

## Outputs

| Path | Contents |
|---|---|
| `results/reanalysis/tables/unit_results.parquet` | Per-unit statistics for all 19,005 units (input to all calibration analyses). |
| `results/reanalysis/tables/*.csv` | Method comparison, latency fits, waveform check, yield and suppression per animal, archetypes, dose, trains, depth, CCGs, sparse-tier Poisson check, variance decomposition, estimator validation. |
| `results/reanalysis/summary.json` | Headline numbers of the reanalysis. |
| `results/reanalysis/figures/` | Figures R1–R6. |
| `results/calibration/` | Calibration and critique analyses, the regression baseline, the resource table with its data dictionary, run logs. |
| `reports/REANALYSIS_RESULTS.md` | Reanalysis write-up. |
| `reports/CALIBRATION_RESULTS.md` | Calibration write-up, including every null or failed analysis. |
| `reports/ANALYSIS_REPORT.md` | Audit of the superseded analysis (now in `archive/`). |
| `reports/PREREGISTRATION_4A.md` | Pre-registered tests for the latency/jitter validity analysis. |

## Headline results (reanalysis)

* **Light activation is rare.** 350 of 19,005 units (1.8%) are light-activated at local FDR < 0.05 (estimated FDR 0.3%), after excluding 25 light-onset artifacts.
* **It replicates and has the expected waveforms.** 92% of these units replicate on independent 5-ms pulse trials, against 3.5% of null units. Narrow-spike enrichment is OR 17.8 for Pvalb and 3.97 for Sst, with none for Vip (0.81).
* **The latency criterion is uninformative.** Latency < 8 ms inside the [1, 9) ms window is passed by 96.7% of null units that fire in the window at all.
* **Light level drives reliability; trains don't adapt.** Activation probability rises with light level (Δρ̂ = +0.41, 24 animals, t(23) = 11.5, p = 5 × 10⁻¹¹). There is no measurable 10-Hz train adaptation (t(23) = −0.70, p = 0.49).
* **PV units show local inhibition.** PV-activated units show short-latency CCG troughs in 7.7% of partners, against 2.5% in the anticausal control.
* **Fragile:** the Cre-line difference in light-activated yield (Pvalb 3.9 ± 1.4%, Sst 1.7 ± 0.5%, Vip 0.46 ± 0.20% per animal) is a **trend**, not a result. The between-animal permutation test gives p = 0.024, but Kruskal–Wallis gives p = 0.10. Dropping one animal at a time moves the Kruskal–Wallis p between 0.03 and 0.17, and the permutation p between 0.006 and 0.07. Six of 28 animals supply 69% of light-activated units, and four have none.

## Headline results (calibration)

Full details, including every null result, are in [`reports/CALIBRATION_RESULTS.md`](reports/CALIBRATION_RESULTS.md).

* **Latency/jitter does not identify cells with inhibitory output.** The "direct-like" latency/jitter criterion (δ̂ < 5 ms, σ̂ < 1.5 ms) does not predict monosynaptic inhibitory output. The animal-level difference is −0.011 [−0.036, +0.013], p = 0.31, across 10 animals. The CCG probe itself is validated by a pre-registered positive control: driven vs matched non-driven units, PV +0.047 [0.012, 0.081], p = 0.018, 6 animals. Narrow waveform and dose slope do predict inhibitory output. Reliability ρ̂ does not, once animal is accounted for.
* **The standard criterion is well calibrated; others are not.** The standard conjunctive heuristic has an implied FDR of about 2% against sham trials, but misses 29% of light-activated units. Latency < 8 ms alone passes 71% of sham trials. A raw-probability reading of the "≥ 4 of 5 pulses" criterion has an implied FDR of 74%; a significance-based reading has 5%.
* **Part of the network response is not opsin-mediated.** Thalamic suppression after a light pulse is Cre-independent (VIP mice included). It overlaps with the visual-flash response beyond chance, but it is faster. Without opsin-negative controls its origin cannot be resolved.

## Data availability

The data are the Allen Brain Observatory **Visual Coding – Neuropixels** dataset (Siegle et al., 2021), publicly available from the Allen Institute (https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels). The NWB files are read directly from `s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/session_<id>/session_<id>.nwb`. AllenSDK is not required; the NWB files are read with h5py 3.16.0. The record of the original pipeline (`results/run_metadata.json`) lists AllenSDK 2.16.2.

The 28 sessions (8 Pvalb, 12 Sst, 8 Vip × Ai32; listed with specimen IDs in `results/cohort/full_28_specimen_manifest.csv`):

`715093703, 719161530, 721123822, 746083955, 751348571, 755434585, 756029989, 758798717, 760345702, 760693773, 762120172, 762602078, 773418906, 786091066, 787025148, 789848216, 791319847, 794812542, 797828357, 798911424, 816200189, 819701982, 829720705, 831882777, 835479236, 839068429, 839557629, 840012044`

Units are those passing the Allen default QC (isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9, quality = good): 19,005 units.

## Code availability

All code is in this repository under the MIT licence. **A Zenodo (or equivalent) DOI for the exact release must be minted at submission.**

## Ethics

This is a secondary analysis of publicly available data. No new animal experiments were performed. All animal procedures for the original recordings were approved by the Allen Institute Institutional Animal Care and Use Committee (Siegle et al., 2021).

## References

Siegle, J. H., Jia, X., Durand, S., et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature* 592, 86–92.

Kvitsiani, D., Ranade, S., Hangya, B., et al. (2013). Distinct behavioural and network functions of two interneuron classes in prefrontal cortex. *Nature* 498, 363–366.

Montijn, J. S., Seignette, K., Howlett, M. H., et al. (2021). A parameter-free statistical test for neuronal responsiveness. *eLife* 10, e71969.

## Legacy code

`src/*.py` (outside `src/reanalysis/`), `scripts/run_*ml*`, the leakage audits and related `results/` folders belong to an earlier machine-learning layer that predicts the operational label from the same features it was defined by. They are not part of the current pipeline and need `requirements-legacy.txt`. This layer has been audited line by line (see the second part of [`archive/README.md`](archive/README.md)):

* The per-unit features in `results/ml_final/master_ml_dataset_28spec.parquet` are computed from the NWB files. They match the independent re-extraction unit by unit (`tests/test_legacy_master_table.py`).
* Scripts and reports whose numbers were typed in, filled with constants, or relabelled were moved to `archive/`. These include the literal cross-validation table, the constant-filled sham controls, the `*_28spec` tables (2-specimen results relabelled), and the typed leakage-audit verdicts.
* The remaining legacy code computes its outputs. Its known defects are listed with file:line in `archive/README.md`.
