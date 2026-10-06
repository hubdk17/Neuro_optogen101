# Calibration and critique of optotagging labels: results

**Scope:** the brief's §4A–§4D, executed on spike times from all 28 Allen Visual Coding Neuropixels Pvalb/Sst/Vip × Ai32 sessions (19,005 units).

**Provenance:** everything here comes from one logged run, `make env calibration` after `make infer population figures regression`, on derived records produced by `scripts/reanalysis/01_extract_cohort.py` (extractor v2). Logs are in `results/calibration/run_log/`; tables in `results/calibration/tables/`; figures C1–C4 in `results/calibration/figures/`.

**Inference:** between-group claims use the animal as the unit of replication. GEE models cluster on animal. With 6 (PV) or 11 (SST) clusters, GEE robust standard errors can be anti-conservative, so animal-level paired tests are reported next to every GEE result, and the animal-level result is the one to quote.

**Pre-registration:** §4A was pre-registered in `reports/PREREGISTRATION_4A.md`, committed in `a8e0d2f` before the extended reference set existed. Deviations are listed in §6.

---

## 0. Summary

| Question | Result | Supports the thesis? |
|---|---|---|
| Is the CCG inhibition probe valid? (§4A positive control) | **Yes, for PV and SST.** Driven units inhibit more partners than rate- and position-matched non-driven units: PV 7.8% vs 1.9%, animal-level +0.047 [0.012, 0.081], p = 0.018 (6 animals); SST 3.8% vs 1.6%, +0.014 [0.003, 0.025], p = 0.017 (11 animals). VIP: no difference. | Enables the test |
| Does δ̂ < 5 ms & σ̂ < 1.5 ms ("direct-like") identify units with inhibitory output? | **No.** OR 1.02 [0.68, 1.51], p = 0.93; animal-level difference −0.011 [−0.036, +0.013], p = 0.31 (10 animals). Minimum detectable difference at 80% power: 0.034 (animal level), 0.040 (unit level), on a 5.9% base rate. | **Yes**, for effects ≥ ~0.035; smaller ones can't be excluded |
| What does predict inhibitory output? | **Narrow waveform** (GEE OR 0.34 per SD; within-animal ρ = −0.27, p = 0.0005) and **dose slope** (OR 1.35, Holm p = 0.001; within-animal ρ = +0.17, p = 0.012). **Not ρ̂** (OR 1.09, Holm p = 0.24) and not train following. | Partly: the brief's ρ̂ association does not survive animal clustering |
| Is part of the "network" response not opsin-mediated? (§4B) | **Likely yes.** Thalamic suppression after a light pulse is Cre-independent (PV 26%, SST 23%, VIP 27%; Kruskal p = 0.80). In VIP mice, which have almost no light-activated cells, 44% of LGd units are suppressed and 28% excited. Opto- and flash-suppression co-occur beyond chance after adjusting for firing rate (OR 2.1–3.3). But opto suppression is **faster** than flash suppression (LGd 27.5 vs 57.5 ms, LP 17.5 vs 62.5 ms), so it is not a copy of the flash response. | Supports a confound; mechanism unresolved |
| Are the field's criteria miscalibrated? (§4C) | **Mixed.** Latency < 8 ms alone is useless (71% of sham trials pass). The raw-pulse reading of the ≥ 4/5-pulse criterion has implied FDR 74%. But the **standard operational heuristic is well calibrated**: implied FDR 1.9%, precision 0.93. It is conservative, missing 29% of light-activated units. | **Only partly.** The headline "labels are miscalibrated" does not hold for the standard heuristic |
| Does label noise propagate into a cell-type classifier? | **Yes, for noisy criteria.** Coefficients shrink towards zero and performance falls to chance (latency < 8 ms, ZETA, SALT). But a classifier trained on **heuristic** labels does at least as well on calibrated test units (AUC 0.685) as one trained on calibrated labels (0.599). | **Partly / against** for the standard heuristic |

**Bottom line:** two results are solid and new:
1. A latency/jitter null against a validated connectivity probe.
2. A Cre-independent, partly visual-like response to the light pulse.

The broad claim that "optotag labels are miscalibrated" is supported only for weak or loosely specified criteria, not for the standard conjunctive heuristic. See §7 for the recommendation.

---

## 1. Regression baseline (§2 of the brief)

`scripts/calibration/00_regression_baseline.py` recomputes every baseline quantity from the committed tables and exits non-zero on any mismatch. It was run twice: on the committed tables, and again after re-running inference from the v2 records. **All 39 checks pass both times** (`results/calibration/tables/regression_baseline.csv`).

| Quantity | Expected | Recomputed |
|---|---|---|
| units / sessions / animals | 19,005 / 28 / 8-12-8 | same |
| max \|exact p − stored\|, π₀ = 0.008/0.468 | ≤ 1e-14 | 3.2e-15 (the wrong 480-ms constant would give up to 0.141) |
| exact BH q < 0.05 / Bonferroni | 396 / 304 | 396 / 304 |
| lfdr < 0.05 / onset artifacts / driven | 375 / 25 / 350 | same |
| 5-ms replication driven vs null; Spearman | 0.92 vs 0.035; 0.84 | 0.920 vs 0.0349; 0.841 |
| dose high − low ρ̂ | +0.41, t(23) = 11.5, p = 5.2e-11 | +0.407, 11.49, 5.2e-11 |
| 10-Hz train log-ratio | −0.05, t(23) = −0.70, p = 0.49 | −0.052, −0.702, 0.490 (see note) |
| narrow-waveform OR PV/SST/VIP | 17.8 / 3.97 / 0.81 | 17.79 / 3.97 / 0.81 |
| yield per animal | 3.94 ± 1.41 / 1.69 ± 0.54 / 0.46 ± 0.20%; Kruskal p = 0.10 | same; Kruskal p = 0.0996 |
| CCG inhibition causal vs anticausal | PV 7.74/2.54, SST 3.90/1.68, VIP 1.05/1.05% | same |
| direct-like | 85/350 = 24.3% | 85, 24.3% |

Notes on the baseline:
* **Train log-ratio definition.** Your value is the per-animal *pooled-count* log ratio, log((Σk₁₀ + ½)/(Σk₁ + ½)). The stored table uses the mean of per-unit log ratios: −0.032, t = −0.53, p = 0.60. Both are null.
* **Yield fragility.** Your leave-one-out range (0.032–0.169) is the Kruskal–Wallis range. With the permutation statistic it is 0.006–0.073. The yield difference is a trend under either.
* **Extraction v2 reproduces v1 exactly.** Aligned spikes, spontaneous counts, CCG counts and valid-epoch masks are identical in all 28 sessions (`extraction_v2_vs_v1_check.csv`).

---

## 2. §4A — Does latency/jitter identify directly activated cells?

**Setup.** As pre-registered. The outcome is each reference unit's *source rate*: the fraction of same-probe partners within 300 µm (both units ≥ 500 spontaneous-epoch spikes) that show a causal short-latency CCG trough. The CCG machinery is unchanged from v1 and reproduces the v1 driven pairs exactly.

### 2.1 Positive control — PASSES for PV and SST (`4A_positive_control.csv`, Fig. C2A)

Pre-registered matching, strict tier:

| | driven rate | matched control rate | GEE OR [95% CI] | animal-level paired diff [95% CI] | t p | Wilcoxon p | animals |
|---|---|---|---|---|---|---|---|
| PV | 0.078 | 0.019 | 4.68 [4.02, 5.46] | +0.047 [+0.012, +0.081] | 0.018 | 0.063 | 6 |
| SST | 0.038 | 0.016 | 2.11 [1.36, 3.25] | +0.014 [+0.003, +0.025] | 0.017 | 0.020 | 11 |
| VIP | 0.010 | 0.008 | 1.18 [0.29, 4.87] | +0.006 [−0.022, +0.034] | 0.59 | 0.63 | 5 |

* **The pre-registered decision rule is met:** PV is significant, so the δ/σ test below is informative.
* **The PV animal-level evidence rests on 6 animals.** With n = 6 the smallest possible two-sided Wilcoxon p is 0.031, and the observed one is 0.063. Treat PV as consistent across animals rather than overwhelming.
* **Matching coverage was incomplete.** 209 of the 344 driven units with valid CCGs (350 driven) got ≥ 1 strict control; 446 strict controls were drawn in total. The unmatched driven units have higher spontaneous rates (PV median 11.3 vs 7.9 Hz) and are more often narrow (PV 90% vs 76%). Their source rates are similar to matched units, though (PV 7.3% vs 8.1%; SST 4.8% vs 4.7%), so the restriction does not appear to bias the comparison (`4A_matching_selection.csv`).
* **Relaxed-matching sensitivity (a deviation; rate ±50%, position ±200 µm, 209 more controls):** PV is unchanged (OR 4.68; animal p = 0.024). The SST animal-level difference shrinks to +0.005 [−0.007, +0.017], p = 0.36. **The SST positive control is therefore fragile; the PV one is robust.**

### 2.2 The latency/jitter test — NULL, with a validated probe (`4A_delta_sigma_tests.csv`, Fig. C2B)

| | direct-like | indirect-like | test |
|---|---|---|---|
| PV source rate | 0.071 (n = 39) | 0.079 (n = 140) | GEE OR 0.92 [0.74, 1.14], p = 0.46 |
| SST source rate | 0.047 (n = 34) | 0.047 (n = 110) | GEE OR 1.17 [0.50, 2.75], p = 0.72 |
| PV + SST (Cre covariate) | | | GEE OR 1.02 [0.68, 1.51], p = 0.93 |
| continuous δ̂ (per SD) | | | OR 1.13 [0.88, 1.45], p = 0.35 |
| continuous σ̂ (per SD) | | | OR 0.96 [0.82, 1.12], p = 0.61 |
| animal-level paired (10 animals with both classes) | | | −0.011 [−0.036, +0.013], t p = 0.31, Wilcoxon p = 0.63 |

**Power** (`4A_power_curve.csv`, Fig. C2C), by beta-binomial simulation at the observed base rate (0.059), between-unit dispersion (ICC 0.094) and the actual group sizes (73 vs 250 units):
* unit-level minimum detectable difference at 80% power: **0.040** (power 0.33 at 0.02, 0.67 at 0.03);
* animal-level minimum detectable difference: **0.034**.

The animal-level CI rules out direct-like units having a source rate more than **0.013** above indirect-like units, about +22% relative to the base rate. The test can therefore exclude a large advantage for "direct-like" units (≥ 0.035, about +60%). It cannot exclude a small one.

Your §4A numbers reproduce: PV 0.071 vs 0.079, SST 0.047 vs 0.047, animal-level −0.011 with Wilcoxon p = 0.625.

### 2.3 Alternative predictors (pre-registered, Holm over four) (`4A_alternative_predictors*.csv`)

| predictor | GEE OR per SD [95% CI] | p | Holm p | unit Spearman | within-animal Spearman, mean [95% CI] (post hoc) | animal p |
|---|---|---|---|---|---|---|
| ρ̂ (chance-corrected reliability) | 1.09 [0.96, 1.23] | 0.16 | 0.24 | +0.20 (p = 0.0003) | +0.08 [−0.04, +0.21] | 0.17 |
| dose slope (high − low ρ̂) | **1.35 [1.14, 1.59]** | 0.0004 | **0.0011** | +0.21 (p = 0.0002) | **+0.17 [+0.05, +0.29]** | 0.012 |
| train following | 0.89 [0.76, 1.03] | 0.12 | 0.24 | +0.05 (p = 0.41) | +0.04 [−0.06, +0.15] | 0.38 |
| waveform duration | **0.34 [0.27, 0.41]** | < 1e-4 | **< 1e-4** | −0.41 (p < 1e-4) | **−0.27 [−0.39, −0.16]** | 0.0005 |

**Correction to the brief.** The brief reported ρ̂ vs inhibition as Spearman +0.179, p = 0.0012, and treated it as the predictor that worked. On this reference set the unit-level correlation is similar (+0.20). But it **does not survive clustering by animal** (GEE Holm p = 0.24; within-animal p = 0.17). It reflects between-animal differences: animals with more reliably driven units also have more inhibitory pairs.

What does predict inhibitory output, within animals, is **narrow waveform**, a cell-type signal that is not part of any tagging criterion, and **dose slope**, i.e. how strongly the response grows with light level.

### 2.4 Direction: sources vs targets (`4A_direction.csv`)

* Driven PV and SST units are mainly **sources**: within driven units, source rate exceeds target rate by OR 3.23 [2.67, 3.91] for PV and 2.42 [1.91, 3.06] for SST.
* Their target rates do **not** differ from matched controls: PV OR 1.01 [0.83, 1.22]; SST 0.74 [0.47, 1.15].
* The pre-registered prediction that indirectly (synaptically) driven units are more often targets **was not supported, and the estimate goes the other way**. Direct-like units have a *higher* target rate (OR 1.42 [1.03, 1.96], p = 0.032). This is a single GEE with 17 clusters and no animal-level replication, so it should be read as "no support for the prediction", not as a positive finding.

---

## 3. §4B — The non-opsin confound

Per unit, suppression and excitation 20–50 / 50–200 ms after 10-ms light pulses and after full-field flashes (150 per session, both polarities) use the same exact test and baseline. "Suppressed" means BH q < 0.05 over units, with Bonferroni over the two windows, excluding driven units (`4B_*.csv`, Fig. C3).

1. **Thalamic suppression is Cre-independent; cortical suppression is not** (animal-level; `4B_cre_dependence_by_region.csv`):

   | region | PV | SST | VIP | Kruskal p | permutation p |
   |---|---|---|---|---|---|
   | visual cortex | 0.40 | 0.35 | 0.20 | 0.024 | 0.035 |
   | thalamus | 0.26 | 0.23 | 0.27 | 0.80 | 0.91 |
   | midbrain | 0.23 | 0.18 | 0.06 | 0.35 | 0.27 |
   | hippocampal formation | 0.04 | 0.08 | 0.03 | 0.89 | 0.52 |

2. **LGd and LP respond to the light pulse in VIP mice**, which have almost no light-activated units (`4B_LGd_LP_by_cre.csv`). LGd suppressed after a light pulse: VIP 0.44 ± 0.13 (5 animals), SST 0.44 ± 0.05 (4), PV 0.27 ± 0.10 (3). LGd excited: VIP 0.28 ± 0.10, SST 0.18 ± 0.13, PV 0.11 ± 0.06. LGd is sampled in only 3–5 animals per line.

3. **Overlap with the visual flash response, beyond detectability** (`4B_overlap_rate_adjusted.csv`). Among opto-suppressed units, 29.5% are also flash-suppressed, against 10.4% of other units. Adjusted for baseline rate (GEE, animal clusters), the overlap OR is:

   | region | overlap OR [95% CI] |
   |---|---|
   | visual cortex | 2.12 [1.76, 2.55] |
   | thalamus | 2.61 [2.07, 3.28] |
   | midbrain | 3.29 [2.32, 4.67] |
   | hippocampal formation | 1.58 [0.84, 2.99] |

   The share of opto-suppressed units that are also flash-suppressed is 32% in visual cortex, 32% in thalamus and 27% in midbrain.

4. **Timing differs** (`4B_onset_opto_vs_flash.csv`, Fig. C3C). The population onset of suppression (rate < 0.8 × baseline for ≥ 3 bins, per animal) comes **earlier** after the light pulse than after the flash:

   | group | opto onset (ms) | flash onset (ms) | Wilcoxon p | animals |
   |---|---|---|---|---|
   | LGd | 27.5 | 57.5 | 0.020 | 11 |
   | LP | 17.5 | 62.5 | 0.007 | 11 |
   | visual cortex | 12.5 | 52.5 | 1e-4 | 28 |

   The flash response in LGd is excitation at about 40 ms, then suppression, then an offset response after the 250-ms flash. The opto response is suppression from about 20–30 ms with no preceding excitation in the population of suppressed units.

**Interpretation.** A Cre-independent thalamic response, present in VIP mice, is not what direct opsin activation predicts. The above-chance overlap with flash suppression suggests a shared, plausibly visual or arousal-related, component. But the light-pulse response is faster than, and shaped differently from, the response to a full-field flash, so "it is just a visual response to the blue light" is **not established** either. The flash is a much brighter, longer, full-field stimulus, so differences in timing are expected even if both reach the retina.

**Limitation (not removable here):** this dataset has no opsin-negative animals, no masking light and no eye occlusion. The non-opsin component can be characterised but **not removed or attributed**. Network-suppression results should be restricted to visual cortex, where suppression is Cre-dependent, and even there a Cre-independent floor of roughly 20% (the VIP level) may not be opsin-mediated.

---

## 4. §4C — Criterion calibration

`scripts/calibration/04_criteria_calibration.py` evaluates every criterion through one code path on:
* **real trials:** the 10-ms pulse trials and the 10-Hz train trials;
* **sham trials:** the same number of pseudo-trials at random onsets inside each unit's valid spontaneous epochs (no light), with identical windows and train geometry.

The recomputed real-trial exact-test and SALT p-values match the stored ones exactly (max difference 0). The calibrated reference uses positives = driven (350) and negatives = lfdr > 0.5 (18,496). CIs come from 2,000 cluster-bootstrap resamples over animals (`4C_criteria_calibration.csv`, Fig. C1).

| criterion | passes (real) | passes (sham) | sham FPR [95% CI] | implied FDR | sensitivity [95% CI] | specificity | precision [95% CI] |
|---|---|---|---|---|---|---|---|
| Allen/Lakunina: ≥ 4/5 train pulses significant + rel ≥ 0.30 + lat < 8 ms | 126 | 6 | 0.0003 [0, 0.0007] | 0.048 | 0.31 [0.21, 0.39] | 0.9994 | 0.85 [0.68, 0.93] |
| Allen/Lakunina, raw per-pulse prob ≥ 0.30 at ≥ 4/5 | 235 | 174 | 0.0092 [0.0073, 0.0110] | **0.74** | 0.15 [0.09, 0.21] | 0.990 | 0.23 [0.09, 0.37] |
| latency < 8 ms alone | 13,472 | 13,418 | **0.71** [0.68, 0.73] | **1.00** | 0.99 | 0.30 | **0.03** |
| reliability ≥ 0.3 & modulation > 2 | 266 | 5 | 0.0003 | 0.019 | 0.71 [0.64, 0.75] | 0.9997 | 0.93 [0.84, 0.97] |
| original operational heuristic (full) | 265 | 5 | 0.0003 | 0.019 | 0.71 [0.64, 0.75] | 0.9998 | 0.93 [0.85, 0.97] |
| SALT p < 0.01 | 454 | 47 | 0.0025 | 0.10 | 0.95 | 0.998 | 0.73 [0.61, 0.80] |
| SALT BH q < 0.05 | 381 | 0 | 0 | 0 | 0.89 | 0.999 | 0.81 [0.70, 0.88] |
| ZETA [1, 9) ms BH | 17 | 0 | 0 | 0 | **0.04** | 1.000 | 0.88 |
| ZETA [1, 51) ms BH | 1,085 | 0 | 0 | 0 | 0.93 | 0.963 | **0.30** [0.20, 0.39] |
| exact test BH* | 396 | 0 | 0 | 0 | 1.00 | 1.000 | 0.88 |
| lfdr < 0.05* | 375 | 2 | 0.0001 | 0.005 | 1.00 | 1.000 | 0.93 |

\* defines or neighbours the reference set; sensitivity and precision are partly circular. Implied FDR = sham FPR × N / passes(real).

Findings:
* **Latency < 8 ms carries no information.** 71% of units pass on sham trials. Among null units that fire in the window, 96.7% pass on real trials and 95.8% on sham trials.
* **The ≥ 4/5-pulse criterion's calibration depends on how it is operationalised.** Requiring a significant per-pulse response gives implied FDR 4.8% but only 31% sensitivity: the 2.5-ms train pulses drive these units weakly, with per-pulse spike probability 0.17–0.26. A raw-probability reading gives implied FDR 74%. Published descriptions do not say which is meant, and that ambiguity is itself a calibration problem.
* **The standard conjunctive heuristic is well calibrated on false positives** (implied FDR 1.9%, precision 0.93) but misses 29% of light-activated units. Its failure mode is sensitivity, not false positives.
* **ZETA p-values carry Monte Carlo noise.** zetapy resamples (250 here). Independently seeded runs give 15 vs 17 positives for [1, 9) ms and 1,091 vs 1,085 for [1, 51) ms (`results/reanalysis/tables/method_comparison.csv` vs this table), so counts near the BH threshold move by about 1%.
* **ZETA detects responsiveness, not tagging.** Over [1, 51) ms it has zero sham false positives but precision 0.30 against light activation, because it flags network-modulated units. Over [1, 9) ms it has almost no power.
* **The lfdr procedure itself** has 2 sham false positives of 19,005 (FPR 0.0001), which validates the empirical-null construction.

### 4.1 Label-noise propagation (`4C_label_noise_*.csv`, Fig. C4)

**Setup.** One model (L2 logistic regression; waveform duration, half-width, PT ratio, log firing rate) is trained per label source to predict PV vs non-PV session, with GroupKFold held out by session. Each model is scored on its own held-out labels and on a common test set of calibrated light-activated units from the held-out sessions.

| label source | training units (PV share) | AUC own labels | AUC on calibrated units | waveform-duration coefficient [95% CI] |
|---|---|---|---|---|
| calibrated (lfdr < 0.05) | 348 (0.52) | 0.60 | 0.60 | −0.52 [−1.77, −0.07] |
| operational heuristic | 264 (0.50) | 0.72 | **0.685** | −1.33 [−2.87, −0.24] |
| rel ≥ 0.3 & mod > 2 | 265 (0.50) | 0.72 | 0.687 | −1.34 [−2.88, −0.30] |
| Allen/Lakunina | 124 (0.40) | 0.58 | 0.63 | −0.96 [−2.53, −0.16] |
| SALT p < 0.01 | 450 (0.45) | 0.52 | 0.53 | −0.39 [−1.55, +0.06] |
| ZETA [1, 51) ms | 1,077 (0.35) | (0.23)† | (0.36)† | −0.16 [−0.49, +0.07] |
| latency < 8 ms | 13,343 (0.24) | (0.13)† | (0.35)† | −0.05 [−0.21, +0.10] |

† AUC < 0.5 with uninformative labels is an artefact of session-held-out cross-validation (class balance shifts between folds), not anti-learning. Read these as "no signal".

* **Noisy label sources propagate.** Training on latency < 8 ms or ZETA [1, 51) ms labels shrinks the waveform coefficient towards zero (−0.05 and −0.16, vs about −0.5 to −1.3 for strict sources), and the classifier carries no information about cell type.
* **Calibrated labels did not train a better classifier than the heuristic.** The heuristic and rel/mod sets are stricter subsets (246 of 264 heuristic units are also in the calibrated set) enriched for strongly driven, narrow-spiking units. The calibrated set adds weakly or indirectly driven units whose waveforms are less cell-type specific. A calibrated *activation* label is not the same as a clean *cell-type* label. That is consistent with §2: light activation is not identity.
* **Scale.** All PV-vs-non-PV AUCs are modest (≤ 0.72) with these four features, and the CIs on coefficients are wide (24 sessions). This is an illustration of label-source sensitivity, **not** evidence that any published classifier (Beau et al. 2025, PhysMAP, HIPPIE) is wrong; those use richer features and other labelling pipelines.

---

## 5. §4D — Resource table

`results/calibration/resource/optotagging_units.{csv,parquet}` has 19,005 units × 147 columns:
* identity and anatomy, waveform metrics and spontaneous rate;
* all `p10_*` and `p5_*` counts and statistics, including latency-model fits with standard errors;
* dose (per-level ρ̂ and latency) and train (per-pulse) responses;
* seeded ZETA, lfdr and evidence, the onset-artifact flag and the driven label;
* every criterion's label on real trials **and** on sham trials;
* opto and flash modulation;
* CCG role with source and target rates.

`DATA_DICTIONARY.md` (and `.csv`) gives the units and definition of every column. The generator refuses to export a column without a dictionary entry.

---

## 6. Nulls, failures and deviations

**Analyses that returned a null or failed their hypothesis:**
1. **The latency/jitter "direct-like" criterion does not predict monosynaptic inhibitory output** (§2.2), with a validated probe and stated power. Differences below about 0.035 in source rate cannot be excluded.
2. **ρ̂ does not predict inhibitory output once animal is accounted for** (§2.3). This contradicts the brief's unit-level result.
3. **Train following does not predict inhibitory output** (§2.3).
4. **The direction prediction failed.** Indirect-like units are not more often targets of inhibition; the estimate goes the other way (§2.4).
5. **The VIP positive control is null** (§2.1). VIP-activated units show no excess inhibitory output, consistent with VIP biology and with few VIP units (21).
6. **The SST positive control does not survive the relaxed-matching sensitivity analysis** (§2.1).
7. **The non-opsin component cannot be attributed.** Opto suppression is not a copy of the flash response: it is faster and differently shaped (§3).
8. **The standard operational heuristic is not miscalibrated on false positives** (§4). The calibration thesis holds only for weak or loosely specified criteria.
9. **Calibrated labels did not train a better waveform classifier than heuristic labels** (§4.1).
10. **ZETA over the direct window [1, 9) ms has almost no power** (sensitivity 0.04).
11. **The Cre-line difference in light-activated yield remains a trend:** Kruskal p = 0.10; six animals supply 69% of driven units and four have none (`per_animal_driven_contributions.csv`).

**Deviations from the pre-registration (all declared, none changes a primary conclusion):**
* **Relaxed-matching tier:** added because 139 of 350 driven units had no strict control. It is reported only as a sensitivity analysis.
* **Post-hoc animal-level robustness for the alternative predictors:** added because GEE standard errors are unreliable with 6–17 clusters. It changes no conclusion; it confirms that ρ̂ is null and that waveform and dose slope hold.
* **A bug corrected before reporting.** The first run of `02_latency_validity.py` reported the Cre covariate instead of the predictor for models with `C(cre)`. It was fixed before these results were written; the per-Cre models were unaffected.
* **Power analysis test:** it uses a Welch t-test on per-unit rates under a beta-binomial model rather than refitting the GEE in every simulation, for speed.

**Other caveats:**
* Driven units are concentrated: 6 of 28 animals supply 69%.
* The latency-model Wald CIs under-cover for weak responses (estimator validation, `results/reanalysis/tables/estimator_validation.csv`).
* The Allen/Lakunina criterion was operationalised from its published description; a different reading changes its FDR from 5% to 74%.

---

## 7. Recommendation: is the thesis strong enough for a high-tier methods venue?

**Not as framed.** The brief's thesis has two parts.

1. **"The standard direct-vs-indirect discriminator (latency/jitter) fails independent validation."** This is **supported**. The connectivity probe is validated against matched non-driven controls (robustly for PV, less so for SST). The null is tight enough to rule out a large advantage for "direct-like" units. The positive alternatives (waveform, dose slope) are coherent with biology. It is a clean, reproducible negative result. Its limits are real, though:
   * one dataset;
   * PV and SST only;
   * 6 PV animals;
   * an outcome (spontaneous-epoch CCG troughs) that measures inhibitory *output*, a consequence of being a PV/SST cell, rather than direct opsin activation itself.

   Without ground truth for direct activation (in-vitro, juxtacellular or collision tests), the claim is "latency/jitter does not predict the expected functional signature of the tagged cell type", not "latency/jitter cannot identify direct activation".

2. **"Optotagging labels are miscalibrated."** This is **not supported for the standard conjunctive heuristic**, which has an implied FDR of about 2% and precision 0.93 here. It is supported for single criteria (latency alone), for loose readings of pulse-following criteria, and for responsiveness tests used as tags (ZETA over 50 ms). The label-noise analysis shows that noisy label sources destroy a downstream classifier. It also shows that the heuristic is, if anything, a better training label for cell type than a broader calibrated activation label. The strongest version of the calibration story is therefore "light activation ≠ cell identity, and loose criteria are dangerous", not "the field's labels are wrong".

The **non-opsin confound** (§3) is a genuine and useful observation: Cre-independent thalamic responses, present in VIP mice. But its mechanism cannot be resolved without controls this dataset lacks.

**Venue.**
* **eLife (Research Article):** a stretch unless the claims are reframed and ideally replicated in a second dataset with stronger ground truth.
* ***Nature Methods* Brief Communication:** **not recommended.** It would need a single crisp, general, externally validated finding. The calibration headline did not materialise for the standard criterion, and the latency/jitter null rests on an indirect validity probe in one cohort.
* ***J. Neurosci. Methods*, eNeuro (Methods/New Research) or *Neurons, Behavior, Data analysis, and Theory*:** a good fit **as currently supported**. A paper there would combine:
  * the measured calibration of the criteria the field uses, including the operationalisation sensitivity of pulse-following criteria and the uselessness of latency < 8 ms;
  * a validated negative result for latency/jitter as a direct/indirect discriminator;
  * the non-opsin confound with its limitation stated;
  * a calibrated, reproducible pipeline (sham-window empirical null, exact tests, lfdr evidence) with the 19,005-unit resource table.

**What would upgrade it:**
1. Replication of §4A in an independent optotagging dataset with collision tests or juxtacellular ground truth. The Neuropixels Opto data (Lakunina et al.) would be the obvious candidate if public.
2. An opsin-negative or masking-light control for §4B.
3. A larger PV cohort, to tighten the animal-level positive control (currently 6 animals).

**Do not claim:**
* that the standard heuristic is miscalibrated;
* that the non-opsin component is visual;
* that any published classifier is wrong;
* a Cre-line difference in yield.
