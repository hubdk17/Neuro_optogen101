# Pre-registration: §4A latency/jitter validity, positive control and alternative predictors

Committed **before** the extended (driven + matched-control) CCG reference set exists, and before any analysis below has been run on it. The brief already reported a δ/σ null and a ρ̂ association on the v1 reference set; those are known, and they are why each test here is specified in advance.

## Units, outcome, partners

* **Driven reference units:** `driven == True` in `results/reanalysis/tables/unit_results.parquet` (lfdr < 0.05, onset artifacts excluded; 350 units).
* **Matched non-driven controls (§3a):** for each driven unit, up to 3 units on the same probe that are:
  * clearly null (`lfdr > 0.5`, not an onset artifact, not driven);
  * within ±25% of the driven unit's `lam_spont_hz`;
  * within ±100 µm in `probe_vertical_position`.

  Units are sampled without replacement within a session, with seed 20261006. **There is no matching on waveform.** The matching keys (driven unit, control unit, both rates, both positions) are written to `results/calibration/tables/ccg_matched_controls.csv`.
* **Partners:** every unit on the same probe within 300 µm vertically, excluding the reference unit itself. A pair is used only if both units have ≥ 500 spontaneous-epoch spikes, as in v1.
* **CCG machinery:** unchanged from v1. Spontaneous epochs only; ±30 ms window; 0.5-ms bins; Stark–Abeles hollow-Gaussian predictor (σ = 10 ms, hollow 0.6); causal test bins +0.8 to +4 ms, anticausal control bins −4 to −0.8 ms; Bonferroni over the tested bins; significance at p < 0.01.
* **Outcome per reference unit (inhibitory output, "source rate"):** the fraction of its partners showing a significant causal trough.
* **Target rate:** the fraction of a unit's CCGs in which another unit inhibits it. That is, an anticausal trough when it is the reference, or a causal trough when it is the partner of another reference.

## Primary analyses

1. **Positive control.** Do driven reference units have a higher source rate than their matched non-driven controls? This is tested within PV and within SST.
   * Model: binomial GEE on partner-level counts per reference unit, clustered by animal (exchangeable), predictor driven vs control.
   * Animal-level check: per-animal mean difference (driven − control), one-sample t-test and Wilcoxon signed-rank, 95% CI.
   * **Decision rule:** if the driven − control difference is not significant (GEE p ≥ 0.05) in PV, the CCG probe is declared to lack demonstrated validity, and the δ/σ null below is reported as **uninformative**, not as evidence against latency/jitter.
2. **Latency/jitter test.** Among driven PV + SST units, direct-like (δ̂ < 5 ms and σ̂ < 1.5 ms) vs indirect-like source rate.
   * Model: binomial GEE clustered by animal, with Cre line as a covariate.
   * Animal-level paired difference with 95% CI, using animals that have both classes.
   * Continuous δ̂ and σ̂ as separate GEE predictors.
3. **Power.** The minimum detectable difference at 80% power and α = 0.05 for test 2, at the observed base rate and n, by simulation from a beta-binomial fitted to the observed between-unit dispersion. The animal-level MDD is also reported.

## Alternative predictors (secondary; Holm correction over these four)

Each is standardised and enters a binomial GEE (clustered by animal, Cre covariate) of source rate among driven PV + SST units. Unit-level Spearman correlations are also reported.

1. ρ̂: `p10_rho` (chance-corrected reliability, 10-ms pulses)
2. dose slope: `p10_high_rho − p10_low_rho`
3. train following: the mean over the 10 train pulses of the chance-corrected per-pulse response, (k_j/n − p₀)/(1 − p₀), with p₀ from `train_lam0_hz`
4. waveform duration: `waveform_duration` (ms)

## Direction (source / target) analysis

* For driven units and their matched controls, compare source rate and target rate (paired within unit).
* Test whether driven units differ from matched controls in target rate (GEE clustered by animal).
* Among driven units, test whether indirect-like units have a higher target rate than direct-like units. The prediction is that synaptically driven units are more often targets.

## Reporting commitments

* Every test above is reported whatever its outcome, with effect size, 95% CI and per-animal contributions (number of reference units and pairs per animal).
* Nulls and failures are listed in a dedicated section of `reports/CALIBRATION_RESULTS.md`.
* Any deviation from this plan is listed explicitly with its reason.
