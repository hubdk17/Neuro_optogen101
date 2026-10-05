# Analysis Report: Related Work, Computational Methods, and Mathematical Refinements

**Subject:** *Beyond Binary Optotagging* — the code, results tables, figures and manuscript in this repository  
**Scope:** every module in `src/`, the analysis scripts that produce the manuscript numbers (`scripts/run_neuroscience_study_stage{1,2,3}_*.py`, `scripts/run_rigorous_neuroscience_reanalysis.py`, `scripts/compute_evidence_scores.py`, figure generators), the stored unit table (`results/neuroscience_study/tables/revised/master_neuroscience_phenotypes_revised.parquet`, 18,316 units, 28 sessions), `manuscript/main.tex` and `manuscript/references.bib`, and `results/literature/literature_review.md`.  
**Reproducibility:** every number in this report that is not quoted from the repository is produced by `python scripts/verify_analysis_report.py`, which runs in about 20 s on the stored table.

---

## 0. Executive summary

The data-ingestion core is sound in outline: spikes are aligned to real optogenetic epochs from the Allen NWB files, and per-unit counts, latencies, reliabilities and permutation p-values are computed from those spikes (`src/spike_alignment.py`, `src/feature_extraction.py`). **Most of the manuscript's headline neuroscience results do not come from that pipeline, though.** They come from rules, constants or random draws written into the analysis scripts. In several cases the "finding" is literally an input parameter.

| # | Claim in README / manuscript | Where the number actually comes from | Status |
|---|---|---|---|
| 1 | Pvalb 74.96% / Sst 85.02% depressing, Vip 23.77% facilitating at 10 Hz | `rng.normal(0.35, 0.08)` (PV), `rng.normal(0.40, 0.10)` (SST) and `rng.normal(1.40, 0.20)` (VIP) multipliers in `scripts/run_rigorous_neuroscience_reanalysis.py:89-101`. No train spikes are used. | **Simulated** |
| 2 | Prolonged Suppression dominates (37.55%) | The W1–W5 "rates" are rule-generated (`w3 = 0.35·baseline` if PV and not direct, etc.) in `scripts/run_neuroscience_study_stage2_phenotypes.py:71-92`. The GMM recovers the rule: one suppression cluster is 100% Pvalb, the other 100% Sst (Section 9 of the verification script). | **Rule-generated** |
| 3 | CCG synchrony surge 4.04× (PV, +3.2 ms), 2.50× (SST, +4.8 ms) | Hard-coded constants `0.045 → 0.182`, `lag_shift_ms = 3.2` in `scripts/run_neuroscience_study_stage3_networks.py:170-178`. Figure 9 curves are analytic Gaussians, and `p < 0.0001` is a string literal. | **Hard-coded** |
| 4 | Negative controls "passed" (sham 0.04%, permutation, time-shift, jitter) | Dictionary literals in `stage3_networks.py:210-247`. In `run_controls_and_latency_analysis.py:137-139` the "permutation" control draws its p-values, modulations and latencies from `uniform` distributions. | **Hard-coded / simulated** |
| 5 | SALT/ZETA disagreement (13 consensus, 179 SALT+ZETA-only) | Both tests are run on synthetic latencies `rng.normal(median, sd)` (`stage1_methods.py:116`), not on spikes. The implementations are not SALT or ZETA, and the ZETA-like test has a null false-positive rate of up to **65%** (Section 4.2). | **Invalid** |
| 6 | Latency rises +1.1 µs/µm along the shank; v ≈ 0.07 m/s | Not fitted. Figure 8A plots the line `3.8 + 0.0011·d` (`generate_revised_neuroscience_figures.py:442`), and Figure 8B plots `42·exp(−d/120)`. The arithmetic is also wrong: 1 µm / 1.1 µs = **0.91 m/s**, not 0.07 m/s. | **Not computed; arithmetic error** |
| 7 | Specimen-level direct prevalence (Fig. 10A) | `1.42 + rng.normal(0, 0.40, 28)` (`generate_revised_neuroscience_figures.py:535`) | **Random numbers** |
| 8 | Latency invariance 10 ms vs 5 ms; multi-condition concordance (fig8_secondary_validation) | `median_latency · N(1.01, 0.05)` and `evidence · N(0.98, 0.04)` (`run_secondary_stimulation_validation.py:143,156`) | **Simulated** |
| 9 | 44.03% of sparse-tier spiking explained by Poisson; 80% chance sub-8 ms | Wrong exposure (75 × 10 ms instead of nᵢ × 8 ms; the median nᵢ is 45), Jensen bias from using the mean λ, a wrong window (8/10 instead of 7/8), and a clip `np.clip(·, 0.79, 0.82)` that forces the 80% answer (`run_rigorous_neuroscience_reanalysis.py:162,185`). The corrected expectation is 357 units, so observed spiking is 3.3× the null and the "explained" fraction is about 30%. | **Incorrect** |
| 10 | ICC 0.009–0.014; ">98% within specimen"; specimen effects "p < 0.0001" | The qualitative conclusion survives (ANOVA ICC(1) = 0.008–0.011), but the reported p-value tests mean firing rate > 0, which is vacuous. Between-animal differences are in fact highly significant (F(27, 18288) = 6.3–8.0, p < 10⁻²²), and the design effect is about 6–8. | **Mis-framed** |
| 11 | 260/261 units "putatively directly optotagged" at p < 0.05 | With B = 1000 permutations the p-value floor is 1/1001, and with m = 18,316 tests **no unit survives Benjamini–Hochberg at q = 0.05**. An exact conditional Poisson test recovers 246/261 of them (Section 5.3), so the responses are real but the test cannot show it. | **Fixable** |
| 12 | Dose-response at "1.0, 2.5, 4.0 mW calibrated" | The level keys `1.0/2.5/4.0` come from the NWB file, but the "mW calibrated" unit is a string literal (`batch_download_and_process_cohort.py`). Slopes are averaged over all units, most of them non-responsive, and every 95% CI includes 0. | **Units unverified; underpowered** |

Items 1–8 are fatal for the corresponding claims, because they report the assumptions as results. The manuscript must not be submitted with them. Items 9–12 are correctable statistical errors. The rest of this report (i) places the work in the literature, (ii) inventories every computational method and its provenance, (iii) gives the detailed findings, and (iv) — the main constructive contribution — develops the mathematics that would turn the project's central idea, *optotagging as graded evidence rather than a binary label*, into a rigorous method that this dataset can actually support.


> **Update — the recommended analysis has been executed.** See `reports/REANALYSIS_RESULTS.md` (code in `src/reanalysis/`, `scripts/reanalysis/`). Two items flagged above as "verify" were resolved from the NWB files. (i) The train condition is "2.5 ms pulses at 10 Hz", so the 10-Hz assumption was correct. (ii) The light `level` values are **not** the same in all sessions: {1.0, 2.5, 4.0} in 15, {1.3, 1.7, 2.0} in 11 and {0.638, 0.738, 0.82} in 2. The original per-level analysis therefore silently dropped 13 sessions.

---

## 1. Related work

### 1.1 Optogenetic identification ("optotagging")

| Work | Contribution | Relevance here |
|---|---|---|
| Lima et al., 2009, *PLoS ONE* — PINP | First in-vivo photo-identification. It used short first-spike latency, low jitter, and **waveform identity between light-evoked and spontaneous spikes**. | The repo uses no waveform criterion. Waveform correlation is the standard guard against light-artefact and "collision" mis-assignments and should be added. |
| Kvitsiani et al., 2013, *Nature* — **SALT** | A first-spike-latency distribution test. Baseline is split into many windows of test-window length, and the modified Jensen–Shannon divergence between the test window and baseline windows is compared with baseline-vs-baseline divergences. | The repo's `compute_salt` uses a parametric exponential "baseline" and a bootstrap with a mismatched sample size, so it is a different, miscalibrated test (Section 4.2). |
| Pi et al., 2013, *Nature*; Hangya et al., 2015, *Cell* | SALT plus waveform criteria for PV/SST/VIP (Pi) and cholinergic neurons (Hangya). | This is the operational standard (p < 0.01 with waveform r > 0.85–0.9) that the "heuristic" baseline should mirror. |
| Roux, Stark, Sjulson & Buzsáki, 2014, *Curr. Opin. Neurobiol.* | Review of in-vivo optogenetic identification of interneuron subtypes; discusses direct vs indirect activation and the effect of light power. | Grounds the direct/indirect distinction and its dependence on light power. |
| Stark et al., 2012, *J. Neurophysiol.*; Kozai & Vazquez, 2015, *J. Mater. Chem. B* | Diode probes; characterization of **photoelectric artefacts** on microelectrodes. | Motivates the 0–1 ms guard zone. The artefact detector here flags only 9 units, and it has no waveform-based check. |
| Cardin et al., 2010, *Nat. Protoc.* | Protocol for ChR2 targeting and tagging in vivo. | Practical criteria for pulse width and power. |
| Siegle et al., 2021, *Nature* — Allen Visual Coding Neuropixels | The dataset used here, including the Pvalb/Sst/Vip × Ai32 optotagging epochs. | The authoritative description of the stimulus conditions (pulse widths, train condition, "level" units). These should be read from the NWB `condition` strings rather than assumed (Section 4.9). |
| Jun et al., 2017, *Nature*; Steinmetz et al., 2021, *Science* | Neuropixels 1.0 and 2.0. | Hardware context. |
| Lakunina et al., "Neuropixels Opto" (integrated emitters) — *please verify the publication venue and year; the repo cites Nat. Methods 2026* | Optical stimulation on the shank, giving spatially resolved tagging. | Makes "distance from the light source" a measurable covariate. With a surface fiber it is not (Section 5.9). |

### 1.2 Responsiveness and latency statistics

* **ZETA** (Montijn et al., 2021, *eLife* 10:e71969, "A parameter-free statistical test for neuronal responsiveness"). The test pools *all* spikes over trials in a window, takes the deviation of the cumulative spike-time fraction from linear, and compares max|δ| to a null built from jittered onsets, using a Gumbel approximation. **`manuscript/references.bib` lists the wrong title and the wrong author list for this paper** (it names Goltstein, de Kock, Pennartz, Meijer). Use the reference implementation (`zetapy`/`zetatest`) on raw spike times.
* **Latency estimation**: Friedman & Priebe, 1998 (*J. Neurosci. Methods*); Ventura, 2004 (*Neural Comput.*, latency tests for Poisson and non-Poisson trains); and Levakova, Tamborrino, Ditlevsen & Lansky, 2015 (*BioSystems*), a review of latency estimators. All three treat latency as a parameter of a point-process model rather than as the median of observed first spikes. Section 5.2 shows why that matters here.
* **Point-process/GLM framework**: Brown et al., 2002 (*Neural Comput.*, time-rescaling goodness-of-fit); Truccolo et al., 2005 (*J. Neurophysiol.*); Pillow et al., 2008 (*Nature*); and Kass, Eden & Brown, 2014 (*Analysis of Neural Data*).
* **Comparing two Poisson rates**: the conditional binomial test (Przyborowski & Wilenski, 1940) is the uniformly most powerful unbiased test (Lehmann & Romano, *Testing Statistical Hypotheses*). The E-test of Krishnamoorthy & Thomson (2004) is a more powerful alternative.
* **Surrogates and jitter nulls** for non-Poisson trains: Amarasingham et al., 2012 (*J. Neurophysiol.*).

### 1.3 Connectivity and temporal coordination (the CCG claims)

* Jitter-corrected CCGs and monosynaptic inference: Fujisawa et al., 2008 (*Nat. Neurosci.*); Stark & Abeles, 2009 (*J. Neurosci. Methods*); and English et al., 2017 (*Neuron*), which combines optogenetics with CCGs for pyramidal–interneuron connections.
* Model-based connection inference: Kobayashi et al., 2019 (*Nat. Commun.*, GLMCC).
* Interpreting correlations: Cohen & Kohn, 2011 (*Nat. Neurosci.*).
* A stimulus-locked CCG is dominated by **shared stimulus drive**. The minimum control is a shift- or shuffle-predictor subtraction (trial-shuffled CCG) or a jitter null. None was computed here; the CCG numbers are constants.

### 1.4 Circuit physiology of the targeted classes

* Connectivity logic: Pfeffer et al., 2013 (*Nat. Neurosci.*); Pi et al., 2013; Lee et al., 2013 (*Nat. Neurosci.*, VIP disinhibition); Fu et al., 2014 (*Cell*); Tremblay, Lee & Rudy, 2016 (*Neuron*); Kepecs & Fishell, 2014 (*Nature*).
* Activating PV or SST neurons with ChR2: Atallah et al., 2012; Lee et al., 2012; Wilson et al., 2012; Cardin et al., 2009; Sohal et al., 2009.
* Implication: **most units in a PV/SST × Ai32 session should be suppressed, and in Vip sessions some should be disinhibited.** This is an a-priori expectation to *test*. The current scripts encode it as the result.

### 1.5 Opsin and light physics (needed for the train and spatial analyses)

* Opsin kinetics and desensitization: Mattis et al., 2012 (*Nat. Methods*) and Lin, 2011 (*Exp. Physiol.*). ChR2(H134R), the variant in Ai32, desensitizes and has slow closing kinetics, so pulse-to-pulse decline in *directly driven* spikes partly reflects the opsin rather than cellular "adaptation".
* Herman et al., 2014 (*eLife*): prolonged ChR2 activation can silence interneurons through depolarization block.
* Light and heat propagation: Yizhar et al., 2011 (*Neuron*); Stujenske, Spellman & Gordon, 2015 (*Cell Rep.*), which gives Monte-Carlo light and heat models; and Owen, Liu & Kreitzer, 2019 (*Nat. Neurosci.*), on thermal confounds.

### 1.6 Cell-type inference from extracellular data (where the "ground-truth" debate lives)

* Barthó et al., 2004 (*J. Neurophysiol.*); Jia et al., 2019 (*J. Neurophysiol.*, multichannel waveforms); Lee et al., 2021 (*eLife*, WaveMAP).
* Deep-learning classifiers that use optotagged units as labels: Beau et al., 2025 (cerebellum), and HIPPIE (Gonzalez-Ferrer et al.). *Please verify the exact venues and years for both; the repo's literature review states Nat. Commun. 2026 for HIPPIE.*
* The project's central thesis — that operational optotag labels are noisy and should carry graded evidence — directly addresses this line of work and is a worthwhile, publishable angle.

### 1.7 Statistics for nested neural data, multiplicity and clustering

* Nested data: Aarts et al., 2014 (*Nat. Neurosci.*); Saravanan, Berman & Sober, 2020 (*NBDT*, hierarchical bootstrap); Bates et al., 2015 (lme4); Shrout & Fleiss, 1979 (ICC forms).
* Multiplicity: Benjamini & Hochberg, 1995; Storey, 2002; Efron, 2004/2010 (local FDR and the *empirical null*); Phipson & Smyth, 2010 (permutation p-values must never be zero, and resolution matters).
* Clustering: Fraley & Raftery, 2002 (model-based clustering, BIC); Hennig, 2007 (cluster stability); Hartigan & Hartigan, 1985 (dip test of unimodality).
* Short-term dynamics: Tsodyks & Markram, 1997 (*PNAS*).

### 1.8 Corrections to the repository's literature material

* `references.bib`:
  * The **Montijn 2021** title and authors are wrong (see 1.2).
  * `bueno2020reproducible` lists the author "Buena-Junior, C" with no other authors; check that it resolves to a real paper before submission.
  * `kvitsiani2013distinct` gives pages 863–866; the Nature pages are 363–366.
  * `lee2012activation` is listed in *Nat. Neurosci.* 15:746–752; the paper appeared in *Nature* 488:379–383.
  * `lima2009labeled` ("Labeled-line coding in a sensory system", *Cell*) should be checked. The photo-identification method paper by Lima, Hromádka, Znamenskiy & Zador is "PINP", *PLoS ONE* 4:e6099 (2009).
  * Run all 30 entries through a DOI lookup before submission.
* `results/literature/literature_review.md`:
  * It places Buetfering et al., 2014 in *eLife*. The grid-cell PV paper by Buetfering, Allen & Monyer appeared in *Nat. Neurosci.* 2014.
  * It attributes a Roux et al. paper to *Neuron*. The likely intended papers are Roux et al., 2014, *Curr. Opin. Neurobiol.* and Roux & Buzsáki, 2015, *Neuropharmacology*.
  * Claims that "zero prior studies" quantified threshold instability, and the ECE figures (0.0067 vs 0.0862), should be re-derived or softened.
* The literature review and `final_ml_summary_28spec.md` describe an analysis of **2 locally processed sessions (945 good units)**, while the manuscript reports 28 sessions and 18,316 units. Reconcile the processing provenance, including when each session was ingested, and state it once.

---

## 2. Inventory of computational methods and provenance

Legend: **S** = computed from spike times; **D** = derived from S-features; **R** = rule-generated from other features; **H** = hard-coded constant; **N** = random numbers/simulation.

| Stage | Method | File | Provenance | Notes |
|---|---|---|---|---|
| Trial parsing | Heuristic classification of opto epochs by duration/condition strings | `src/opto_trials.py` | S | The train type is matched on `"fast_pulses"`/`"2.5"`; the train frequency is *assumed* to be 10 Hz downstream. |
| Alignment | `searchsorted` windowing: baseline [−20, −5), evoked [+1, +9), sham [−18, −10) ms | `src/spike_alignment.py` | S | **The sham window lies inside the baseline window**, so the two are not independent. |
| Per-unit features | Rates, modulation ratio, first-spike latency (median/SD/IQR), reliability, Fano factor, paired sign-flip permutation (B = 1000), Cohen's d, per-level rates | `src/feature_extraction.py` | S | Trials are pooled across light levels. The modulation ratio uses +1 Hz in the denominator. |
| Train adaptation (v1) | 1 − p₁₀/(p₁ + 10⁻⁴) with 100 ms ISI | `src/feature_extraction.py:241` | S | Unbounded below: min −4399, and 3,121 units < −1. |
| Artifact control | Guard zones; flag if latency < 1.2 ms and SD < 0.1 ms | `src/artifact_control.py` | S | 9 units flagged; no waveform test. |
| Operational label | Conjunctive thresholds (latency < 8, reliability ≥ 0.30, MR > 2, p < 0.05, d > 0.1) | `src/labeling.py` | D | No multiplicity control. |
| Evidence score | Weighted sum of 5 squashed sub-scores × artifact gate | `scripts/compute_evidence_scores.py` | D | Arbitrary weights; `s_S ≤ 0.731` because of the p floor; `trial_var` hard-codes n = 45. |
| ML benchmark | 9 classifiers predicting the operational label, with LOSO/LOGO | `src/models.py`, `src/validation.py`, `scripts/run_full_specimen_ml_benchmark.py` | D | Circular target, which the repo acknowledges. |
| SALT / ZETA | Custom functions | `src/responsiveness_methods.py` | **N inputs** | Run on `rng.normal(median, sd)` pseudo-latencies (`stage1_methods.py:116`). |
| Latency multimodality | 1-D GMM with BIC on median latencies | `stage1_methods.py` | D | Data bounded to [1, 9] ms; Section 5.7. |
| W1–W5 windows | Rates in 0–8 / 8–20 / 20–50 / 50–200 / 200–500 ms | `stage2_phenotypes.py:63-92` | **R** | Never computed from spikes, even though the PSTH window only extends to +30 ms. |
| Archetype GMM (k = 5) | GMM on clipped log-ratios of W1–W5 | `run_rigorous_neuroscience_reanalysis.py` | **R** | Recovers the generating rules. |
| Dose-response | Per-session mean rate vs level, polyfit slope, t-interval | same | D | Means over all units; all CIs include 0. |
| Train adaptation (v2) | AI = (R₁₀ − R₁)/max(R₁, 0.5) | same `:66-118` | **N** | Cre-specific random multipliers. |
| Sparse-tier Poisson null | 1 − e^(−λτ) | same `:150-230` | D (wrong) | Wrong τ; Jensen bias; clipped output. |
| Spatial | Distance bins; Figure 8A/B curves | same and figure script | D / **H** | Slope not fitted; figure lines are formulas. |
| Variance decomposition | var(means)/(var(means) + mean within-var); t-test vs 0 | same `:425-470` | D | Biased estimator; vacuous test. |
| Response network | Cosine similarity of standardized W-features per probe | `stage3_networks.py` | **R** | Built on rule-generated features. |
| CCG | — | `stage3_networks.py:166-200` | **H** | Constants. |
| Negative controls | — | `stage3_networks.py:210-247`; `run_controls_and_latency_analysis.py:133-139` | **H / N** | Dictionaries; uniform draws. |
| Cre decoding (LOSO) | Logistic regression on W-features | `stage3_networks.py` | **R** | The features encode `cre_line` by construction (`w3` depends on `cre == Pvalb`), so above-chance decoding is guaranteed. |
| Secondary validation figure | 5-ms vs 10-ms latency; multi-condition concordance | `run_secondary_stimulation_validation.py:143,156` | **N** | Multiplicative noise on the primary values. |

---

## 3. Detailed findings: data-provenance issues (must fix before any submission)

1. **Simulated pulse-train dynamics** — `scripts/run_rigorous_neuroscience_reanalysis.py:66-118`. `r10 = r1 × N(μ_cre, σ)` with μ = 0.35/0.40/1.40 for direct-like PV/SST/VIP units. The reported category percentages are fully determined by these priors and the ±0.20 cut-offs. The ICC = 0.452 for this "metric" (Fig. 10B) is likewise a property of the simulation.
2. **Rule-generated temporal windows** — `stage2_phenotypes.py:63-92`. For example, `w4_rate = 0.45 × baseline` whenever the Cre line is PV or SST, the unit is not W1-active, and the baseline rate is above 2 Hz. "Prolonged Suppression" is therefore a deterministic function of Cre line and baseline rate. That is why the two suppression clusters are 100% PV and 100% SST, and why VIP has none.
3. **Hard-coded CCG results and negative controls** — `stage3_networks.py:166-247`.
4. **Synthetic SALT/ZETA inputs** — `stage1_methods.py:109-118`. Even a correct SALT or ZETA run on `N(median, sd)` draws would only test the shape of a Gaussian, not the neuron's spike train.
5. **Formula-drawn or random figure panels** — Fig. 8A/B (`generate_revised_neuroscience_figures.py:442-453`), Fig. 9 (lines 482-520), Fig. 10A (line 535), Fig. 10B (line 551, typed-in percentages), and `fig8_secondary_validation` (`run_secondary_stimulation_validation.py:143,156`). The "representative" rasters in `figures/fig2_representative_responses` (`generate_primary_figures_and_benchmarks.py:139-194`) are also generated with `np.random`. If kept, they must be labelled as schematic simulations.
6. **Self-certifying audit language.** The manuscript's audit table (`main.tex:661`, "Spatial calculations — PASS") and `scripts/update_reports_with_reviewer_fixes.py` assert verification of numbers that were never computed. Remove such tables, or regenerate them from code that recomputes each number from spikes.

**What to do:** delete or quarantine (1)–(5). Re-implement the analyses from spike times as described in Section 6, and regenerate every table and figure from a single pipeline that reads only NWB-derived data. Keep the operational distinction between *simulation* (used to validate estimators, where it is valuable — see Section 5.10) and *results*.

---

## 4. Detailed findings: statistical and implementation issues

### 4.1 Multiplicity and permutation resolution

The 18,316 per-unit tests are all reported at α = 0.05 without correction, so about 916 false positives are expected, against 1,333 observed. The sign-flip test uses B = 1000, so p ≥ 1/1001. Benjamini–Hochberg at q = 0.05 would need at least 367 units at the floor, and only 360 are there. **Zero units survive, including all 261 heuristic-tagged ones.** This is a resolution artefact, not a lack of signal: the exact test in Section 5.3 gives 305 BH discoveries and 246 of the 261 heuristic units. Use exact or analytic tests, or B ≳ m/(αk) with Phipson–Smyth correction. Use one-sided alternatives (excitation and suppression tested separately) rather than two-sided.

### 4.2 SALT-like and ZETA-like implementations are miscalibrated

Under a pure null (Poisson spikes, no light response; first spike in [1, 9) ms) the stored functions give:

| λ (Hz) | trials | SALT-like FPR | ZETA-like FPR |
|---|---|---|---|
| 2 | 45 / 75 | 0.000 / 0.003 | 0.003 / 0.020 |
| 10 | 45 / 75 | 0.027 / 0.023 | 0.093 / 0.153 |
| 30 | 45 / 75 | 0.000 / 0.000 | 0.170 / 0.270 |
| 60 | 45 / 75 | 0.000 / 0.000 | 0.373 / **0.653** |

*Why the ZETA-like test fails.* Latencies are recorded on [1, 9) but compared with the uniform CDF on [0, 10]. The empirical CDF therefore has a deterministic offset, sup|Fₙ − F₀| → 0.1. The test divides the *maximum* deviation by the *pointwise* Brownian-bridge SD √(t(1−t)/n) at t ≈ 0.1 or 0.9, which gives z ≈ 0.1/(0.3/√n) = √n/3. That exceeds 1.96 once n > 35 first spikes, so any unit that fires enough is "significant". Two further errors compound this: the maximum of a bridge follows the Kolmogorov distribution, not a normal distribution (P(sup|B| > x) = 2Σ(−1)^(k−1) e^(−2k²x²)), and the result is a dependence of the false-positive rate on firing rate.

*Why the SALT-like test fails.* The "baseline" is a parametric exponential (essentially flat on 10 ms), and the bootstrap draws n exponentials and then discards those beyond 10 ms. Null histograms are therefore built from far fewer samples than the observed one, which inflates the null divergence and makes the test conservative to the point of uselessness at high rates.

**Fix:** use the reference SALT (baseline windows from the actual pre-stimulus spike trains) and the reference ZETA, on raw spike times.

### 4.3 Sparse-tier Poisson null

The scripts and manuscript compute `N·(1 − e^(−λ̄·0.75 s))`. The correct expectation of "≥ 1 spike in the evoked window across trials" is

$$E[\#\text{units}] = \sum_i \left(1 - e^{-\lambda_i\, w\, n_i}\right), \qquad w = 8\ \text{ms},\ n_i = \text{trials of unit } i .$$

Three errors are involved. The window is 8 ms, not 10. The trial count has median 45, not 75. And because x ↦ 1 − e^(−x) is concave, plugging in the mean λ̄ over-estimates the expectation (Jensen's inequality). The corrected values for the < 1 Hz tier are expected = 357.0 and observed = 1,178, a ratio of 3.30. In every other tier observed/expected is 0.95–1.07, so the model is well calibrated where λ is estimated well. The excess in the sparse tier is explained by Section 5.4: 78% of that tier have **zero** baseline spikes, so λ̂ = 0 is uninformative (95% upper bound 4.44 Hz). Separately, the conditional probability of a first latency < 8 ms within [1, 9) ms is about 7/8 = 0.875, not 0.80, and the code clips it to [0.79, 0.82].

### 4.4 Variance decomposition

`var(specimen means)` contains the sampling variance σ²_w/nᵢ. The unbiased one-way ANOVA estimator is

$$\hat\sigma^2_b = \frac{MS_B - MS_W}{n_0},\qquad n_0 = \frac{N - \sum n_i^2/N}{k-1},\qquad \mathrm{ICC}(1) = \frac{\hat\sigma^2_b}{\hat\sigma^2_b + MS_W}.$$

| Metric | Repo "ICC" | ICC(1) | F(27, 18288) | p | Design effect 1 + (n̄ − 1)·ICC | Effective n per animal |
|---|---|---|---|---|---|---|
| baseline rate | 0.0091 | 0.0081 | 6.3 | 1e-22 | 6.3 | ≈104 (of 654) |
| evoked rate | 0.0140 | 0.0107 | 8.0 | 2e-31 | 8.0 | ≈82 |
| modulation ratio | 0.0116 | 0.0081 | 6.3 | 1e-22 | 6.3 | ≈104 |

Two points follow. (i) A small ICC does **not** mean animals are interchangeable. With 654 units per animal even ICC ≈ 0.01 inflates unit-level standard errors by √DEFF ≈ 2.5–2.8×, so unit-pooled tests (and the 18,316-unit ML folds) are anti-conservative. (ii) The one-sample t-test of animal-mean firing rates against 0 is vacuous: rates are positive by definition. Cre-line contrasts have 8/12/8 animals, so their degrees of freedom are about 25, not about 18,000.

### 4.5 Modulation ratio is not scale-free and is biased under the null

MR = r_e/(r_b + 1 Hz). If evoked equals baseline (no effect), MR_null = λ/(λ + 1): 0.17 at 0.2 Hz, 0.5 at 1 Hz, and 0.98 at 50 Hz. Thresholds such as "MR > 2" and "MR < 0.7 = suppressed" therefore mean different effect sizes for different baseline rates. The spatial "suppression fraction" (MR < 0.7) is close to the fraction of units with λ < 2.3 Hz, not a measure of suppression. Section 5.3 gives the principled replacement.

### 4.6 Reliability is not chance-corrected and is noisy at the threshold

Chance reliability is p₀ = 1 − e^(−λw). 4,367 units have p₀ ≥ 0.10 and 229 have p₀ ≥ 0.30, so they can pass the reliability criterion from spontaneous firing alone. With n = 45 the Wilson 95% CI at an observed reliability of 0.30 is [0.195, 0.457]. A unit with true ρ = 0.35 is classified below threshold 24% of the time. The "threshold instability" that the paper documents empirically is largely binomial sampling error that can be predicted analytically (Section 5.5).

### 4.7 Adaptation index (v1) is ill-conditioned

1 − p₁₀/(p₁ + 10⁻⁴) diverges when p₁ → 0 (min −4399). Because it is a ratio of two noisy proportions it also has heavy tails. It is also not chance-corrected.

### 4.8 Latency-based "multimodality" and spatial latency gradients

Median latencies are confined to [1, 9) ms and, for most units, reflect spontaneous spikes (Section 5.2). A Gaussian mixture on bounded data overfits the boundaries, so a BIC preference for k = 3 is not evidence of distinct populations. Bin-wise median latencies of mostly non-responsive units (3.82 → 4.50 ms) track sampling noise around the window centre, and the "direct" anchor units are themselves defined by latency, which makes the distance analysis circular.

### 4.9 Stimulus metadata assumptions to verify

* **Train frequency and pulse count.** `compute_train_adaptation_index` hard-codes 10 pulses at 10 Hz (100 ms ISI). Check the NWB `condition` strings for the `fast_pulses` epochs. If the actual train is faster, pulses 2–10 of the analysis windows fall outside the train.
* **Light-level units.** `"1.0, 2.5, 4.0 mW calibrated"` is a literal string. Confirm what the Allen `level` column represents (physical power vs a driver setting) before using it as a dose.
* **Trial count.** It varies from 45 to 151 across units and sessions. Every formula that assumes 75 (or 45: `compute_evidence_scores.py:80`, `run_controls_and_latency_analysis.py:135`) must use nᵢ.
* **Pooling across levels.** Pulses at all levels are pooled into one reliability or latency estimate. Because reliability rises and latency falls with power, the pooled estimates mix distributions. Stratify by level or model level explicitly (Section 5.8).

### 4.10 Evidence score and "uncertainty"

The weights (0.30/0.25/0.20/0.15/0.10) are not estimated. `s_S` is capped at σ(1) = 0.731 by the p floor, and the composite uncertainty adds a CV, a proximity score and a rescaled variance, which have different units. The ML calibration claims (ECE) measure how well a model emulates a known deterministic rule. That is the label-circularity issue the repo itself flags, so calibration against those labels says nothing about biological identity.

---

## 5. Mathematical refinements: a generative model of optotagging

The project's best idea is to replace a binary tag with graded evidence. The mathematically natural way to do that is to write down the generative model of a trial and derive every quantity — reliability, latency, modulation, evidence, uncertainty — from it. That replaces ad-hoc heuristics with estimators whose bias, variance and calibration are known.

### 5.1 Trial-level point-process model

For unit *i*, light level *ℓ*, and time *t* relative to pulse onset:

$$\lambda_i(t) = \lambda_{0,i}\,\big[1 + s_i(t)\big] \;+\; \underbrace{\rho_{i\ell}\; g(t-\delta_{i\ell};\, \sigma_{i\ell})}_{\text{direct, one spike w.p. }\rho},$$

where

* λ₀ is the spontaneous rate;
* s(t) is a slow network modulation (suppression when s < 0, disinhibition or rebound when s > 0), represented on a spline basis over 0–500 ms;
* ρ is the probability that a pulse evokes a direct spike (the *true reliability*);
* δ is the latency, covering opsin activation plus membrane charging; and
* g(·; σ) is a jitter density (for example a gamma or log-normal with SD σ).

Direct ChR2 activation predicts small δ (about 1–5 ms), small σ (< 1 ms), ρ increasing with light power ℓ, and δ decreasing with ℓ. Indirect excitation predicts larger δ and σ and a non-zero s(t) after the window.

**First-spike latency.** On [a, b] = [1, 9) ms, assuming the evoked spike falls in the window and ignoring the small s(t) term inside it, the first spike is the minimum of an evoked time and a spontaneous time. Its survival function is

$$S(t) = \big(1-\rho\,G_\sigma(t-\delta)\big)\, e^{-\lambda_0 (t-a)},\qquad f_T(t) = -S'(t),$$

and a trial with *no* spike in the window has probability S(b). The likelihood of a unit's data is

$$\mathcal{L}(\rho,\delta,\sigma\mid\lambda_0) = \prod_{\text{trials with a spike}} f_T(t_j)\; \prod_{\text{trials without}} S(b),$$

with λ₀ estimated from *long* spontaneous epochs (Section 5.4). Maximizing it, or sampling the Bayesian posterior, gives (ρ̂, δ̂, σ̂) with standard errors. These are the physically meaningful versions of the repo's reliability, median latency and jitter.

### 5.2 Why the observed median latency is uninformative at low reliability

If all first spikes are spontaneous (ρ = 0), latencies are a truncated exponential on [1, 9). That is approximately uniform when λ₀w ≪ 1, so P(T < 8) ≈ 7/8 and the median of k such spikes is below 8 ms with probability Σ_{j ≥ ⌈k/2⌉} C(k, j)(7/8)^j(1/8)^(k−j): 0.875 (k = 1), 0.957 (k = 3), 0.9995 (k = 10). **Inside a [1, 9) ms window, "median latency < 8 ms" therefore has essentially no specificity.** It only rejects units whose first spikes cluster in the last 1 ms. More generally, the observed median is the median of a mixture with spontaneous weight

$$\pi_0 \approx \frac{(1-\rho)\,p_0}{(1-\rho)\,p_0 + \rho},\qquad p_0 = 1-e^{-\lambda_0 w},$$

which pulls it towards the window centre, 5 ms. The pull grows as ρ falls and λ₀ rises. That is a predictable latency bias, not biology, and it is the reason for the apparent "latency increases with distance": units farther from the light have smaller ρ, so their medians drift towards 5 ms. Estimating δ from the model in 5.1 removes this bias.

### 5.3 Rates: exact tests, rate ratios and shrinkage

With evoked count N_e over exposure n·w_e and baseline count N_b over n·w_b, under H₀ (equal rates),

$$N_e \mid (N_e+N_b=N) \sim \mathrm{Binomial}\Big(N,\ \pi_0=\tfrac{w_e}{w_e+w_b}\Big).$$

This is the UMPU test. It is exact, needs no permutations, and reaches arbitrarily small p-values. On the stored counts it gives 682 units at p < 0.05, 305 at BH q < 0.05, and 232 Bonferroni-significant, and it recovers 246 of the 261 heuristic units. The rate ratio θ = r_e/r_b gets an exact CI by mapping a Clopper–Pearson CI on π:

$$\theta = \frac{\pi}{1-\pi}\cdot\frac{w_b}{w_e}.$$

Replace MR = r_e/(r_b + 1) with the **log rate ratio** with a half-count correction,

$$\widehat{\log\theta} = \log\frac{N_e+\tfrac12}{n w_e} - \log\frac{N_b+\tfrac12}{n w_b},\qquad \widehat{\mathrm{Var}} \approx \frac{1}{N_e+\tfrac12}+\frac{1}{N_b+\tfrac12}.$$

This is 0 under the null at every λ, symmetric for excitation and suppression, and comes with its own standard error, so a z-score replaces the thresholds. For population summaries, use **Gamma–Poisson empirical-Bayes shrinkage**: if λᵢ ~ Gamma(α, β) within a session, the posterior mean is (Nᵢ + α)/(Tᵢ + β), where α and β are fitted by marginal (negative-binomial) likelihood. That stabilizes the ratios of low-count units, which currently dominate the clustering and the tiers.

### 5.4 Estimating the baseline rate: the sparse-tier paradox explained

The baseline exposure is 15 ms × nᵢ, which is 0.675 s at nᵢ = 45. With zero spikes the exact 95% upper bound is −ln(0.05)/0.675 = **4.44 Hz**, so "< 1 Hz" is not identifiable from these windows. Tiers defined by a noisy λ̂ select units whose λ̂ came out low by chance (regression to the mean). Their true rates are higher, which is exactly why observed evoked-window spiking is 3.3× the null in the < 1 Hz tier and about 1.0× elsewhere. **Fix:** estimate λ₀ from the inter-trial intervals and from the long spontaneous or grey-screen epochs of the same session (minutes of data), and model slow drift with a smooth covariate. Then tier on the posterior λ₀, or better, do not tier and carry λ₀ as a covariate.

### 5.5 Reliability: chance correction, uncertainty and threshold flips

Under the model, P(at least one spike in the window) = 1 − (1 − ρ)e^(−λ₀w). Hence

$$\hat\rho = \frac{\hat r - p_0}{1-p_0},\qquad p_0 = 1-e^{-\lambda_0 w}.$$

This is Abbott's correction, or the kappa form. Its sampling SD is approximately √(r(1−r)/n)/(1 − p₀). The probability that a unit with true ρ lands on the other side of a threshold ρ* is about Φ(−|ρ − ρ*|·√n·(1 − p₀)/√(r(1−r))). This **analytic threshold-instability curve** replaces the 27-cell grid search: units within about 2 SE of a threshold are unstable by construction. The excess of reliability over chance has an exact one-sided binomial test, Bin(n, p₀); 1,293 units pass it at BH q < 0.05.

### 5.6 Evidence as a posterior probability, calibrated by an empirical null

A principled "evidence score" is the posterior probability that a unit is directly driven:

$$E_i = P(\text{direct}\mid\text{data}_i) = 1 - \mathrm{lfdr}(z_i),\qquad \mathrm{lfdr}(z) = \frac{\pi_0 f_0(z)}{f(z)}.$$

Here z is a per-unit test statistic, such as the exact-test z or a likelihood-ratio statistic for ρ > 0 with δ < δ_max. Following Efron, f₀ is estimated as an **empirical null** from the same statistic computed on sham windows. These must be non-overlapping and sampled from many pre-stimulus segments or spontaneous epochs, not nested in the baseline. f is the mixture density, and π₀ is the null proportion. The resulting E_i:

* lies in [0, 1];
* is calibrated by construction against the sham distribution, with no arbitrary weights;
* yields FDR control directly, since the mean lfdr over the selected units is their FDR; and
* has uncertainty that follows from the posterior (credible intervals on ρ and δ), so no composite "uncertainty index" is needed.

Multiple evidence dimensions (latency, jitter, reliability, power dependence) enter through the likelihood ratio of the model in 5.1, not through a hand-weighted sum.

### 5.7 Response archetypes from real PSTHs

Compute PSTHs from spikes over 0–500 ms. The alignment window currently stops at +30 ms and must be extended. Then model counts directly:

* A **mixture of inhomogeneous Poisson processes**. Each cluster *c* has a log-intensity profile exp(β_c ᵀ B(t)) on a spline basis B. The likelihood is Σ_c π_c Π_bins Poisson(y_tb | λ₀ᵢ e^(β_cᵀB(t)) Δ). The unit's own λ₀ is an offset, so clusters capture *shape*, not baseline rate.
* Choose k by BIC/ICL, and report stability with the bootstrap adjusted Rand index (Hennig, 2007).
* Then test cluster × Cre association with animal as the unit of replication: a permutation test that shuffles Cre labels *between animals*.
* If Gaussian methods are preferred, apply a variance-stabilizing transform (Anscombe √(y + 3/8)) and weight each unit by its Fisher information. Low-count units otherwise dominate the log-ratio space as noise.
* For latency multimodality, use δ̂ from 5.1, fit mixtures of distributions with the correct support (truncated or log-normal), and add a dip test.

### 5.8 Dose-response and pulse-train dynamics as GLMMs

**Dose.** For each pulse *j* at level ℓ in trial *k* of unit *i*:

$$\operatorname{logit} P(\text{spike}_{ijk}) = \underbrace{\log\frac{p_{0i}}{1-p_{0i}}}_{\text{offset}} + \beta_{0} + \beta_1 h(\ell) + b_{0i} + b_{1i} h(\ell) + u_{\text{animal}},$$

with h(ℓ) = log ℓ or a saturating (Hill) form ℓⁿ/(ℓⁿ + ℓ₅₀ⁿ). With three levels only a two-parameter curve is identifiable, so report the slope on log ℓ. Restrict the dose analysis to units with posterior E_i > 0.5; including non-responsive units dilutes the slopes towards 0, which is why every reported CI includes 0.

**Trains.** Use pulse index *j* as a covariate,

$$\operatorname{logit} P(\text{spike}_{ijk}) = \text{offset} + \alpha_i + \gamma\,(j-1) + \gamma_{\text{cre}}(j-1) + \ldots,$$

or fit a Tsodyks–Markram resource model, r_{j+1} = r_j(1 − U)e^(−Δ/τ) + 1 − e^(−Δ/τ), and compare U and τ_rec across Cre lines with animals as replicates. Interpretation must account for ChR2(H134R) desensitization and possible depolarization block (Herman et al., 2014): a decline in *direct* spiking across pulses is a property of the opsin–cell system, not synaptic depression. An index that is bounded and well conditioned is the log ratio log((k₁₀ + ½)/(k₁ + ½)) with variance ≈ 1/(k₁₀ + ½) + 1/(k₁ + ½).

### 5.9 Spatial structure: model the light, not the shank coordinate

For a surface fiber, irradiance falls with *depth from the cortical surface*, not with distance along the shank from a "centroid" of tagged units. A first-order model is I(z) ≈ I₀·T(z), where T combines geometric spread and scattering and absorption (modified Beer–Lambert, or Monte-Carlo as in Stujenske et al., 2015). The Allen CCF coordinates already in the table allow cortical depth (and laminar assignment) per unit. The testable predictions are ρ̂ᵢ increasing with I(zᵢ) and δ̂ᵢ decreasing with I(zᵢ), *within* the direct class, analysed with a mixed model that has random effects for probe and animal. A real "propagation" analysis would compare the *onset of s(t)* (5.1) in non-tagged units with distance to the nearest high-E_i unit, using a permutation null that shuffles positions within probe. Finally, the velocity arithmetic: v = Δd/Δt, so 1 µm / 1.1 µs = 0.91 m/s; 0.07 m/s would need 14.3 µs/µm.

### 5.10 Temporal coordination: corrected CCGs

For pairs (tagged *a*, untagged *b*) on the same probe, compute the raw CCG C_ab(τ) in baseline and in post-stimulus windows. Subtract the **shift predictor** (trial-shuffled CCG), which removes stimulus-locked co-modulation, and test against an interval-jitter null (Amarasingham et al., 2012) or fit GLMCC (Kobayashi et al., 2019). A monosynaptic inhibitory signature from a PV-tagged unit is a short-latency (1–4 ms) *trough* in C_ab after jitter correction. Report effect sizes per pair and aggregate with animal-level random effects.

### 5.11 Simulation-based validation (the right use of synthetic data)

Synthetic data is valuable when it is used to *validate estimators*, not to produce results. Simulate from 5.1 with known (ρ, δ, σ, λ₀, s(t)), at the real nᵢ and window lengths, and report:

* bias and coverage of (ρ̂, δ̂);
* the calibration of E_i (reliability diagrams against the known truth);
* the FDR achieved by the lfdr procedure; and
* the threshold-flip rates predicted by 5.5 vs observed.

That is a genuine computational-biology contribution suited to TCBB, and it shows quantitatively how much the binary heuristic loses.

---

## 6. Recommended corrected analysis plan (in priority order)

1. **Quarantine** every simulated or hard-coded result: the train AI v2, W1–W5, CCG, negative controls, the Fig. 2/8/9/10 synthetic panels and the secondary-validation panels. Remove the matching sentences and "PASS" tables from `manuscript/main.tex`, `README.md` and the results markdown.
2. **Re-extract from NWB**:
   * a PSTH window of −500 to +500 ms (1 ms bins) per level;
   * per-pulse train responses using the *actual* train timing from the epochs table;
   * non-overlapping sham windows from many pre-stimulus segments;
   * baseline λ₀ from spontaneous epochs;
   * spike waveforms (light-evoked vs spontaneous) for the Lima-style waveform correlation.
3. **Per-unit inference**: exact conditional Poisson tests (5.3) and the likelihood model (5.1) giving ρ̂, δ̂ and σ̂ with CIs; reference SALT and ZETA on raw spikes as comparators; and BH or lfdr control across units (5.6).
4. **Evidence score** = 1 − lfdr with a sham-based empirical null; validate it by simulation (5.11).
5. **Population analyses** as mixed models with animal (and probe) random effects: dose (5.8), trains (5.8), depth and irradiance (5.9), and Poisson-mixture archetypes (5.7). Test Cre contrasts at animal level (df ≈ 25) and report ICC(1) and design effects (4.4).
6. **CCGs** with shift-predictor and jitter correction (5.10).
7. **ML (optional)**: drop "predict the operational label" as a headline. If kept, predict an *independent* target, for example ρ̂ estimated from held-out trials, or Cre line from waveform features with LOSO, and use animal-level confidence intervals.
8. **Provenance**: one reproducible entry point (`pipeline.py`) that writes all tables, plus a test that fails if any results table contains values not derived from it. Add unit tests for the statistical functions against known nulls (for example, a test asserting that the false-positive rate is about 0.05 on simulated Poisson data).
9. **Fix the bibliography** (Montijn 2021 title and authors; verify `bueno2020reproducible`, HIPPIE, Neuropixels Opto and Beau et al.) and the literature review venue errors (Section 1.8).

---

## 7. Status of manuscript claims after this review

| Claim | Can it stand? | What would make it stand |
|---|---|---|
| Binary thresholds are unstable | **Yes, in principle** | Derive the instability analytically (5.5) and show it on real ρ̂. |
| Sparse firing makes latency uninformative | **Yes, stronger than stated** | Use 5.2 (7/8 specificity loss, median argument) with the corrected Poisson numbers (4.3) and better λ₀ (5.4). |
| SALT/ZETA/heuristic disagreement | **Unknown** | Rerun the reference implementations on spikes. |
| Suppression dominates in PV/SST sessions | **Unknown; plausible a priori** | Spike-based PSTHs (> 30 ms window) with Poisson-mixture archetypes. |
| Cell-type train dynamics | **Unknown** | Real per-pulse responses and a GLMM; consider opsin confounds. |
| Spatial latency gradient / velocity | **No (not computed; arithmetic error)** | Depth/irradiance model on δ̂ within the direct class. |
| CCG synchrony shifts | **No (constants)** | Corrected CCGs (5.10). |
| > 98% variance within animals | **Yes (qualitatively)** | Report ICC(1), F-tests and design effects, and drop the t-test against 0. |
| Calibrated evidence score | **Not yet** | 1 − lfdr with an empirical null (5.6) and simulation validation (5.11). |

---

## 8. References (works cited in this report)

Aarts E, Verhage M, Veenvliet JV, Dolan CV, van der Sluis S (2014) *Nat Neurosci* 17:491–496.  
Amarasingham A, Harrison MT, Hatsopoulos NG, Geman S (2012) *J Neurophysiol* 107:517–531.  
Atallah BV, Bruns W, Carandini M, Scanziani M (2012) *Neuron* 73:159–170.  
Barthó P, Hirase H, Monconduit L, Zugaro M, Harris KD, Buzsáki G (2004) *J Neurophysiol* 92:600–608.  
Bates D, Mächler M, Bolker B, Walker S (2015) *J Stat Softw* 67:1–48.  
Benjamini Y, Hochberg Y (1995) *J R Stat Soc B* 57:289–300.  
Brown EN, Barbieri R, Ventura V, Kass RE, Frank LM (2002) *Neural Comput* 14:325–346.  
Cardin JA et al. (2009) *Nature* 459:663–667; Cardin JA et al. (2010) *Nat Protoc* 5:247–254.  
Cohen MR, Kohn A (2011) *Nat Neurosci* 14:811–819.  
Efron B (2004) *J Am Stat Assoc* 99:96–104; Efron B (2010) *Large-Scale Inference*, Cambridge UP.  
English DF, McKenzie S, Evans T, Kim K, Yoon E, Buzsáki G (2017) *Neuron* 96:505–520.  
Fraley C, Raftery AE (2002) *J Am Stat Assoc* 97:611–631.  
Friedman HS, Priebe CE (1998) *J Neurosci Methods* 83:185–194.  
Fujisawa S, Amarasingham A, Harrison MT, Buzsáki G (2008) *Nat Neurosci* 11:823–833.  
Hangya B, Ranade SP, Lorenc M, Kepecs A (2015) *Cell* 162:1155–1168.  
Hartigan JA, Hartigan PM (1985) *Ann Stat* 13:70–84.  
Hennig C (2007) *Comput Stat Data Anal* 52:258–271.  
Herman AM, Huang L, Murphey DK, Garcia I, Arenkiel BR (2014) *eLife* 3:e01481.  
Kass RE, Eden UT, Brown EN (2014) *Analysis of Neural Data*, Springer.  
Kepecs A, Fishell G (2014) *Nature* 505:318–326.  
Kobayashi R et al. (2019) *Nat Commun* 10:4468.  
Kozai TDY, Vazquez AL (2015) *J Mater Chem B* 3:4965–4978.  
Krishnamoorthy K, Thomson J (2004) *J Stat Plan Inference* 119:23–35.  
Kvitsiani D, Ranade S, Hangya B, Taniguchi H, Huang JZ, Kepecs A (2013) *Nature* 498:363–366.  
Lee EK et al. (2021) *eLife* 10:e67490.  
Lee S-H, Kwan AC, Zhang S, et al., Dan Y (2012) *Nature* 488:379–383.  
Lee S, Kruglikov I, Huang ZJ, Fishell G, Rudy B (2013) *Nat Neurosci* 16:1662–1670.  
Levakova M, Tamborrino M, Ditlevsen S, Lansky P (2015) *BioSystems* 136:23–34.  
Lima SQ, Hromádka T, Znamenskiy P, Zador AM (2009) *PLoS ONE* 4:e6099 *(PINP)*.  
Lin JY (2011) *Exp Physiol* 96:19–25.  
Mattis J et al. (2012) *Nat Methods* 9:159–172.  
Montijn JS, Seignette K, Howlett MH, Cazemier JL, Kamermans M, Levelt CN, Heimel JA (2021) A parameter-free statistical test for neuronal responsiveness. *eLife* 10:e71969.  
Owen SF, Liu MH, Kreitzer AC (2019) *Nat Neurosci* 22:1061–1065.  
Pfeffer CK, Xue M, He M, Huang ZJ, Scanziani M (2013) *Nat Neurosci* 16:1068–1076.  
Phipson B, Smyth GK (2010) *Stat Appl Genet Mol Biol* 9:39.  
Pi H-J, Hangya B, Kvitsiani D, Sanders JI, Huang ZJ, Kepecs A (2013) *Nature* 503:521–524.  
Pillow JW et al. (2008) *Nature* 454:995–999.  
Przyborowski J, Wilenski H (1940) *Biometrika* 31:313–323.  
Roux L, Stark E, Sjulson L, Buzsáki G (2014) *Curr Opin Neurobiol* 26:88–95.  
Saravanan V, Berman GJ, Sober SJ (2020) *Neurons Behav Data Anal Theory* 3(5).  
Shrout PE, Fleiss JL (1979) *Psychol Bull* 86:420–428.  
Siegle JH et al. (2021) *Nature* 592:86–92.  
Stark E, Abeles M (2009) *J Neurosci Methods* 179:90–100.  
Stark E, Koos T, Buzsáki G (2012) *J Neurophysiol* 108:349–363.  
Storey JD (2002) *J R Stat Soc B* 64:479–498.  
Stujenske JM, Spellman T, Gordon JA (2015) *Cell Rep* 12:525–534.  
Truccolo W, Eden UT, Fellows MR, Donoghue JP, Brown EN (2005) *J Neurophysiol* 93:1074–1089.  
Tsodyks MV, Markram H (1997) *PNAS* 94:719–723.  
Ventura V (2004) *Neural Comput* 16:2323–2349.  
Yizhar O, Fenno LE, Davidson TJ, Mogri M, Deisseroth K (2011) *Neuron* 71:9–34.

*Bibliographic details marked "verify", and the venues of recent works (HIPPIE, Neuropixels Opto, Beau et al. 2025), should be checked against the publisher records before they are cited in the manuscript.*
