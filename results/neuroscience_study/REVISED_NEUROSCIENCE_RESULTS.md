# REVISED NEUROPHYSIOLOGICAL RESULTS

**Dataset**: Allen Visual Coding Neuropixels Cohort (28 Specimens, 159 Probes, 18,316 Units)  
**Cre Lines**: `Pvalb-IRES-Cre` ($N=8$), `Sst-IRES-Cre` ($N=12$), `Vip-IRES-Cre` ($N=8$)  
**Status**: All values audited, verified, and recomputed with strict statistical controls.

---

## 1. Responsiveness Methods Comparison & Physiological Divergence

We cross-classified all 18,316 units across four response identification frameworks: the Operational Heuristic, SALT (Kvitsiani et al. 2013), ZETA (Montijn et al. 2021), and Continuous Evidence ($E_i$).

### Table 1: Comprehensive 8-Way Venn Disagreement Partitioning
*Source: [`results/neuroscience_study/tables/revised/revised_method_disagreement_detailed.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_method_disagreement_detailed.csv)*

| Disagreement Group | Unit Count | Cohort % | Mean Baseline (Hz) | Mean Evoked (Hz) | Median Latency (ms) | Mean Reliability | Mean Effect Size | Mean Modulation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Three (Heuristic + SALT + ZETA)** | 13 | 0.07% | 17.09 | 131.67 | 2.71 | 0.646 | 1.044 | 66.00 |
| **Heuristic + SALT (ZETA Negative)** | 9 | 0.05% | 0.40 | 78.58 | 7.01 | 0.507 | 0.980 | 64.55 |
| **Heuristic + ZETA (SALT Negative)** | 147 | 0.80% | 14.06 | 120.89 | 4.03 | 0.630 | 1.007 | 14.76 |
| **Heuristic Only** | 92 | 0.50% | 11.62 | 69.29 | 5.98 | 0.478 | 0.718 | 10.35 |
| **SALT + ZETA (Heuristic Negative)** | 179 | 0.98% | 4.10 | 10.72 | 2.97 | 0.082 | 0.169 | 3.69 |
| **SALT Only** | 510 | 2.78% | 8.66 | 7.75 | 6.04 | 0.060 | -0.013 | 1.37 |
| **ZETA Only** | 1,800 | 9.83% | 13.62 | 19.82 | 3.74 | 0.148 | 0.119 | 1.72 |
| **None (Non-Responsive Across All)** | 15,566 | 84.99% | 8.05 | 6.82 | 4.93 | 0.053 | -0.048 | 0.72 |

### Methodological Interpretation:
- **Heuristic Criteria**: Exclusively captures high-firing, high-reliability units (mean evoked rate $100.2$ Hz, mean reliability $0.57$). However, it rejects 179 units that achieve dual statistical significance under SALT and ZETA (mean modulation $3.69\times$, latency $2.97$ ms) simply because their trial reliability ($0.082$) falls below the arbitrary $0.30$ cutoff.
- **SALT**: Detects units with tightly phase-locked spike timing relative to stimulus onset even when total spike counts are modest (identifying 510 SALT-only units where spikes are reorganized in time without large rate increases).
- **ZETA**: Sensitive to cumulative deviations over the entire stimulus and post-stimulus interval, detecting 1,800 ZETA-only units characterized by sustained, lower-amplitude rate shifts ($19.82$ Hz vs. $13.62$ Hz baseline).

---

## 2. Latency Distributions & Sparse-Firing Reliability

Across 13,643 active units with detectable post-stimulus spikes:
- **Borderline Zone**: **3,193 units (23.40%)** exhibit median latencies between $6.0$ ms and $10.0$ ms.
- **GMM Model Comparison**:
  - $k=1$ Component BIC: $53,114.6$
  - $k=2$ Components BIC: $53,017.9$
  - $k=3$ Components BIC: $52,900.4$ (Optimal, $\Delta\text{BIC} = -117.5$ vs $k=2$, $-214.2$ vs $k=1$)
  - Bootstrap median latency 95% CI: $[4.668\text{ ms}, 4.734\text{ ms}]$.

### Table 2: Sparse-Firing Stability & Poisson Null Comparison
*Source: [`results/neuroscience_study/tables/revised/revised_sparse_firing_stability.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_sparse_firing_stability.csv)*

| Baseline Firing Tier | Total Units in Tier | Spiking Units in Window | Spiking Fraction (%) | Theoretical Poisson Null (%) | Sub-8ms Spiking Units | Sub-8ms Rate (Spiking Units) | Sub-8ms Rate (All Units) | Heuristic Pass Rate (%) | Latency Bootstrap SE (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$< 1\text{ Hz}$** | 3,867 | 1,178 | 30.46% | 13.41% | 1,096 | **93.04%** | 28.34% | **0.59%** | 0.092 |
| **$1-2\text{ Hz}$** | 1,842 | 993 | 53.91% | 69.41% | 937 | 94.36% | 50.87% | 0.71% | 0.085 |
| **$2-4\text{ Hz}$** | 1,973 | 1,441 | 73.04% | 89.62% | 1,369 | 95.00% | 69.39% | 0.56% | 0.067 |
| **$4-8\text{ Hz}$** | 3,711 | 3,230 | 87.04% | 98.83% | 3,100 | 95.98% | 83.54% | 1.51% | 0.040 |
| **$> 8\text{ Hz}$** | 6,920 | 6,798 | 98.24% | 100.0% | 6,696 | 98.50% | 96.76% | 2.28% | 0.023 |

### Methodological Interpretation:
In quiescent units ($<1$ Hz, mean baseline $0.19$ Hz), 1,178 of 3,867 units recorded at least one spike during the 75 trials $\times$ 10 ms window. Of these, 93.04% had a median latency $<8$ ms. Under a Poisson null model with $\lambda = 0.192$ Hz, spontaneous spikes account for nearly half ($13.41\%$) of the observed spiking fraction ($30.46\%$). When an isolated spike occurs within a 10-ms window, its expected arrival time under uniform chance is $5.0$ ms ($<8$ ms with probability $0.80$). Latency alone is therefore structurally uninformative in low-firing units without corroborating trial reliability and effect-size statistics.

---

## 3. Mutually Exclusive Unsupervised Response Archetypes

Unsupervised Gaussian Mixture Modeling on normalized 5-window trajectories $[W_1, W_2, W_3, W_4, W_5]$ partitioned the cohort into 5 mutually exclusive dynamical archetypes:

### Table 3: Unsupervised Response Archetype Properties
*Source: [`results/neuroscience_study/tables/revised/revised_population_response_archetypes.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_population_response_archetypes.csv)*

| Archetype Label | Unit Count | Cohort % | Mean Baseline (Hz) | Mean Evoked (Hz) | Median Latency (ms) | Pvalb % | Sst % | Vip % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rapid Direct-Like Excitation** | 386 | 2.11% | 9.42 | 58.20 | 2.85 | 42.49% | 43.52% | 13.99% |
| **Prolonged Network Suppression** | 2,379 | 12.99% | 11.20 | 5.80 | 4.80 | 100.0% | 0.00% | 0.00% |
| **Prolonged Network Suppression (Variant 2)**| 4,499 | 24.56% | 9.15 | 5.40 | 4.95 | 0.00% | 100.0% | 0.00% |
| **Non-Responsive / Stationary** | 7,222 | 39.43% | 7.80 | 7.95 | 4.82 | 17.79% | 37.75% | 44.46% |
| **Non-Responsive / Stationary (Variant 2)** | 3,830 | 20.91% | 8.10 | 8.20 | 4.90 | 15.48% | 28.59% | 55.93% |

*Total: Exactly 18,316 units ($100.0\%$).*

---

## 4. Optical Intensity-Response Relationships

Optical power dependence was evaluated across calibrated levels $P \in \{1.0, 2.5, 4.0\}\text{ mW}$ using a hierarchical mixed-effects model with specimen random effects:

### Table 4: Hierarchical Intensity-Response Slopes by Cre Line
*Source: [`results/neuroscience_study/tables/revised/revised_intensity_response_hierarchical.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_intensity_response_hierarchical.csv)*

| Cre Line | Specimens ($N$) | Mean Rate 1.0 mW (Hz) | Mean Rate 2.5 mW (Hz) | Mean Rate 4.0 mW (Hz) | Hierarchical Slope (Hz/mW) | Slope SEM | Slope 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Pvalb-IRES-Cre`** | 8 | 4.29 | 4.38 | 4.52 | **+0.076** | 0.066 | $[-0.080, +0.232]$ |
| **`Sst-IRES-Cre`** | 12 | 5.98 | 6.02 | 6.10 | **+0.039** | 0.069 | $[-0.113, +0.191]$ |
| **`Vip-IRES-Cre`** | 8 | 3.30 | 3.01 | 3.27 | **-0.009** | 0.012 | $[-0.038, +0.021]$ |

*For candidate directly driven units specifically, slopes were substantially steeper: $+57.38\text{ Hz/mW}$ in Pvalb, $+22.83\text{ Hz/mW}$ in Sst, and $+1.84\text{ Hz/mW}$ in Vip.*

---

## 5. Standardized Pulse-Train Dynamics (10-Hz Train)

Adaptation was quantified using the unified index:
$$AI = \frac{R_{10} - R_1}{\max(R_1, 0.5)}$$
evaluated across responsive units ($\text{Evoked Rate} \ge 2.0\text{ Hz}$):

### Table 5: Standardized 10-Hz Train Adaptation by Cell Line
*Source: [`results/neuroscience_study/tables/revised/revised_pulse_train_adaptation.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_pulse_train_adaptation.csv)*

| Cre Line | Adaptation Category | Unit Count | Within-Cre % | Mean $AI$ | Median $AI$ | Mean $R_1$ (Hz) | Mean $R_{10}$ (Hz) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Pvalb-IRES-Cre`** | **Depressing ($AI < -0.20$)** | 2,185 | **74.96%** | -0.378 | -0.358 | 20.88 | 11.33 |
| | Stable ($-0.20 \le AI \le +0.20$) | 728 | 24.97% | -0.109 | -0.126 | 14.96 | 13.35 |
| | Facilitating ($AI > +0.20$) | 2 | 0.07% | +0.243 | +0.243 | 16.02 | 20.17 |
| **`Sst-IRES-Cre`** | **Depressing ($AI < -0.20$)** | 4,823 | **85.02%** | -0.397 | -0.383 | 16.34 | 9.51 |
| | Stable ($-0.20 \le AI \le +0.20$) | 849 | 14.97% | -0.123 | -0.140 | 13.48 | 11.81 |
| | Facilitating ($AI > +0.20$) | 1 | 0.02% | +0.239 | +0.239 | 19.26 | 23.86 |
| **`Vip-IRES-Cre`** | Depressing ($AI < -0.20$) | 391 | 10.50% | -0.293 | -0.272 | 14.41 | 10.11 |
| | **Stable ($-0.20 \le AI \le +0.20$)** | 2,447 | **65.73%** | +0.010 | +0.014 | 13.84 | 13.97 |
| | **Facilitating ($AI > +0.20$)** | 885 | **23.77%** | +0.322 | +0.297 | 15.68 | 21.00 |

---

## 6. Distance-Dependent Response Structure along Neuropixels Shanks

### Table 6: Spatial Structure Controlled for Distance Bins
*Source: [`results/neuroscience_study/tables/revised/revised_spatial_propagation_controlled.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_spatial_propagation_controlled.csv)*

| Cre Line | Distance Bin ($\mu$m) | Unit Count | Mean Baseline (Hz) | Mean Evoked (Hz) | Median Latency (ms) | Suppression Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Pvalb-IRES-Cre`** | $0–50\ \mu\text{m}$ | 36 | 13.06 | 16.31 | 3.82 | 55.56% |
| | $50–150\ \mu\text{m}$ | 98 | 10.05 | 11.45 | 4.15 | 46.94% |
| | $150–300\ \mu\text{m}$ | 170 | 9.56 | 9.94 | 4.30 | 47.65% |
| | $300–600\ \mu\text{m}$ | 398 | 8.84 | 8.92 | 4.41 | 51.51% |
| | $> 600\ \mu\text{m}$ | 3,719 | 8.84 | 8.78 | 4.50 | 52.29% |
| **`Sst-IRES-Cre`** | $0–50\ \mu\text{m}$ | 48 | 12.15 | 14.80 | 4.62 | 47.91% |
| | $50–150\ \mu\text{m}$ | 185 | 9.85 | 11.20 | 4.70 | 48.65% |
| | $150–300\ \mu\text{m}$ | 341 | 9.12 | 9.50 | 4.83 | 53.09% |
| | $300–600\ \mu\text{m}$ | 812 | 8.65 | 8.72 | 4.80 | 51.23% |
| | $> 600\ \mu\text{m}$ | 7,102 | 8.48 | 8.35 | 4.74 | 49.04% |
| **`Vip-IRES-Cre`** | $0–50\ \mu\text{m}$ | 16 | 9.20 | 11.05 | 5.06 | 37.50% |
| | $50–150\ \mu\text{m}$ | 75 | 9.10 | 9.85 | 4.95 | 44.00% |
| | $150–300\ \mu\text{m}$ | 193 | 8.85 | 9.10 | 4.88 | 47.15% |
| | $300–600\ \mu\text{m}$ | 452 | 8.70 | 8.72 | 4.85 | 46.02% |
| | $> 600\ \mu\text{m}$ | 4,671 | 8.58 | 8.52 | 4.84 | 46.23% |

*Interpretation: Apparent latency increases weakly with distance along probe ($3.82\text{ ms} \to 4.50\text{ ms}$ in PV), corresponding to an apparent linear gradient of $\approx 0.0011\text{ ms}/\mu\text{m}$ ($v \approx 0.07\text{ m/s}$). Network suppression is broadly distributed along the shank, reflecting extensive lateral and laminar inhibitory arborization.*

---

## 7. Specimen-Level Replication & Corrected Variance Decomposition

### Table 7: Hierarchical Variance Decomposition Across 28 Specimens
*Source: [`results/neuroscience_study/tables/revised/revised_cross_specimen_reproducibility.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_cross_specimen_reproducibility.csv)*

| Metric | Grand Mean | Total Var | Between-Specimen Var | Within-Specimen Var | Within-Specimen % | ICC | Specimen-Level $t$-stat | Specimen $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Rate** | 8.64 Hz | 106.21 | 0.98 | 106.38 | **99.09%** | 0.0091 | 46.57 | $< 0.0001$ |
| **Evoked Rate** | 9.52 Hz | 287.69 | 4.28 | 300.98 | **98.60%** | 0.0140 | 24.74 | $< 0.0001$ |
| **Modulation Ratio** | 1.10 | 18.27 | 0.23 | 19.13 | **98.84%** | 0.0116 | 12.45 | $< 0.0001$ |
| **Adaptation Index ($AI$)** | -0.199 | 0.057 | 0.026 | 0.032 | **54.80%** | 0.4520 | -6.59 | $< 0.0001$ |

*Corrected Statistical Interpretation: Low ICC values ($0.01–0.02$) demonstrate that the overwhelming majority of variance ($>98\%$) reflects cellular and laminar heterogeneity within each individual mouse, whereas animal-to-animal technical variance is small ($<2\%$). Crucially, one-sample $t$-tests on specimen-level means confirm that population-level effects are highly reproducible across specimens ($p < 0.0001$).*
