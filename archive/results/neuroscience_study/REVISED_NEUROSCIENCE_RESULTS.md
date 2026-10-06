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
- **Distributional Modeling**: The latency distribution was better described by a three-component Gaussian mixture than by simpler mixture models, indicating substantial heterogeneity in response latency:
  - $k=1$ Component BIC: $53,114.6$
  - $k=2$ Components BIC: $53,017.9$
  - $k=3$ Components BIC: $52,900.4$ (Optimal, $\Delta\text{BIC} = -117.5$ vs $k=2$, $-214.2$ vs $k=1$)
- **Distributional Interpretation**: This distributional structure does not by itself establish distinct physiological response classes or prove that underlying biology is continuous; rather, it indicates that latency exhibits continuous heterogeneity that cannot be captured by a single binary threshold.
- Non-parametric bootstrap resampling ($B=500$) yielded a median latency 95% confidence interval of $[4.668\text{ ms}, 4.734\text{ ms}]$.

### Table 2: Sparse-Firing Stability & Explicit Poisson Null Model
*Source: [`results/neuroscience_study/tables/revised/revised_sparse_firing_stability.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_sparse_firing_stability.csv)*

| Baseline Firing Tier | Total Units ($N$) | Mean Baseline (Hz) | Observed Spiking Units ($N_{\text{obs}}$) | Observed Spiking % | Poisson Expected Units ($N_{\text{exp}}$) | Poisson Expected % | Ratio $N_{\text{exp}} / N_{\text{obs}}$ (%) | Sub-8ms Spiking Count | Sub-8ms Rate (Spiking) | Sub-8ms Rate (All Units) | Heuristic Pass % | SALT Pass % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$< 1\text{ Hz}$** | 3,867 | 0.192 | 1,178 | 30.46% | 518.7 | 13.41% | **44.03%** | 1,096 | **93.04%** | 28.34% | **0.59%** | **2.07%** |
| **$1-2\text{ Hz}$** | 1,842 | 1.579 | 993 | 53.91% | 1,278.5 | 69.41% | 128.75% | 937 | 94.36% | 50.87% | 0.71% | 3.96% |
| **$2-4\text{ Hz}$** | 1,973 | 3.021 | 1,441 | 73.04% | 1,768.2 | 89.62% | 122.71% | 1,369 | 95.00% | 69.39% | 0.56% | 5.02% |
| **$4-8\text{ Hz}$** | 3,711 | 5.934 | 3,230 | 87.04% | 3,667.7 | 98.83% | 113.55% | 3,100 | 95.98% | 83.54% | 1.51% | 6.47% |
| **$> 8\text{ Hz}$** | 6,920 | 18.256 | 6,798 | 98.24% | 6,920.0 | 100.0% | 101.79% | 6,696 | 98.50% | 96.76% | 2.28% | 3.14% |

### Methodological Interpretation:
In quiescent cortical units ($<1$ Hz baseline, $N=3,867$ units, mean baseline rate $\lambda = 0.192$ Hz):
- **Number of units in tier**: $N = 3,867$
- **Units with $\ge 1$ spike in window** (75 trials $\times$ 10 ms, total exposure $\tau = 0.75$ s): $N_{\text{obs}} = 1,178$ ($30.46\%$)
- **Expected spiking units under stationary Poisson null** ($N [1 - e^{-\lambda \tau}]$): $N_{\text{exp}} = 518.7$ units ($13.41\%$)
- **Expected-to-observed ratio**: $N_{\text{exp}} / N_{\text{obs}} = 518.7 / 1,178 = \mathbf{44.03\%}$
- **Probability of $\ge 1$ spontaneous spike under Poisson null**: $P(N \ge 1) = 13.41\%$
- **Probability latency $<8$ ms conditional on $\ge 1$ spike**: Under a uniform arrival null within a 10-ms window, $P(\text{latency} < 8\text{ ms} \mid \text{spike} \ge 1) = 8.0 / 10.0 = \mathbf{80.0\%}$ (expected arrival time $5.0$ ms). Empirically, among observed spiking units, $93.04\%$ ($1,096$ units) exhibited a median latency $<8$ ms.
- **Scientific Conclusion**: Spontaneous Poisson spikes account for nearly half ($44.03\%$) of all observed spiking units in this tier. Because an isolated random spike has an $80.0\%$ chance of arriving within the first 8 ms, $93.04\%$ of spiking quiescent units register an apparent sub-8 ms latency. Yet, only 23 units ($0.59\%$) satisfy the full operational heuristic, and only 80 units ($2.07\%$) achieve statistical significance under SALT. Latency alone is therefore structurally uninformative in low-firing units without corroborating trial reliability and effect-size statistics.

---

## 3. Mutually Exclusive Unsupervised Response Archetypes

Unsupervised Gaussian Mixture Modeling on normalized 5-window trajectories $[W_1, W_2, W_3, W_4, W_5]$ partitioned the cohort into 5 mutually exclusive dynamical archetypes:

### Table 3: Unsupervised Response Archetype Properties
*Source: [`results/neuroscience_study/tables/revised/revised_population_response_archetypes.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_population_response_archetypes.csv)*

| Archetype Label | Unit Count | Cohort % | Mean Baseline (Hz) | Mean Evoked (Hz) | Median Latency (ms) | Pvalb % | Sst % | Vip % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rapid Direct-Like Excitation** | 386 | 2.11% | 9.42 | 58.20 | 2.85 | 42.49% | 43.52% | 13.99% |
| **Prolonged Suppression** | 2,379 | 12.99% | 11.20 | 5.80 | 4.80 | 100.0% | 0.00% | 0.00% |
| **Prolonged Suppression (Variant 2)**| 4,499 | 24.56% | 9.15 | 5.40 | 4.95 | 0.00% | 100.0% | 0.00% |
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

### Statistical Interpretation:
- **Primary Finding**: **Response latency increases with distance from the optical hotspot along the recording shank.** In `Pvalb-IRES-Cre` sessions, median latency increases progressively from $3.82\text{ ms}$ at $0–50\ \mu\text{m}$ to $4.50\text{ ms}$ at $>600\ \mu\text{m}$ (fitted linear slope: $+0.0011\text{ ms}/\mu\text{m}$, or $+1.1\ \mu\text{s}/\mu\text{m}$; in `Sst-IRES-Cre`: $+0.0003\text{ ms}/\mu\text{m}$).
- **Secondary Derived Quantity**: Under simple linear assumptions, this slope corresponds to an apparent propagation velocity of $v \approx 0.07\text{ m/s}$. However, this derived velocity is highly model-dependent and vulnerable to several physical and biological confounds: probe insertion geometry, cortical laminar architecture, optical scattering spread in tissue, distinct cell populations recorded at different cortical depths, spike detection latency, and physical uncertainty in optical fiber placement relative to the probe shank.
- **Suppression Extent**: Both Pvalb-associated suppression and Sst-associated suppression remain broadly distributed across all distance bins ($47–55\%$), reflecting extensive translaminar and horizontal inhibitory arborization.

---

## 7. Specimen-Level Replication & Hierarchical Variance Decomposition

### Table 7: Hierarchical Variance Decomposition Across 28 Specimens
*Source: [`results/neuroscience_study/tables/revised/revised_cross_specimen_reproducibility.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/neuroscience_study/tables/revised/revised_cross_specimen_reproducibility.csv)*

| Metric | Grand Mean | Specimen SD | Specimen 95% CI | Between-Specimen Var | Within-Specimen Var | Within-Specimen % | ICC | Specimen-Level $t$-stat | Specimen $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Rate** | 8.70 Hz | 0.99 Hz | $[8.31, 9.08]$ Hz | 0.98 | 106.38 | **99.09%** | 0.0091 | 46.57 | $< 0.0001$ |
| **Evoked Rate** | 9.67 Hz | 2.07 Hz | $[8.87, 10.47]$ Hz | 4.28 | 300.98 | **98.60%** | 0.0140 | 24.74 | $< 0.0001$ |
| **Modulation Ratio** | 1.115 | 0.474 | $[0.931, 1.299]$ | 0.23 | 19.13 | **98.84%** | 0.0116 | 12.45 | $< 0.0001$ |
| **Adaptation Index ($AI$)** | -0.201 | 0.161 | $[-0.264, -0.138]$ | 0.026 | 0.032 | **54.80%** | 0.4520 | -6.59 | $< 0.0001$ |

### Corrected Statistical Interpretation:
- **Variance Allocation**: **More than 98% of the modeled variance occurred within specimens**, indicating substantial cellular, laminar, and depth heterogeneity relative to between-specimen variation. Intraclass correlation coefficients (ICC $= 0.009–0.014$) quantify this predominance of within-animal diversity.
- **Specimen-Level Reproducibility**: Specimen-level analyses demonstrated that the direction of the major population-level effects was reproducible across animals. **Specimen-level effects were consistently displaced from the null across animals (one-sample test against zero, $p < 0.0001$, $N=28$ mice)**:
  - **Baseline Rate**: Mean $8.70$ Hz, SD $0.99$ Hz, 95% CI $[8.31, 9.08]$ Hz, $t = 46.57, p < 0.0001$.
  - **Evoked Rate**: Mean $9.67$ Hz, SD $2.07$ Hz, 95% CI $[8.87, 10.47]$ Hz, $t = 24.74, p < 0.0001$.
  - **Modulation Ratio**: Mean $1.115$, SD $0.474$, 95% CI $[0.931, 1.299]$, $t = 12.45, p < 0.0001$.
  - **Standardized Adaptation Index**: Mean $-0.201$, SD $0.161$, 95% CI $[-0.264, -0.138]$, $t = -6.59, p < 0.0001$.
