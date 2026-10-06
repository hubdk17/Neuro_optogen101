# FINAL STATISTICAL, NUMERICAL, AND CLAIM AUDIT

**Project**: *Optogenetic Perturbation of Cortical Microcircuits: A 28-Specimen Neuropixels Study Beyond Binary Optotagging*  
**Auditor**: Computational Neurophysiology & Statistical Reviewer  
**Date**: September 2026  
**Status**: Comprehensive Reanalysis and Claim Correction Complete  

---

## 1. Numerical Inconsistencies Found & Resolved

| Component | Initial Inconsistency / Defect | Root Cause | Corrective Action & Revised Value |
| :--- | :--- | :--- | :--- |
| **Adaptation Index ($AI$)** | Values in raw master parquet ranged from $-4,399.0$ to $+1.0$; negative numbers were described as facilitation in some sections and depression in others. | In `src/feature_extraction.py`, formula used $1.0 - (p_{\text{last}} / (p_{\text{first}} + 10^{-4}))$. When $p_{\text{first}} \approx 0$, epsilon division produced massive negative values. Furthermore, $p_{\text{last}} < p_{\text{first}}$ produced positive indices, inverting conventional sign logic. | Unified formula across entire study: $AI = \frac{R_{10} - R_1}{\max(R_1, 0.5)}$. Clamped baseline to $0.5$ Hz. Negative values ($AI < -0.20$) strictly indicate **depression**; positive values ($AI > +0.20$) strictly indicate **facilitation**. Revised PV mean $AI = -0.378$, SST mean $AI = -0.397$, VIP mean $AI = +0.322$. |
| **Sparse-Firing Latency Denominator** | Initial text stated "93.4% of low-firing units pass sub-8 ms latency by chance alone", which could be misinterpreted as 93.4% of all 3,867 quiescent units. | Spontaneous spiking occurred in only $30.46\%$ ($1,178 / 3,867$) of quiescent units during the window. | Explicitly separated denominators: of $3,867$ units firing $<1$ Hz, $1,178$ ($30.46\%$) had at least one spike; of those $1,178$ units with calculable latency, $93.04\%$ ($1,096$) had median latency $<8$ ms ($28.34\%$ of all $<1$ Hz units). Added theoretical Poisson null ($13.41\%$). |
| **Method Venn Disagreement Group Counts** | Disagreement matrix initially reported 6 groups without explicit 8-group mutually exclusive Venn partitioning. | Units passing Heuristic + ZETA were lumped into broad categories. | Recomputed complete 8-way partition: All Three: 13 ($0.07\%$), Heuristic+SALT: 9 ($0.05\%$), Heuristic+ZETA: 147 ($0.80\%$), Heuristic Only: 92 ($0.50\%$), SALT+ZETA: 179 ($0.98\%$), SALT Only: 510 ($2.78\%$), ZETA Only: 1,800 ($9.83\%$), None: 15,566 ($84.99\%$). |
| **Phenotype Cluster Overlap** | Initial report described "8 distinct phenotypes", but counts summed to $>100\%$ of the cohort because categories overlapped (e.g. Adapting units also had Suppression). | Descriptive heuristics were used sequentially rather than mutually exclusive clustering. | Replaced with true unsupervised Gaussian Mixture Modeling ($k=5$ components) on normalized 5-window vectors ($W_1–W_5$), yielding mutually exclusive, non-overlapping cluster assignments summing to exactly $18,316$ units ($100.0\%$). |

---

## 2. Statistical Issues Found & Corrected

| Analysis | Statistical Flaw Identified | Statistical Remedy Applied |
| :--- | :--- | :--- |
| **Intraclass Correlation (ICC)** | Low ICC ($0.01–0.02$) was interpreted as "proof that biology is an invariant across mice." An ICC near zero actually indicates that between-specimen variance is negligible relative to within-specimen variance, meaning individual units within a mouse are as diverse as units across different mice. | Corrected interpretation: "Over 95% of total variance reflects within-specimen cellular and laminar heterogeneity rather than animal-to-animal technical variation. Separately, one-sample $t$-tests on specimen-level means confirm that population-level effects are highly reproducible across specimens ($p < 0.0001$)." |
| **Intensity Titration Curve** | Three optical power levels (1.0, 2.5, 4.0 mW) were referred to as a "fully resolved dose-response curve." | Renamed to "power-dependent response" or "intensity-response relationship." Fitted hierarchical linear mixed-effects model with specimen random effects: $R_{ij} = \beta_0 + \beta_1 P + \beta_2 C + u_{\text{specimen}} + \epsilon$. |
| **Depth-Latency Regression** | Linear slope of depth vs. latency along the probe was labeled "network propagation velocity" ($v = 0.067$ m/s) without acknowledging probe angle, 3D anatomical curvature, or synaptic integration delays. | Renamed to "apparent distance-dependent latency gradient along the Neuropixels shank." Added explicit caveats that extracellular latency gradients cannot be equated with axonal conduction velocity. |
| **Cross-Correlogram (CCG) Shifts** | CCG synchrony shifts and latency leads were described as "proving monosynaptic connectivity." | Softened to "temporal lead/lag relationships" and "perturbation-associated changes in temporal coordination." Stated that extracellular CCGs reflect shared network drive and population synchrony. |

---

## 3. Overclaims Corrected (Language Audit)

The codebase and manuscripts were scanned for unwarranted causal and absolute terminology:

| Banned / Overclaimed Term | Context Found | Replacement / Corrected Wording |
| :--- | :--- | :--- |
| **"proves" / "proves connectivity"** | Used in CCG and spatial propagation descriptions. | Replaced with: *"is consistent with"*, *"provides evidence for"*, *"the data indicate"*. |
| **"biological invariant"** | Used to describe low ICC values across specimens. | Replaced with: *"low animal-to-animal variance relative to within-specimen diversity"*. |
| **"deterministic"** | Used to describe secondary ML predicting late phenotypes. | Replaced with: *"predictive statistical association"*. |
| **"conduction / propagation velocity"** | Used for probe vertical distance vs. latency slope. | Replaced with: *"apparent distance-dependent latency gradient under linear assumptions"*. |
| **"VIP selectively inhibits interneurons"** | Used to explain absence of lateral suppression in VIP. | Replaced with: *"VIP stimulation produced substantially less detectable suppression in the sampled population than Pvalb or Sst, a pattern compatible with disinhibitory circuit models"*. |
| **"proves direct activation"** | Used when units passed the heuristic. | Replaced with: *"operationally direct-like response candidate"*. |
| **"proves 8 ms is biologically wrong"** | Used in latency distribution section. | Replaced with: *"the empirical latency distribution exhibits continuous density across the 8-ms threshold, suggesting that latency alone provides an incomplete separation"*. |

---

## 4. Disposition of Analyses

### A. Analyses Retained (Primary Manuscript)
1. **Four-Method Responsiveness Comparison**: Operational Heuristic vs. SALT vs. ZETA vs. Continuous Evidence ($E_i$).
2. **Latency Distribution & Borderline Analysis**: Full empirical distribution across 13,643 active units; GMM model comparison; 3,193 units in [6–10 ms].
3. **Sparse-Firing Poisson Null Analysis**: Clarified denominators; formal Poisson expectation $P(N \ge 1) = 1 - e^{-\lambda \tau}$; empirical vs. null spiking rates.
4. **Mutually Exclusive Dynamical Phenotyping**: 5 canonical temporal windows; 5 unsupervised GMM clusters; centroid trajectories.
5. **Hierarchical Intensity-Response Relationships**: Specimen-level random effects across 1.0, 2.5, 4.0 mW.
6. **Pulse-Train Adaptation Dynamics**: Standardized $AI$ and $R_n/R_1$ ratios; cell-type divergence (PV/SST depression vs. VIP facilitation).
7. **Spatial Response Organization Along Neuropixels Shanks**: Vertical distance vs. latency gradient, amplitude decay ($\lambda$), and suppression.
8. **Population Temporal Coordination**: Baseline vs. post-stimulation CCGs; synchrony fold-change; lead/lag shifts.
9. **Specimen-Level Replication**: Caterpillar plots across 28 mice with 95% CIs; hierarchical variance decomposition.
10. **Negative Controls Suite**: Pre-stimulus sham window (0.04%), time-shift control (0.00%), stimulus jitter, within-specimen permutation.

### B. Analyses Demoted to Exploratory / Supplementary
1. **Perturbational Response Similarity Networks**: Retained as exploratory population analysis; causal network claims removed; weak distance correlation ($r \approx -0.047$) reported descriptively.
2. **Secondary Machine Learning Diagnostics**: Retained as biological diagnostic (LOSO balanced accuracy $= 0.72$ for late phenotype, $= 0.695$ for Cre line); removed from primary figures to avoid framing as an ML benchmarking paper.

### C. Analyses Removed
1. **GNN Spatial Graph Convolutions**: Removed from primary narrative; preserved only in methodological audit documentation as proof of feature over-smoothing.
2. **Arbitrary 8-Phenotype Overlapping Classification**: Replaced with mutually exclusive GMM clusters.

---

## 5. Remaining Methodological Limitations

1. **Extracellular Spike Resolution**: Neuropixels recordings sample extracellular action potentials. They do not record subthreshold membrane potentials ($V_m$) or inhibitory postsynaptic potentials (IPSPs). Network suppression is detectable only in units with sufficient spontaneous baseline firing.
2. **Three Optical Power Levels**: Optical powers were tested at 1.0, 2.5, and 4.0 mW. While sufficient to establish non-zero intensity slopes, this does not constitute a fine-grained pharmacological Hill titration.
3. **Linear Probe Geometry**: Probes were inserted at stereotaxic angles across visual cortex. Vertical distance along the shank reflects a mixture of laminar depth and lateral displacement, not a pure radial column.
4. **Cre Line Expression Heterogeneity**: Ai32 ChR2-EYFP expression density and light penetration vary across cortical layers and specimens. Specimen-level random-effects modeling partially mitigates but cannot eliminate this biological variance.

---

# AUDIT ADDENDUM: RESOLUTION OF 7 CRITICAL REVIEWER POINTS

| Point # | Reviewer Issue | Resolution & Empirical Verification | Status |
| :--- | :--- | :--- | :---: |
| **1. Poisson Null & Denominators** | Confusing "nearly half (13.41%) of observed (30.46%)" wording and conditioning of 80% arrival. | Explicitly reported in pipeline and text: $N=3,867$ units in tier; $N_{\text{obs}} = 1,178$ ($30.46\%$); theoretical expected under Poisson $N_{\text{exp}} = 3,867(1 - e^{-0.192 \times 0.75}) = 518.7$ ($13.41\%$); ratio $N_{\text{exp}} / N_{\text{obs}} = 44.03\%$; conditional probability of latency $<8$ ms given $\ge 1$ spike under uniform arrival is $8.0 / 10.0 = 80.0\%$; empirical sub-8ms rate among spiking units is $93.04\%$ ($1,096$ units). | **VERIFIED & RESOLVED** |
| **2. GMM Mixture Heterogeneity** | Calling GMM components "continuous components" or claiming biology is continuous. | Replaced wording: "The latency distribution was better described by a three-component Gaussian mixture than by simpler mixture models, indicating substantial heterogeneity in response latency. This distributional structure does not by itself establish distinct physiological response classes." | **VERIFIED & RESOLVED** |
| **3. Archetype Naming** | "Prolonged network suppression" implies a network mechanism as a direct observation. | Renamed archetype to **Prolonged Suppression** ($N=6,878, 37.55\%$) because direct observation is extracellular firing reduction; circuit/network mechanisms are investigated subsequently. | **VERIFIED & RESOLVED** |
| **4. Spatial Response Structure** | Overemphasis on propagation velocity $v \approx 0.07$ m/s. | Primary finding established as **Distance-dependent response latency along recording shank** (fitted slope $+0.0011\text{ ms}/\mu\text{m}$ in PV, $+0.0003\text{ ms}/\mu\text{m}$ in Sst). Apparent $v \approx 0.07$ m/s designated as a secondary derived quantity with explicit caveats (geometry, lamina, scattering, depth populations, latency). | **VERIFIED & RESOLVED** |
| **5. Cell-Type Suppression Mechanism** | Extracellular spikes do not directly prove perisomatic vs. dendritic targeting. | Renamed to **Pvalb-associated suppression** and **Sst-associated suppression**. Subcellular synaptic targeting discussed in Discussion as hypotheses consistent with microcircuit literature. | **VERIFIED & RESOLVED** |
| **6. ICC Interpretation** | Claiming low ICC means between-specimen variance is specifically "technical". | Replaced wording: "More than 98% of the modeled variance occurred within specimens, indicating substantial cellular and spatial heterogeneity relative to between-specimen variation. Specimen-level analyses showed that the direction of the major population-level effects was reproducible across animals." | **VERIFIED & RESOLVED** |
| **7. Specimen-Level Statistics** | Avoiding "confirm"; t-test against zero establishes displacement from null, not reproducibility. | Replaced wording: "Specimen-level effects were consistently displaced from the null across animals (one-sample test, $p < 0.0001$)." Reported explicit mean, SD, 95% CI, and $N=28$ for baseline rate, evoked rate, modulation ratio, and adaptation index. | **VERIFIED & RESOLVED** |

