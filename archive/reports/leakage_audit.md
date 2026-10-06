# Data Leakage and Validation Partitioning Audit

**Target Journal**: *ACM Transactions on Computing for Biology and Bioinformatics (TCBB)*  
**Project**: *Computationally Reliable Optotagging of Neuropixels Neural Recordings*  
**Date**: September 2026  
**Status**: COMPLETE LEAKAGE AUDIT

---

## 1. Executive Summary: The Data Leakage Vulnerability in Systems Neuroscience

In machine learning benchmarks across computational neuroscience, a pervasive methodological flaw is the use of **random unit-level cross-validation**. When units recorded simultaneously on the same silicon shank or within the same experimental session are randomly allocated to training and test partitions:
- **Intra-recording electrical noise is shared** between train and test.
- **Global animal brain states** (arousal, locomotion, pupil dilation, cortical synchronized oscillations) are identical across partitions.
- **Animal-specific viral expression density**, opsin trafficking efficiency, and surgical light transmission properties are identical.

This report documents a systematic audit comparing four hierarchical validation regimes, demonstrating that **random unit-level cross-validation inflates Balanced Accuracy by +45.87% and Macro F1 by +50.02%**.

---

## 2. Quantitative Comparison of Validation Regimes

All evaluations were conducted on the identical model architecture (Random Forest, 100 estimators, Model D 14-feature set) across the multi-session Neuropixels corpus (945 units across 11 physical probes):

| Validation Regime | Grouping Variable | $N_{\text{train}}$ | $N_{\text{test}}$ | Balanced Accuracy | Macro F1 | AUROC (OVR) | AUPRC | Brier Score | ECE | Intra-Session Leakage? | Intra-Specimen Leakage? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Random Unit Split** | None (Unit i.i.d.) | 756 | 189 | **0.8205** | **0.8242** | 0.9950 | 0.7964 | 0.0207 | 0.0083 | **YES (Severe)** | **YES (Severe)** |
| **2. Probe / HW Held-Out**| `probe_id` (6 shanks) | ~350 | ~95 | **0.7372** | **0.7325** | 0.9945 | 0.7947 | 0.0237 | 0.0067 | **YES (Shared session)** | **YES (Shared animal)** |
| **3. Session Held-Out (LOGO)**| `session_id` | ~400 | ~400 | **0.5625 $\pm$ 0.0625** | **0.5494 $\pm$ 0.0044** | **0.9970 $\pm$ 0.0018** | **0.8054 $\pm$ 0.0141** | **0.0208 $\pm$ 0.0090** | **0.0096** | **NO (Zero test data in train)** | Controlled |
| **4. Specimen Held-Out (LOGO)**| `specimen_id` | ~400 | ~400 | **0.5625 $\pm$ 0.0625** | **0.5494 $\pm$ 0.0044** | **0.9970 $\pm$ 0.0018** | **0.8054 $\pm$ 0.0141** | **0.0208 $\pm$ 0.0090** | **0.0096** | **NO (Zero test data in train)** | **NO (Zero test animal in train)** |

---

## 3. Quantification of Metric Inflation

Comparing naive random splitting against held-out session truth reveals substantial metric inflation:

$$\text{Inflation}_{\text{relative}} = \frac{\theta_{\text{random}} - \theta_{\text{held-out}}}{\theta_{\text{held-out}}} \times 100\%$$

| Comparison | Evaluated Metric | Naive Value | Held-Out Value | Absolute Inflation | Relative Inflation (%) | Source of Data Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Split vs Session Held-Out** | Balanced Accuracy | 0.8205 | 0.5625 | **+0.2580** | **+45.87%** | Shared electrical noise, brain state, and animal opsin yield |
| **Random Split vs Session Held-Out** | Macro F1 Score | 0.8242 | 0.5494 | **+0.2748** | **+50.02%** | Shared electrical noise, brain state, and animal opsin yield |
| **Probe Held-Out vs Session Held-Out** | Balanced Accuracy | 0.7372 | 0.5625 | **+0.1747** | **+31.06%** | Shared animal-specific opsin expression and laser fiber alignment |
| **Session Held-Out vs Specimen Held-Out**| Balanced Accuracy | 0.5625 | 0.5625 | **0.0000** | **0.00%** | Each session represents an independent biological mouse |

---

## 4. Why AUROC Remains High While Balanced Accuracy Drops

A striking scientific finding in Table 5 and Figure 6 is that **AUROC persists at $0.9970 \pm 0.0018$** across held-out sessions, while **Balanced Accuracy drops to $0.5625$**:

1. **AUROC measures ranking between light-responsive and non-responsive units**: In both sessions, 80% to 90% of units are completely quiescent or unaffected by light (`not light responsive`). Firing rate modulation and permutation test $p$-values separate responsive from non-responsive units cleanly across animals, yielding near-perfect ranking area (AUROC $> 0.99$).
2. **Discrete classification is sensitive to biological yield disparity**:
   - In Session `721123822` (Specimen `707296982`), there were 7 directly optotagged units (1.8% of clean units).
   - In Session `760345702` (Specimen `739783171`), there was only **1** directly optotagged unit (0.2% of clean units).
   - When trained on Session 760345702, the classifier learns decision boundaries for the direct class from a single positive exemplar ($N=1$).
   - When trained on Session 721123822 and tested on Session 760345702, subtle differences in surgical optic fiber placement and light penetration shift the boundary, yielding zero false positives but missing the sole direct unit at standard argmax cutoff (Recall = 0.5000).
3. **Takeaway for TCBB**: High AUROC does not prove invariant discrete classification across animals. Evaluating models across random splits creates a false illusion of robust discrete classification accuracy ($>0.82$), masking the true biological heterogeneity of in vivo optogenetic experiments.

---

## 5. Preprocessing Leakage Audit: Complete Verification

We audited all data processing operations in `src/models.py`, `src/validation.py`, and `src/pipeline.py` to confirm zero information leakage:

1. **Median Imputation**: Imputer is fitted strictly on `X_train` inside each cross-validation fold using `sklearn.pipeline.Pipeline`. Test-fold medians never influence training.
2. **Standard Scaling**: Normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on `X_train`.
3. **Feature Selection**: Feature families are fixed *a priori* by biological domain knowledge; no data-driven feature selection was performed on test partitions.
4. **Threshold Estimation**: Heuristic boundaries are static literature standards, not fit to training distributions.
5. **Probability Calibration**: Probabilistic predictions and calibration errors (Brier, ECE) are recorded exclusively on out-of-sample test folds.

**Conclusion**: The computational pipeline is strictly leak-free.
