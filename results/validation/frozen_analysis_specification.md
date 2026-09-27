# Frozen Analysis Specification: Reliability-Aware Multifeature Optotagging Pipeline

**Target Journal**: *ACM Transactions on Computing for Biology and Bioinformatics* (TCBB)  
**Document Status**: **FROZEN** (Pre-registered prior to cohort expansion)  
**Date**: September 27, 2026  
**Pipeline Version**: 1.0.0 (Deterministic CPU-first pipeline)  

---

## 1. Quality Control & Unit Inclusion Criteria

1. **Spike Sorting Quality**:
   - Only units passing the Allen Institute automated spike-sorting quality assurance are processed: `quality == "good"`.
   - Signal-to-Noise Ratio (SNR) $\ge 1.5$.
   - Presence ratio $\ge 0.90$.
   - Inter-Spike Interval (ISI) violation rate $\le 0.5\%$.
2. **Firing-Rate Guard & Insufficient Evidence Stratification**:
   - Baseline spontaneous firing rate $r_{\text{baseline}} < 0.10\text{ Hz}$ flags a unit as having `"insufficient evidence"` unless it demonstrates unequivocal direct activation ($\ge 5$ evoked spikes in window with $p < 0.001$).
   - Units with insufficient evidence are explicitly categorized and retained in the denominator for population yield calculations; they are NOT silently discarded.
3. **Artifact Contamination Flag**:
   - Any unit exhibiting $>5\%$ of evoked spikes strictly clustered within the photoelectric artifact windows ($[0.0, 1.0\text{ ms}]$ post-onset or $[9.0, 11.0\text{ ms}]$ post-offset) is flagged (`artifact_flag = True`) and subjected to multiplicative artifact gating penalty $g_A(A_i)$.

---

## 2. Stimulation Protocols & Windows

1. **Primary Protocol**:
   - Single square optical pulses of **10 ms** duration.
   - Delivered at calibrated optical power levels (1.0 mW, 2.5 mW, 4.0 mW).
   - 45 trials per intensity condition (180 total trials per session including baseline catch trials).
2. **Secondary Protocols (Validation Only)**:
   - 5-ms single optical pulse (latency invariance test).
   - 2.5-ms pulses delivered at 10 Hz for 1 s (train adaptation dynamics).
   - 1-s raised-cosine stimulation (smooth ramp dynamics).
   - *Constraint*: Secondary protocols are analyzed strictly for response consistency and are never merged into primary 10-ms pulse operational classification.
3. **Temporal Windows**:
   - **Baseline Spontaneous Window**: $[-20.0, -5.0\text{ ms}]$ relative to optical stimulus onset (duration: $15.0\text{ ms}$).
   - **Primary Evoked Response Window**: $[+1.0, +9.0\text{ ms}]$ relative to optical stimulus onset (duration: $8.0\text{ ms}$). Spikes during the first $1.0\text{ ms}$ are excluded from evoked counts to prevent photoelectric transient contamination.
   - **Matched Pre-Stimulus Sham Control Window**: $[-18.0, -10.0\text{ ms}]$ relative to optical stimulus onset (duration: $8.0\text{ ms}$, strictly matched in duration and preceding stimulus onset by $\ge 10\text{ ms}$).
   - **PSTH Binning**: $1.0\text{ ms}$ uniform bins across $[-20.0, +30.0\text{ ms}]$.

---

## 3. Physiological Feature Definitions

1. **Baseline Firing Rate ($r_{\text{baseline}}$)**:
   $$r_{\text{baseline}} = \frac{\sum_{k=1}^K N_{\text{baseline}, k}}{K \cdot \Delta t_{\text{baseline}}} \quad [\text{Hz}]$$
   where $K$ is total trials ($K=45$ per intensity), $\Delta t_{\text{baseline}} = 0.015\text{ s}$.

2. **Evoked Firing Rate ($r_{\text{evoked}}$)**:
   $$r_{\text{evoked}} = \frac{\sum_{k=1}^K N_{\text{evoked}, k}}{K \cdot \Delta t_{\text{evoked}}} \quad [\text{Hz}]$$
   where $\Delta t_{\text{evoked}} = 0.008\text{ s}$ ($[+1, +9\text{ ms}]$).

3. **Optogenetic Modulation Ratio ($M_i$)**:
   $$M_i = \frac{r_{\text{evoked}} + \epsilon}{r_{\text{baseline}} + \epsilon}$$
   where $\epsilon = 1.0\text{ Hz}$ provides variance stabilization for low-rate units.

4. **Trial-to-Trial Response Reliability ($R_i$)**:
   $$R_i = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(N_{\text{evoked}, k} \ge 1) \in [0, 1]$$
   Fraction of trials eliciting at least one spike within $[+1, +9\text{ ms}]$.

5. **First-Spike Latency ($L_i$) & Temporal Dispersion**:
   - For trials with $N_{\text{evoked}, k} \ge 1$, let $t_{k}^{(1)}$ be the timestamp of the first spike relative to onset.
   - **Median First-Spike Latency ($L_i$)**: $\text{median}(\{t_k^{(1)}\})$. If no trials elicited spikes, imputed at $25.0\text{ ms}$.
   - **Latency Standard Deviation ($J_i$, Jitter)**: $\text{std}(\{t_k^{(1)}\})$.
   - **Latency Interquartile Range**: $\text{IQR}(\{t_k^{(1)}\}) = Q_3 - Q_1$.
   - **Latency Coefficient of Variation (CV)**: $J_i / L_i$.

6. **Statistical Significance ($p_i$) & Effect Size ($d_i$)**:
   - **Significance**: Non-parametric two-sided Wilcoxon signed-rank test (or paired permutation test across $K$ trials) comparing $N_{\text{evoked}, k}$ against matched pre-stimulus counts $N_{\text{sham}, k}$.
   - **Effect Size (Cohen's $d$)**:
     $$d_i = \frac{\bar{N}_{\text{evoked}} - \bar{N}_{\text{sham}}}{\sqrt{\frac{s_{\text{evoked}}^2 + s_{\text{sham}}^2}{2}}}$$

7. **Optical Intensity Slope**:
   - Linear regression slope of evoked spike count across calibrated optical power levels (1.0, 2.5, 4.0 mW).

8. **Train Adaptation Index**:
   $$\text{AI} = \frac{N_{\text{pulse 1}} - \bar{N}_{\text{pulses 2-10}}}{N_{\text{pulse 1}} + \bar{N}_{\text{pulses 2-10}} + \epsilon} \in [-1, 1]$$

---

## 4. Predefined 27-Condition Threshold Grid

The complete $3 \times 3 \times 3$ grid is evaluated without post-hoc selection:
* **Latency ($L$)**: $6.0, 8.0, 10.0\text{ ms}$
* **Reliability ($R$)**: $0.20, 0.30, 0.50$
* **Modulation ($M$)**: $1.5\times, 2.0\times, 3.0\times$
* **Statistical Anchor (Fixed)**: $p < 0.05$, Cohen's $d > 0.10$.

**Operational Classification Rule**:
* **Putatively Directly Optotagged**: $L_i < L_{\text{th}} \land R_i \ge R_{\text{th}} \land M_i > M_{\text{th}} \land p_i < 0.05 \land d_i > 0.10 \land \neg \text{artifact\_flag}$.
* **Not Light Responsive**: $p_i \ge 0.05 \lor M_i \le 1.0 \lor R_i < 0.05$.
* **Light-Responsive / Indirect or Uncertain**: Statistically responsive ($p < 0.05$) but failing one or more direct criteria ($L \ge L_{\text{th}}$, $R < R_{\text{th}}$, or $M \le M_{\text{th}}$).
* **Insufficient Evidence**: $r_{\text{baseline}} < 0.10\text{ Hz}$ and failing direct criteria.

---

## 5. Continuous Response Evidence Score Formulation

$$E_i = g_A(A_i) \cdot \left[ w_S s_S(S_i) + w_R s_R(R_i) + w_M s_M(M_i) + w_L s_L(L_i) + w_J s_J(J_i) \right]$$

### Normalized Subscores ($s_k \in [0, 1]$):
1. **Statistical Significance Subscore**:
   $$s_S(S_i) = \frac{1}{1 + \exp\left(-\frac{-\log_{10}(p_i) - 2.0}{1.0}\right)}$$
2. **Reliability Subscore**:
   $$s_R(R_i) = \text{clip}(R_i, 0.0, 1.0)$$
3. **Modulation Subscore**:
   $$s_M(M_i) = \tanh\left(\frac{\max(0, \log_2 M_i)}{2.0}\right)$$
4. **Latency Subscore**:
   $$s_L(L_i) = \frac{1}{1 + \exp\left(\frac{L_i - 8.0}{2.0}\right)}$$
5. **Jitter Subscore**:
   $$s_J(J_i) = \exp\left(-\frac{J_i}{2.5}\right)$$
6. **Artifact Penalty Gating**:
   $$g_A(A_i) = 1.0 - \text{clip}(2.0 \cdot A_i, 0.0, 1.0)$$

### Frozen Weighting Variants (Sensitivity Testing Suite):
* **Model A (Proposed Physiological)**: $(w_S, w_R, w_M, w_L, w_J) = (0.30, 0.25, 0.20, 0.15, 0.10)$
* **Model B (Equal Weights)**: $(0.20, 0.20, 0.20, 0.20, 0.20)$
* **Model C1 (Minus Statistical)**: $(0.00, 0.36, 0.29, 0.21, 0.14)$
* **Model C2 (Minus Reliability)**: $(0.40, 0.00, 0.27, 0.20, 0.13)$
* **Model C3 (Minus Modulation)**: $(0.38, 0.31, 0.00, 0.19, 0.12)$
* **Model C4 (Minus Latency)**: $(0.35, 0.29, 0.24, 0.00, 0.12)$
* **Model C5 (Minus Jitter)**: $(0.33, 0.28, 0.22, 0.17, 0.00)$
* **Model D1 (Heavy-Statistical)**: $(0.50, 0.15, 0.15, 0.10, 0.10)$
* **Model D2 (Heavy-Reliability)**: $(0.15, 0.45, 0.15, 0.15, 0.10)$
* **Model D3 (Heavy-Latency)**: $(0.15, 0.20, 0.15, 0.40, 0.10)$
* **Model D4 (Heavy-Modulation)**: $(0.15, 0.15, 0.45, 0.15, 0.10)$

---

## 6. Multi-Source Uncertainty Characterization

$$U_i = 0.40 \cdot \text{Prox}_i + 0.35 \cdot \text{CV}_{L, i} + 0.25 \cdot \left(\frac{\text{Var}(R_i)}{0.0055}\right) \in [0, 1]$$

1. **Trial-to-Trial Reliability Sampling Variance**:
   $$\text{Var}(R_i) = \frac{R_i (1 - R_i)}{K}, \quad K=45$$
2. **Normalized Decision Boundary Proximity**:
   $$\text{Dist}_i = \sqrt{\left(\frac{L_i - 8.0}{4.0}\right)^2 + \left(\frac{R_i - 0.30}{0.20}\right)^2 + \left(\frac{\log_2 M_i - 1.0}{1.0}\right)^2}$$
   $$\text{Prox}_i = \exp(-\text{Dist}_i) \in [0, 1]$$
3. **Latency Uncertainty**:
   $$\text{CV}_{L, i} = \text{clip}\left(\frac{J_i}{\max(L_i, 1.0)}, 0.0, 2.0\right)$$

---

## 7. Machine Learning Specifications & Leakage Prevention

1. **Classifiers**:
   - Logistic Regression: `class_weight='balanced'`, $C=1.0$, `max_iter=1000`.
   - Random Forest: `n_estimators=200`, `max_depth=6`, `class_weight='balanced'`, `random_state=42`, `n_jobs=1`.
   - XGBoost: `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `tree_method='hist'`, `random_state=42`, `n_jobs=1`.
2. **Preprocessing Pipeline (Strictly within Training Folds)**:
   $$\text{Pipeline} = [\text{SimpleImputer(strategy='median')}, \text{StandardScaler()}, \text{Classifier}]$$
   *No test-fold data touches imputation, scaling, or hyperparameter selection.*
3. **Partitioning Regimes**:
   - **Primary**: Leave-One-Session-Out (`LeaveOneGroupOut(groups=session_id)`).
   - **Secondary**: Leave-One-Specimen-Out (`LeaveOneGroupOut(groups=specimen_id)`).
   - **Hardware Control**: Probe-Held-Out (`GroupKFold(n_splits=5, groups=probe_id)`).
   - **Leakage Baseline**: Random Stratified 5-Fold (`StratifiedKFold(n_splits=5, shuffle=True)`).
4. **Metrics**:
   - Balanced Accuracy, Macro F1, Precision, Recall, Specificity, AUROC (One-vs-Rest), AUPRC, Brier Score.

---

## 8. Controls and Statistical Inference

1. **Matched Pre-Stimulus Sham Control**:
   - Extracted identically on $[-18.0, -10.0\text{ ms}]$.
   - Reported as observed false-positive count $k / N$.
   - Evaluated with **exact Clopper-Pearson binomial 95% confidence interval**:
     $$\text{CI}_{95\%} = \left[ B\left(\frac{\alpha}{2}; k, n-k+1\right), B\left(1 - \frac{\alpha}{2}; k+1, n-k\right) \right]$$
2. **Permutation Control**:
   - 1,000 Monte Carlo shuffles of trial onset timestamps.
   - Empirical null distribution of pseudo-direct yields and evidence scores.
3. **Reporting Rule**:
   - If $k=0$, write: *"No false positives were observed among $N$ tested units; exact 95% CI is $[0.00\%, \text{CI}_{\text{high}}]$"*. Never write *"false positive rate = 0"*.

---

*This specification is now locked and constitutes the pre-registered benchmark for all subsequent cohort analyses.*
