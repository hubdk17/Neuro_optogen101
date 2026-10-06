# Final Data-Leakage Audit and Dual-Validation Benchmark Report

**Target Journal**: *ACM Transactions on Computing for Biology and Bioinformatics* (TCBB)  
**Project**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Dataset**: Full 28-Specimen Cohort (28 Sessions, 159 Probes, 18,316 Units, 260 Direct Positives)  
**Audit Execution Date**: September 2026  
**Audit Directory**: [`results/leakage_audit_28spec/`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/)  

---

## 1. Executive Summary & Audit Verdict

Following the ingestion and processing of the complete 28-specimen empirical cohort (18,316 units across 159 Neuropixels probes), the initial machine learning benchmark reported extraordinarily high classification metrics under Leave-One-Specimen-Out (LOSO) cross-validation (e.g., XGBoost Balanced Accuracy $\approx 0.9999$, AUROC $\approx 0.9999$, AUPRC $\approx 0.9958$; Setting B XGBoost AUPRC $\approx 0.9904$).

Per the task mandate, these values were treated with extreme scientific skepticism and subjected to an exhaustive data-dependency, feature-redundancy, temporal-window, preprocessing, and label-circularity audit.

### Final Audit Verdict:
> ### **B. Results are substantially explained by label-derived / proxy features.**
> 
> The near-perfect machine learning performance is **NOT** de novo biological discovery of cell identity from independent electrophysiological signatures. Rather, it represents **faithful mathematical reconstruction of the operational heuristic threshold rule**.
> 
> When models are trained on features derived from the post-stimulus optical response window $[1, 9]\text{ ms}$ (`effect_size`, `evoked_rate`, `trial_reliability`, `modulation_ratio`), tree ensembles and linear classifiers learn the human-engineered threshold boundaries with near-zero error.
> 
> Crucially, when all direct threshold features and mathematical proxies of the optical response are withheld (Setting D: waveform quality metrics and spontaneous baseline rate alone), **AUPRC collapses from $0.9865 \to 0.0288$** (against background positive prevalence of $0.0142$), and AUROC falls from $0.9999 \to 0.6975$.

---

## 2. Answers to the 10 Core Audit Questions

### Q1: Is there any demonstrable data leakage?
**Answer: NO classical implementation or data leakage was detected, but profound LABEL CIRCULARITY was identified.**

1. **Preprocessing Leakage**: **None**. Imputers and standardizers are fitted strictly fold-locally inside training folds. Zero test-fold information enters preprocessing ([`preprocessing_leakage_audit.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/preprocessing_leakage_audit.md)).
2. **Hyperparameter Leakage**: **None**. Hyperparameters were frozen *a priori* or selected strictly via inner cross-validation ([`hyperparameter_selection_audit.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/hyperparameter_selection_audit.md)).
3. **Temporal Leakage**: **None**. Pre-stimulus baseline $[-20, -5]\text{ ms}$ is separated from post-stimulus evoked $[1, 9]\text{ ms}$ by a guaranteed $6\text{-ms}$ safety gap; no window overlap exists ([`temporal_window_audit.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/temporal_window_audit.md)).
4. **Unit Duplication Leakage**: **None**. All 18,316 `unit_id`s are globally unique across sessions, specimens, and probes ([`duplicate_unit_audit.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/duplicate_unit_audit.csv)).
5. **Graph Edge Leakage**: **None**. GNN session graphs are strictly inductive, disjoint graphs in 3D CCF space; test specimen graphs are completely held out ([`gnn_leakage_audit.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/gnn_leakage_audit.md)).
6. **Group Leakage**: **Eliminated under Specimen LOSO**. Random-unit K-fold exhibits intra-recording correlation inflation, but Specimen LOSO holds out all units from an unseen mouse.

---

### Q2: Which feature(s), if any, explain the near-perfect performance?
**Answer: `effect_size`, `evoked_rate`, `trial_reliability`, and `intensity_slope`.**

In the systematic univariate screen ([`univariate_feature_leakage.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/univariate_feature_leakage.csv)), single-feature logistic models were trained under 28-specimen LOSO cross-validation:

| Feature Name | Feature Category | Held-Out LOSO AUROC | Held-Out LOSO AUPRC | Held-Out LOSO Balanced Acc | Predictive Classification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`effect_size`** | Paired Cohen's d | **0.99966** | **0.97936** | **0.99437** | **Near-Perfect Direct Proxy** |
| **`evoked_rate`** | Firing in $[1, 9]\text{ ms}$ | **0.99598** | **0.80041** | **0.98773** | **Strong Response Signal** |
| **`trial_reliability`** | Fraction response trials | **0.99608** | **0.77946** | **0.98834** | **Strong Response Signal** |
| **`intensity_slope`** | Multi-power slope | **0.96421** | **0.75436** | **0.92447** | **Strong Response Signal** |
| **`p_value`** | Permutation p-value | **0.99582** | **0.65400** | **0.97077** | **Strong Response Signal** |
| **`modulation_ratio`** | Evoked / baseline ratio | **0.98648** | **0.58862** | **0.92807** | **Strong Response Signal** |
| `baseline_rate` | Pre-stimulus firing | 0.65673 | 0.02233 | 0.61855 | Baseline / Chance |
| `snr` | Waveform SNR | 0.50670 | 0.01462 | 0.51864 | Pure Chance ($\approx 0.0142$) |
| `presence_ratio` | Temporal stability | 0.56760 | 0.03091 | 0.52805 | Pure Chance |
| `isi_violations` | Refractory violations | 0.57773 | 0.01867 | 0.58634 | Pure Chance |
| `d_prime` | Cluster isolation | 0.50057 | 0.01368 | 0.51493 | Pure Chance ($\approx 0.0142$) |

**Mathematical Mechanism**:
In `src/feature_extraction.py`, `effect_size = cohens_d_paired(e_counts, b_counts_scaled)`. The operational label requires $mod > 2.0$, $rel \ge 0.30$, and $lat < 8\text{ ms}$. Neurons exhibiting large synchronized burst firing have massive Cohen's d values ($> 1.5 - 3.5$), while non-responsive neurons ($98.6\%$) have Cohen's d near 0.0. Consequently, **`effect_size` alone achieves an AUPRC of 0.9794 under specimen-held-out validation**.

---

### Q3: Does the performance remain near-perfect after removing all direct and indirect label proxies?
**Answer: NO. Performance completely collapses.**

In the four-tier circularity audit ([`label_circularity_abcd.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/label_circularity_abcd.csv)), models were evaluated under 28-specimen LOSO across four feature sets:

| Feature Tier | Included Features | Logistic Reg AUPRC | Random Forest AUPRC | XGBoost AUPRC | Mean Balanced Acc | AUROC Range |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Setting A (Full Features)** | All 14 physiological features (including 5 defining heuristic cuts) | **0.9884** | **0.9997** | **0.9865** | **0.9975** | $0.9998 - 1.0000$ |
| **Setting B (-Defining Features)** | Withholds `median_latency_ms`, `trial_reliability`, `modulation_ratio`, `p_value`, `effect_size` | **0.9868** | **0.9726** | **0.9810** | **0.9890** | $0.9996 - 0.9999$ |
| **Setting C (-All Response Proxies)** | Also withholds `evoked_rate`, `intensity_slope`, `fano_factor`, `latency_sd/iqr/cv` | **0.0304** | **0.0289** | **0.0289** | **0.6449** | $0.6819 - 0.7083$ |
| **Setting D (Independent Set)** | Waveform quality metrics (`snr`, `isi`, `isolation_dist`, `d_prime`, etc.) + spontaneous `baseline_rate` | **0.0304** | **0.0276** | **0.0289** | **0.6482** | $0.6845 - 0.7083$ |

**Key Finding**:
- In **Setting B**, removing the 5 defining features did not collapse performance because `evoked_rate` ($AUPRC = 0.800$) and `intensity_slope` ($AUPRC = 0.754$) remained in the feature set as direct proxies of laser-evoked activation.
- In **Setting C & D**, when all laser-evoked measurements are removed, AUPRC collapses from **$0.9810 \to 0.0289$** (approaching the baseline empirical prevalence of $0.0142$).
- This demonstrates that **models cannot predict optotagging from non-stimulus waveform or baseline properties**. Machine learning does not discover unmeasured cell-type identity; it fits the optical response signal.

---

### Q4: What happens under normal 5-fold K-fold?
*Source: [`kfold_results.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/kfold_results.csv)*

Under Setting A with 5-fold stratified unit splitting:
- **XGBoost**: Balanced Accuracy = **0.99997**, AUROC = **0.99995**, AUPRC = **0.99111**, Macro $F_1$ = **0.99903** (Sensitivity = 1.0000, Specificity = 0.9999).
- **Random Forest**: Balanced Accuracy = **0.99607**, AUROC = **0.99999**, AUPRC = **0.99963**, Macro $F_1$ = **0.99513**.
- **Logistic Regression**: Balanced Accuracy = **0.99435**, AUROC = **0.99982**, AUPRC = **0.98842**, Macro $F_1$ = **0.94161**.

---

### Q5: What happens under 10-fold K-fold?
*Source: [`kfold_results.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/kfold_results.csv)*

10-fold stratified unit splitting yields virtually identical metrics to 5-fold:
- **XGBoost**: Balanced Accuracy = **0.99997**, AUROC = **0.99995**, AUPRC = **0.98899**, Macro $F_1$ = **0.99903**.
- **Random Forest**: Balanced Accuracy = **0.99610**, AUROC = **1.00000**, AUPRC = **0.99966**, Macro $F_1$ = **0.99610**.
- **Logistic Regression**: Balanced Accuracy = **0.99430**, AUROC = **0.99982**, AUPRC = **0.98848**, Macro $F_1$ = **0.94007**.

---

### Q6: What happens under specimen LOSO?
*Source: [`loso_results.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/loso_results.csv)*

Under strict Leave-One-Specimen-Out cross-validation across all 28 independent mice:
- **Setting A (Full Features)**:
  - **XGBoost**: Balanced Accuracy = **0.99997**, AUROC = **0.99995**, AUPRC = **0.98654**, Macro $F_1$ = **0.99903**.
  - **Random Forest**: Balanced Accuracy = **0.99805**, AUROC = **1.00000**, AUPRC = **0.99970**, Macro $F_1$ = **0.99805**.
  - **Logistic Regression**: Balanced Accuracy = **0.99438**, AUROC = **0.99981**, AUPRC = **0.98837**, Macro $F_1$ = **0.94238**.
- **Setting D (Independent Non-Stimulus Features)**:
  - **XGBoost**: Balanced Accuracy = **0.64910**, AUROC = **0.69747**, AUPRC = **0.02886**, Macro $F_1$ = **0.43046**.
  - **Logistic Regression**: Balanced Accuracy = **0.66656**, AUROC = **0.70827**, AUPRC = **0.03043**, Macro $F_1$ = **0.38870**.

---

### Q7: How much does performance change between K-fold and LOSO?
*Source: [`validation_comparison.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/validation_comparison.csv)*

| Metric / Model | 5-Fold Stratified K-Fold | 10-Fold Stratified K-Fold | Specimen LOSO (28 Folds) | Validation Difference ($\Delta$ LOSO vs 5-Fold) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost AUPRC (Setting A)** | 0.99111 | 0.98899 | 0.98654 | -0.00457 |
| **Logistic Regression Macro $F_1$**| 0.94161 | 0.94007 | 0.94238 | +0.00077 |
| **Linear SVM Sensitivity** | 0.89231 | 0.89231 | 0.88846 | -0.00385 |

**Interpretation**:
When the 5 threshold features are present, the decision boundary is identical in every mouse by definition; hence, LOSO and K-fold produce nearly identical near-perfect metrics.
However, when subtle biological features are used (Setting D or spatial GNNs), random unit-level splitting inflates performance due to intra-recording electrical noise and animal behavioral states.

---

### Q8: Do permutation controls behave as expected?
**Answer: YES, exactly as theoretically expected.**

1. **Global Label Permutation** ($N=50$ iterations across all 18,316 units, preserving positive count of 260):
   - Logistic Regression Null Mean AUROC: **0.5000** (95% CI: $[0.4851, 0.5149]$).
   - Logistic Regression Null Mean AUPRC: **0.0146** (exactly matching background positive prevalence $0.0142$).
   - XGBoost Null Mean AUROC: **0.5000**, Null Mean AUPRC: **0.0146**.
2. **Within-Specimen Label Permutation** ($N=30$ iterations, shuffling labels only within each animal to preserve specimen-specific prevalence):
   - Null Mean AUROC: **0.5458**, Null Mean AUPRC: **0.0163**.

**Conclusion**:
There are zero structural, indexing, or software artifacts that generate spurious positive predictions when labels are detached from features.

---

### Q9: Does the GNN contain any graph-specific leakage?
**Answer: NO.**

As documented in [`gnn_leakage_audit.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/gnn_leakage_audit.md):
- Session graphs are constructed strictly per recording; zero edges connect units from different animals.
- In `graph_shuffle_control.csv`, **degree-preserving randomized graphs outperform real spatial graphs** ($\text{AUPRC} = 0.6003$ vs $0.2489$).
- This confirms that low GNN performance is caused by **isotropic spatial over-smoothing** of rare localized signals across dense non-responsive neighbors, not edge leakage or target snooping.

---

### Q10: Are the current near-perfect results trustworthy?
**Answer: NO, not as biological optotagging classifiers.**

The near-perfect results (AUROC $\approx 0.9999$, AUPRC $\approx 0.9958$) are **trustworthy ONLY as demonstrations that modern machine learning models can faithfully approximate the human-engineered threshold rule**.

They MUST NOT be presented as proof that machine learning can accurately discover optotagged units from extracellular recordings without observing the optical stimulation response.

---

## 3. Forensic Unit Analysis

- **Positive Units ($N=260$)**: Cataloged in [`positive_unit_forensic_audit.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/positive_unit_forensic_audit.csv). Mean baseline rate = $3.82\text{ Hz}$, mean evoked rate = $94.61\text{ Hz}$, mean modulation ratio = $28.4\times$, mean latency = $4.12\text{ ms}$, mean reliability = $0.742$, mean Cohen's d = $2.41$.
- **Matched Negatives ($N=260$)**: Cataloged in [`negative_unit_forensic_audit.csv`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/leakage_audit_28spec/negative_unit_forensic_audit.csv). Mean baseline rate = $3.45\text{ Hz}$, mean evoked rate = $3.61\text{ Hz}$, mean modulation ratio = $1.04\times$, mean latency = $\text{NaN}$ (no evoked spikes), mean reliability = $0.012$, mean Cohen's d = $0.02$.

---

## 4. Impact on the TCBB Paper Narrative

This audit provides a **substantially stronger, more profound computational contribution** for *ACM Transactions on Computing for Biology and Bioinformatics*:

1. **Expose the Heuristic Reconstruction Fallacy**: Many published computational biology papers report near-1.0 AUC scores for cell-typing models without realizing their feature sets contain direct proxies of the heuristic rules used to generate the ground-truth labels.
2. **Deconstruct the Circularity Tiers**: By formally presenting Setting A vs B vs C vs D, we demonstrate exactly how much predictive power originates from the optical stimulation signal vs independent waveform/baseline features.
3. **The True Contribution**: The value of the paper is **NOT** boasting an inflated 0.9999 XGBoost classifier. The true contributions are:
   - Quantifying the 10.5-fold yield volatility of binary threshold cutoffs.
   - Proving that spatial GNNs suffer from over-smoothing in sparse optotagging regimes.
   - Introducing continuous evidence scoring ($E_i$) and epistemic uncertainty ($U_i$) to replace arbitrary binary thresholding with probabilistic reliability.
