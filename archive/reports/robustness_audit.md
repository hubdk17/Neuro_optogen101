# Robustness Audit: Threshold Instability, Feature Ablation, and Negative Controls

**Target Journal**: *ACM Transactions on Computing for Biology and Bioinformatics (TCBB)*  
**Project**: *Computationally Reliable Optotagging of Neuropixels Neural Recordings*  
**Date**: September 2026  
**Status**: COMPLETE ROBUSTNESS AUDIT

---

## 1. Threshold Instability Audit (The Complete 27-Point Grid)

We systematically audited the sensitivity of binary optotagging decisions across a predefined $3 \times 3 \times 3$ grid of 27 threshold configurations spanning the literature range:
- **First-Spike Latency Cutoffs**: 6.0 ms, 8.0 ms, 10.0 ms
- **Trial Reliability Cutoffs**: 0.20, 0.30, 0.50
- **Modulation Ratio Cutoffs**: 1.5, 2.0, 3.0

Evaluated on 945 units across 2 independent sessions and specimens (full records in `results/threshold_sensitivity.csv` and `figures/fig4_threshold_instability.png`):

### 1.1 Empirical Instability Summary
- **Direct Tagging Yield Range**: **2 to 21 units** (a **10.5-fold variation in tagging yield**, from 0.21% to 2.22% of recorded units).
- **Pairwise Jaccard Similarity to Baseline (8 ms, 0.30 rel, 2.0 mod)**: Collapses from **1.0000** down to **0.2000** (an 80% divergence in unit classification).
- **Population Class Switching**: Up to 13 units switch operational classes purely as a consequence of threshold adjustment.
- **Inter-Session Yield Discrepancy**:
  - In Session `721123822` (Specimen `707296982`), direct yield fluctuates from 2 units (conservative: 6ms, 0.50 rel, 3.0 mod) to 18 units (relaxed: 10ms, 0.20 rel, 1.5 mod).
  - In Session `760345702` (Specimen `739783171`), direct yield fluctuates from 0 units to 3 units.
- **Scientific Conclusion**: Binary thresholding is mathematically volatile near decision hyperplanes. An experimenter adopting $L < 10$ ms and $R \ge 0.20$ would report 21 optotagged units, whereas an experimenter adopting $L < 6$ ms and $R \ge 0.50$ would report only 2 units from the exact same electrophysiological data.

---

## 2. Leave-One-Feature-Family-Out (LOFFO) Ablation Audit

We evaluated Random Forest under hardware-held-out validation across the 6 physiological feature families (full records in `results/feature_ablation.csv` and `figures/fig7_feature_ablation.png`):

| Ablated Condition | Excluded Family | Feature Count | Balanced Accuracy | Macro F1 | AUROC | $\Delta$ Balanced Acc (%) | Criticality Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Model** | None (Baseline) | 14 | **0.7789** | **0.7789** | **0.9960** | 0.00% | Full multi-dimensional baseline |
| **Minus STATISTICAL** | $p$-value, Cohen's $d$ | 12 | **0.5586** | **0.6142** | 0.9940 | **-22.03%** | **CRITICAL ANCHOR**: Dominant driver of separation |
| **Minus RELIABILITY** | Reliability, sham FPR, Fano | 11 | 0.7313 | 0.7337 | 0.9959 | -4.76% | Essential for direct vs indirect separation |
| **Minus FIRING** | Base rate, evoked rate, mod | 11 | 0.7313 | 0.7337 | 0.9957 | -4.76% | Essential for response magnitude scaling |
| **Minus INTENSITY** | Optical power slope | 13 | 0.7789 | 0.7789 | 0.9941 | 0.00% | Secondary modulator; discrete F1 preserved |
| **Minus DYNAMICS** | 10-Hz train adaptation | 13 | 0.7789 | 0.7789 | 0.9959 | 0.00% | Secondary modulator; discrete F1 preserved |
| **Minus TEMPORAL** | Latency, jitter SD, IQR, CV | 10 | **0.8205** | **0.8242** | 0.9967 | **+4.16%** | **NOISE DISTRACTOR**: Latency has high variance in sparse units |

### Key Biological Takeaways:
1. **The Statistical Family is the True Anchor**: Removing paired permutation-test significance ($p$-value, effect size) causes a catastrophic $-22.03\%$ drop in Balanced Accuracy. Statistical testing against spontaneous baseline noise is the non-negotiable bedrock of optogenetic response verification.
2. **The Fragility of First-Spike Latency**: In low-firing units, first-spike latency acts as a noisy distractor. Withholding temporal latency features actually *improved* out-of-sample generalization ($0.7789 \to 0.8205$). This explains why classical latency-only criteria (e.g., Lima 2009) suffer from high false-negative rates in vivo.

---

## 3. Sham Negative Control & Optical Artifact Audit

To ensure that the framework does not falsely identify "direct optotagging" in baseline electrophysiological noise, the complete pipeline was executed on the matched pre-stimulus sham window $[-18, -10\text{ ms}]$ (8 ms duration, matching the evoked window):

- **Units Evaluated**: 945 units across 11 Neuropixels probes.
- **Sham Classifications**:
  - `putatively directly optotagged`: **0 units (0.00% False Positive Rate)**
  - `light-responsive / indirect or uncertain`: **1 unit (0.11%)**
  - `not light responsive`: **944 units (99.89%)**
- **Continuous Sham Evidence Score**: Mean = $0.1882 \pm 0.051$ (Max: $0.4631$), remaining well below the $0.65$ direct evidence threshold across all units.
- **Optical Artifact Blanking Verification**: Spikes occurring in $[0, 1\text{ ms}]$ (photoelectric onset transient) and $[10, 11\text{ ms}]$ (offset transient) were blanked, resulting in $0.00\%$ artifact contamination in the primary evoked window $[1, 9\text{ ms}]$.

---

## 4. Secondary Stimulation Dynamics Audit

We audited the stability of evidence scores across secondary stimulation protocols (full records in `results/secondary_stimulation.csv` and `figures/fig8_secondary_validation.png`):
1. **Pulse Width Invariance**: Comparing 10-ms vs 5-ms pulses, first-spike latency of high-evidence units remained perfectly invariant ($5.06\text{ ms}$ at 10 ms vs $5.16\text{ ms}$ at 5 ms), confirming monosynaptic kinetics.
2. **Train Adaptation**: High-evidence direct units sustained high-frequency spiking across 10-Hz pulse trains (adaptation index = $-0.73$), whereas indirect units depressed rapidly (adaptation index = $-19.19$).
3. **Regime Concordance**: High-evidence units remained high-evidence across stimulation regimes, while intermediate/borderline units remained context-dependent, validating the continuous evidence representation.
