# Temporal Window & Stimulus Contamination Audit

**Project**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Analysis Date**: 2026-09-28  

---

## 1. Specification of Time Windows

In `src/feature_extraction.py` and `config.yaml`, the temporal windows relative to optical pulse onset ($t=0.0\text{ s}$) are configured as follows:

| Epoch Window | Start Time ($t_0$) | Stop Time ($t_1$) | Duration ($\Delta t$) | Purpose & Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Pre-stimulus Baseline** | $-0.020\text{ s}$ ($-20\text{ ms}$) | $-0.005\text{ s}$ ($-5\text{ ms}$) | $0.015\text{ s}$ ($15\text{ ms}$) | Uncontaminated spontaneous activity before laser shutter opens. |
| **Safety Transition Gap** | $-0.005\text{ s}$ ($-5\text{ ms}$) | $0.000\text{ s}$ ($0\text{ ms}$) | $0.005\text{ s}$ ($5\text{ ms}$) | **Buffer buffer**: Pre-pulse electrical transients or optical onset jitter cannot contaminate baseline. |
| **Photoelectric Artifact Gate** | $0.000\text{ s}$ ($0\text{ ms}$) | $0.001\text{ s}$ ($1\text{ ms}$) | $0.001\text{ s}$ ($1\text{ ms}$) | Photoelectric/photovoltaic artifact transient immediately upon laser onset. Spikes here trigger artifact flag. |
| **Direct Evoked Window** | $0.001\text{ s}$ ($1\text{ ms}$) | $0.009\text{ s}$ ($9\text{ ms}$) | $0.008\text{ s}$ ($8\text{ ms}$) | Monosynaptic ChR2-evoked primary action potentials. |
| **Matched Pre-Onset Sham** | $-0.017\text{ s}$ ($-17\text{ ms}$) | $-0.009\text{ s}$ ($-9\text{ ms}$) | $0.008\text{ s}$ ($8\text{ ms}$) | Identical $8\text{-ms}$ duration window placed entirely inside pre-stimulus baseline to measure false-positive rate. |

---

## 2. Mathematical Verification of Temporal Separation

1. **Zero Overlap between Baseline and Evoked**:
   $$\text{Baseline } [-20\text{ ms}, -5\text{ ms}] \cap \text{Evoked } [1\text{ ms}, 9\text{ ms}] = \emptyset$$
   There is a guaranteed $6\text{-ms}$ separation gap ($[-5\text{ ms}, 1\text{ ms}]$) between baseline counting and evoked response counting.

2. **Spike Duration Normalization**:
   Baseline duration $b\_dur = 0.015\text{ s}$. Evoked duration $e\_dur = 0.008\text{ s}$.
   In `src/feature_extraction.py`:
   $$\text{baseline\_rate} = \frac{\sum \text{baseline\_count}}{b\_dur \times n\_trials}$$
   $$\text{evoked\_rate} = \frac{\sum \text{evoked\_count}}{e\_dur \times n\_trials}$$
   $$\text{modulation\_ratio} = \frac{\text{evoked\_rate}}{\text{baseline\_rate} + 1.0}$$
   For the paired permutation test, baseline counts are scaled by $(e\_dur / b\_dur) = (0.008 / 0.015)$ to ensure exact duration matching.

3. **No Off-By-One Indexing**:
   Spike times are continuous floating-point timestamps ($t_{spike} \in \mathbb{R}$) directly subtracted from optical pulse onset timestamp ($t_{stim}$).
   A spike is evoked if:
   $$0.001000 \le (t_{spike} - t_{stim}) < 0.009000$$
   Half-open intervals prevent boundary double-counting.

---

## 3. Findings

- **Temporal Contamination**: **None**. Spontaneous baseline spikes are strictly segregated from evoked spikes.
- **Window Overlap**: **None**.
- **Sham Control Alignment**: Exactly matched $8\text{-ms}$ window allows direct empirical false-alarm audit.
