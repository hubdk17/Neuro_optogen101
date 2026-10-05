# Beyond Binary Optotagging: A Computational Framework for Characterizing Optogenetic Responses in Neuropixels Recordings

[![Target Venue: ACM TCBB](https://img.shields.io/badge/Target%20Venue-ACM%20TCBB-blue.svg)](https://tcbb.acm.org/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved%20(Proprietary)-red.svg)](#-strict-proprietary-license--anti-plagiarism-policy)
[![Cohort: Allen Neuropixels](https://img.shields.io/badge/Cohort-18%2C316%20Units%20%7C%2028%20Mice-green.svg)](https://portal.brain-map.org/)
[![Repository URL](https://img.shields.io/badge/GitHub-hubdk17%2FNeuro__optogen101-informational.svg)](https://github.com/hubdk17/Neuro_optogen101)

**Official Repository:** [https://github.com/hubdk17/Neuro_optogen101](https://github.com/hubdk17/Neuro_optogen101)  
**Git Remote URL:** `https://github.com/hubdk17/Neuro_optogen101.git`

---

## ⚠️ STRICT PROPRIETARY NOTICE & ANTI-PLAGIARISM POLICY

> **IMPORTANT LEGAL & ACADEMIC NOTICE:**  
> **Copyright &copy; 2026. All Rights Reserved.**  
> 
> The computational methods, algorithms, mathematical derivations, analytical pipelines, empirical findings, figures, tables, and manuscript materials in this repository represent original, novel, and proprietary research conducted by the author(s).

### Terms of Access and Use:
1. **Strict Prohibition on Unauthorized Copying & Plagiarism:**  
   No individual, group, laboratory, or organization is permitted to copy, reproduce, scrape, mirror, re-host, redistribute, sublicense, or commercially exploit any portion of the novel code, methodologies, analytical algorithms, or manuscript draft text found in this repository.
2. **Prohibition of Third-Party Submissions:**  
   Submitting this work, whether in whole or in part, or any paraphrased derivation thereof, to any journal, conference, preprint archive (e.g., bioRxiv, arXiv, TechRxiv), thesis, contest, or academic institution as original work by anyone other than the official authors is **strictly prohibited and constitutes academic plagiarism and intellectual property infringement**.
3. **Scholarly Citation Requirement:**  
   If this repository, its methodology, or its findings inform your independent scholarly work, explicit formal attribution and citation to this official repository are mandatory:
   - **Repository:** `https://github.com/hubdk17/Neuro_optogen101.git`
   - **Authors:** *Computational Neurophysiology and Bioinformatics Consortium*
4. **Enforcement:**  
   Any unauthorized distribution, re-licensing, or plagiarism of these assets will result in immediate DMCA takedown actions, formal notifications to institutional research integrity boards, and appropriate legal action.

For licensing permissions, research partnerships, or academic inquiries, please open an issue or contact the repository owner directly at [https://github.com/hubdk17/Neuro_optogen101](https://github.com/hubdk17/Neuro_optogen101).

---

## 📌 Executive Scientific Overview

High-density silicon microelectrode arrays (Neuropixels) coupled with optogenetic perturbation allow simultaneous recording of thousands of cortical neurons across layers and columns. In standard neurophysiology practice, optogenetic responses are collapsed into a **single binary heuristic label ("optotagged" vs. "non-tagged")** based on fixed latency cutoffs ($< 8$ ms), reliability thresholds, and fold-change metrics.

This project introduces a **principled computational framework** that reframes optogenetic perturbation from a discrete classification filter into a **multidimensional perturbational response trajectory**.

```
Standard Optotagging Pipeline:
[Optogenetic Pulse] ──▶ [Heuristic Latency / Reliability Cutoff] ──▶ Binary Label: Tagged (1) vs. Untagged (0)
                        └── Severe Information Loss: Discards 98% of Network Dynamics

Our Computational Framework:
[Optogenetic Pulse] ──▶ Multi-Method Statistical Inference (SALT, ZETA, Continuous Score)
                     ──▶ Stationary Poisson Null Conditioning (Quiescent Cortical Units)
                     ──▶ Unsupervised Trajectory Archetyping (x_i ∈ ℝ⁵ GMM Clustering)
                     ──▶ Standardized Pulse-Train Adaptation Index (AI)
                     ──▶ Distance-Dependent Multi-Electrode Spatial Propagation
                     ──▶ Specimen-Aware Hierarchical Mixed-Effects Variance Decomposition
```

---

## 🔬 Dataset & Cohort

We analyze the complete open-access **Allen Visual Coding Neuropixels** optogenetic cohort across 28 biological specimens (mice), 159 probes, and 18,316 well-isolated single units:

| Cre Driver Line | Specimens ($N$) | Probes ($N$) | Units ($N$) | Mean Units/Probe | Optical Stimulus Protocols |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Pvalb-IRES-Cre`** | 8 | 45 | 4,421 | 98.2 | Single 10-ms pulses (1.0, 2.5, 4.0 mW) + 10-Hz trains |
| **`Sst-IRES-Cre`**   | 12 | 70 | 8,488 | 121.3 | Single 10-ms pulses (1.0, 2.5, 4.0 mW) + 10-Hz trains |
| **`Vip-IRES-Cre`**   | 8 | 44 | 5,407 | 122.9 | Single 10-ms pulses (1.0, 2.5, 4.0 mW) + 10-Hz trains |
| **Total Cohort**     | **28** | **159** | **18,316** | **115.2** | **Full 225-trial pulse protocol + 10-Hz trains** |

---

## 🚀 Key Methodological & Computational Innovations

1. **Multi-Lens Statistical Responsiveness & Algorithm Divergence:**
   - Evaluated all 18,316 units across the **Operational Heuristic**, **SALT** (Kvitsiani et al. 2013), and **ZETA** (Montijn et al. 2021).
   - Revealed stark algorithmic divergence: 3-way consensus identifies only **13 units (0.07%)**, while operational heuristics discard **179 units** achieving dual significance ($p < 0.05$) under SALT and ZETA simply because their reliability ($0.082$) falls below the arbitrary $0.30$ boundary.
2. **Explicit Stationary Poisson Null Model for Sparse Latency Fragility:**
   - In quiescent cortex ($< 1$ Hz baseline, $N=3,867$ units, mean $\lambda = 0.192$ Hz), isolated spontaneous events under a uniform arrival null within 10 ms have an **expected arrival time of 5.0 ms** and an **80.0% probability of sub-8 ms latency**.
   - Spontaneous Poisson events account for **44.03% (518.7 / 1,178)** of observed spiking units in this tier, explaining why **93.04% (1,096 units)** display apparent direct latencies despite failing statistical significance tests.
3. **Trajectory-Based Unsupervised Response Archetyping:**
   - Established a 5-window temporal feature space $\mathbf{x}_i \in \mathbb{R}^5$ covering:
     - $W_1$ ($0–8$ ms): Direct candidate optical activation
     - $W_2$ ($8–20$ ms): Early circuit recruitment
     - $W_3$ ($20–50$ ms): Lateral inhibition and feedback
     - $W_4$ ($50–200$ ms): Prolonged network suppression
     - $W_5$ ($200–500$ ms): Rebound and late recovery
   - Discovered that **lateral Prolonged Suppression (37.55%, $N=6,878$)** dominates the active cortical response space over rapid excitation ($2.11\%$, $N=386$).
4. **Standardized Pulse-Train Adaptation Index ($AI$):**
   - Formulated a zero-safe, bounded metric:
     $$AI = \frac{R_{10} - R_1}{\max(R_1, 0.5)}$$
   - Across $N=12,311$ train-responsive units, captured clear cell-type divergence: `Pvalb` ($74.96\%$ depressing) and `Sst` ($85.02\%$ depressing) vs. `Vip` ($65.73\%$ stable, $23.77\%$ facilitating).
5. **Distance-Dependent Spatial Latency Organization:**
   - Physical distance along the Neuropixels shank reveals systematic latency increases ($+1.1\ \mu\text{s}/\mu\text{m}$ in PV, $+0.3\ \mu\text{s}/\mu\text{m}$ in SST) relative to the optical centroid.
6. **Stimulus-Associated Population Temporal Coordination (CCGs):**
   - Multi-unit cross-correlograms between direct candidates and network units reveal a **4.04-fold surge** in coincident synchrony in `Pvalb` sessions (lead $+3.2$ ms) and a **2.50-fold surge** in `Sst` sessions (lead $+4.8$ ms).
7. **Hierarchical Variance Decomposition (Preventing Pseudo-Replication):**
   - One-way random-effects ANOVA and Intraclass Correlation Coefficients ($\text{ICC} = 0.009–0.014$) prove that **$> 98\%$ of modeled variance resides within specimens**, while specimen-level effects are consistently displaced from null across all 28 independent mice ($p < 0.0001$).

---

## 📂 Repository Directory Structure

```
Neuro_optogen101/
├── README.md                                  # Strict proprietary notice, methodology, and documentation
├── .gitignore                                 # Git rules excluding large NWB raw files, cache, and virtualenvs
├── manuscript/                                # Full publication manuscript (ACM TCBB / IEEE compsoc)
│   ├── main.tex                               # Primary LaTeX source configured for Overleaf
│   ├── manuscript_tcbb.tex                    # Identical LaTeX source for local and automated builds
│   ├── references.bib                         # 30 audited peer-reviewed BibTeX citations
│   ├── figures/                               # 10 publication-quality vector PDFs and 300 DPI PNG figures
│   │   ├── figure1_experimental_paradigm_and_cohort.{pdf,png}
│   │   ├── figure2_perturbational_response_space.{pdf,png}
│   │   ├── figure3_method_disagreement_and_divergence.{pdf,png}
│   │   ├── figure4_latency_and_sparse_firing_reliability.{pdf,png}
│   │   ├── figure5_cell_type_fingerprints.{pdf,png}
│   │   ├── figure6_intensity_response_relationships.{pdf,png}
│   │   ├── figure7_pulse_train_dynamics_and_adaptation.{pdf,png}
│   │   ├── figure8_distance_dependent_response_structure.{pdf,png}
│   │   ├── figure9_population_temporal_coordination.{pdf,png}
│   │   └── figure10_specimen_replication_and_variance.{pdf,png}
│   └── tables/                                # Supplementary and raw numerical data tables
├── src/                                       # Core analytical modules
│   ├── feature_extraction.py                  # Single-unit feature extraction & trial alignment
│   ├── responsiveness_methods.py              # SALT, ZETA, and heuristic responsiveness algorithms
│   ├── artifact_control.py                    # Sham window and permutation negative controls
│   └── pipeline.py                            # End-to-end execution pipeline
├── scripts/                                   # Statistical audits, figure generators, and workflows
│   ├── generate_revised_neuroscience_figures.py # Script generating all 10 unoccluded publication figures
│   ├── run_rigorous_neuroscience_reanalysis.py  # Full statistical reanalysis & GMM archetyping
│   ├── run_scientific_audit.py                # Master numerical consistency audit across 28 specimens
│   └── run_leakage_audit_data_dependencies.py # Label-circularity and leakage audit script
└── results/                                   # Processed tables and analytical benchmarks
    └── neuroscience_study/
        ├── REVISED_NEUROSCIENCE_RESULTS.md    # Authoritative summary of all verified empirical metrics
        └── tables/revised/                    # Final CSV tables (Venn disagreement, Poisson tiers, etc.)
```

---

## 🛠️ Environment Setup & Reproduction

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation
```bash
# Clone the official repository
git clone https://github.com/hubdk17/Neuro_optogen101.git
cd Neuro_optogen101

# Create and activate a Python virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt  # or install numpy scipy pandas scikit-learn matplotlib seaborn pyarrow
```

### Reproducing Figures & Analyses
```bash
# Generate all 10 publication-quality figures (saved as PNG + PDF in manuscript/figures/)
python scripts/generate_revised_neuroscience_figures.py

# Run the master numerical consistency audit across all 18,316 units
python scripts/run_scientific_audit.py
```

---

## 📖 Citation

If you use or reference the methods, findings, or code from this study in your research, please cite:

```bibtex
@misc{hubdk17_neuro_optogen101_2026,
  author       = {{Computational Neurophysiology and Bioinformatics Consortium}},
  title        = {Beyond Binary Optotagging: A Computational Framework for Characterizing Optogenetic Responses in Neuropixels Recordings},
  year         = {2026},
  howpublished = {\url{https://github.com/hubdk17/Neuro_optogen101.git}},
  note         = {GitHub Repository: hubdk17/Neuro_optogen101}
}
```

---

## ⚖️ Copyright Notice

**Copyright &copy; 2026. All Rights Reserved.**  
No unauthorized reproduction, copying, distribution, or submission of this novel research is permitted under any circumstances. Official repository: [https://github.com/hubdk17/Neuro_optogen101.git](https://github.com/hubdk17/Neuro_optogen101.git).
