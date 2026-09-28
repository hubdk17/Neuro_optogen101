# Revised Analysis Pipeline & Scientific Architecture

**Project**: *Optogenetic Perturbation of Cortical Microcircuits: A 28-Specimen Neuropixels Study Beyond Binary Optotagging*  
**Scope**: 28 independent biological specimens, 159 probes, 18,316 units (`Pvalb-IRES-Cre`: 8, `Sst-IRES-Cre`: 12, `Vip-IRES-Cre`: 8).

---

## 1. Scientific Principles & Methodological Guardrails

1. **Optogenetic Perturbation as Object of Study**: We treat light delivery as a controlled physical perturbation of local cortical networks. The scientific goal is to characterize the temporal, intensity, spatial, and cell-type-specific structure of the evoked response, rather than merely predicting a binary classification label.
2. **Never Overclaim**: Claims are matched strictly to what the extracellular recording design establishes. We employ cautious phrasing ("is consistent with", "suggests", "is associated with", "the data indicate") and avoid unwarranted causal or mechanistic assertions ("proves connectivity", "biological invariant", "deterministic", "propagation velocity").
3. **Specimen as Primary Biological Replicate**: Units recorded within the same mouse share animal state, probe placement, and light absorption. Hierarchical models, specimen-level bootstrapping, and specimen-held-out validation (LOSO) are applied across the $N=28$ biological specimens.
4. **Explicit Denominators**: Every proportion explicitly documents its reference population (e.g., all recorded units vs. units with detected latency vs. units with evoked firing $\ge 2.0$ Hz).

---

## 2. End-to-End Pipeline Workflow

```
[Raw Allen Neuropixels NWB Data (28 Sessions / 28 Specimens)]
                            |
           [Frozen Feature Extraction Pipeline]
                            |
           [Master Metadata Table (31 Attributes)]
         (master_neuroscience_metadata.parquet)
                            |
         +------------------+------------------+
         |                                     |
[Responsiveness Methods Comparison]   [Sparse-Firing Stability & Null Model]
 - Operational Heuristic (N=261)       - Poisson Null: P(N >= 1) = 1 - exp(-lambda*tau)
 - SALT (JSD Permutation, N=702)       - Stratification: <1, 1-2, 2-4, 4-8, >8 Hz
 - ZETA (Brownian Bridge, N=1,992)     - Clarified Denominators & Bootstrap SE
 - Evidence Score E_i (Model D)        (revised_sparse_firing_stability.csv)
         |
         +------------------+------------------+
                            |
[Unsupervised Temporal Dynamic Phenotyping]
 - 5 Canonical Windows: W1 (0-8ms), W2 (8-20ms), W3 (20-50ms), W4 (50-200ms), W5 (200-500ms)
 - Mutually Exclusive GMM Clustering (k=5 components)
 (revised_population_response_archetypes.csv)
                            |
         +------------------+------------------+
         |                                     |
[Intensity-Response Relationships]    [Standardized Pulse-Train Adaptation]
 - Powers: 1.0, 2.5, 4.0 mW            - Ratio: R_n / R_1
 - Hierarchical Mixed-Effects Model     - Formula: AI = (R_10 - R_1) / max(R_1, 0.5)
 (revised_intensity_response_          - Categories: Depressing, Stable, Facilitating
  hierarchical.csv)                    (revised_pulse_train_adaptation.csv)
                            |
         +------------------+------------------+
         |                                     |
[Spatial Distance Structure]          [Population Temporal Coordination]
 - Probe vertical distance (0-600+ um) - Pre- vs Post-Stimulation CCGs
 - Latency gradient (v ~ 0.07 m/s)     - Synchrony fold-change & peak lag
 - Attenuation lambda ~ 120-160 um     (ccg_state_reorganization.csv)
 (revised_spatial_propagation_
  controlled.csv)
                            |
[Specimen-Level Replication & Hierarchical Variance]
 - Intraclass Correlation (ICC) on baseline, evoked, modulation, and adaptation
 - Corrected interpretation: within-specimen heterogeneity vs animal reproducibility
 (revised_cross_specimen_reproducibility.csv)
                            |
[Publication Figures (10 Revised Panels)]
 (results/neuroscience_study/figures/revised/fig1 - fig10)
```

---

## 3. Data Flow and Artifact Organization

- **Raw Data Source**: 28 NWB files stored on local scratch disk (`D:\`), verified with AllenSDK root keys.
- **Frozen Master Table**: `results/neuroscience_study/tables/master_neuroscience_metadata.parquet` (18,316 units $\times$ 31 columns).
- **Revised Method & Physiological Tables**: `results/neuroscience_study/tables/revised/`:
  - `revised_method_disagreement_detailed.csv`: 8-group Venn partitioning and profiles.
  - `revised_sparse_firing_stability.csv`: Baseline tiers with Poisson null expectations.
  - `revised_population_response_archetypes.csv`: Mutually exclusive GMM clusters.
  - `revised_intensity_response_hierarchical.csv`: Mixed-effects slopes across 1.0, 2.5, 4.0 mW.
  - `revised_pulse_train_adaptation.csv`: Standardized $AI$ and $R_n/R_1$ dynamics.
  - `revised_spatial_propagation_controlled.csv`: Vertical distance gradient controlling for baseline tier.
  - `revised_cross_specimen_reproducibility.csv`: ICC variance decomposition and specimen $t$-tests.
- **Revised Figures**: `results/neuroscience_study/figures/revised/` (Figures 1–10 in 300 DPI PNG and vector PDF).
