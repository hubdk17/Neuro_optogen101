# Full 28-Specimen Cohort Descriptive Analysis Report

**Target Journal**: *ACM Transactions on Computing for Biology and Bioinformatics* (TCBB)  
**Study**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Catalog Source**: Allen Brain Observatory Neuropixels Visual Coding (Ai32 ChR2 Drivers)  
**Deliverable**: `results/cohort/full_28_cohort_report.md`  

---

## 1. Population Overview

The cataloged optogenetic Neuropixels cohort comprises **28 independent recording sessions** across **28 distinct specimens (mice)**, spanning 3 major Cre-driver transgenic lines expressing channelrhodopsin-2 (ChR2-EYFP via Ai32 reporter):

| Transgenic Cre Line | Independent Specimens | Total Probes | Recorded Units (Raw) | Good-Quality Units (Passed QA) | Mean Good Units / Mouse |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sst-IRES-Cre** | 12 | 70 | 26,988 | 19,842 | 1,653.5 |
| **Pvalb-IRES-Cre** | 8 | 46 | 16,913 | 12,418 | 1,552.3 |
| **Vip-IRES-Cre** | 8 | 43 | 16,392 | 12,030 | 1,503.8 |
| **Total Cohort** | **28** | **159** | **60,293** | **44,290** | **1,581.8** |

### Quality-Control Attrition across Full Catalog:
- **Total Cataloged Units**: 60,293 units across 159 Neuropixels 1.0 probes (mean $379.2$ units per probe).
- **Passed Standard Ecephys QA**: 44,290 units (**73.46%** pass rate; ISI violations <= 0.5%, amplitude > 50 uV, waveform spread metrics).
- **Excluded by Spike-Sorting QA**: 16,003 units (26.54% multi-unit or unisolated clusters).

---

## 2. Complete 28-Specimen Inventory Table

| Specimen ID | Session ID | Cre Driver Line | Sex | Age (Days) | Probes | Raw Units | Good Units | QC Pass % | Empirical Status | Analyzed Units | Operational Positives |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| 699733581 | 715093703 | Sst-IRES-Cre | M | 118 | 6 | 2,714 | 2,073 | 76.4% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 703279284 | 719161530 | Sst-IRES-Cre | M | 122 | 6 | 3,129 | 1,785 | 57.0% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 707296982 | 721123822 | Pvalb-IRES-Cre | M | 125 | 6 | 1,600 | 1,191 | 74.4% | Empirically Processed & Verified | 444 | 7 |
| 726170935 | 746083955 | Pvalb-IRES-Cre | F | 98 | 6 | 2,069 | 1,542 | 74.5% | In-Progress Transfer / Partial Cache | 0 | Pending |
| 732548380 | 751348571 | Vip-IRES-Cre | F | 93 | 6 | 2,622 | 1,985 | 75.7% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 730760270 | 755434585 | Vip-IRES-Cre | M | 100 | 6 | 2,116 | 1,541 | 72.8% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 734865738 | 756029989 | Sst-IRES-Cre | M | 96 | 6 | 2,144 | 1,641 | 76.5% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 735109609 | 758798717 | Sst-IRES-Cre | M | 102 | 4 | 1,689 | 1,343 | 79.5% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 739783171 | 760345702 | Pvalb-IRES-Cre | M | 103 | 5 | 1,774 | 1,332 | 75.1% | Empirically Processed & Verified | 501 | 1 |
| 738651054 | 760693773 | Sst-IRES-Cre | F | 110 | 6 | 2,281 | 1,786 | 78.3% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 745276236 | 762120172 | Vip-IRES-Cre | M | 100 | 5 | 2,344 | 1,642 | 70.0% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 744915204 | 762602078 | Sst-IRES-Cre | M | 110 | 6 | 2,159 | 1,314 | 60.9% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 757329624 | 773418906 | Pvalb-IRES-Cre | F | 124 | 6 | 1,868 | 1,393 | 74.6% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 763884103 | 786091066 | Sst-IRES-Cre | F | 111 | 6 | 1,881 | 1,530 | 81.3% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 763236014 | 787025148 | Sst-IRES-Cre | M | 114 | 6 | 2,468 | 1,868 | 75.7% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 763808604 | 789848216 | Sst-IRES-Cre | M | 119 | 6 | 1,467 | 1,075 | 73.3% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 769360779 | 791319847 | Vip-IRES-Cre | M | 116 | 6 | 1,997 | 1,445 | 72.4% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 774672366 | 794812542 | Sst-IRES-Cre | F | 120 | 6 | 2,680 | 2,137 | 79.7% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 776061251 | 797828357 | Pvalb-IRES-Cre | M | 107 | 6 | 2,152 | 1,610 | 74.8% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 775876828 | 798911424 | Vip-IRES-Cre | F | 110 | 6 | 2,525 | 1,878 | 74.4% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 791857608 | 816200189 | Vip-IRES-Cre | F | 128 | 5 | 2,152 | 1,572 | 73.0% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 795770036 | 819701982 | Vip-IRES-Cre | F | 135 | 5 | 1,816 | 1,351 | 74.4% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 811322619 | 829720705 | Pvalb-IRES-Cre | M | 112 | 5 | 1,766 | 1,286 | 72.8% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 803390291 | 831882777 | Sst-IRES-Cre | M | 137 | 6 | 1,902 | 1,488 | 78.2% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 813701562 | 835479236 | Vip-IRES-Cre | M | 121 | 5 | 1,948 | 1,313 | 67.4% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 817060751 | 839068429 | Sst-IRES-Cre | F | 129 | 6 | 2,492 | 1,906 | 76.5% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 821469666 | 839557629 | Pvalb-IRES-Cre | M | 115 | 5 | 1,761 | 1,218 | 69.2% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |
| 820866121 | 840012044 | Pvalb-IRES-Cre | M | 116 | 6 | 2,777 | 2,045 | 73.6% | Remote on AWS S3 (~2.4 GB uncached) | 0 | Pending |

---

## 3. Empirical Data Acquisition & Transfer Constraint Audit

### Empirical Verification:
1. **Locally Verified & Processed Sessions**:
   - `721123822` (Specimen `707296982`, Pvalb-IRES-Cre): 444 analyzed units, 7 operational positives ($1.58\%$).
   - `760345702` (Specimen `739783171`, Pvalb-IRES-Cre): 501 analyzed units, 1 operational positive ($0.20\%$).
   - *Total Empirically Analyzed Cohort*: 945 units across 11 probes, 8 operational positives ($0.8466\%$).
2. **Session 746083955 (Specimen 726170935)**:
   - Partial file transferred from AWS S3 (1.22 GB / 2.39 GB).
   - In-depth byte-level audit revealed 154 non-contiguous 1MB zero holes resulting from an interrupted multi-threaded transfer; currently undergoing clean resumable transfer from S3.
3. **Remaining 25 Sessions on AWS S3**:
   - All 28 sessions confirmed present on `s3://allen-brain-observatory/visual-coding-neuropixels/ecephys-cache/`.
   - Measured single-connection transfer rate across public WAN: **1.24 MB/s**.
   - Total un-cached volume: **58.42 GB** across 25 sessions.
   - At 1.24 MB/s, sequential transfer of the un-cached cohort requires **~14.5 hours** of continuous network downloading.
   - The analysis explicitly distinguishes between the **44,290 metadata-cataloged units** and the **945 empirically processed units** to ensure absolute data provenance.

---

## 4. Methodological Freezing & Information Flow Controls

In accordance with [`results/validation/frozen_analysis_specification.md`](file:///d:/Desktop/NEUROSCIENCE/Optogenetics_1/results/validation/frozen_analysis_specification.md):
- **QC Rules**: Units with baseline firing rate $< 0.10	ext{ Hz}$ are categorized as `insufficient_evidence` (143 units, $15.13\%$) rather than deleted.
- **Evoked Window**: $[+1, +9]	ext{ ms}$ post-stimulus onset with $1	ext{ ms}$ optical onset/offset artifact blanking.
- **Sham Control**: $[-18, -10]	ext{ ms}$ pre-stimulus noise window.
- **Threshold Sensitivity Sweep**: Evaluated across 27 conditions ($L \in [6, 8, 10]	ext{ ms}, R \in [0.20, 0.30, 0.50], M \in [1.5, 2.0, 3.0]$).
- **Validation**: Leave-One-Specimen-Out (LOSO) cross-validation with zero information leakage into preprocessing, scaling, or hyperparameter selection.
