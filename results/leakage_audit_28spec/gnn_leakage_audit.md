# Graph Neural Network (GNN) Leakage & Graph Construction Audit

**Project**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Analysis Date**: 2026-09-28  

---

## 1. Graph Construction Audit Criteria

Graph Neural Networks present unique data-leakage risks that do not exist for independent-sample models:
1. **Transductive vs Inductive Formulation**:
   - In a transductive graph, train and test nodes belong to the **same graph**, allowing message passing across train-test edges during forward passes.
   - In an inductive graph, test specimens are **entirely separate disjoint graphs**, with zero edges connecting test nodes to training nodes.
2. **Edge Construction Integrity**:
   - Edges must be computed strictly from physical recording coordinates (3D CCF coordinates $\mu\text{m}$), with zero dependence on target labels $y$ or evidence scores $E_i$.
   - No label-informed neighborhood pruning or graph rewiring.

---

## 2. Implementation Verification in Codebase

In `scripts/run_full_specimen_ml_benchmark.py` and `src/graph_benchmark.py`:

```python
# Graph construction for each session independently:
coords = df_session[["anterior_posterior_ccf_coordinate",
                     "dorsal_ventral_ccf_coordinate",
                     "left_right_ccf_coordinate"]].values

# k-NN graph constructed using strictly 3D spatial Euclidean distance:
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree").fit(coords)
distances, indices = nbrs.kneighbors(coords)

# Construct edge index (undirected, self-loops excluded):
edges = []
for i in range(len(coords)):
    for j in indices[i][1:]: # exclude self
        edges.append((i, j))
        edges.append((j, i))
```

### Audit Findings:

| Audit Check | Implementation Rule | Observed in Code | Status |
| :--- | :--- | :--- | :---: |
| **Cross-Specimen Edge Leakage** | Edges only connect nodes within the same recording session | Confirmed: Graphs are constructed strictly per session | **PASS** |
| **Specimen LOSO Graph Isolation** | Held-out test mouse graph is completely evaluated inductively | Confirmed: Test specimen is a separate `Data` object | **PASS** |
| **Label-Informed Edges** | Edges must not use labels or evidence scores | Confirmed: Edges use strictly 3D CCF coordinates | **PASS** |
| **Node Feature Leakage** | Node features normalized only using training sessions | Confirmed: `StandardScaler` fitted on training graphs only | **PASS** |
| **Random-Node K-Fold Caveat** | Random unit splits within a graph share edges across train/test | Documented: Random node CV is leakage-prone for GNNs | **NOTED** |

---

## 3. Graph Topology Controls: Real Spatial vs Degree-Preserving Randomized Graphs

In `graph_shuffle_control.csv`:
- **Real Spatial Graph ($k=5$)**: Mean $\text{BA} = 0.8241$, $\text{AUPRC} = 0.2489$.
- **Degree-Preserving Randomized Graph**: Mean $\text{BA} = 0.8496$, $\text{AUPRC} = 0.6003$.

### Scientific Interpretation:
The fact that degree-preserving randomized graphs outperform the real physical spatial graph confirms that the poor performance of GCN/GAT is **NOT** caused by edge leakage. Rather, physical spatial clustering on Neuropixels shanks groups rare direct positives ($1.42\%$) with their non-responsive neighbors, causing spatial message passing to act as a **low-pass smoothing filter that dilutes sparse activation signals**.
