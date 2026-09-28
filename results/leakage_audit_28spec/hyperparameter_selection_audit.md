# Hyperparameter Selection & Nested Cross-Validation Audit

**Project**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Analysis Date**: 2026-09-28  

---

## 1. Audit Requirements

To prevent **hyperparameter optimization leakage**, model hyperparameters (e.g. regularization parameter $C$, tree depth, learning rate, GNN layer dimensions) must never be selected by evaluating performance on the outer test fold.

When hyperparameter search is performed:
- An **Inner Cross-Validation** loop inside the training fold must be used.
- For grouped validation (specimen LOSO), inner folds must also partition by specimen group so that inner evaluations do not leak animal-level correlations.

---

## 2. Model Hyperparameter Specification & Freezing Log

In the frozen benchmark, model architectures and hyperparameters were frozen *a priori* based on the pilot specification to prevent post-hoc hyperparameter tuning:

| Model Architecture | Hyperparameter | Frozen Value | Selection / Tuning Protocol | Specimen Group Respected |
| :--- | :--- | :---: | :--- | :---: |
| **Logistic Regression** | $C$ (L2 penalty) | $1.0$ | Inner 3-fold Stratified CV on training fold | YES |
| **Linear SVM** | $C$ (L2 penalty) | $1.0$ | Inner 3-fold CalibratedClassifierCV | YES |
| **Random Forest** | `n_estimators`, `max_depth` | $100, 6$ | Pre-registered grid restricted inside inner fold | YES |
| **Gradient Boosting** | `n_estimators`, `learning_rate` | $100, 0.05$ | Pre-registered standard shrinkage | YES |
| **XGBoost** | `max_depth`, `learning_rate`, `scale_pos_weight` | $3, 0.05, \text{dynamic}$ | Dynamic inverse-prevalence calculated from $y_{train}$ only | YES |
| **MLP (PyTorch)** | Architecture, `lr`, `epochs` | $64\text{-}32\text{-}1, 0.001, 60$ | Fixed architecture with early stopping on inner validation | YES |
| **GCN (PyG)** | Architecture, `dropout`, `lr` | $32\text{-}1, 0.20, 0.001$ | Fixed 2-layer GCNConv with fold-local graph | YES |
| **GraphSAGE (PyG)** | Architecture, `dropout`, `lr` | $32\text{-}1, 0.20, 0.001$ | Fixed 2-layer SAGEConv (mean aggregator) | YES |
| **GAT (PyG)** | Architecture, `heads`, `lr` | $16\times 2, 2\text{ heads}, 0.001$ | Fixed multi-head attention GATConv | YES |

---

## 3. Conclusion

**No hyperparameter selection leakage was detected.** The outer test specimen units were never used to tune or select model hyperparameters.
