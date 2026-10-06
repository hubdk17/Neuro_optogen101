# Preprocessing Leakage Audit

**Project**: Computationally Reliable Optotagging of Neuropixels Neural Recordings  
**Analysis Date**: 2026-09-28  

---

## 1. Audit Principles

In supervised machine learning, **preprocessing leakage** occurs when transformations applied to input features (such as missing-value imputation, standardization, scaling, or feature selection) compute statistics (mean, variance, median, min/max, percentiles) across the entire dataset prior to splitting into train and test folds.

---

## 2. Code-Level Inspection of Benchmark Preprocessing

In `scripts/run_full_specimen_ml_benchmark.py`:

```python
# Fold-by-fold execution loop:
for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
    X_tr = X.iloc[tr_idx].copy()
    X_te = X.iloc[te_idx].copy()
    
    # Imputer fitted strictly on X_tr:
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)  # transform ONLY on test fold
    
    # Scaler fitted strictly on X_tr_imp:
    scaler = StandardScaler()
    X_tr_scl = scaler.fit_transform(X_tr_imp)
    X_te_scl = scaler.transform(X_te_imp)  # transform ONLY on test fold
    
    # Probability calibration (Platt scaling) fitted strictly on X_tr:
    model.fit(X_tr_scl, y_tr)
```

### Verification Findings for Preprocessing Steps:

| Transformation Step | Scikit-Learn Class | Fitted on Test Data? | Fold-Local Pipeline? | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Missing-Value Imputation** | `SimpleImputer(strategy='median')` | **NO** | **YES** | **PASS** |
| **Feature Standardization** | `StandardScaler()` | **NO** | **YES** | **PASS** |
| **Scale Pos Weight / Class Weights**| Dynamic `(y_tr == 0).sum() / (y_tr == 1).sum()` | **NO** | **YES** | **PASS** |
| **Platt Probability Scaling** | `CalibratedClassifierCV(cv=3)` | **NO** | **YES** | **PASS** |
| **Decision Threshold Optimization** | Evaluated on validation set | **NO** | **YES** | **PASS** |

---

## 3. Conclusion

**No preprocessing leakage was detected** in the standard sklearn/PyTorch training loop:
1. Feature imputers and scalers were instantiated afresh in every outer cross-validation fold.
2. Training fold statistics were never computed with test-specimen units included.
3. Class weights and loss penalty balances were calculated dynamically from $y_{train}$ only.
