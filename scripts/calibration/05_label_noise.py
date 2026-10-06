"""
05_label_noise.py
=================
§4C (second part): how much does a downstream waveform cell-type classifier
inherit from its label source?

This is an illustration of label-noise sensitivity, in the spirit of
classifiers trained on optotagged units (e.g. Beau et al. 2025, PhysMAP,
HIPPIE). It is not a claim that any published classifier is wrong.

Task: given a unit labelled "tagged" by a label source, predict which Cre line
the session was (the cell type the tag is supposed to identify) from
extracellular features only: waveform duration, half-width, peak-to-trough
ratio, log10 firing rate. Primary: PV vs non-PV (binary); secondary: 3-class.

Model: L2-regularised logistic regression (C = 1, standardised features),
identical for every label source. Validation: GroupKFold by session (5 folds).

For each label source L (criteria from 04_criteria_calibration.py, real trials):
  * own-label score: balanced accuracy / AUC on held-out sessions' L-positive units
    (what a study using L would report);
  * common-test score: the same models scored on the calibrated light-activated
    units (lfdr < 0.05, artifacts excluded) of the held-out sessions;
  * coefficients of the model fit to all L-positive units, with session-bootstrap CIs.

Outputs: results/calibration/tables/4C_label_noise_*.csv, summary_4C_label_noise.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/calibration/tables"
FEATURES = ["waveform_duration", "waveform_halfwidth", "PT_ratio", "log_firing_rate"]
SOURCES = {"calibrated (lfdr<0.05, artifacts excluded)": "calibrated",
           "operational heuristic (original pipeline)": "crit_real_heuristic_full",
           "Allen/Lakunina (>=4/5 pulses, rel>=0.3, lat<8)": "crit_real_allen_lakunina",
           "reliability>=0.3 & modulation>2": "crit_real_rel030_mod2",
           "SALT p<0.01": "crit_real_salt_p01",
           "ZETA [1,51) ms BH": "crit_real_zeta51_bh",
           "latency<8 ms alone": "crit_real_latency_lt8"}
SEED = 20261006
warnings.filterwarnings("ignore")


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def main():
    df = pd.read_parquet(T / "4C_unit_criteria.parquet")
    u = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet",
                        columns=["unit_id", "waveform_duration", "waveform_halfwidth", "PT_ratio", "firing_rate"])
    df = df.merge(u, on="unit_id")
    df["log_firing_rate"] = np.log10(df.firing_rate.clip(lower=0.01))
    df["calibrated"] = df.driven.astype(bool)
    df["is_pv"] = (df.cre_line == "Pvalb-IRES-Cre").astype(int)
    df["cre3"] = df.cre_line.map({"Pvalb-IRES-Cre": 0, "Sst-IRES-Cre": 1, "Vip-IRES-Cre": 2})
    df = df.dropna(subset=FEATURES)
    sessions = np.sort(df.session_id.unique())
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(sessions, groups=sessions))

    rows, coefs, sizes = [], [], []
    for name, col in SOURCES.items():
        lab = df[df[col].astype(bool)]
        sizes.append(dict(label_source=name, n_units=len(lab), n_sessions=lab.session_id.nunique(),
                          frac_pv=lab.is_pv.mean(), frac_sst=(lab.cre3 == 1).mean(), frac_vip=(lab.cre3 == 2).mean(),
                          n_also_calibrated=int(lab.calibrated.sum())))
        for task, y in (("PV vs non-PV", "is_pv"), ("3-class Cre", "cre3")):
            own_true, own_pred, own_prob, com_true, com_pred, com_prob = [], [], [], [], [], []
            for tr_s, te_s in folds:
                trs, tes = sessions[tr_s], sessions[te_s]
                tr = lab[lab.session_id.isin(trs)]
                if tr[y].nunique() < 2:
                    continue
                m = model().fit(tr[FEATURES], tr[y])
                for test, T_, P_, PR_ in ((lab[lab.session_id.isin(tes)], own_true, own_pred, own_prob),
                                          (df[df.calibrated & df.session_id.isin(tes)], com_true, com_pred, com_prob)):
                    if len(test) == 0:
                        continue
                    T_.extend(test[y]); P_.extend(m.predict(test[FEATURES]))
                    pr = np.zeros((len(test), 3 if y == "cre3" else 2))
                    pr[:, m.classes_.astype(int)] = m.predict_proba(test[FEATURES])
                    PR_.extend(pr)
            r = dict(label_source=name, task=task, n_train_units=len(lab))
            for tag, T_, P_, PR_ in (("own", own_true, own_pred, own_prob), ("common_test", com_true, com_pred, com_prob)):
                T_, P_, PR_ = np.array(T_), np.array(P_), np.array(PR_)
                if len(np.unique(T_)) < 2:
                    continue
                r[f"{tag}_n"] = len(T_)
                r[f"{tag}_balanced_accuracy"] = balanced_accuracy_score(T_, P_)
                try:
                    r[f"{tag}_auc"] = (roc_auc_score(T_, PR_[:, 1]) if y == "is_pv"
                                       else roc_auc_score(T_, PR_, multi_class="ovr", average="macro"))
                except ValueError:
                    r[f"{tag}_auc"] = np.nan
            rows.append(r)
        # coefficients (PV vs non-PV), full fit + session bootstrap
        if lab.is_pv.nunique() == 2:
            m = model().fit(lab[FEATURES], lab.is_pv)
            beta = m[-1].coef_[0]
            rng = np.random.default_rng(SEED)
            boots = []
            ls = lab.session_id.unique()
            for _ in range(500):
                pick = rng.choice(ls, len(ls), replace=True)
                b = pd.concat([lab[lab.session_id == s] for s in pick])
                if b.is_pv.nunique() < 2:
                    continue
                boots.append(model().fit(b[FEATURES], b.is_pv)[-1].coef_[0])
            boots = np.array(boots)
            for i, f in enumerate(FEATURES):
                coefs.append(dict(label_source=name, feature=f, coef_std=beta[i],
                                  ci_low=np.percentile(boots[:, i], 2.5), ci_high=np.percentile(boots[:, i], 97.5)))
    perf = pd.DataFrame(rows)
    perf.to_csv(T / "4C_label_noise_performance.csv", index=False)
    pd.DataFrame(coefs).to_csv(T / "4C_label_noise_coefficients.csv", index=False)
    pd.DataFrame(sizes).to_csv(T / "4C_label_noise_training_sets.csv", index=False)
    with open(ROOT / "results/calibration/summary_4C_label_noise.json", "w") as fh:
        json.dump(dict(performance=perf.to_dict("records")), fh, indent=2, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(sizes).round(3).to_string(index=False))
        print(perf.round(3).to_string(index=False))
        print(pd.DataFrame(coefs).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
