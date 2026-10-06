"""
03_flash_confound.py
====================
§4B: how much of the opto-evoked "network" suppression is shared with the same
sessions' response to a full-field visual flash?

Rationale: thalamic suppression after 10-ms light pulses is 23-29% in all three
Cre lines, including regions that surface light cannot reach, so part of the
response may not be opsin-mediated (a retinal/visual response to the blue
light, or a global state change). This dataset has no opsin-negative or
masking-light control, so the confound can be characterised but not removed.

Per unit, both for 10-ms opto pulses and for flashes (both polarities pooled),
with identical windows and baseline [-480, -20) ms:
  * exact conditional tests for suppression and excitation in 20-50 and 50-200 ms,
  * log rate ratio in 20-200 ms,
  * a 5-ms-binned PSTH from -100 to +400 ms normalised to baseline.
Then, per region group and Cre line, with the animal as replicate:
  * fraction suppressed (opto, flash), overlap vs independence,
  * correlation of opto and flash modulation across units,
  * suppression onset latency (population PSTH, per animal) for opto vs flash,
    with LGd and LP reported separately.

Outputs: results/calibration/tables/4B_*.csv, results/calibration/summary_4B.json,
         results/calibration/tables/4B_unit_flash_opto.parquet
"""

import gzip
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import exact_rate_test, log_rate_ratio, bh  # noqa: E402

T = ROOT / "results/calibration/tables"
BASE = (-0.480, -0.020)
WINS = {"w20_50": (0.020, 0.050), "w50_200": (0.050, 0.200), "w20_200": (0.020, 0.200)}
EDGES = np.arange(-0.100, 0.4001, 0.005)
CRES = {"Pvalb-IRES-Cre": "PV", "Sst-IRES-Cre": "SST", "Vip-IRES-Cre": "VIP"}
S = {}


def region(a):
    a = str(a)
    if a.startswith("VIS"):
        return "visual cortex"
    if a in ("LGd", "LGv", "LP", "PO", "POL", "TH", "Eth", "IGL", "LD", "SGN", "VPM", "VPL", "MGv", "MGd", "MGm",
             "IntG", "PP", "PIL", "RT", "SPF", "VL", "CL", "LT"):
        return "thalamus"
    if a in ("CA1", "CA2", "CA3", "DG", "ProS", "SUB", "HPF", "POST", "PRE") or a.startswith("CA"):
        return "hippocampal formation"
    if a.startswith("SC") or a in ("APN", "MB", "NOT", "PPT", "MRN", "OP", "NB", "PAG", "SCig", "SCiw", "SCop", "SCsg", "SCzo"):
        return "midbrain"
    return "other / unassigned"


def block(rel, ep, trials):
    n = len(trials)
    m = np.isin(ep, trials)
    r = rel[m].astype(np.float64)
    out = {"n": n, "nb": int(np.count_nonzero((r >= BASE[0]) & (r < BASE[1])))}
    for k, (lo, hi) in WINS.items():
        out[k] = int(np.count_nonzero((r >= lo) & (r < hi)))
    out["psth"] = np.histogram(r, EDGES)[0].astype(np.int32)
    return out


def session(path):
    with gzip.open(path, "rb") as fh:
        R = pickle.load(fh)
    assert R.get("extract_version", 1) >= 2, f"{path} needs v2 extraction"
    opto = R["opto"]
    rows, psth = [], {}
    for _, unit in R["units"].iterrows():
        uid = int(unit.unit_id)
        p = unit.probe_name
        p10 = opto.index[(opto.kind == "pulse10").values & R["valid_epochs"][p]].values
        rel, ep = R["aligned"][uid]
        o = block(rel, ep, p10)
        if len(R["flashes"]) and uid in R["aligned_flash"]:
            fl = np.nonzero(R["valid_flash"][p])[0]
            frel, fep = R["aligned_flash"][uid]
            f = block(frel, fep, fl)
        else:
            f = None
        row = dict(session_id=R["session_id"], unit_id=uid, structure=unit.structure)
        for tag, b in (("opto", o), ("flash", f)):
            if b is None or b["n"] == 0:
                continue
            row[f"{tag}_n"] = b["n"]
            row[f"{tag}_nb"] = b["nb"]
            for k, (lo, hi) in WINS.items():
                row[f"{tag}_{k}"] = b[k]
            psth[(uid, tag)] = b["psth"]
        rows.append(row)
    return pd.DataFrame(rows), psth


def onset_latency(curve, thr):
    """First post-onset bin (centre, ms) after which the curve stays below thr for >= 3 bins."""
    centres = (EDGES[:-1] + EDGES[1:]) / 2 * 1000
    post = np.nonzero(centres > 0)[0]
    below = curve[post] < thr
    for i in range(len(post) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            return centres[post[i]]
    return np.nan


def main():
    files = sorted((ROOT / "data/derived").glob("session_*.pkl.gz"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(session, files))
    df = pd.concat([o[0] for o in out], ignore_index=True)
    ps = {}
    for o in out:
        ps.update(o[1])
    u = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet",
                        columns=["unit_id", "specimen_id", "cre_line", "driven", "lfdr"])
    df = df.merge(u, on="unit_id")
    df["cre"] = df.cre_line.map(CRES)
    df["region"] = df.structure.map(region)
    df.loc[df.structure.isin(["LGd", "LP"]), "region_detail"] = df.structure
    W_B = BASE[1] - BASE[0]
    for tag in ("opto", "flash"):
        n = df[f"{tag}_n"]
        for k, (lo, hi) in WINS.items():
            pe, pse, _ = exact_rate_test(df[f"{tag}_{k}"].fillna(0).astype(int), df[f"{tag}_nb"].fillna(0).astype(int),
                                         n * (hi - lo), n * W_B)
            df[f"{tag}_p_sup_{k}"] = np.where(n > 0, pse, np.nan)
            df[f"{tag}_p_exc_{k}"] = np.where(n > 0, pe, np.nan)
        lrr, se = log_rate_ratio(df[f"{tag}_w20_200"].fillna(0), n * 0.18, df[f"{tag}_nb"].fillna(0), n * W_B)
        df[f"{tag}_lrr_20_200"] = np.where(n > 0, lrr, np.nan)
        # suppressed: BH over units on the min of the two windows' p (x2, Bonferroni over windows)
        pmin = np.minimum(df[f"{tag}_p_sup_w20_50"], df[f"{tag}_p_sup_w50_200"]) * 2
        df[f"{tag}_suppressed"] = (bh(np.clip(pmin.fillna(1), 0, 1).values) < 0.05) & ~df.driven
        pmin_e = np.minimum(df[f"{tag}_p_exc_w20_50"], df[f"{tag}_p_exc_w50_200"]) * 2
        df[f"{tag}_excited"] = (bh(np.clip(pmin_e.fillna(1), 0, 1).values) < 0.05) & ~df.driven
    df = df[df.flash_n.notna() & df.opto_n.notna()].copy()
    S["units_with_both"] = int(len(df))
    S["flashes_per_session_median"] = float(df.groupby("session_id").flash_n.max().median())
    df.drop(columns=[]).to_parquet(T / "4B_unit_flash_opto.parquet", index=False)

    # ---- region x Cre prevalence, animal level
    rows = []
    for (reg, cre), g in df.groupby(["region", "cre"]):
        a = g.groupby("specimen_id").agg(opto_sup=("opto_suppressed", "mean"), flash_sup=("flash_suppressed", "mean"),
                                        n=("unit_id", "size"))
        both = (g.opto_suppressed & g.flash_suppressed).mean()
        exp_ind = g.opto_suppressed.mean() * g.flash_suppressed.mean()
        # overlap: P(flash-suppressed | opto-suppressed) vs P(flash-suppressed | not opto-suppressed)
        p_f_given_o = g[g.opto_suppressed].flash_suppressed.mean() if g.opto_suppressed.any() else np.nan
        p_f_given_no = g[~g.opto_suppressed].flash_suppressed.mean()
        tab = pd.crosstab(g.opto_suppressed, g.flash_suppressed).reindex(index=[False, True], columns=[False, True], fill_value=0)
        orr, pf = st.fisher_exact(tab.values) if tab.values.min() >= 0 else (np.nan, np.nan)
        rho = st.spearmanr(g.opto_lrr_20_200, g.flash_lrr_20_200, nan_policy="omit")
        rows.append(dict(region=reg, cre=cre, n_units=len(g), n_animals=len(a),
                         opto_suppressed_frac_animal_mean=a.opto_sup.mean(), opto_sem=a.opto_sup.sem(),
                         flash_suppressed_frac_animal_mean=a.flash_sup.mean(), flash_sem=a.flash_sup.sem(),
                         both_frac=both, both_expected_if_independent=exp_ind,
                         frac_opto_suppressed_also_flash_suppressed=p_f_given_o,
                         frac_flash_suppressed_among_not_opto=p_f_given_no, overlap_OR=orr, overlap_fisher_p=pf,
                         spearman_opto_vs_flash_lrr=rho.statistic, spearman_p=rho.pvalue))
    prev = pd.DataFrame(rows)
    prev.to_csv(T / "4B_region_prevalence_overlap.csv", index=False)

    # ---- is opto suppression Cre-dependent within each region? (animal level, Kruskal + permutation)
    rng = np.random.default_rng(20261006)
    cdep = []
    for reg, g in df.groupby("region"):
        a = g.groupby(["specimen_id", "cre"]).opto_suppressed.mean().reset_index()
        groups = [a[a.cre == c].opto_suppressed.values for c in ["PV", "SST", "VIP"] if (a.cre == c).sum() >= 2]
        kw = st.kruskal(*groups).pvalue if len(groups) == 3 else np.nan
        v, lab = a.opto_suppressed.values, a.cre.values
        def stat(lb):
            gm = v.mean()
            return sum((lb == c).sum() * (v[lb == c].mean() - gm) ** 2 for c in np.unique(lb))
        obs = stat(lab)
        null = [stat(rng.permutation(lab)) for _ in range(5000)]
        cdep.append(dict(region=reg, n_animals=len(a), kruskal_p=kw, perm_p=(1 + np.sum(np.array(null) >= obs)) / 5001,
                         **{f"opto_sup_{c}": a[a.cre == c].opto_suppressed.mean() for c in ["PV", "SST", "VIP"]},
                         **{f"flash_sup_{c}": g[g.cre == c].groupby("specimen_id").flash_suppressed.mean().mean() for c in ["PV", "SST", "VIP"]}))
    pd.DataFrame(cdep).to_csv(T / "4B_cre_dependence_by_region.csv", index=False)

    # ---- timing: population PSTH per animal x region, onset of suppression (< 0.8 x baseline)
    centres = (EDGES[:-1] + EDGES[1:]) / 2 * 1000
    tim, curves = [], []
    df["grp"] = df.region_detail.fillna(df.region)
    for (sp, cre, grp), g in df.groupby(["specimen_id", "cre", "grp"]):
        for tag in ("opto", "flash"):
            sub = g[g[f"{tag}_suppressed"]]
            if len(sub) < 3:
                continue
            norm = []
            for _, r in sub.iterrows():
                lam = r[f"{tag}_nb"] / (r[f"{tag}_n"] * (BASE[1] - BASE[0]))
                if lam <= 0:
                    continue
                norm.append(ps[(r.unit_id, tag)] / (r[f"{tag}_n"] * 0.005) / lam)
            if len(norm) < 3:
                continue
            c = np.mean(norm, axis=0)
            curves.append(dict(specimen_id=sp, cre=cre, group=grp, stim=tag, n_units=len(norm), **{f"t{t:.1f}": v for t, v in zip(centres, c)}))
            tim.append(dict(specimen_id=sp, cre=cre, group=grp, stim=tag, n_units=len(norm),
                            onset_ms=onset_latency(c, 0.8),
                            min_ratio=float(np.min(c[(centres > 0) & (centres < 200)])),
                            mean_ratio_20_200=float(np.mean(c[(centres > 20) & (centres < 200)]))))
    tim = pd.DataFrame(tim)
    tim.to_csv(T / "4B_suppression_timing_per_animal.csv", index=False)
    pd.DataFrame(curves).to_csv(T / "4B_population_psth_suppressed.csv", index=False)
    # paired opto vs flash onset, per group, animals with both
    pt = []
    for grp, g in tim.groupby("group"):
        w = g.pivot_table(index="specimen_id", columns="stim", values="onset_ms")
        if {"opto", "flash"} <= set(w.columns):
            w = w.dropna()
            if len(w) >= 3:
                d = w.opto - w.flash
                pt.append(dict(group=grp, n_animals=len(w), opto_onset_median=w.opto.median(),
                               flash_onset_median=w.flash.median(), median_diff_opto_minus_flash=d.median(),
                               wilcoxon_p=st.wilcoxon(d).pvalue if np.any(d != 0) else 1.0))
            else:
                pt.append(dict(group=grp, n_animals=len(w), note="fewer than 3 animals with both"))
    pd.DataFrame(pt).to_csv(T / "4B_onset_opto_vs_flash.csv", index=False)

    # ---- overlap adjusted for detectability (baseline rate), GEE clustered by animal
    df["lograte"] = np.log10((df.opto_nb / (df.opto_n * (BASE[1] - BASE[0]))).clip(lower=0.05))
    df["fs"] = df.flash_suppressed.astype(int)
    df["os"] = df.opto_suppressed.astype(int)
    adj = []
    for reg, g in df.groupby("region"):
        if g.specimen_id.nunique() < 5 or g.os.sum() < 10:
            continue
        m = smf.gee("fs ~ os + lograte", "specimen_id", g, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = np.exp(m.conf_int().loc["os"].values)
        adj.append(dict(region=reg, n_units=len(g), n_animals=g.specimen_id.nunique(),
                        overlap_OR_rate_adjusted=np.exp(m.params["os"]), ci_low=ci[0], ci_high=ci[1], p=m.pvalues["os"]))
    pd.DataFrame(adj).to_csv(T / "4B_overlap_rate_adjusted.csv", index=False)
    # ---- excitation as well as suppression (a visual response to the light would excite LGd/LP)
    ex = df.groupby(["region", "cre", "specimen_id"])[["opto_excited", "opto_suppressed", "flash_excited", "flash_suppressed"]].mean()
    ex = ex.groupby(["region", "cre"]).agg(["mean", "sem"])
    ex.columns = ["_".join(c) for c in ex.columns]
    ex.reset_index().to_csv(T / "4B_excitation_suppression_by_region.csv", index=False)
    lp = df[df.structure.isin(["LGd", "LP"])].groupby(["structure", "cre", "specimen_id"])[
        ["opto_excited", "opto_suppressed", "flash_excited", "flash_suppressed"]].mean().groupby(["structure", "cre"]).agg(["mean", "sem", "count"])
    lp.columns = ["_".join(c) for c in lp.columns]
    lp.reset_index().to_csv(T / "4B_LGd_LP_by_cre.csv", index=False)
    S["overlap_rate_adjusted"] = adj

    # ---- shared fraction summary
    sup = df[df.opto_suppressed]
    S["opto_suppressed_units"] = int(len(sup))
    S["frac_opto_suppressed_also_flash_suppressed"] = float(sup.flash_suppressed.mean())
    S["frac_flash_suppressed_among_non_opto_suppressed"] = float(df[~df.opto_suppressed].flash_suppressed.mean())
    S["by_region_shared"] = prev.groupby("region").apply(
        lambda g: float((g.frac_opto_suppressed_also_flash_suppressed * g.n_units).sum() / g.n_units.sum())).to_dict()
    with open(ROOT / "results/calibration/summary_4B.json", "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(prev.round(3).to_string(index=False))
        print(pd.DataFrame(cdep).round(3).to_string(index=False))
        print(pd.DataFrame(pt).round(3).to_string(index=False))
        print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
