"""
02_latency_validity.py
======================
§4A, as pre-registered in reports/PREREGISTRATION_4A.md (commit a8e0d2f):
does the latency/jitter "direct-like" criterion identify units with
monosynaptic inhibitory output, and is the CCG probe valid at all?

Inputs : results/calibration/tables/ccg_pairs_extended.csv (01_matched_ccg.py)
         results/reanalysis/tables/unit_results.parquet
Outputs: results/calibration/tables/4A_*.csv, results/calibration/summary_4A.json

Inference: pair-level binary outcome (trough yes/no) in a binomial GEE with
exchangeable working correlation clustered by animal (specimen), so standard
errors are cluster-robust at the animal level; plus animal-level paired tests.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/calibration/tables"
U = ROOT / "results/reanalysis/tables/unit_results.parquet"
ALPHA_PAIR = 0.01
CRES = {"Pvalb-IRES-Cre": "PV", "Sst-IRES-Cre": "SST", "Vip-IRES-Cre": "VIP"}
S = {}
rng = np.random.default_rng(20261006)
warnings.filterwarnings("ignore")


def gee(formula, data, label):
    """Binomial GEE clustered by animal. Returns dict with OR, CI, p for the first non-intercept term."""
    data = data.copy()
    if data.specimen_id.nunique() < 3 or data.iloc[:, 0].size < 20:
        return dict(model=label, n_pairs=len(data), n_animals=data.specimen_id.nunique(), note="too few clusters")
    m = smf.gee(formula, groups="specimen_id", data=data, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    term = [t for t in m.params.index if t != "Intercept"][0]
    b, se = m.params[term], m.bse[term]
    return dict(model=label, term=term, n_pairs=len(data), n_ref_units=data.ref_unit.nunique(),
                n_animals=data.specimen_id.nunique(), log_or=b, se=se, odds_ratio=np.exp(b),
                or_ci_low=np.exp(b - 1.96 * se), or_ci_high=np.exp(b + 1.96 * se), p=m.pvalues[term])


def t_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return dict(n=len(x), mean=float(np.mean(x)) if len(x) else np.nan, ci_low=np.nan, ci_high=np.nan, t_p=np.nan, wilcoxon_p=np.nan)
    t = st.ttest_1samp(x, 0)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    w = st.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return dict(n=len(x), mean=x.mean(), ci_low=x.mean() - h, ci_high=x.mean() + h, t_p=t.pvalue, wilcoxon_p=w)


def main():
    u = pd.read_parquet(U).set_index("unit_id")
    pr = pd.read_csv(T / "ccg_pairs_extended.csv")
    pr["inh"] = (pr.p_inh < ALPHA_PAIR).astype(int)
    pr["inh_in"] = (pr.p_inh_anticausal < ALPHA_PAIR).astype(int)   # partner inhibits the reference unit
    pr["cre"] = pr.cre_line.map(CRES)

    # ------------------------------------------------ per-reference-unit table
    ref = pr.groupby(["ref_unit", "ref_type"]).agg(
        session_id=("session_id", "first"), specimen_id=("specimen_id", "first"), cre=("cre", "first"),
        matched_to=("matched_to", "first"), n_partners=("partner", "size"),
        n_inh=("inh", "sum"), n_inh_in=("inh_in", "sum")).reset_index()
    ref["source_rate"] = ref.n_inh / ref.n_partners
    ref["target_rate"] = ref.n_inh_in / ref.n_partners
    for c in ["p10_fit_delta", "p10_fit_sigma", "p10_rho", "p10_high_rho", "p10_low_rho", "waveform_duration",
              "lam_spont_hz", "train_n", "train_lam0_hz"] + [f"train_k{j}" for j in range(1, 11)]:
        ref[c] = ref.ref_unit.map(u[c])
    ref["direct_like"] = (ref.p10_fit_delta < 5.0) & (ref.p10_fit_sigma < 1.5)
    ref["dose_slope"] = ref.p10_high_rho - ref.p10_low_rho
    p0 = 1 - np.exp(-ref.train_lam0_hz * 0.008)
    K = ref[[f"train_k{j}" for j in range(1, 11)]].values
    ref["train_following"] = np.nanmean((K / ref.train_n.values[:, None] - p0.values[:, None]) / (1 - p0.values[:, None]), axis=1)
    ref.to_csv(T / "4A_reference_units.csv", index=False)

    # per-animal contributions
    contrib = ref.groupby(["specimen_id", "cre", "ref_type"]).agg(ref_units=("ref_unit", "size"),
                                                                  pairs=("n_partners", "sum")).reset_index()
    contrib.to_csv(T / "4A_per_animal_contributions.csv", index=False)

    rows = []
    # ------------------------------------------------ 1. positive control
    for tier, ctype in [("strict (pre-registered)", ["control"]), ("strict + relaxed (sensitivity, deviation)", ["control", "control_relaxed"])]:
        for cre in ["PV", "SST", "VIP"]:
            refs_c = ref[(ref.cre == cre) & ref.ref_type.isin(ctype)]
            drv = refs_c.matched_to.dropna().astype(int).unique()
            d = pr[(pr.cre == cre) & (((pr.ref_type == "driven") & pr.ref_unit.isin(drv)) | pr.ref_type.isin(ctype))].copy()
            d["is_driven"] = (d.ref_type == "driven").astype(int)
            r = gee("inh ~ is_driven", d, f"positive control {cre}, {tier}")
            r.update(driven_rate=d[d.is_driven == 1].inh.mean(), control_rate=d[d.is_driven == 0].inh.mean())
            # animal-level paired: per driven unit, rate - mean rate of its controls; averaged per animal
            dr = ref[(ref.ref_type == "driven") & ref.ref_unit.isin(drv)].set_index("ref_unit")
            cr = refs_c.groupby("matched_to").source_rate.mean()
            diff = (dr.source_rate - dr.index.map(cr)).groupby(dr.specimen_id).mean()
            a = t_ci(diff.values)
            r.update({f"animal_{k}": v for k, v in a.items()})
            rows.append(r)
            if cre == "PV" and tier.startswith("strict ("):
                S["positive_control_PV_p"] = r.get("p")
                S["positive_control_PV_OR"] = r.get("odds_ratio")
    pc = pd.DataFrame(rows)
    pc.to_csv(T / "4A_positive_control.csv", index=False)
    S["positive_control_passes_PV"] = bool(S.get("positive_control_PV_p", 1) < 0.05 and S.get("positive_control_PV_OR", 0) > 1)

    # selection check: driven units with vs without a strict control
    has = set(ref[ref.ref_type == "control"].matched_to.dropna().astype(int))
    dref = ref[ref.ref_type == "driven"].copy()
    dref["has_strict_control"] = dref.ref_unit.isin(has)
    sel = dref.groupby(["cre", "has_strict_control"]).agg(n=("ref_unit", "size"), source_rate=("source_rate", "mean"),
                                                          lam_spont_median=("lam_spont_hz", "median"),
                                                          narrow_frac=("waveform_duration", lambda x: np.mean(x < 0.4))).reset_index()
    sel.to_csv(T / "4A_matching_selection.csv", index=False)

    # ------------------------------------------------ 2. delta/sigma test among driven PV+SST
    dd = pr[(pr.ref_type == "driven") & pr.cre.isin(["PV", "SST"])].merge(
        ref[ref.ref_type == "driven"][["ref_unit", "direct_like", "p10_fit_delta", "p10_fit_sigma", "p10_rho", "dose_slope",
                                       "train_following", "waveform_duration"]], on="ref_unit")
    dd = dd.dropna(subset=["p10_fit_delta"])
    dd["direct_like_i"] = dd.direct_like.astype(int)
    res2 = [gee("inh ~ direct_like_i + C(cre)", dd, "direct-like vs indirect-like (PV+SST)")]
    for c in ["p10_fit_delta", "p10_fit_sigma"]:
        dd[f"z_{c}"] = (dd[c] - dd[c].mean()) / dd[c].std()
        res2.append(gee(f"inh ~ z_{c} + C(cre)", dd, f"continuous {c} (per SD)"))
    for cre in ["PV", "SST"]:
        res2.append(gee("inh ~ direct_like_i", dd[dd.cre == cre], f"direct-like vs indirect-like ({cre})"))
    dr2 = ref[(ref.ref_type == "driven") & ref.cre.isin(["PV", "SST"])].dropna(subset=["p10_fit_delta"])
    g = dr2.groupby(["specimen_id", "direct_like"]).source_rate.mean().unstack()
    g = g.dropna()
    a = t_ci((g[True] - g[False]).values)
    res2.append(dict(model="animal-level paired difference direct - indirect (PV+SST)", n_animals=a["n"],
                     mean_diff=a["mean"], ci_low=a["ci_low"], ci_high=a["ci_high"], p=a["t_p"], wilcoxon_p=a["wilcoxon_p"]))
    tab = dr2.groupby(["cre", "direct_like"]).agg(n=("ref_unit", "size"), mean_source_rate=("source_rate", "mean"),
                                                  pairs=("n_partners", "sum")).reset_index()
    tab.to_csv(T / "4A_direct_vs_indirect_table.csv", index=False)
    pd.DataFrame(res2).to_csv(T / "4A_delta_sigma_tests.csv", index=False)
    S["delta_sigma_gee_p"] = res2[0].get("p")
    S["delta_sigma_gee_OR"] = res2[0].get("odds_ratio")
    S["delta_sigma_animal_diff"] = [a["mean"], a["ci_low"], a["ci_high"], a["t_p"], a["n"]]

    # ------------------------------------------------ 3. power / minimum detectable difference
    # beta-binomial fitted (method of moments) to driven PV+SST reference units
    x = dr2.n_inh.values.astype(float)
    n = dr2.n_partners.values.astype(float)
    p_hat = x.sum() / n.sum()
    # moment estimate of intra-unit correlation
    s2 = np.sum(n * (x / n - p_hat) ** 2) / (len(n) - 1)
    n_bar = n.mean()
    icc = float(np.clip((s2 / (p_hat * (1 - p_hat)) - 1) / (n_bar - 1), 1e-6, 0.99))
    is_dl = dr2.direct_like.values

    def sim_power(delta, nsim=1000):
        hits = 0
        a_ = p_hat * (1 - icc) / icc
        b_ = (1 - p_hat) * (1 - icc) / icc
        for _ in range(nsim):
            pdl = np.clip(p_hat + delta, 1e-4, 0.999)
            a_dl = pdl * (1 - icc) / icc
            b_dl = (1 - pdl) * (1 - icc) / icc
            pu = np.where(is_dl, rng.beta(a_dl, b_dl, len(n)), rng.beta(a_, b_, len(n)))
            k = rng.binomial(n.astype(int), pu)
            r = k / n
            hits += st.ttest_ind(r[is_dl], r[~is_dl], equal_var=False).pvalue < 0.05
        return hits / nsim
    grid = np.round(np.arange(0.01, 0.16, 0.01), 3)
    pw = [(dlt, sim_power(dlt)) for dlt in grid]
    mdd = next((dlt for dlt, p in pw if p >= 0.8), np.nan)
    pd.DataFrame(pw, columns=["true_difference_direct_minus_indirect", "power"]).to_csv(T / "4A_power_curve.csv", index=False)
    diffs = (g[True] - g[False]).values
    df_ = len(diffs) - 1
    mdd_animal = (st.t.ppf(0.975, df_) + st.t.ppf(0.8, df_)) * diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 2 else np.nan
    S["power"] = dict(base_rate=p_hat, intra_unit_icc=icc, n_direct_like=int(is_dl.sum()), n_indirect_like=int((~is_dl).sum()),
                      mdd_unit_level_80pct=mdd, mdd_animal_level_80pct=mdd_animal, n_animals_paired=len(diffs),
                      method="beta-binomial simulation, Welch t on unit rates; animal-level from observed SD of paired differences")
    # positive-control MDD at animal level (PV, strict)
    pcv = pc[(pc.model == "positive control PV, strict (pre-registered)")]

    # ------------------------------------------------ 4. alternative predictors (Holm over 4)
    preds = {"rho_hat": "p10_rho", "dose_slope": "dose_slope", "train_following": "train_following",
             "waveform_duration": "waveform_duration"}
    alt = []
    for name, c in preds.items():
        d4 = dd.dropna(subset=[c]).copy()
        d4["z"] = (d4[c] - d4[c].mean()) / d4[c].std()
        r = gee("inh ~ z + C(cre)", d4, name)
        rr = dr2.dropna(subset=[c])
        sp = st.spearmanr(rr[c], rr.source_rate)
        r.update(spearman_rho_unit=sp.statistic, spearman_p_unit=sp.pvalue, n_units=len(rr))
        alt.append(r)
    alt = pd.DataFrame(alt)
    order = np.argsort(alt.p.values)
    m = len(alt)
    holm = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * alt.p.values[i]))
        holm[i] = run
    alt["p_holm"] = holm
    alt.to_csv(T / "4A_alternative_predictors.csv", index=False)
    S["alternative_predictors"] = alt[["model", "odds_ratio", "or_ci_low", "or_ci_high", "p", "p_holm",
                                       "spearman_rho_unit", "spearman_p_unit"]].to_dict("records")

    # ------------------------------------------------ 5. direction: source vs target
    dirn = []
    for cre in ["PV", "SST", "VIP"]:
        for typ in ["driven", "control"]:
            s = ref[(ref.cre == cre) & (ref.ref_type == typ)]
            dirn.append(dict(cre=cre, ref_type=typ, n=len(s), source_rate=s.n_inh.sum() / max(s.n_partners.sum(), 1),
                             target_rate=s.n_inh_in.sum() / max(s.n_partners.sum(), 1)))
    dirn = pd.DataFrame(dirn)
    tests = []
    for cre in ["PV", "SST"]:
        d = pr[(pr.cre == cre) & pr.ref_type.isin(["driven", "control"])].copy()
        d["is_driven"] = (d.ref_type == "driven").astype(int)
        r = gee("inh_in ~ is_driven", d, f"target rate driven vs control ({cre})")
        tests.append(r)
        # within driven units: source vs target (stacked, unit-level paired via GEE)
        dd5 = pr[(pr.cre == cre) & (pr.ref_type == "driven")]
        st_ = pd.concat([dd5.assign(y=dd5.inh, is_source=1), dd5.assign(y=dd5.inh_in, is_source=0)])
        tests.append(gee("y ~ is_source", st_, f"driven units: source vs target ({cre})"))
    d6 = dd.copy()
    tests.append(gee("inh_in ~ direct_like_i + C(cre)", d6, "target rate: direct-like vs indirect-like (PV+SST)"))
    pd.concat([dirn.assign(kind="rates"), pd.DataFrame(tests).assign(kind="test")]).to_csv(T / "4A_direction.csv", index=False)

    # unit counts behind the positive control
    S["matching"] = dict(
        driven_units=int((ref.ref_type == "driven").sum()),
        driven_with_strict_control=int(len(has)),
        strict_controls=int((ref.ref_type == "control").sum()),
        relaxed_controls=int((ref.ref_type == "control_relaxed").sum()))
    with open(ROOT / "results/calibration/summary_4A.json", "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pc.round(4).to_string(index=False))
        print(pd.DataFrame(res2).round(4).to_string(index=False))
        print(alt.round(4).to_string(index=False))
        print(pd.DataFrame(tests).round(4).to_string(index=False))
        print(dirn.round(4).to_string(index=False))
        print(json.dumps(S["power"], indent=1, default=float))


if __name__ == "__main__":
    main()
