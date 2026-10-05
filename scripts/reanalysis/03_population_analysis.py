"""
03_population_analysis.py
=========================
Population-level spike-based reanalysis (report Sec. 6, steps 3-7).

Reads results/reanalysis/tables/unit_inference.parquet and the CCGs in
data/derived/, writes tables to results/reanalysis/tables/ and a flat
dictionary of headline numbers to results/reanalysis/summary.json.

All between-group comparisons treat the animal (session) as the unit of
replication.
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import (  # noqa: E402
    bh, empirical_null_lfdr, fit_multinomial_mixture, adjusted_rand, ccg_jitter_test,
)
from src.reanalysis.extract_nwb import CCG_BIN, CCG_HALF  # noqa: E402

T = ROOT / "results/reanalysis/tables"
S = {}  # headline numbers
CRES = ["Pvalb-IRES-Cre", "Sst-IRES-Cre", "Vip-IRES-Cre"]
rng = np.random.default_rng(7)


def animal_level(df, col, by="cre_line"):
    """Per-animal means of col, then mean +/- SEM across animals per group."""
    a = df.groupby(["specimen_id", by])[col].mean().reset_index()
    g = a.groupby(by)[col].agg(["mean", "sem", "count"]).reset_index()
    return a, g


def perm_test_groups(a, col, by="cre_line", n_perm=20000):
    """Permutation test (labels shuffled across animals) for between-group variance."""
    vals = a[col].values
    labs = a[by].values

    def stat(lb):
        gm = vals.mean()
        return sum(np.sum(lb == c) * (vals[lb == c].mean() - gm) ** 2 for c in np.unique(lb))
    obs = stat(labs)
    null = np.array([stat(rng.permutation(labs)) for _ in range(n_perm)])
    return float(obs), float((1 + np.sum(null >= obs)) / (n_perm + 1))


def main():
    df = pd.read_parquet(T / "unit_inference.parquet")
    S["n_units"] = int(len(df))
    S["n_sessions"] = int(df.session_id.nunique())
    S["units_by_cre"] = df.cre_line.value_counts().to_dict()
    S["animals_by_cre"] = df.groupby("cre_line").specimen_id.nunique().to_dict()
    S["trials_p10_median"] = float(df.p10_n.median())
    S["baseline_exposure_s_median"] = float((df.p10_n * 0.46).median())

    # ------------------------------------------------------------ A. evidence (lfdr)
    z = df["p10_z"].values
    z_null = np.concatenate(df["p10_sham_z"].values)
    lf, pi0, diag = empirical_null_lfdr(z, z_null, n_null_per_real=10)
    df["lfdr"] = lf
    df["evidence"] = 1 - lf
    df["q_exact"] = bh(df["p10_p_exc"].values)
    df["driven"] = df["lfdr"] < 0.05
    S["pi0"] = pi0
    S["n_lfdr_lt_0.05"] = int(df.driven.sum())
    S["n_lfdr_lt_0.2"] = int((df.lfdr < 0.2).sum())
    S["est_FDR_of_driven_set"] = float(df.loc[df.driven, "lfdr"].mean())
    S["n_exact_BH_q05"] = int((df.q_exact < 0.05).sum())
    S["sham_fpr_exact_p05"] = float(np.mean(st.norm.sf(z_null) < 0.05))
    pd.DataFrame({"z_mid": diag["mids"], "f": diag["f"], "f0": diag["f0"], "lfdr": diag["lfdr_curve"],
                  "count_real": diag["c_real"], "count_null": diag["c_null"]}).to_csv(T / "lfdr_curve.csv", index=False)

    # ------------------------------------------------------------ B. method comparison
    salt_p = df["p10_salt_p"].clip(lower=1 / 1771).values
    df["salt_sig"] = df["p10_salt_p"] < 0.01
    df["salt_q"] = bh(salt_p)
    df["zeta_q"] = bh(df["zeta_p"].values)
    df["zeta_sig_pos"] = (df["zeta_q"] < 0.05) & (df["zeta_sign"] > 0)
    df["zeta50_q"] = bh(df["zeta50_p"].values)
    df["zeta50_sig"] = df["zeta50_q"] < 0.05
    rep = pd.read_parquet(ROOT / "results/neuroscience_study/tables/revised/master_neuroscience_phenotypes_revised.parquet",
                          columns=["unit_id", "method_heuristic_direct", "method_salt_sig", "method_zeta_sig",
                                   "unsupervised_response_archetype", "reference_class"])
    df = df.merge(rep, on="unit_id", how="left")
    S["units_matched_to_repo"] = int(df.method_heuristic_direct.notna().sum())
    heur = df.method_heuristic_direct.fillna(0).astype(bool)
    methods = {
        "lfdr<0.05 (this work)": df.driven,
        "exact test BH q<0.05": df.q_exact < 0.05,
        "SALT (reference port) p<0.01": df.salt_sig,
        "SALT BH q<0.05": df.salt_q < 0.05,
        "ZETA (zetapy) [1,9) ms, BH q<0.05, positive": df.zeta_sig_pos,
        "ZETA (zetapy) [1,51) ms, BH q<0.05, any sign": df.zeta50_sig,
        "repo heuristic label": heur,
        "repo 'SALT'": df.method_salt_sig.fillna(0).astype(bool),
        "repo 'ZETA'": df.method_zeta_sig.fillna(0).astype(bool),
    }
    rows = []
    for name, m in methods.items():
        m = np.asarray(m, bool)
        rows.append(dict(method=name, n_positive=int(m.sum()),
                         overlap_with_lfdr=int((m & df.driven.values).sum()),
                         jaccard_with_lfdr=float((m & df.driven.values).sum() / max((m | df.driven.values).sum(), 1)),
                         **{f"n_{c.split('-')[0]}": int(m[df.cre_line.values == c].sum()) for c in CRES}))
    mc = pd.DataFrame(rows)
    mc.to_csv(T / "method_comparison.csv", index=False)
    S["zeta_sham_fpr_p05"] = float(np.mean(df.zeta_sham_p < 0.05))
    S["zeta_real_p05"] = float(np.mean(df.zeta_p < 0.05))
    S["zeta50_sham_fpr_p05"] = float(np.mean(df.zeta50_sham_p < 0.05))
    S["zeta50_real_p05"] = float(np.mean(df.zeta50_p < 0.05))
    S["zeta50_sham_BH_q05"] = int((bh(df.zeta50_sham_p.values) < 0.05).sum())
    core = ["lfdr<0.05 (this work)", "SALT (reference port) p<0.01", "ZETA (zetapy) [1,51) ms, BH q<0.05, any sign", "repo heuristic label"]
    M = np.column_stack([np.asarray(methods[k], bool) for k in core])
    combos = pd.Series([tuple(r) for r in M]).value_counts().reset_index()
    combos.columns = ["pattern", "n_units"]
    combos[core] = pd.DataFrame(combos.pattern.tolist(), index=combos.index)
    combos.drop(columns="pattern").to_csv(T / "method_overlap_patterns.csv", index=False)

    # ------------------------------------------------------------ C. 5-ms replication
    d = df[df.driven]
    nd = df[df.lfdr > 0.5]
    S["rep5_driven_p05"] = float(np.mean(d.p5_p_exc < 0.05))
    S["rep5_null_p05"] = float(np.mean(nd.p5_p_exc < 0.05))
    S["rep5_rho_spearman"] = float(st.spearmanr(d.p10_rho, d.p5_rho, nan_policy="omit").statistic)
    both = d.dropna(subset=["p10_fit_delta", "p5_fit_delta"])
    S["rep5_delta_n"] = int(len(both))
    if len(both) > 5:
        S["rep5_delta_spearman"] = float(st.spearmanr(both.p10_fit_delta, both.p5_fit_delta).statistic)
        S["rep5_delta_median_diff_ms"] = float((both.p5_fit_delta - both.p10_fit_delta).median())

    # ------------------------------------------------------------ D. latency / jitter characterization
    fit = d.dropna(subset=["p10_fit_delta"])
    char = fit.groupby("cre_line").agg(n=("unit_id", "size"),
                                       rho_median=("p10_fit_rho", "median"),
                                       delta_median_ms=("p10_fit_delta", "median"),
                                       delta_q25=("p10_fit_delta", lambda x: x.quantile(0.25)),
                                       delta_q75=("p10_fit_delta", lambda x: x.quantile(0.75)),
                                       sigma_median_ms=("p10_fit_sigma", "median"),
                                       naive_median_latency_ms=("p10_median_first_ms", "median"),
                                       naive_reliability_median=("p10_rel_raw", "median")).reset_index()
    char.to_csv(T / "driven_unit_latency_characterisation.csv", index=False)
    S["driven_delta_median_ms"] = float(fit.p10_fit_delta.median())
    S["driven_sigma_median_ms"] = float(fit.p10_fit_sigma.median())
    S["driven_frac_direct_like"] = float(np.mean((fit.p10_fit_delta < 5.0) & (fit.p10_fit_sigma < 1.5)))

    # latency criterion specificity among null-like units
    nl = df[(df.lfdr > 0.5) & (df.p10_k >= 1)]
    S["null_units_with_spike_frac_median_lat_lt8"] = float(np.mean(nl.p10_median_first_ms < 8.0))
    S["null_units_with_spike_n"] = int(len(nl))

    # ------------------------------------------------------------ E. waveform validation
    df["narrow"] = df.waveform_duration < 0.4
    wf = df.groupby(["cre_line", "driven"]).narrow.agg(["mean", "size"]).reset_index()
    wf.to_csv(T / "waveform_narrow_fraction.csv", index=False)
    for c in CRES:
        sub = df[df.cre_line == c]
        tab = pd.crosstab(sub.driven, sub.narrow)
        if tab.shape == (2, 2):
            orr, p = st.fisher_exact(tab.values)
            S[f"narrow_OR_{c.split('-')[0]}"] = float(orr)
            S[f"narrow_p_{c.split('-')[0]}"] = float(p)
        S[f"narrow_frac_driven_{c.split('-')[0]}"] = float(sub[sub.driven].narrow.mean()) if sub.driven.any() else np.nan
        S[f"narrow_frac_all_{c.split('-')[0]}"] = float(sub.narrow.mean())

    # ------------------------------------------------------------ F. yield per animal
    df["driven_f"] = df.driven.astype(float)
    a_y, g_y = animal_level(df, "driven_f")
    g_y.to_csv(T / "yield_by_cre_animal_level.csv", index=False)
    a_y.to_csv(T / "yield_per_animal.csv", index=False)
    S["yield_perm_p"] = perm_test_groups(a_y, "driven_f")[1]
    S["yield_by_cre"] = {r.cre_line: [float(r["mean"]), float(r["sem"]), int(r["count"])] for _, r in g_y.iterrows()}
    S["yield_by_cre_counts"] = df.groupby("cre_line").driven.sum().astype(int).to_dict()

    # ------------------------------------------------------------ G. archetypes (multinomial mixture)
    cols = ["base", "w1", "w2", "w3", "w4", "w5"]
    ok = df[[f"cnt_{c}" for c in cols]].notna().all(1)
    Y = df.loc[ok, [f"cnt_{c}" for c in cols]].values.astype(float)
    E = df.loc[ok, [f"exp_{c}" for c in cols]].values.astype(float)
    fits, bic = {}, {}
    for k in range(1, 17):
        fits[k] = fit_multinomial_mixture(Y, E, k, n_init=4, seed=k)
        bic[k] = fits[k]["bic"]
    S["archetype_k_bic_min"] = int(min(bic, key=bic.get))
    S["archetype_bic_drop_1to2"] = float(bic[1] - bic[2])
    S["archetype_bic_drop_last"] = float(bic[max(bic) - 1] - bic[max(bic)])

    def assign(logm, w):
        from scipy.special import logsumexp
        lp = np.log(E[:, None, :]) + logm[None]
        lp = lp - logsumexp(lp, axis=2, keepdims=True)
        return (np.einsum("nw,nkw->nk", Y, lp) + np.log(w)[None]).argmax(1)
    stab = {}
    for kk in (4, 6, 8):
        lab_k = fits[kk]["resp"].argmax(1)
        aris = []
        for b in range(4):
            idx = rng.integers(0, len(Y), len(Y))
            fb = fit_multinomial_mixture(Y[idx], E[idx], kk, n_init=2, seed=100 + b)
            aris.append(adjusted_rand(lab_k, assign(fb["logm"], fb["weights"])))
        stab[kk] = aris
    S["archetype_bootstrap_ARI"] = {str(k): [float(x) for x in v] for k, v in stab.items()}
    kb = 6  # descriptive summary; BIC does not identify a discrete number of types
    S["archetype_k_displayed"] = kb
    best = fits[kb]
    lab = best["resp"].argmax(1)
    arch = pd.DataFrame(np.exp(best["logm"][:, 1:]), columns=[f"rate_ratio_{c}" for c in cols[1:]])
    arch["weight"] = best["weights"]
    arch["n_units"] = np.bincount(lab, minlength=kb)
    arch["archetype"] = np.arange(kb)
    df.loc[ok, "archetype"] = lab
    comp = pd.crosstab(df.loc[ok, "archetype"], df.loc[ok, "cre_line"], normalize="columns")
    for c in CRES:
        arch[f"frac_of_{c.split('-')[0]}_units"] = comp[c].values if c in comp else np.nan
    # animal-level association of archetype fractions with Cre
    ptab = []
    for k in range(kb):
        df["_ak"] = (df.archetype == k).astype(float)
        a_k, _ = animal_level(df[ok], "_ak")
        ptab.append(perm_test_groups(a_k, "_ak", n_perm=5000)[1])
    arch["cre_assoc_perm_p_animal_level"] = ptab
    arch["cre_assoc_q"] = bh(np.array(ptab))
    arch.to_csv(T / "archetypes.csv", index=False)

    # ------------------------------------------------------------ H. suppression prevalence
    df["q_sup_w3"] = bh(df.p_sup_w3.fillna(1).values)
    df["q_sup_w4"] = bh(df.p_sup_w4.fillna(1).values)
    df["suppressed"] = ((df.q_sup_w3 < 0.05) | (df.q_sup_w4 < 0.05)) & ~df.driven
    df["detectable"] = (df.lam_spont_hz * 0.15 * df.p10_n) >= 20  # expected W4 count under no change
    df["suppressed_f"] = df.suppressed.astype(float)
    a_s, g_s = animal_level(df, "suppressed_f")
    a_s2, g_s2 = animal_level(df[df.detectable], "suppressed_f")
    g_s.assign(subset="all units").to_csv(T / "suppression_by_cre.csv", index=False)
    g_s2.assign(subset="detectable").to_csv(T / "suppression_by_cre.csv", mode="a", header=False, index=False)
    S["suppressed_n"] = int(df.suppressed.sum())
    S["suppressed_by_cre_all"] = {r.cre_line: [float(r["mean"]), float(r["sem"])] for _, r in g_s.iterrows()}
    S["suppressed_by_cre_detectable"] = {r.cre_line: [float(r["mean"]), float(r["sem"])] for _, r in g_s2.iterrows()}
    S["suppression_perm_p"] = perm_test_groups(a_s, "suppressed_f")[1]
    S["suppression_detectable_perm_p"] = perm_test_groups(a_s2, "suppressed_f")[1]
    S["repo_prolonged_suppression_frac"] = float(df.unsupervised_response_archetype.fillna("").str.contains("Suppression").mean())

    # by brain-region group (light is delivered over visual cortex)
    def region(a):
        a = str(a)
        if a.startswith("VIS"):
            return "visual cortex"
        if a in ("CA1", "CA2", "CA3", "DG", "ProS", "SUB", "HPF", "POST", "PRE") or a.startswith("CA"):
            return "hippocampal formation"
        if a in ("LGd", "LGv", "LP", "PO", "POL", "TH", "Eth", "IGL", "LD", "SGN", "VPM", "VPL", "MGv", "MGd", "MGm", "IntG", "PP", "PIL", "RT", "SPF", "VL", "CL", "LT"):
            return "thalamus"
        return "midbrain / other / unassigned"
    df["region_group"] = df.structure.map(region)
    reg = df.groupby(["cre_line", "region_group"]).agg(n_units=("unit_id", "size"), driven_frac=("driven_f", "mean"),
                                                       suppressed_frac=("suppressed_f", "mean")).reset_index()
    reg.to_csv(T / "driven_suppressed_by_region.csv", index=False)
    vc = df[df.region_group == "visual cortex"]
    a_v, g_v = animal_level(vc, "suppressed_f")
    S["suppressed_visctx_by_cre"] = {r.cre_line: [float(r["mean"]), float(r["sem"])] for _, r in g_v.iterrows()}
    S["suppressed_visctx_perm_p"] = perm_test_groups(a_v, "suppressed_f")[1]
    a_vy, g_vy = animal_level(vc, "driven_f")
    S["driven_visctx_by_cre"] = {r.cre_line: [float(r["mean"]), float(r["sem"])] for _, r in g_vy.iterrows()}
    S["driven_frac_in_visctx"] = float(df[df.driven].region_group.eq("visual cortex").mean())

    # ------------------------------------------------------------ I. dose-response (driven units)
    lv = [1.0, 2.5, 4.0]
    x = np.log2(lv)
    dd = df[df.driven].copy()
    R = dd[[f"p10_L{L}_rho" for L in lv]].values
    dd["dose_slope_rho_per_log2"] = [np.polyfit(x, r, 1)[0] if np.isfinite(r).all() else np.nan for r in R]
    Lt = dd[[f"p10_L{L}_median_first_ms" for L in lv]].values
    dd["dose_slope_latency_ms_per_log2"] = [np.polyfit(x, r, 1)[0] if np.isfinite(r).all() else np.nan for r in Lt]
    dose_rows = []
    for col in ["dose_slope_rho_per_log2", "dose_slope_latency_ms_per_log2"]:
        a_d = dd.groupby(["specimen_id", "cre_line"])[col].mean().reset_index().dropna()
        for c in CRES + ["all"]:
            v = a_d[col] if c == "all" else a_d[a_d.cre_line == c][col]
            if len(v) >= 3:
                t, p = st.ttest_1samp(v, 0)
                dose_rows.append(dict(metric=col, cre_line=c, n_animals=len(v), mean=v.mean(), sem=v.sem(), t=t, p=p))
            elif len(v):
                dose_rows.append(dict(metric=col, cre_line=c, n_animals=len(v), mean=v.mean(), sem=np.nan, t=np.nan, p=np.nan))
    pd.DataFrame(dose_rows).to_csv(T / "dose_response_animal_level.csv", index=False)
    dd[["unit_id", "specimen_id", "cre_line"] + [f"p10_L{L}_rho" for L in lv] + [f"p10_L{L}_median_first_ms" for L in lv]
       ].groupby("cre_line").median(numeric_only=True).to_csv(T / "dose_response_medians.csv")
    S["dose_rho_by_level_median"] = {str(L): float(dd[f"p10_L{L}_rho"].median()) for L in lv}
    S["dose_latency_by_level_median"] = {str(L): float(dd[f"p10_L{L}_median_first_ms"].median()) for L in lv}

    # ------------------------------------------------------------ J. 10-Hz trains (driven units)
    tt = dd[dd.train_n > 0].copy()
    n = tt.train_n.values
    p0 = 1 - np.exp(-tt.train_lam0_hz.values * 0.008)
    K = tt[[f"train_k{j}" for j in range(1, 11)]].values
    rho_j = (K / n[:, None] - p0[:, None]) / (1 - p0[:, None])
    tt["train_log_ratio_10_1"] = np.log((K[:, 9] + 0.5) / (K[:, 0] + 0.5))
    tt["train_log_ratio_se"] = np.sqrt(1 / (K[:, 9] + 0.5) + 1 / (K[:, 0] + 0.5))
    slopes = []
    for i in range(len(tt)):
        yk, nn = K[i], n[i]
        Xd = sm.add_constant(np.arange(10.0))
        try:
            g = sm.GLM(np.column_stack([yk, nn - yk]), Xd, family=sm.families.Binomial(),
                       offset=np.full(10, 0.0)).fit()
            slopes.append(g.params[1])
        except Exception:
            slopes.append(np.nan)
    tt["train_logit_slope_per_pulse"] = slopes
    traj = pd.DataFrame(rho_j, columns=[f"rho_pulse{j}" for j in range(1, 11)])
    traj["cre_line"] = tt.cre_line.values
    traj.groupby("cre_line").median().to_csv(T / "train_trajectory_median_rho.csv")
    tr_rows = []
    for col in ["train_log_ratio_10_1", "train_logit_slope_per_pulse"]:
        a_t = tt.groupby(["specimen_id", "cre_line"])[col].mean().reset_index().dropna()
        for c in CRES + ["all"]:
            v = a_t[col] if c == "all" else a_t[a_t.cre_line == c][col]
            if len(v) >= 3:
                t_, p_ = st.ttest_1samp(v, 0)
            else:
                t_, p_ = np.nan, np.nan
            tr_rows.append(dict(metric=col, cre_line=c, n_animals=len(v), n_units=int((tt.cre_line == c).sum()) if c != "all" else len(tt),
                                mean=v.mean() if len(v) else np.nan, sem=v.sem() if len(v) > 1 else np.nan, t=t_, p=p_))
        groups = [a_t[a_t.cre_line == c][col].values for c in CRES if (a_t.cre_line == c).sum() >= 2]
        if len(groups) >= 2:
            tr_rows.append(dict(metric=col, cre_line="Kruskal-Wallis across Cre (animal level)", n_animals=len(a_t),
                                n_units=len(tt), mean=np.nan, sem=np.nan, t=st.kruskal(*groups).statistic,
                                p=st.kruskal(*groups).pvalue))
    pd.DataFrame(tr_rows).to_csv(T / "train_dynamics_animal_level.csv", index=False)
    S["train_units"] = int(len(tt))
    S["train_frac_units_decline"] = float(np.mean(tt.train_logit_slope_per_pulse < 0))

    # ------------------------------------------------------------ K. depth
    ctx = df[df.cortical_depth_um.notna()].copy()
    ctx["depth100"] = ctx.cortical_depth_um / 100.0
    ctx["probe_key"] = ctx.session_id.astype(str) + "_" + ctx.probe_name
    S["cortex_units"] = int(len(ctx))
    try:
        lg = smf.glm("driven_f ~ depth100 + C(session_id)", data=ctx, family=sm.families.Binomial()).fit()
        S["depth_logit_driven_per100um"] = [float(lg.params["depth100"]), float(lg.bse["depth100"]), float(lg.pvalues["depth100"])]
    except Exception as e:  # noqa: BLE001
        S["depth_logit_error"] = str(e)
    cd = ctx[ctx.driven].dropna(subset=["p10_fit_delta"])
    dep_rows = []
    for yv in ["p10_rho", "p10_fit_delta", "p10_fit_sigma"]:
        if len(cd) > 10 and cd.probe_key.nunique() > 2:
            mm = smf.mixedlm(f"{yv} ~ depth100", cd, groups=cd["probe_key"]).fit()
            dep_rows.append(dict(outcome=yv, n_units=len(cd), n_probes=cd.probe_key.nunique(),
                                 slope_per_100um=mm.params["depth100"], se=mm.bse["depth100"], p=mm.pvalues["depth100"]))
    pd.DataFrame(dep_rows).to_csv(T / "depth_models.csv", index=False)
    ctx["depth_bin"] = pd.cut(ctx.cortical_depth_um, [0, 200, 400, 600, 800, 1200])
    ctx.groupby(["cre_line", "depth_bin"], observed=True).driven_f.agg(["mean", "size"]).reset_index().to_csv(
        T / "driven_fraction_by_depth.csv", index=False)

    # ------------------------------------------------------------ L. connectivity CCGs
    drv = set(df.loc[df.driven, "unit_id"].astype(int))
    ccg_rows = []
    for f in sorted((ROOT / "data/derived").glob("session_*.pkl.gz")):
        with gzip.open(f, "rb") as fh:
            R = pickle.load(fh)
        for _, r in R["ccg"].iterrows():
            if int(r.unit_a) not in drv or r.n_a < 500 or r.n_b < 500:
                continue
            c = np.asarray(r.counts)
            pe, pi, pk, tr_ = ccg_jitter_test(c, CCG_BIN, CCG_HALF)
            pe_c, pi_c, _, _ = ccg_jitter_test(c, CCG_BIN, CCG_HALF, win=(-0.0040, -0.0008))
            ccg_rows.append(dict(session_id=R["session_id"], unit_a=int(r.unit_a), unit_b=int(r.unit_b),
                                 dy_um=r.dy_um, n_a=r.n_a, n_b=r.n_b, p_exc=pe, p_inh=pi, peak_ratio=pk,
                                 trough_ratio=tr_, p_exc_anticausal=pe_c, p_inh_anticausal=pi_c,
                                 b_driven=int(r.unit_b) in drv))
    cg = pd.DataFrame(ccg_rows)
    if len(cg):
        cg = cg.merge(df[["session_id", "cre_line", "specimen_id"]].drop_duplicates(), on="session_id")
        cg.to_csv(T / "ccg_pairs.csv", index=False)
        summ = cg.groupby("cre_line").agg(pairs=("p_inh", "size"), ref_units=("unit_a", "nunique"),
                                          inh_frac=("p_inh", lambda x: np.mean(x < 0.01)),
                                          inh_frac_anticausal=("p_inh_anticausal", lambda x: np.mean(x < 0.01)),
                                          exc_frac=("p_exc", lambda x: np.mean(x < 0.01)),
                                          exc_frac_anticausal=("p_exc_anticausal", lambda x: np.mean(x < 0.01))).reset_index()
        summ.to_csv(T / "ccg_summary.csv", index=False)
        S["ccg_pairs"] = int(len(cg))
        cg["dist_bin"] = pd.cut(cg.dy_um.abs(), [-1, 50, 100, 200, 300])
        cg.groupby(["cre_line", "dist_bin"], observed=True).agg(
            pairs=("p_inh", "size"), inh_frac=("p_inh", lambda x: np.mean(x < 0.01))).reset_index().to_csv(
            T / "ccg_by_distance.csv", index=False)

    # ------------------------------------------------------------ M. revisiting repo claims
    lam = df.lam_spont_hz.values
    has = (df.p10_k >= 1).values
    tier_rows = []
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]:
        m = (lam >= lo) & (lam < hi)
        exp_ = np.sum(1 - np.exp(-lam[m] * 0.008 * df.p10_n.values[m]))
        exp_nd = np.sum((1 - np.exp(-lam[m] * 0.008 * df.p10_n.values[m]))[~df.driven.values[m]])
        tier_rows.append(dict(tier_hz=f"[{lo},{hi})", n_units=int(m.sum()), observed_with_spike=int(has[m].sum()),
                              expected_poisson=float(exp_), obs_over_exp=float(has[m].sum() / exp_) if exp_ else np.nan,
                              observed_nondriven=int((has & ~df.driven.values)[m].sum()),
                              expected_nondriven=float(exp_nd), driven=int(df.driven.values[m].sum())))
    pd.DataFrame(tier_rows).to_csv(T / "sparse_tier_poisson_check.csv", index=False)

    ct = pd.crosstab(df.reference_class.fillna("not in repo table"), df.driven.map({True: "lfdr<0.05", False: "lfdr>=0.05"}))
    ct.to_csv(T / "repo_label_vs_lfdr.csv")
    S["heuristic_direct_also_driven"] = int((heur & df.driven).sum())
    S["heuristic_direct_n"] = int(heur.sum())
    S["driven_missed_by_heuristic"] = int((~heur & df.driven).sum())

    icc_rows = []
    for col, tf in [("lam_spont_hz", np.log1p), ("p10_lrr", lambda v: v), ("driven_f", lambda v: v)]:
        v = tf(df[col].astype(float))
        g = v.groupby(df.specimen_id)
        ni = g.size().values
        k = len(ni)
        N = ni.sum()
        msb = np.sum(ni * (g.mean().values - v.mean()) ** 2) / (k - 1)
        msw = np.sum(g.var(ddof=1).values * (ni - 1)) / (N - k)
        n0 = (N - np.sum(ni ** 2) / N) / (k - 1)
        s2b = max(0, (msb - msw) / n0)
        icc = s2b / (s2b + msw)
        F = msb / msw
        icc_rows.append(dict(metric=col, icc1=icc, F=F, p=st.f.sf(F, k - 1, N - k), design_effect=1 + (N / k - 1) * icc))
    pd.DataFrame(icc_rows).to_csv(T / "variance_decomposition.csv", index=False)

    # analytic threshold instability for driven units at the 0.30 raw-reliability cut
    r = dd.p10_rel_raw.values
    nn = dd.p10_n.values
    pflip = st.norm.cdf(-np.abs(r - 0.30) / np.sqrt(np.clip(r * (1 - r), 1e-6, None) / nn))
    S["driven_mean_flip_prob_rel030"] = float(np.mean(pflip))
    S["driven_frac_flip_prob_gt_0.2"] = float(np.mean(pflip > 0.2))

    keep = [c for c in df.columns if c not in ("p10_sham_z", "p5_sham_z", "_ak")]
    df[keep].to_parquet(T / "unit_results.parquet", index=False)
    with open(ROOT / "results/reanalysis/summary.json", "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
