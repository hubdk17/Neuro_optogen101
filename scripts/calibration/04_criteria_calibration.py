"""
04_criteria_calibration.py
==========================
§4C: calibration of the optotagging criteria the field uses, on real spikes.

Every criterion is evaluated through ONE code path on two trial sets per unit:
  real : the 10-ms pulse trials (and 10-Hz train trials) of the session;
  sham : the same number of pseudo-trials placed at random onsets inside the
         unit's valid spontaneous epochs (no light), with the same windows and
         the same 10-pulse/100-ms train geometry. Sham onsets are shared by all
         units on a probe, as real trials are. Seed 20261006.

Criteria (operationalisation stated explicitly, because published wording is terse):
  allen_lakunina      >= 4 of the first 5 train pulses with a significant per-pulse
                      response (one-sided binomial vs chance p0, alpha 0.05)
                      AND raw 10-ms trial reliability >= 0.30
                      AND median first-spike latency < 8 ms
  allen_lakunina_raw  variant: per-pulse raw response probability >= 0.30 at >= 4 of 5
                      pulses (no significance), + reliability >= 0.30 + latency < 8 ms
  latency_lt8         median first-spike latency in [1, 9) ms < 8 ms (alone)
  rel030_mod2         raw reliability >= 0.30 AND modulation ratio > 2, with the
                      original pipeline's definitions (evoked [1,9) ms rate over
                      baseline [-20,-5) ms rate + 1 Hz)
  heuristic_full      the original operational rule: rel >= 0.30, latency < 8 ms,
                      modulation > 2, sign-flip permutation p < 0.05 (B = 1000),
                      paired Cohen's d > 0.10
  salt_p01            SALT port p < 0.01;   salt_bh  BH q < 0.05 across units
  zeta9_bh            zetapy ZETA, window [1, 9) ms, positive deviation, BH q < 0.05
  zeta51_bh           zetapy ZETA, window [1, 51) ms, any sign, BH q < 0.05
  exact_bh            exact conditional test BH q < 0.05
  lfdr05              local FDR < 0.05 using the lfdr curve fitted on the real data
                      (src/reanalysis/inference.empirical_null_lfdr) - for sham trials
                      this measures the FPR of the calibrated procedure itself

Metrics:
  (i)  sham null: per-unit false-positive rate; implied FDR = FPR x N / N_pass(real)
  (ii) against the calibrated reference: positives = driven (lfdr < 0.05, artifacts
       excluded); negatives = lfdr > 0.5. Sensitivity, specificity, precision.
       NOTE: exact_bh and lfdr05 define/neighbour the reference set, so their
       scores in (ii) are partly circular and are flagged as such.
  95% CIs by cluster bootstrap over animals (2,000 resamples).

Outputs: results/calibration/tables/4C_unit_criteria.parquet, 4C_criteria_calibration.csv,
         results/calibration/summary_4C.json
"""

import gzip
import json
import logging
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import exact_rate_test, salt, bh, empirical_null_lfdr  # noqa: E402

T = ROOT / "results/calibration/tables"
SEED = 20261006
EV = (0.001, 0.009)
BASE = (-0.480, -0.020)
HBASE = (-0.020, -0.005)          # original heuristic baseline
W_E, W_B = EV[1] - EV[0], BASE[1] - BASE[0]
CRITERIA = ["allen_lakunina", "allen_lakunina_raw", "latency_lt8", "rel030_mod2", "heuristic_full",
            "salt_p01", "salt_bh", "zeta9_bh", "zeta51_bh", "exact_bh", "lfdr05"]
CIRCULAR = {"exact_bh", "lfdr05"}


def realign(spk, onsets, pre=-0.6, post=1.3):
    lo = np.searchsorted(spk, onsets + pre)
    hi = np.searchsorted(spk, onsets + post)
    rel = np.concatenate([spk[a:b] - t for a, b, t in zip(lo, hi, onsets)]) if len(onsets) else np.zeros(0)
    ep = np.concatenate([np.full(b - a, k) for k, (a, b) in enumerate(zip(lo, hi))]) if len(onsets) else np.zeros(0, int)
    return rel, ep.astype(int)


def sham_onsets(intervals, n, rng, margin_pre=0.6, margin_post=1.3, min_gap=2.0):
    """n random onsets inside spontaneous intervals, windows fully inside, >= min_gap apart."""
    iv = [(s + margin_pre, e - margin_post) for s, e in intervals if e - margin_post > s + margin_pre]
    if not iv:
        return np.zeros(0)
    lens = np.array([b - a for a, b in iv])
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        k = rng.choice(len(iv), p=lens / lens.sum())
        t = rng.uniform(*iv[k])
        if all(abs(t - x) >= min_gap for x in out):
            out.append(t)
    return np.sort(np.array(out))


def pulse_stats(rel, ep, n, rng_seed):
    """Statistics on n pulse trials (trial index 0..n-1)."""
    out = {}
    em = (rel >= EV[0]) & (rel < EV[1])
    ne = int(em.sum())
    nb = int(np.count_nonzero((rel >= BASE[0]) & (rel < BASE[1])))
    first = pd.Series(rel[em]).groupby(ep[em]).min() * 1000
    k = len(first)
    lam0 = nb / (n * W_B)
    pe, _, z = exact_rate_test(ne, nb, n * W_E, n * W_B)
    out.update(n=n, ne=ne, nb=nb, k=k, rel=k / n, med_lat=float(first.median()) if k else np.nan,
               p_exact=float(pe), z=float(np.ravel(z)[0]), lam0=lam0)
    # original heuristic quantities
    e_cnt = np.bincount(ep[em], minlength=n)[:n].astype(float)
    hb = (rel >= HBASE[0]) & (rel < HBASE[1])
    b_cnt = np.bincount(ep[hb], minlength=n)[:n].astype(float)
    hdur = HBASE[1] - HBASE[0]
    ev_rate = e_cnt.mean() / W_E
    b_rate = b_cnt.mean() / hdur
    out["mod_ratio"] = ev_rate / (b_rate + 1.0)
    diffs = e_cnt - b_cnt * (W_E / hdur)
    if np.all(diffs == 0):
        out["perm_p"], out["cohen_d"] = 1.0, 0.0
    else:
        rng = np.random.default_rng(rng_seed)
        signs = rng.choice([-1.0, 1.0], size=(1000, n))
        perm = np.abs((signs * diffs).mean(1))
        out["perm_p"] = float((np.sum(perm >= abs(diffs.mean())) + 1) / 1001)
        sd = diffs.std(ddof=1)
        out["cohen_d"] = float(diffs.mean() / sd) if sd > 1e-9 else 0.0
    out["salt_p"], _ = salt(rel, ep, n)
    return out


def train_stats(rel, ep, n):
    nb = int(np.count_nonzero((rel >= BASE[0]) & (rel < BASE[1])))
    p0 = 1 - np.exp(-(nb / (n * W_B)) * W_E)
    sig, raw = 0, 0
    for j in range(5):
        lo, hi = 0.1 * j + EV[0], 0.1 * j + EV[1]
        kj = len(np.unique(ep[(rel >= lo) & (rel < hi)]))
        if st.binom.sf(kj - 1, n, min(max(p0, 1e-9), 1 - 1e-9)) < 0.05:
            sig += 1
        if kj / n >= 0.30:
            raw += 1
    return dict(train_sig_pulses=sig, train_raw_pulses=raw)


def zeta(rel_abs, events, dur, seed):
    from zetapy import zetatest
    np.random.seed(seed % (2 ** 32))
    try:
        d = zetatest(np.unique(rel_abs), events + 0.001, max_duration=dur, resampling_number=250)[1]
        return float(d["zeta_p_value"]), float(np.sign(d.get("zeta_deviation", np.nan)))
    except Exception:
        return 1.0, np.nan


def session(path):
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    with gzip.open(path, "rb") as fh:
        R = pickle.load(fh)
    assert R.get("extract_version", 1) >= 2
    sid = R["session_id"]
    rng = np.random.default_rng(SEED + sid % 100000)
    opto = R["opto"]
    on_all = opto.start_time.values
    rows = []
    sham_cache = {}
    for _, unit in R["units"].iterrows():
        uid = int(unit.unit_id)
        p = unit.probe_name
        valid = R["valid_epochs"][p]
        p10 = opto.index[(opto.kind == "pulse10").values & valid].values
        tr = opto.index[(opto.kind == "train10hz").values & valid].values
        if p not in sham_cache:
            iv = R["spont_intervals"][p]
            sham_cache[p] = (sham_onsets(iv, len(p10), rng), sham_onsets(iv, len(tr), rng))
        sh_p, sh_t = sham_cache[p]
        rel, ep = R["aligned"][uid]
        rel = rel.astype(np.float64)
        spont = R["spont_spikes"][uid]
        row = dict(session_id=sid, unit_id=uid)
        sets = {}
        # real
        m = np.isin(ep, p10)
        sets["real"] = (rel[m], np.searchsorted(np.sort(p10), ep[m]), len(p10), on_all[np.sort(p10)],
                        rel[np.isin(ep, tr)], np.searchsorted(np.sort(tr), ep[np.isin(ep, tr)]), len(tr))
        # sham
        r_s, e_s = realign(spont, sh_p)
        r_t, e_t = realign(spont, sh_t)
        sets["sham"] = (r_s, e_s, len(sh_p), sh_p, r_t, e_t, len(sh_t))
        for tag, (r, e, n, onsets, rt, et, nt) in sets.items():
            if n == 0:
                continue
            ps = pulse_stats(r, e, n, uid + (0 if tag == "real" else 7))
            row.update({f"{tag}_{k}": v for k, v in ps.items()})
            if nt:
                row.update({f"{tag}_{k}": v for k, v in train_stats(rt, et, nt).items()})
            abs_spk = np.sort(onsets[e] + r) if len(r) else np.zeros(0)
            row[f"{tag}_zeta9_p"], row[f"{tag}_zeta9_sign"] = zeta(abs_spk, onsets, 0.008, uid + 11 + (0 if tag == "real" else 1))
            row[f"{tag}_zeta51_p"], row[f"{tag}_zeta51_sign"] = zeta(abs_spk, onsets, 0.050, uid + 13 + (0 if tag == "real" else 1))
        rows.append(row)
    return pd.DataFrame(rows)


def apply_criteria(df, tag, lf_curve):
    c = pd.DataFrame(index=df.index)
    lat8 = df[f"{tag}_med_lat"] < 8.0
    rel3 = df[f"{tag}_rel"] >= 0.30
    c["allen_lakunina"] = (df[f"{tag}_train_sig_pulses"] >= 4) & rel3 & lat8
    c["allen_lakunina_raw"] = (df[f"{tag}_train_raw_pulses"] >= 4) & rel3 & lat8
    c["latency_lt8"] = lat8
    c["rel030_mod2"] = rel3 & (df[f"{tag}_mod_ratio"] > 2)
    c["heuristic_full"] = rel3 & lat8 & (df[f"{tag}_mod_ratio"] > 2) & (df[f"{tag}_perm_p"] < 0.05) & (df[f"{tag}_cohen_d"] > 0.10)
    c["salt_p01"] = df[f"{tag}_salt_p"] < 0.01
    c["salt_bh"] = bh(np.clip(df[f"{tag}_salt_p"].values, 1 / 1771, 1)) < 0.05
    c["zeta9_bh"] = (bh(df[f"{tag}_zeta9_p"].values) < 0.05) & (df[f"{tag}_zeta9_sign"] > 0)
    c["zeta51_bh"] = bh(df[f"{tag}_zeta51_p"].values) < 0.05
    c["exact_bh"] = bh(df[f"{tag}_p_exact"].values) < 0.05
    mids, lfc = lf_curve
    idx = np.clip(np.searchsorted(mids, df[f"{tag}_z"].values) - 1, 0, len(mids) - 1)
    c["lfdr05"] = lfc[idx] < 0.05
    return c.fillna(False).astype(bool)


def metrics(real, sham, pos, neg, n_units):
    out = {}
    for k in CRITERIA:
        r, s = real[k].values, sham[k].values
        fpr = s.mean()
        npass = r.sum()
        out[k] = dict(n_pass_real=int(npass), sham_fpr=fpr, sham_n_pass=int(s.sum()),
                      implied_fdr=min(1.0, fpr * n_units / npass) if npass else np.nan,
                      sensitivity=r[pos].mean(), specificity=1 - r[neg].mean(),
                      precision=(r & pos).sum() / npass if npass else np.nan)
    return out


def main():
    T.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data/derived").glob("session_*.pkl.gz"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        df = pd.concat(list(ex.map(session, files)), ignore_index=True)
    u = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet",
                        columns=["unit_id", "specimen_id", "cre_line", "driven", "lfdr", "onset_artifact", "p10_z", "p10_sham_z"]
                        if "p10_sham_z" in pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet").columns
                        else ["unit_id", "specimen_id", "cre_line", "driven", "lfdr", "onset_artifact", "p10_z"])
    df = df.merge(u, on="unit_id")
    # lfdr curve as fitted on the real data (same procedure as 03_population_analysis)
    lc = pd.read_csv(ROOT / "results/reanalysis/tables/lfdr_curve.csv")
    curve = (lc.z_mid.values - (lc.z_mid.values[1] - lc.z_mid.values[0]) / 2, lc.lfdr.values)
    real = apply_criteria(df, "real", curve)
    sham = apply_criteria(df, "sham", curve)
    # consistency check: the recomputed real exact test must reproduce the stored driven set's p-values
    df["real_driven_ref"] = df.driven
    pos = df.driven.values
    neg = (df.lfdr > 0.5).values
    N = len(df)
    M = metrics(real, sham, pos, neg, N)

    # cluster bootstrap over animals
    rng = np.random.default_rng(SEED)
    animals = df.specimen_id.unique()
    boot = {k: {m: [] for m in ("sham_fpr", "sensitivity", "specificity", "precision")} for k in CRITERIA}
    idx_by_animal = {a: np.nonzero(df.specimen_id.values == a)[0] for a in animals}
    for _ in range(2000):
        pick = rng.choice(animals, len(animals), replace=True)
        ii = np.concatenate([idx_by_animal[a] for a in pick])
        Mb = metrics(real.iloc[ii], sham.iloc[ii], pos[ii], neg[ii], len(ii))
        for k in CRITERIA:
            for m in boot[k]:
                boot[k][m].append(Mb[k][m])
    rows = []
    for k in CRITERIA:
        r = dict(criterion=k, circular_with_reference=k in CIRCULAR, **M[k])
        for m in boot[k]:
            v = np.array(boot[k][m], float)
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(T / "4C_criteria_calibration.csv", index=False)
    # per-animal pass counts (who contributes)
    pa = pd.concat([real.add_prefix("real_"), sham.add_prefix("sham_")], axis=1)
    pa["specimen_id"] = df.specimen_id.values
    pa["cre_line"] = df.cre_line.values
    pa.groupby(["specimen_id", "cre_line"]).sum().reset_index().to_csv(T / "4C_per_animal_pass_counts.csv", index=False)
    out = pd.concat([df, real.add_prefix("crit_real_"), sham.add_prefix("crit_sham_")], axis=1)
    out.drop(columns=[c for c in out.columns if c == "p10_sham_z"], errors="ignore").to_parquet(T / "4C_unit_criteria.parquet", index=False)

    # latency<8 among null units with any evoked spike (real trials)
    nl = df[(df.lfdr > 0.5) & (df.real_k >= 1)]
    S = dict(n_units=N, n_driven=int(pos.sum()), n_null_ref=int(neg.sum()),
             latency_lt8_among_null_with_spike=float((nl.real_med_lat < 8).mean()), n_null_with_spike=len(nl),
             latency_lt8_sham_among_units_with_spike=float((df[df.sham_k >= 1].sham_med_lat < 8).mean()),
             recomputed_exact_p_max_abs_diff_vs_stored=None)
    st_ = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet", columns=["unit_id", "p10_p_exc"])
    chk = df[["unit_id", "real_p_exact"]].merge(st_, on="unit_id")
    S["recomputed_exact_p_max_abs_diff_vs_stored"] = float((chk.real_p_exact - chk.p10_p_exc).abs().max())
    S["criteria"] = tab.set_index("criterion").to_dict("index")
    with open(ROOT / "results/calibration/summary_4C.json", "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab[["criterion", "n_pass_real", "sham_fpr", "implied_fdr", "sensitivity", "specificity", "precision"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in S.items() if k != "criteria"}, indent=1, default=float))


if __name__ == "__main__":
    main()
