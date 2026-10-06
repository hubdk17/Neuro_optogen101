"""
00_regression_baseline.py
=========================
Regression check of the committed spike-based reanalysis (Layer B) before any
new work. Every quantity is recomputed from
results/reanalysis/tables/unit_results.parquet and ccg_pairs.csv and compared
with the expected values agreed in the calibration brief. Exits non-zero if
any check fails.

Output: results/calibration/tables/regression_baseline.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import bh  # noqa: E402

T = ROOT / "results/reanalysis/tables"
OUT = ROOT / "results/calibration/tables"
OUT.mkdir(parents=True, exist_ok=True)
CRES = ["Pvalb-IRES-Cre", "Sst-IRES-Cre", "Vip-IRES-Cre"]
rows = []


def check(name, value, expected, tol, note=""):
    ok = abs(value - expected) <= tol
    rows.append(dict(quantity=name, computed=value, expected=expected, tolerance=tol, ok=bool(ok), note=note))


def main():
    d = pd.read_parquet(T / "unit_results.parquet")
    check("units", len(d), 19005, 0)
    check("sessions", d.session_id.nunique(), 28, 0)
    animals = d.groupby("cre_line").specimen_id.nunique()
    for c, n in zip(CRES, (8, 12, 8)):
        check(f"animals {c}", animals[c], n, 0)

    # exact conditional test, pi0 = w_e/(w_e + w_b) with w_b = [-480, -20) ms = 460 ms
    pi0 = 0.008 / (0.008 + 0.460)
    N = d.p10_ne + d.p10_nb
    p = np.where(N > 0, st.binom.sf(d.p10_ne - 1, N, pi0), 1.0)
    check("max |recomputed p - stored p| (pi0 = 0.008/0.468)", float(np.max(np.abs(p - d.p10_p_exc))), 0.0, 1e-14)
    pi0_wrong = 0.008 / (0.008 + 0.480)
    p_wrong = np.where(N > 0, st.binom.sf(d.p10_ne - 1, N, pi0_wrong), 1.0)
    rows.append(dict(quantity="max |p - stored| with WRONG pi0 = 0.008/0.488 (480-ms baseline)",
                     computed=float(np.max(np.abs(p_wrong - d.p10_p_exc))), expected=np.nan, tolerance=np.nan,
                     ok=True, note="informational: the 480-ms constant is wrong; baseline is 460 ms"))
    check("exact test BH q<0.05", int((bh(p) < 0.05).sum()), 396, 0)
    check("exact test Bonferroni", int((p < 0.05 / len(d)).sum()), 304, 0)
    check("lfdr<0.05", int((d.lfdr < 0.05).sum()), 375, 0)
    check("onset artifacts within lfdr<0.05", int(((d.lfdr < 0.05) & d.onset_artifact).sum()), 25, 0)
    check("driven", int(d.driven.sum()), 350, 0)

    dr, nl = d[d.driven], d[d.lfdr > 0.5]
    check("5-ms replication, driven (p5<0.05)", float((dr.p5_p_exc < 0.05).mean()), 0.92, 0.005)
    check("5-ms replication, null (lfdr>0.5)", float((nl.p5_p_exc < 0.05).mean()), 0.035, 0.001)
    check("Spearman rho10 vs rho5 (driven)", float(st.spearmanr(dr.p10_rho, dr.p5_rho).statistic), 0.84, 0.005)

    dd = dr.copy()
    dd["dose"] = dd.p10_high_rho - dd.p10_low_rho
    a = dd.groupby("specimen_id").dose.mean().dropna()
    t = st.ttest_1samp(a, 0)
    check("dose: animals", len(a), 24, 0)
    check("dose: mean high-low rho", float(a.mean()), 0.41, 0.005)
    check("dose: t(23)", float(t.statistic), 11.5, 0.05)
    check("dose: p", float(t.pvalue), 5.2e-11, 0.1e-11)

    tr = dr[dr.train_n > 0]
    g = tr.groupby("specimen_id")[["train_k1", "train_k10"]].sum()
    a = np.log((g.train_k10 + 0.5) / (g.train_k1 + 0.5))
    t = st.ttest_1samp(a, 0)
    check("train log-ratio, pooled counts per animal", float(a.mean()), -0.05, 0.005,
          "definition: log((sum k10 + 0.5)/(sum k1 + 0.5)) per animal")
    check("train t(23), pooled", float(t.statistic), -0.70, 0.01)
    check("train p, pooled", float(t.pvalue), 0.49, 0.005)
    u = np.log((tr.train_k10 + 0.5) / (tr.train_k1 + 0.5)).groupby(tr.specimen_id).mean()
    t2 = st.ttest_1samp(u, 0)
    rows.append(dict(quantity="train log-ratio, mean of unit log-ratios per animal (stored-table definition)",
                     computed=float(u.mean()), expected=np.nan, tolerance=np.nan, ok=True,
                     note=f"t(23)={t2.statistic:.3f}, p={t2.pvalue:.3f}; null under both definitions"))

    for c, exp in zip(CRES, (17.8, 3.97, 0.81)):
        s = d[d.cre_line == c]
        tab = pd.crosstab(s.driven, s.waveform_duration < 0.4)
        orr = st.contingency.odds_ratio(tab.values, kind="sample").statistic
        check(f"narrow-waveform OR {c}", float(orr), exp, 0.01)

    y = d.groupby(["specimen_id", "cre_line"]).driven.mean().reset_index()
    for c, m, se in zip(CRES, (0.0394, 0.0169, 0.0046), (0.0141, 0.0054, 0.0020)):
        v = y[y.cre_line == c].driven
        check(f"yield mean {c}", float(v.mean()), m, 0.0001)
        check(f"yield SEM {c}", float(v.sem()), se, 0.0001)
    kw = st.kruskal(*[y[y.cre_line == c].driven for c in CRES])
    check("yield Kruskal-Wallis p", float(kw.pvalue), 0.10, 0.01, "fragile: treat Cre yield difference as a trend")

    cg = pd.read_csv(T / "ccg_pairs.csv")
    cg = cg[cg.unit_a.isin(set(dr.unit_id))]
    for c, inh, anti in zip(CRES, (0.0774, 0.0390, 0.0105), (0.0254, 0.0168, 0.0105)):
        s = cg[cg.cre_line == c]
        check(f"CCG inhibition causal {c}", float((s.p_inh < 0.01).mean()), inh, 0.0005)
        check(f"CCG inhibition anticausal {c}", float((s.p_inh_anticausal < 0.01).mean()), anti, 0.0005)

    dl = dr.dropna(subset=["p10_fit_delta"])
    n_dl = int(((dl.p10_fit_delta < 5) & (dl.p10_fit_sigma < 1.5)).sum())
    check("direct-like count (delta<5, sigma<1.5)", n_dl, 85, 0)
    check("direct-like fraction of driven", n_dl / len(dr), 0.243, 0.001)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regression_baseline.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_colwidth", 70, "display.max_rows", 100):
        print(df.to_string(index=False))
    if not df.ok.all():
        print("\nREGRESSION FAILURES:\n", df[~df.ok].to_string(index=False))
        sys.exit(1)
    print("\nAll regression checks passed.")


if __name__ == "__main__":
    main()
