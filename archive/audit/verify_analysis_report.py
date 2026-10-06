"""
verify_analysis_report.py
=========================
AUDIT OF ARCHIVED (non-data-derived) MATERIAL. Reproduces every number quoted in
reports/ANALYSIS_REPORT.md about the superseded analysis, reading the archived
unit table. Kept with the archive for provenance; nothing in src/ or scripts/
uses it. Run from the repository root:

    python archive/audit/verify_analysis_report.py [--sims 300]

Sections:
 1. Multiple-comparison control of the per-unit permutation p-values
 2. Exact conditional (binomial) test for the Poisson rate ratio
 3. Corrected Poisson null for the sparse-firing tier
 4. Latency-threshold specificity inside the [1, 9] ms evoked window
 5. ICC: naive vs one-way ANOVA estimator, design effect
 6. Chance-corrected reliability and binomial threshold instability
 7. Suppression detectability (minimum detectable effect)
 8. Adaptation-index pathologies
 9. Cluster/Cre cross-tabulation of the GMM archetypes
10. Spatial "conduction velocity" arithmetic
11. Null calibration of archive/src/responsiveness_methods.py (SALT-/ZETA-like tests)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from archive.src.responsiveness_methods import compute_salt, compute_zeta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "archive/results/neuroscience_study/tables/revised/master_neuroscience_phenotypes_revised.parquet"
W_EVOKED_S = 0.008   # [+1, +9) ms
W_BASE_S = 0.015     # [-20, -5) ms


def bh_reject(p, alpha):
    """Benjamini-Hochberg step-up procedure."""
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    return rej


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main(n_sims):
    df = pd.read_parquet(TABLE)
    N = len(df)
    n = df["n_trials"].values
    print(f"Units: {N}  sessions: {df.session_id.nunique()}  "
          f"trials/unit median={np.median(n):.0f} (min {n.min()}, max {n.max()})")

    section("1. Multiple comparisons on stored permutation p-values (B = 1000)")
    p = df["p_value"].values
    print(f"p < 0.05: {(p < 0.05).sum()}   expected false positives if all null: {0.05 * N:.0f}")
    print(f"p-value floor: {p.min():.4f} (= 1/1001); units at floor: {(p <= 0.001).sum()}")
    print(f"BH survivors needed at floor: k >= {int(np.ceil(p.min() * N / 0.05))}")
    for a in (0.05, 0.10):
        print(f"BH q < {a:.2f}: {bh_reject(p, a).sum()} units")
    h = df["method_heuristic_direct"].values == 1
    print(f"Heuristic 'direct' units surviving BH q<0.05: {(h & bh_reject(p, 0.05)).sum()} / {h.sum()}")

    section("2. Exact conditional binomial test (UMPU test for a Poisson rate ratio)")
    Ne = np.round(df["evoked_rate"].values * W_EVOKED_S * n).astype(int)
    Nb = np.round(df["baseline_rate"].values * W_BASE_S * n).astype(int)
    pi0 = W_EVOKED_S / (W_EVOKED_S + W_BASE_S)
    p_exact = np.where(Ne + Nb > 0, st.binom.sf(Ne - 1, Ne + Nb, pi0), 1.0)
    print(f"pi0 = w_e/(w_e+w_b) = {pi0:.4f}")
    print(f"one-sided exact p < 0.05: {(p_exact < 0.05).sum()}")
    print(f"BH q < 0.05: {bh_reject(p_exact, 0.05).sum()}   Bonferroni: {(p_exact < 0.05 / N).sum()}")
    print(f"exact p < 1e-3: {(p_exact < 1e-3).sum()}   exact p < 1e-6: {(p_exact < 1e-6).sum()}")
    print(f"heuristic 'direct' units with exact BH q<0.05: {(h & bh_reject(p_exact, 0.05)).sum()} / {h.sum()}")

    section("3. Poisson null for the sparse (<1 Hz) tier")
    lam = df["baseline_rate"].values
    has_lat = df["median_latency_ms"].notna().values
    tier = lam < 1.0
    lam_t, n_t = lam[tier], n[tier]
    repo = tier.sum() * (1 - np.exp(-lam_t.mean() * 0.75))
    unitwise = np.sum(1 - np.exp(-lam_t * W_EVOKED_S * n_t))
    print(f"N tier = {tier.sum()}, observed units with >=1 evoked-window spike = {has_lat[tier].sum()}")
    print(f"repo expectation (tau = 75 x 10 ms, mean lambda): {repo:.1f}")
    print(f"corrected expectation (tau_i = n_i x 8 ms, unit-wise): {unitwise:.1f}")
    print(f"observed / corrected expected = {has_lat[tier].sum() / unitwise:.2f}")
    print(f"fraction of tier with zero baseline spikes (lambda_hat = 0): {(lam_t == 0).mean():.3f}")
    tau_b = W_BASE_S * np.median(n)
    print(f"95% upper bound on lambda given 0 baseline spikes in {tau_b:.3f} s: {-np.log(0.05) / tau_b:.2f} Hz")
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, np.inf)]:
        m = (lam >= lo) & (lam < hi)
        e = np.sum(1 - np.exp(-lam[m] * W_EVOKED_S * n[m]))
        print(f"  tier [{lo},{hi}) Hz: N={m.sum():5d} obs={has_lat[m].sum():5d} "
              f"exp={e:8.1f} obs/exp={has_lat[m].sum() / e:5.2f}")

    section("4. Latency threshold (< 8 ms) inside the [1, 9) ms evoked window")
    print(f"P(first spontaneous spike < 8 ms | spike in window) ~ 7/8 = {7 / 8:.3f} (repo uses 0.80)")
    for k in (1, 2, 3, 5, 10, 20):
        m = (k + 1) // 2
        print(f"  k={k:2d} spontaneous first-spikes: P(median latency < 8 ms) = {st.binom.sf(m - 1, k, 7 / 8):.4f}")

    section("5. Variance decomposition: naive ratio vs ANOVA ICC(1); design effect")
    for col in ("baseline_rate", "evoked_rate", "modulation_ratio"):
        g = df.groupby("session_id")[col]
        ni, means, k = g.size().values, g.mean().values, g.ngroups
        Nt, gm = ni.sum(), df[col].mean()
        msb = np.sum(ni * (means - gm) ** 2) / (k - 1)
        msw = np.sum(g.var(ddof=1).values * (ni - 1)) / (Nt - k)
        n0 = (Nt - np.sum(ni ** 2) / Nt) / (k - 1)
        s2b = max(0.0, (msb - msw) / n0)
        icc = s2b / (s2b + msw)
        naive = np.var(means, ddof=1) / (np.var(means, ddof=1) + g.var(ddof=1).mean())
        F = msb / msw
        deff = 1 + (Nt / k - 1) * icc
        print(f"  {col:17s} naive={naive:.4f} ICC(1)={icc:.4f} F({k - 1},{Nt - k})={F:.1f} "
              f"p={st.f.sf(F, k - 1, Nt - k):.1e} design effect={deff:.2f} "
              f"n_eff/animal={(Nt / k) / deff:.0f}")
    print("  modulation ratio evoked/(baseline+1) under exact null (evoked == baseline):")
    for l in (0.2, 1, 5, 20, 50):
        print(f"    lambda={l:5.1f} Hz -> MR_null={l / (l + 1):.3f}")

    section("6. Reliability: chance level and binomial threshold instability")
    r0 = 1 - np.exp(-lam * W_EVOKED_S)
    print(f"units whose chance reliability p0 = 1-exp(-lambda*8ms) >= 0.10: {(r0 >= 0.10).sum()}, >= 0.30: {(r0 >= 0.30).sum()}")
    for nn in (45, 75):
        lo, hi = st.binomtest(int(round(0.3 * nn)), nn).proportion_ci(method="wilson")
        print(f"  n={nn}: observed reliability 0.30 -> Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    for rho in (0.25, 0.30, 0.35, 0.40):
        flip = st.binom.cdf(np.ceil(0.30 * 45) - 1, 45, rho)
        print(f"  true rho={rho:.2f}, n=45: P(observed rel < 0.30) = {flip:.3f}")
    k_rel = np.round(df["trial_reliability"].values * n).astype(int)
    p_rel = st.binom.sf(k_rel - 1, n, np.clip(r0, 1e-12, 1))
    print(f"units with reliability > chance (one-sided binomial, BH q<0.05): {bh_reject(p_rel, 0.05).sum()}")

    section("7. Suppression detectability in an 8-ms window")
    for l in (2, 5, 10, 20, 40):
        mu = l * W_EVOKED_S * 45
        print(f"  lambda={l:2d} Hz, 45 trials: expected baseline count={mu:5.2f}; "
              f"P(0 spikes | no change)={np.exp(-mu):.3f} -> complete silencing "
              f"{'detectable' if np.exp(-mu) < 0.05 else 'NOT detectable'} at alpha=0.05")

    section("8. Adaptation index stored by src/feature_extraction.py")
    ai = df["adaptation_index"]
    print(ai.describe().to_string())
    print(f"units with AI < -1: {(ai < -1).sum()}")

    section("9. GMM archetype x Cre line")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(pd.crosstab(df["unsupervised_response_archetype"], df["cre_line"]))

    section("10. Spatial gradient arithmetic")
    print(f"slope 1.1 us/um -> apparent velocity = 1/1.1 = {1 / 1.1:.2f} m/s (manuscript states 0.07 m/s)")
    print(f"0.07 m/s would require {1 / 0.07:.1f} us/um")

    section("11. Null calibration of SALT-/ZETA-like functions (no light response)")
    rng = np.random.default_rng(0)

    def null_first_lats(lam_hz, ntr):
        out = []
        for _ in range(ntr):
            k = rng.poisson(lam_hz * W_EVOKED_S)
            if k > 0:
                out.append(rng.uniform(1.0, 9.0, k).min())
        return np.array(out)

    for lam_hz in (2, 10, 30, 60):
        for ntr in (45, 75):
            fs = fz = 0
            for r in range(n_sims):
                L = null_first_lats(lam_hz, ntr)
                fs += compute_salt(L, lam_hz, lam_hz, stim_win_ms=10.0, random_state=r)["salt_significant"]
                fz += compute_zeta(L, lam_hz, lam_hz, len(L) / ntr, 1.0, 0.0, stim_win_ms=10.0)["zeta_significant"]
            print(f"  lambda={lam_hz:2d} Hz, trials={ntr}: SALT-like FPR={fs / n_sims:.3f}  "
                  f"ZETA-like FPR={fz / n_sims:.3f}  (nominal 0.05)")
    print("  analytic: first latencies on [1,9] vs uniform[0,10] give sup|F_n-F_0| -> 0.1,"
          " z ~ sqrt(n)/3 > 1.96 once n > 35 first-spikes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    main(ap.parse_args().sims)
