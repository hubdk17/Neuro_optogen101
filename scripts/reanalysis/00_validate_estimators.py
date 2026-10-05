"""
00_validate_estimators.py
=========================
Simulation-based validation of every estimator in src/reanalysis/inference.py
(report Sec. 5.11). Simulated units use the real trial structure: 45 trials,
baseline [-480, -20) ms, evoked window [+1, +9) ms, and spontaneous rates
drawn from a log-normal matched to the cohort (median ~5 Hz).

Outputs results/reanalysis/tables/estimator_validation.csv and prints a summary.
Synthetic data here is used only to check calibration; no result in the
reanalysis is simulated.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import (  # noqa: E402
    exact_rate_test, salt, fit_latency_model, empirical_null_lfdr, bh,
    fit_multinomial_mixture, adjusted_rand, ccg_jitter_test, chance_corrected_reliability,
)

OUT = ROOT / "results/reanalysis/tables"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
N_TR = 45
BASE = (-0.480, -0.020)
EV = (0.001, 0.009)
SHAMS = [(-0.480 + 0.04 * i, -0.472 + 0.04 * i) for i in range(10)]


def sim_unit(lam0, rho=0.0, delta=0.003, sigma=0.0007):
    """Return (rel, trial) for one unit over [-0.5, 0.05] s."""
    rel, tr = [], []
    for k in range(N_TR):
        n = rng.poisson(lam0 * 0.55)
        s = list(rng.uniform(-0.5, 0.05, n))
        if rng.random() < rho:
            s.append(rng.normal(delta, sigma))
        rel += s
        tr += [k] * len(s)
    return np.array(rel), np.array(tr, int)


def counts(rel, lo, hi):
    return int(np.sum((rel >= lo) & (rel < hi)))


def unit_stats(rel, tr):
    ne = counts(rel, *EV)
    nb = counts(rel, *BASE)
    pe, ps, z = exact_rate_test(ne, nb, 0.008 * N_TR, 0.460 * N_TR)
    zs = []
    for a, b in SHAMS:
        nsh = counts(rel, a, b)
        nbb = nb - nsh
        zs.append(exact_rate_test(nsh, nbb, 0.008 * N_TR, 0.452 * N_TR)[2])
    return float(pe), float(z), np.array(zs, float).ravel(), ne, nb


rows = []

# 1. null calibration of exact test and SALT
for lam0 in (1, 5, 20, 60):
    pe_l, ps_l = [], []
    for _ in range(400):
        rel, tr = sim_unit(lam0)
        pe, z, _, _, _ = unit_stats(rel, tr)
        pe_l.append(pe)
        ps_l.append(salt(rel, tr, N_TR)[0])
    rows.append(dict(check="null FPR exact test (p<0.05)", lam0=lam0, value=np.mean(np.array(pe_l) < 0.05)))
    rows.append(dict(check="null FPR SALT port (p<0.05)", lam0=lam0, value=np.mean(np.array(ps_l) < 0.05)))

# 2. power at rho = 0.3
for lam0 in (1, 5, 20, 60):
    hits_e, hits_s = 0, 0
    for _ in range(200):
        rel, tr = sim_unit(lam0, rho=0.3)
        hits_e += unit_stats(rel, tr)[0] < 0.05
        hits_s += salt(rel, tr, N_TR)[0] < 0.05
    rows.append(dict(check="power exact test, rho=0.3", lam0=lam0, value=hits_e / 200))
    rows.append(dict(check="power SALT port, rho=0.3", lam0=lam0, value=hits_s / 200))

# 3. latency model: bias and 95% coverage
for rho in (0.2, 0.5, 0.9):
    for lam0 in (5, 30):
        est = []
        for _ in range(150):
            rel, tr = sim_unit(lam0, rho=rho, delta=0.0035, sigma=0.0008)
            m = (rel >= EV[0]) & (rel < EV[1])
            first = pd.Series(rel[m] * 1000).groupby(tr[m]).min().values
            lam_hat = counts(rel, *BASE) / (0.46 * N_TR)
            f = fit_latency_model(first, N_TR, lam_hat)
            est.append(f)
        e = pd.DataFrame(est).dropna(subset=["rho"])
        cov_rho = np.mean(np.abs(e.rho - rho) <= 1.96 * e.se_rho)
        cov_d = np.mean(np.abs(e.delta - 3.5) <= 1.96 * e.se_delta)
        rows += [
            dict(check=f"latency model rho bias (true {rho})", lam0=lam0, value=e.rho.mean() - rho),
            dict(check=f"latency model rho 95% coverage (true {rho})", lam0=lam0, value=cov_rho),
            dict(check=f"latency model delta bias ms (true 3.5, rho {rho})", lam0=lam0, value=e.delta.median() - 3.5),
            dict(check=f"latency model delta 95% coverage (rho {rho})", lam0=lam0, value=cov_d),
        ]
        # naive estimators for comparison
        rr = []
        for _ in range(150):
            rel, tr = sim_unit(lam0, rho=rho, delta=0.0035, sigma=0.0008)
            m = (rel >= EV[0]) & (rel < EV[1])
            first = pd.Series(rel[m] * 1000).groupby(tr[m]).min()
            rr.append((len(first) / N_TR, first.median()))
        rr = np.array(rr)
        rows += [dict(check=f"naive reliability bias (true {rho})", lam0=lam0, value=rr[:, 0].mean() - rho),
                 dict(check=f"naive median-latency bias ms (rho {rho})", lam0=lam0, value=np.nanmean(rr[:, 1]) - 3.5)]

# 4. lfdr calibration on a simulated cohort (3% responders, rho ~ U(0.1, 0.9))
n_units = 6000
lam_all = np.exp(rng.normal(np.log(5), 1.1, n_units))
truth = rng.random(n_units) < 0.03
z_real, z_null, pvals = [], [], []
for i in range(n_units):
    rho = rng.uniform(0.1, 0.9) if truth[i] else 0.0
    rel, tr = sim_unit(lam_all[i], rho=rho)
    pe, z, zs, _, _ = unit_stats(rel, tr)
    z_real.append(float(np.ravel(z)[0]))
    z_null.append(zs)
    pvals.append(pe)
z_real = np.array(z_real)
z_null = np.concatenate(z_null)
lf, pi0, _ = empirical_null_lfdr(z_real, z_null, n_null_per_real=len(SHAMS))
sel = lf < 0.05
rows += [
    dict(check="lfdr: estimated pi0 (true 0.97)", lam0=np.nan, value=pi0),
    dict(check="lfdr<0.05: realised FDR", lam0=np.nan, value=float(np.mean(~truth[sel])) if sel.any() else np.nan),
    dict(check="lfdr<0.05: mean lfdr of selected (estimated FDR)", lam0=np.nan, value=float(lf[sel].mean()) if sel.any() else np.nan),
    dict(check="lfdr<0.05: sensitivity", lam0=np.nan, value=float(np.mean(sel[truth]))),
    dict(check="BH q<0.05 on exact p: realised FDR", lam0=np.nan,
         value=float(np.mean(~truth[bh(np.array(pvals)) < 0.05]))),
    dict(check="BH q<0.05 on exact p: sensitivity", lam0=np.nan,
         value=float(np.mean((bh(np.array(pvals)) < 0.05)[truth]))),
]
for lo, hi in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
    m = (lf >= lo) & (lf < hi)
    if m.sum() > 20:
        rows.append(dict(check=f"lfdr calibration: P(null | lfdr in [{lo},{hi}))", lam0=np.nan,
                         value=float(np.mean(~truth[m]))))

# 5. multinomial mixture recovers planted archetypes
W = 6
E = np.tile(np.array([0.48, 0.008, 0.009, 0.030, 0.150, 0.300]) * N_TR, (1500, 1))
M = np.array([[1, 1, 1, 1, 1, 1], [1, 6, 1.5, 1, 1, 1], [1, 1, 0.6, 0.3, 0.4, 0.9], [1, 1, 1, 1.6, 1.4, 1.1]])
lab = rng.integers(0, 4, 1500)
lam_u = np.exp(rng.normal(np.log(8), 0.8, 1500))
Y = rng.poisson(lam_u[:, None] * E * M[lab])
bics = {}
fits = {}
for k in range(1, 7):
    fits[k] = fit_multinomial_mixture(Y, E, k, n_init=4, seed=k)
    bics[k] = fits[k]["bic"]
kb = min(bics, key=bics.get)
rows += [dict(check="mixture: BIC-selected k (true 4)", lam0=np.nan, value=kb),
         dict(check="mixture: ARI vs truth at selected k", lam0=np.nan,
              value=adjusted_rand(lab, fits[kb]["resp"].argmax(1)))]

# 6. CCG test FPR on independent trains, power on planted inhibition
fp_e = fp_i = 0
pw = 0
T = 1200.0
for r in range(200):
    a = np.sort(rng.uniform(0, T, rng.poisson(15 * T)))
    b = np.sort(rng.uniform(0, T, rng.poisson(10 * T)))
    from src.reanalysis.extract_nwb import _ccg_counts, CCG_BIN, CCG_HALF  # noqa: E402
    c = _ccg_counts(a, b)
    pe, pi, _, _ = ccg_jitter_test(c, CCG_BIN, CCG_HALF)
    fp_e += pe < 0.01
    fp_i += pi < 0.01
    # planted inhibition: delete b spikes 1-4 ms after a spikes with prob 0.5
    lo = np.searchsorted(a, b - 0.004)
    hi = np.searchsorted(a, b - 0.001)
    hit = (hi > lo) & (rng.random(len(b)) < 0.5)
    c2 = _ccg_counts(a, b[~hit])
    pw += ccg_jitter_test(c2, CCG_BIN, CCG_HALF)[1] < 0.01
rows += [dict(check="CCG test FPR excitation (p<0.01)", lam0=np.nan, value=fp_e / 200),
         dict(check="CCG test FPR inhibition (p<0.01)", lam0=np.nan, value=fp_i / 200),
         dict(check="CCG test power, 50% inhibition 1-4 ms", lam0=np.nan, value=pw / 200)]

df = pd.DataFrame(rows)
df.to_csv(OUT / "estimator_validation.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(df.to_string(index=False))
