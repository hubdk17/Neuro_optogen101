"""
inference.py
============
Statistical machinery for the spike-based optotagging reanalysis.

All functions operate on spike counts / spike times extracted from NWB files
(src/reanalysis/extract_nwb.py). Section numbers refer to
reports/ANALYSIS_REPORT.md, Section 5.

* exact_rate_test            conditional binomial (UMPU) test for a Poisson rate ratio (5.3)
* log_rate_ratio             half-count corrected log rate ratio and its SE (5.3)
* chance_corrected_reliability  rho = (r - p0)/(1 - p0) with SE (5.5)
* fit_latency_model          ML fit of (rho, delta, sigma) of the first-spike model (5.1)
* salt                       port of the Kvitsiani et al. (2013) SALT test
* empirical_null_lfdr        Efron local FDR with an empirical null from sham windows (5.6)
* fit_multinomial_mixture    mixture of multinomials over response windows (5.7)
* ccg_jitter_test            Stark & Abeles (2009) convolution-predictor CCG test (5.10)
* bh                         Benjamini-Hochberg q-values
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.special import logsumexp


# ---------------------------------------------------------------- multiplicity
def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------- rate tests
def exact_rate_test(n_e, n_b, w_e: float, w_b: float):
    """
    Conditional binomial test of H0: rate_e == rate_b, given N = n_e + n_b.
    Returns one-sided p (excitation), one-sided p (suppression), and mid-p
    z-score for excitation (used for the empirical-null analysis).
    """
    n_e = np.asarray(n_e)
    n_b = np.asarray(n_b)
    N = n_e + n_b
    pi0 = w_e / (w_e + w_b)
    p_exc = np.where(N > 0, st.binom.sf(n_e - 1, N, pi0), 1.0)
    p_sup = np.where(N > 0, st.binom.cdf(n_e, N, pi0), 1.0)
    mid = np.where(N > 0, st.binom.sf(n_e, N, pi0) + 0.5 * st.binom.pmf(n_e, N, pi0), 0.5)
    z = st.norm.isf(np.clip(mid, 1e-300, 1 - 1e-16))
    return p_exc, p_sup, z


def log_rate_ratio(n_e, T_e, n_b, T_b):
    """Half-count corrected log rate ratio and approximate SE."""
    n_e = np.asarray(n_e, float)
    n_b = np.asarray(n_b, float)
    lrr = np.log((n_e + 0.5) / T_e) - np.log((n_b + 0.5) / T_b)
    se = np.sqrt(1.0 / (n_e + 0.5) + 1.0 / (n_b + 0.5))
    return lrr, se


def chance_corrected_reliability(k, n, lam0, w):
    """
    rho_hat = (r - p0)/(1 - p0), p0 = 1 - exp(-lam0 * w).
    Returns rho_hat, its SE, p0, and one-sided binomial p for r > p0.
    """
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    r = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    p0 = 1.0 - np.exp(-np.asarray(lam0, float) * w)
    rho = (r - p0) / (1.0 - p0)
    se = np.sqrt(np.clip(r * (1 - r), 1e-12, None) / np.maximum(n, 1)) / (1.0 - p0)
    p = st.binom.sf(k - 1, n, np.clip(p0, 1e-12, 1 - 1e-12))
    return rho, se, p0, p


# ---------------------------------------------------------------- latency model
def _latency_nll(theta, t, n_none, lam0, a, b):
    rho = 1.0 / (1.0 + np.exp(-theta[0]))
    delta = theta[1]
    sigma = np.exp(theta[2])
    Ga = st.norm.cdf((a - delta) / sigma)
    Gt = st.norm.cdf((t - delta) / sigma)
    gt = st.norm.pdf((t - delta) / sigma) / sigma
    Gb = st.norm.cdf((b - delta) / sigma)
    dens = (rho * gt + lam0 * (1.0 - rho * (Gt - Ga))) * np.exp(-lam0 * (t - a))
    s_b = (1.0 - rho * (Gb - Ga)) * np.exp(-lam0 * (b - a))
    ll = np.sum(np.log(np.clip(dens, 1e-300, None))) + n_none * np.log(np.clip(s_b, 1e-300, None))
    return -ll


def fit_latency_model(first_lat_ms: np.ndarray, n_trials: int, lam0_hz: float,
                      a: float = 1.0, b: float = 9.0):
    """
    Maximum-likelihood fit of the first-spike model (report Sec. 5.1):

        S(t) = (1 - rho * [G(t) - G(a)]) * exp(-lam0 (t - a)),  t in [a, b)

    with G the N(delta, sigma^2) CDF. Times in ms, lam0 in Hz.
    Returns dict(rho, delta, sigma, se_rho, se_delta, se_logsigma, nll, converged).
    """
    t = np.asarray(first_lat_ms, float)
    t = t[(t >= a) & (t < b)]
    n_none = n_trials - len(t)
    lam = lam0_hz / 1000.0
    out = dict(rho=np.nan, delta=np.nan, sigma=np.nan, se_rho=np.nan, se_delta=np.nan,
               se_logsigma=np.nan, nll=np.nan, converged=False)
    if len(t) < 3:
        return out
    best = None
    med = float(np.median(t))
    for d0 in (med, float(np.min(t)) + 0.5, (a + b) / 2):
        for s0 in (0.5, 1.5):
            x0 = np.array([0.0, d0, np.log(s0)])
            r = minimize(_latency_nll, x0, args=(t, n_none, lam, a, b), method="L-BFGS-B",
                         bounds=[(-8, 8), (a - 1.0, b), (np.log(0.1), np.log(5.0))])
            if best is None or r.fun < best.fun:
                best = r
    th = best.x
    # observed information by central differences
    h = np.array([1e-3, 1e-3, 1e-3])
    H = np.zeros((3, 3))
    f0 = best.fun
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3); ei[i] = h[i]
            ej = np.zeros(3); ej[j] = h[j]
            fpp = _latency_nll(th + ei + ej, t, n_none, lam, a, b)
            fpm = _latency_nll(th + ei - ej, t, n_none, lam, a, b)
            fmp = _latency_nll(th - ei + ej, t, n_none, lam, a, b)
            fmm = _latency_nll(th - ei - ej, t, n_none, lam, a, b)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    rho = 1 / (1 + np.exp(-th[0]))
    out.update(rho=rho, delta=th[1], sigma=float(np.exp(th[2])),
               se_rho=float(se[0] * rho * (1 - rho)), se_delta=float(se[1]), se_logsigma=float(se[2]),
               nll=float(f0), converged=bool(best.success))
    return out


# ---------------------------------------------------------------- SALT
def _first_bin(rel: np.ndarray, trial: np.ndarray, n_trials: int, t0: float, wn: float, dt: float):
    """Index (1..nb) of first spike bin per trial in [t0, t0+wn); 0 if none."""
    nb = int(round(wn / dt))
    m = (rel >= t0) & (rel < t0 + wn)
    first = np.zeros(n_trials, dtype=int)
    if m.any():
        tr = trial[m]
        bins = np.floor((rel[m] - t0) / dt).astype(int) + 1
        order = np.lexsort((bins, tr))
        tr, bins = tr[order], bins[order]
        keep = np.r_[True, tr[1:] != tr[:-1]]
        first[tr[keep]] = np.clip(bins[keep], 1, nb)
    return np.bincount(first, minlength=nb + 1)[: nb + 1]


def _jsd_dist(P, Q):
    M = 0.5 * (P + Q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl1 = np.nansum(np.where(P > 0, P * np.log2(P / M), 0.0), axis=-1)
        kl2 = np.nansum(np.where(Q > 0, Q * np.log2(Q / M), 0.0), axis=-1)
    return np.sqrt(np.clip(0.5 * (kl1 + kl2), 0, None) * 2)


def salt(rel: np.ndarray, trial: np.ndarray, n_trials: int,
         test_start: float = 0.001, wn: float = 0.008, dt: float = 0.001,
         base_start: float = -0.480, base_stop: float = -0.020):
    """
    Port of salt.m (Kvitsiani et al. 2013): first-spike-latency histograms
    (dt bins, plus a 'no spike' bin) in baseline windows of length wn and in
    the test window; null = pairwise baseline-baseline JS distances; test =
    median test-baseline distance. Returns (p, I).
    """
    starts = np.arange(base_start, base_stop - wn + 1e-9, wn)
    H = [_first_bin(rel, trial, n_trials, s, wn, dt) for s in starts]
    H.append(_first_bin(rel, trial, n_trials, test_start, wn, dt))
    H = np.array(H, float)
    H /= H.sum(axis=1, keepdims=True)
    kn = len(H)
    D = _jsd_dist(H[:, None, :], H[None, :, :])
    iu = np.triu_indices(kn - 1, k=1)
    null = D[: kn - 1, : kn - 1][iu]
    test = np.median(D[: kn - 1, kn - 1])
    p = float(np.mean(null >= test))
    return p, float(test - np.median(null))


# ---------------------------------------------------------------- empirical null / lfdr
def empirical_null_lfdr(z_real: np.ndarray, z_null: np.ndarray, n_null_per_real: float,
                        bins: np.ndarray | None = None, pi0_region=(-np.inf, 0.5)):
    """
    Efron-style local FDR for an excitation statistic z (larger = more evidence).

    f0 is estimated from sham-window statistics (an empirical null), f from the
    real statistics, both by Lindsey's method (Poisson GLM on histogram counts
    with a natural-spline-like polynomial basis). pi0 is the ratio of real to
    rescaled null mass in pi0_region. lfdr is made monotone non-increasing in z
    above the null median. Returns (lfdr, pi0, diagnostics).
    """
    z_real = np.asarray(z_real, float)
    z_null = np.asarray(z_null, float)
    if bins is None:
        lo = min(np.percentile(z_real, 0.1), np.percentile(z_null, 0.1)) - 0.25
        hi = max(z_real.max(), z_null.max()) + 0.25
        bins = np.linspace(lo, hi, 90)
    mids = 0.5 * (bins[1:] + bins[:-1])
    width = np.diff(bins)
    c_real, _ = np.histogram(z_real, bins)
    c_null, _ = np.histogram(z_null, bins)

    def lindsey(counts, deg=7):
        x = (mids - mids.mean()) / mids.std()
        X = np.vander(x, deg + 1, increasing=True)
        beta = np.zeros(deg + 1)
        beta[0] = np.log(max(counts.mean(), 1e-3))
        for _ in range(100):  # IRLS for Poisson GLM
            eta = np.clip(X @ beta, -50, 50)
            mu = np.exp(eta)
            W = mu
            zz = eta + (counts - mu) / np.maximum(mu, 1e-12)
            A = X.T @ (W[:, None] * X) + 1e-6 * np.eye(deg + 1)
            new = np.linalg.solve(A, X.T @ (W * zz))
            if np.max(np.abs(new - beta)) < 1e-8:
                beta = new
                break
            beta = new
        mu = np.exp(np.clip(X @ beta, -50, 50))
        return mu / (mu.sum() * width)

    f = lindsey(c_real)
    f0 = lindsey(c_null)
    in_reg = (mids > pi0_region[0]) & (mids < pi0_region[1])
    pi0 = float(np.clip(c_real[in_reg].sum() / max(c_null[in_reg].sum() / n_null_per_real, 1e-9), 0, 1))
    lf = np.clip(pi0 * f0 / np.maximum(f, 1e-300), 0, 1)
    # monotone non-increasing above the null median
    zmed = np.median(z_null)
    start = np.searchsorted(mids, zmed)
    lf[start:] = np.minimum.accumulate(lf[start:])
    lf[:start] = np.maximum(lf[:start], lf[start] if start < len(lf) else 1.0)
    idx = np.clip(np.searchsorted(bins, z_real) - 1, 0, len(mids) - 1)
    return lf[idx], pi0, dict(mids=mids, f=f, f0=f0, lfdr_curve=lf, c_real=c_real, c_null=c_null)


# ---------------------------------------------------------------- archetypes
def fit_multinomial_mixture(Y: np.ndarray, E: np.ndarray, k: int, n_init: int = 8,
                            max_iter: int = 500, tol: float = 1e-7, seed: int = 0):
    """
    Mixture of multinomials for window counts (report Sec. 5.7).

    Y: (n_units, W) counts with column 0 = baseline window.
    E: (n_units, W) exposures (trials x window length).
    Cluster c has rate multipliers m_c (m_c0 = 1); conditionally on the unit's
    total count the window counts are multinomial with probabilities
    proportional to E_w m_cw, so the unit's baseline rate cancels.
    Returns dict(logm, weights, resp, loglik, bic).
    """
    rng = np.random.default_rng(seed)
    n, W = Y.shape
    Ntot = Y.sum(1)
    keep = Ntot > 0
    best = None
    for init in range(n_init):
        logm = np.zeros((k, W))
        logm[:, 1:] = rng.normal(0, 1.0, size=(k, W - 1))
        w = np.full(k, 1.0 / k)
        prev = -np.inf
        for it in range(max_iter):
            # log-likelihood per unit per cluster (multinomial kernel, constants dropped)
            logp = np.log(E[:, None, :]) + logm[None, :, :]
            logp = logp - logsumexp(logp, axis=2, keepdims=True)
            ll = np.einsum("nw,nkw->nk", Y, logp) + np.log(w)[None, :]
            L = logsumexp(ll, axis=1)
            R = np.exp(ll - L[:, None])
            tot = L[keep].sum()
            # M-step: weights
            w = np.clip(R.mean(0), 1e-12, None)
            w /= w.sum()
            # M-step for m_c: fixed-point (minorization) update per cluster
            for c in range(k):
                r = R[:, c]
                for _ in range(20):
                    pr = E * np.exp(logm[c])[None, :]
                    pr /= pr.sum(1, keepdims=True)
                    num = (r[:, None] * Y).sum(0)
                    den = (r[:, None] * Ntot[:, None] * pr).sum(0)
                    upd = np.log(np.maximum(num, 1e-12)) - np.log(np.maximum(den, 1e-12))
                    logm[c] = logm[c] + upd
                    logm[c] -= logm[c, 0]
                    if np.max(np.abs(upd)) < 1e-6:
                        break
            if tot - prev < tol * abs(tot):
                break
            prev = tot
        if best is None or tot > best["loglik"]:
            best = dict(logm=logm.copy(), weights=w.copy(), resp=R.copy(), loglik=float(tot))
    n_par = k * (W - 1) + (k - 1)
    best["bic"] = -2 * best["loglik"] + n_par * np.log(keep.sum())
    return best


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    from math import comb
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)), int)
    np.add.at(C, (ia, ib), 1)
    s = sum(comb(int(x), 2) for x in C.ravel())
    sa = sum(comb(int(x), 2) for x in C.sum(1))
    sb = sum(comb(int(x), 2) for x in C.sum(0))
    n = comb(len(a), 2)
    exp = sa * sb / n
    mx = 0.5 * (sa + sb)
    return float((s - exp) / (mx - exp)) if mx != exp else 1.0


# ---------------------------------------------------------------- CCG
def hollow_gaussian_predictor(ccg: np.ndarray, bin_s: float, sigma_s: float = 0.010,
                              hollow: float = 0.6) -> np.ndarray:
    """Stark & Abeles (2009): CCG convolved with a partially hollow Gaussian."""
    half = int(np.ceil(3 * sigma_s / bin_s))
    x = np.arange(-half, half + 1) * bin_s
    k = np.exp(-0.5 * (x / sigma_s) ** 2)
    k[half] *= (1 - hollow)
    k /= k.sum()
    pad = np.pad(ccg.astype(float), half, mode="reflect")
    return np.convolve(pad, k, mode="valid")


def ccg_jitter_test(ccg: np.ndarray, bin_s: float, half_s: float,
                    win=(0.0008, 0.0040), sigma_s: float = 0.010):
    """
    Short-latency excitation/inhibition test for b relative to a.
    Bins whose centres fall in win (b after a) are tested against the
    convolution predictor with a continuity-corrected Poisson test,
    Bonferroni-corrected over the tested bins. Returns
    (p_excit, p_inhib, peak_ratio, trough_ratio).
    """
    lam = hollow_gaussian_predictor(ccg, bin_s, sigma_s)
    centres = -half_s + (np.arange(len(ccg)) + 0.5) * bin_s
    sel = (centres >= win[0]) & (centres <= win[1])
    n = ccg[sel].astype(float)
    mu = np.maximum(lam[sel], 1e-9)
    p_hi = st.poisson.sf(n - 1, mu) - 0.5 * st.poisson.pmf(n, mu)
    p_lo = st.poisson.cdf(n, mu) - 0.5 * st.poisson.pmf(n, mu)
    m = sel.sum()
    return (float(min(1, np.min(p_hi) * m)), float(min(1, np.min(p_lo) * m)),
            float(np.max(n / mu)), float(np.min(n / mu)))
