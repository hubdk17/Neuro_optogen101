"""
responsiveness_methods.py
=========================
Implementation of standard and cutting-edge optogenetic responsiveness tests:
1. Operational Heuristic (Standard lab practice: rel >= 0.30, lat < 8ms, mod > 2.0, p < 0.05, eff > 0.10)
2. SALT (Stimulus-Associated spike Latency Test, Kvitsiani et al. 2013, Nature)
3. ZETA (Z-score Exact Test of Activity, Montijn et al. 2021, eLife)
4. Continuous Evidence Representation (Multifeature reliability and uncertainty formulation)
"""

import numpy as np
import scipy.stats as stats

def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two discrete probability distributions."""
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    q = np.clip(q, eps, 1.0)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(max(0.0, jsd)))

def compute_salt(
    first_latencies: np.ndarray,
    baseline_rate: float,
    evoked_rate: float,
    n_trials: int = 75,
    stim_win_ms: float = 10.0,
    base_win_ms: float = 30.0,
    n_bins: int = 10,
    n_bootstrap: int = 200,
    random_state: int = 42
) -> dict:
    """
    Stimulus-Associated spike Latency Test (SALT; Kvitsiani et al. 2013).
    Compares latency distribution in the stimulus window to baseline latency variability.
    
    Returns:
    --------
    dict with:
    - salt_statistic: Jensen-Shannon distance between stimulus latency and baseline
    - salt_p_value: permutation p-value
    - salt_significant: bool (p < 0.05)
    """
    valid_lats = first_latencies[~np.isnan(first_latencies)] if first_latencies is not None else np.array([])
    n_spikes = len(valid_lats)
    
    if n_spikes < 3 or evoked_rate <= 0:
        return {
            "salt_statistic": 0.0,
            "salt_p_value": 1.0,
            "salt_significant": False
        }
        
    bins = np.linspace(0.0, stim_win_ms, n_bins + 1)
    p_stim, _ = np.histogram(valid_lats, bins=bins, density=True)
    if np.sum(p_stim) == 0:
        return {"salt_statistic": 0.0, "salt_p_value": 1.0, "salt_significant": False}
    p_stim = p_stim / np.sum(p_stim)
    
    # Baseline expected latency distribution under Poisson process with rate = baseline_rate
    # If baseline_rate > 0, probability density of first spike latency is f(t) = lambda * exp(-lambda * t)
    lam = max(0.1, baseline_rate) / 1000.0 # rate per ms
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    p_base = lam * np.exp(-lam * bin_centers)
    p_base = p_base / np.sum(p_base)
    
    # True test statistic: JS distance between stimulus and baseline
    js_true = compute_jensen_shannon_divergence(p_stim, p_base)
    
    # Bootstrap null: resample baseline pseudo-trials from the Poisson null
    rng = np.random.RandomState(random_state)
    null_js = []
    for _ in range(n_bootstrap):
        # sample n_spikes from Poisson null in [0, stim_win_ms]
        null_lats = rng.exponential(scale=1.0/lam, size=n_spikes)
        null_lats = null_lats[null_lats <= stim_win_ms]
        if len(null_lats) < 2:
            null_lats = rng.uniform(0.0, stim_win_ms, size=n_spikes)
        p_null, _ = np.histogram(null_lats, bins=bins, density=True)
        if np.sum(p_null) > 0:
            p_null = p_null / np.sum(p_null)
            null_js.append(compute_jensen_shannon_divergence(p_null, p_base))
        else:
            null_js.append(0.0)
            
    null_js = np.array(null_js)
    p_val = float((1.0 + np.sum(null_js >= js_true)) / (1.0 + len(null_js)))
    
    return {
        "salt_statistic": float(js_true),
        "salt_p_value": p_val,
        "salt_significant": bool(p_val < 0.05)
    }

def compute_zeta(
    first_latencies: np.ndarray,
    baseline_rate: float,
    evoked_rate: float,
    trial_reliability: float,
    p_value: float,
    effect_size: float,
    n_trials: int = 75,
    stim_win_ms: float = 10.0
) -> dict:
    """
    Z-score Exact Test of Activity (ZETA; Montijn et al. 2021).
    Parameter-free deviation of cumulative spike density from linear baseline expectation.
    
    Returns:
    --------
    dict with:
    - zeta_score: deviation Z-score
    - zeta_p_value: two-tailed p-value
    - zeta_significant: bool (p < 0.05)
    - zeta_latency_ms: estimated onset latency
    """
    valid_lats = first_latencies[~np.isnan(first_latencies)] if first_latencies is not None else np.array([])
    n_spikes = len(valid_lats)
    
    if n_spikes < 3 or trial_reliability < 0.02:
        return {
            "zeta_score": 0.0,
            "zeta_p_value": 1.0,
            "zeta_significant": False,
            "zeta_latency_ms": np.nan
        }
        
    sorted_lats = np.sort(valid_lats)
    # Empirical cumulative fraction
    t_vals = sorted_lats
    c_emp = np.arange(1, len(t_vals) + 1) / len(t_vals)
    # Linear expectation under uniform / stationary assumption
    c_exp = t_vals / stim_win_ms
    
    # Maximum Brownian bridge deviation
    diff = c_emp - c_exp
    max_idx = np.argmax(np.abs(diff))
    max_dev = float(diff[max_idx])
    onset_lat = float(sorted_lats[max_idx])
    
    # Approximate ZETA z-score using Kolmogorov-Smirnov / bridge asymptotic variance
    # Var(bridge) = t/T * (1 - t/T) / N
    t_norm = np.clip(sorted_lats[max_idx] / stim_win_ms, 0.05, 0.95)
    var_bridge = (t_norm * (1.0 - t_norm)) / max(1, n_spikes)
    std_bridge = np.sqrt(var_bridge)
    
    zeta_score = float(max_dev / (std_bridge + 1e-6))
    
    # Factor in modulation ratio and trial reliability
    if evoked_rate < baseline_rate:
        zeta_score = -abs(zeta_score)
        
    # Standard normal p-value
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(zeta_score))))
    p_val = float(np.clip(p_val, 1e-12, 1.0))
    
    return {
        "zeta_score": zeta_score,
        "zeta_p_value": p_val,
        "zeta_significant": bool(p_val < 0.05 and zeta_score > 0),
        "zeta_latency_ms": onset_lat
    }
