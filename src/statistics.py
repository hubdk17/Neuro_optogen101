"""
statistics.py
=============
Statistical testing, permutation procedures, effect size calculations,
and session-level bootstrap confidence intervals for optotagging analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import numpy as np
import scipy.stats as stats
import pandas as pd

logger = logging.getLogger(__name__)


def paired_permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Two-sided paired permutation test on difference (a - b).
    
    Parameters:
    -----------
    a : 1D array of test values (e.g. evoked counts)
    b : 1D array of reference values (e.g. scaled baseline counts)
    n_permutations : number of sign-flip permutations
    random_state : random seed
    
    Returns:
    --------
    diff_mean : observed mean difference mean(a) - mean(b)
    p_value : permutation p-value
    """
    diffs = a - b
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
        
    obs_diff = np.mean(diffs)
    if np.all(diffs == 0):
        return 0.0, 1.0
        
    rng = np.random.default_rng(random_state)
    # Randomly assign +1 or -1 to each pair difference
    # (n_permutations, n)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    perm_means = np.mean(signs * diffs, axis=1)
    
    # Two-sided p-value: proportion of permuted differences as or more extreme than observed
    p_value = (np.sum(np.abs(perm_means) >= np.abs(obs_diff)) + 1.0) / (n_permutations + 1.0)
    return float(obs_diff), float(p_value)


def paired_wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Nonparametric paired Wilcoxon signed-rank test.
    Fallback for large trial counts or fast vectorization.
    """
    diffs = a - b
    if len(diffs) == 0 or np.all(diffs == 0):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(np.mean(diffs)), float(res.pvalue)
    except Exception:
        return float(np.mean(diffs)), 1.0


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Cohen's d for paired samples:
    d = mean(a - b) / sd(a - b)
    """
    diffs = a - b
    sd = np.std(diffs, ddof=1)
    if sd < 1e-9:
        return 0.0
    return float(np.mean(diffs) / sd)


def rank_biserial_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate matched-pairs rank biserial correlation (r) as a nonparametric effect size.
    Ranges from -1.0 to +1.0.
    """
    diffs = (a - b).astype(float)
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(non_zero))
    w_pos = np.sum(ranks[non_zero > 0])
    w_neg = np.sum(ranks[non_zero < 0])
    total = w_pos + w_neg
    if total < 1e-9:
        return 0.0
    return float((w_pos - w_neg) / total)


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable[[np.ndarray], float] = np.median,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute empirical bootstrap confidence interval for arbitrary 1D data.
    """
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return np.nan, np.nan
    if len(valid) == 1:
        return float(valid[0]), float(valid[0])
        
    rng = np.random.default_rng(random_state)
    boot_indices = rng.integers(0, len(valid), size=(n_bootstraps, len(valid)))
    boot_samples = valid[boot_indices]
    boot_stats = np.apply_along_axis(stat_func, 1, boot_samples)
    
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(boot_stats, alpha * 100)
    high = np.percentile(boot_stats, (1.0 - alpha) * 100)
    return float(low), float(high)


def session_level_cluster_bootstrap(
    df: pd.DataFrame,
    stat_func: Callable[[pd.DataFrame], float],
    group_col: str = "session_id",
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Hierarchical cluster bootstrap where SESSION is the unit of replication,
    preventing pseudo-replication across co-recorded units.
    """
    sessions = df[group_col].unique()
    n_sessions = len(sessions)
    if n_sessions == 0:
        return np.nan, np.nan
    if n_sessions == 1:
        val = stat_func(df)
        return val, val
        
    rng = np.random.default_rng(random_state)
    boot_stats = []
    
    for _ in range(n_bootstraps):
        resampled_sessions = rng.choice(sessions, size=n_sessions, replace=True)
        # Concatenate units from resampled sessions
        parts = [df[df[group_col] == s] for s in resampled_sessions]
        boot_df = pd.concat(parts, ignore_index=True)
        boot_stats.append(stat_func(boot_df))
        
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(boot_stats, alpha * 100)
    high = np.percentile(boot_stats, (1.0 - alpha) * 100)
    return float(low), float(high)
