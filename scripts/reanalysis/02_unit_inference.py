"""
02_unit_inference.py
====================
Per-unit inference from the extracted spike data (data/derived/session_*.pkl.gz).

For every unit and for 10-ms and 5-ms single pulses:
  * spontaneous rate from the pre-stimulus baseline [-480, -20) ms of the same
    trials (46x the exposure of the original 15-ms baseline) and from the
    session's spontaneous epochs;
  * exact conditional test (excitation / suppression) for [+1, +9) ms, mid-p z,
    and the same statistic in 10 sham windows (empirical null);
  * chance-corrected reliability, per light level;
  * ML fit of the first-spike model (rho, delta, sigma);
  * SALT (port of Kvitsiani et al. 2013); reference ZETA comes from extraction.
Window counts over 0-500 ms after 10-ms pulses (archetypes, suppression) and
per-pulse responses to the 10-Hz trains are tabulated as well.

Output: results/reanalysis/tables/unit_inference.parquet
"""

import gzip
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.reanalysis.inference import (  # noqa: E402
    exact_rate_test, chance_corrected_reliability, fit_latency_model, salt, log_rate_ratio,
)

BASE = (-0.480, -0.020)
EV = (0.001, 0.009)
SHAMS = [(-0.480 + 0.04 * i, -0.472 + 0.04 * i) for i in range(10)]
WINDOWS = {"w1": (0.001, 0.009), "w2": (0.011, 0.020), "w3": (0.020, 0.050),
           "w4": (0.050, 0.200), "w5": (0.200, 0.500)}
W_E = EV[1] - EV[0]
W_B = BASE[1] - BASE[0]
FIT_P = 0.01  # latency model is fitted only where a response is detectable


def cnt(rel, lo, hi):
    return int(np.count_nonzero((rel >= lo) & (rel < hi)))


def pulse_block(rel, ep, trials, prefix, fit_model=True):
    """Statistics for one pulse type given the unit's aligned spikes."""
    n = len(trials)
    out = {f"{prefix}_n": n}
    if n == 0:
        return out, None
    m = np.isin(ep, trials)
    r, e = rel[m], ep[m]
    ti = np.searchsorted(np.sort(trials), e)
    nb = cnt(r, *BASE)
    ne = cnt(r, *EV)
    lam0 = nb / (n * W_B)
    pe, ps, z = exact_rate_test(ne, nb, n * W_E, n * W_B)
    lrr, se = log_rate_ratio(ne, n * W_E, nb, n * W_B)
    em = (r >= EV[0]) & (r < EV[1])
    first = pd.Series(r[em] * 1000.0).groupby(ti[em]).min()
    k = len(first)
    rho, rho_se, p0, p_rel = chance_corrected_reliability(k, n, lam0, W_E)
    out.update({
        f"{prefix}_nb": nb, f"{prefix}_ne": ne, f"{prefix}_lam0_hz": lam0,
        f"{prefix}_p_exc": float(pe), f"{prefix}_p_sup": float(ps), f"{prefix}_z": float(np.ravel(z)[0]),
        f"{prefix}_lrr": float(lrr), f"{prefix}_lrr_se": float(se),
        f"{prefix}_k": k, f"{prefix}_rel_raw": k / n, f"{prefix}_rho": float(rho), f"{prefix}_rho_se": float(rho_se),
        f"{prefix}_p0": float(p0), f"{prefix}_p_rel": float(p_rel),
        f"{prefix}_median_first_ms": float(first.median()) if k else np.nan,
    })
    zs = []
    for a, b in SHAMS:
        nsh = cnt(r, a, b)
        zs.append(float(np.ravel(exact_rate_test(nsh, nb - nsh, n * W_E, n * (W_B - W_E))[2])[0]))
    out[f"{prefix}_sham_z"] = zs
    sp, si = salt(r, ti, n)
    if fit_model and float(pe) < FIT_P:
        f = fit_latency_model(first.values, n, lam0)
        out.update({f"{prefix}_fit_{k_}": v for k_, v in f.items()})
    out[f"{prefix}_salt_p"] = sp
    out[f"{prefix}_salt_I"] = si
    return out, (r, ti, first)


def zeta_from_aligned(rel, ep, onsets, trials, shift=0.0, start=0.001, dur=0.050, n_res=250):
    """Reference ZETA (zetapy) on absolute spike times rebuilt from the aligned record."""
    import logging
    import warnings
    from zetapy import zetatest
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    s = np.sort(onsets[ep] + rel)
    s = np.unique(s)
    ev = np.sort(onsets[trials]) + start + shift
    try:
        out = zetatest(s, ev, max_duration=dur, resampling_number=n_res)
        d = out[1]
        return float(d["zeta_p_value"]), float(np.sign(d.get("zeta_deviation", np.nan)))
    except Exception:
        return 1.0, np.nan


def process(path):
    with gzip.open(path, "rb") as fh:
        R = pickle.load(fh)
    units, opto = R["units"], R["opto"]
    sid = R["session_id"]
    zeta = R["zeta"].set_index("unit_id")
    spont = R["spont"].set_index("unit_id")
    rows = []
    for _, u in units.iterrows():
        uid = int(u["unit_id"])
        rel, ep = R["aligned"][uid]
        rel = rel.astype(np.float64)
        valid = R["valid_epochs"][u["probe_name"]]
        row = dict(session_id=sid, unit_id=uid)
        row.update({c: u[c] for c in ["probe_name", "probe_id", "structure", "probe_vertical_position",
                                      "ccf_ap", "ccf_dv", "ccf_lr", "cortical_depth_um", "waveform_duration",
                                      "waveform_halfwidth", "PT_ratio", "firing_rate", "snr"]})
        row["lam_spont_hz"] = spont.loc[uid, "spont_count"] / max(spont.loc[uid, "spont_duration_s"], 1e-9)
        row["spont_duration_s"] = spont.loc[uid, "spont_duration_s"]
        for c in ["zeta_p", "zeta_z", "zeta_sign", "zeta_latency_s", "zeta_sham_p", "zeta_sham_sign"]:
            row[c] = zeta.loc[uid, c]

        p10 = opto.index[(opto["kind"] == "pulse10").values & valid].values
        p5 = opto.index[(opto["kind"] == "pulse5").values & valid].values
        b10, d10 = pulse_block(rel, ep, p10, "p10")
        row.update(b10)
        if len(p10):
            on = opto["start_time"].values
            row["zeta50_p"], row["zeta50_sign"] = zeta_from_aligned(rel, ep, on, p10)
            row["zeta50_sham_p"], row["zeta50_sham_sign"] = zeta_from_aligned(rel, ep, on, p10, shift=-0.5)
        b5, _ = pulse_block(rel, ep, p5, "p5")
        row.update(b5)

        if d10 is not None:
            r, ti, first = d10
            n = len(p10)
            lam0 = row["p10_lam0_hz"]
            for name, (lo, hi) in WINDOWS.items():
                row[f"cnt_{name}"] = cnt(r, lo, hi)
                row[f"exp_{name}"] = n * (hi - lo)
                ps = np.ravel(exact_rate_test(cnt(r, lo, hi), row["p10_nb"], n * (hi - lo), n * W_B)[1])[0]
                row[f"p_sup_{name}"] = float(ps)
            row["cnt_base"] = row["p10_nb"]
            row["exp_base"] = n * W_B
            # per light level
            lv = opto.loc[p10, "level"].values
            # light levels differ between sessions ({1,2.5,4}, {1.3,1.7,2}, {0.638,0.738,0.82});
            # index them by within-session rank
            uniq = np.sort(np.unique(np.round(opto.loc[opto["kind"] == "pulse10", "level"].values, 4)))
            for rank, L in zip(("low", "mid", "high"), uniq[:3]):
                row[f"p10_{rank}_level"] = float(L)
                sel = np.nonzero(np.isclose(lv, L))[0]
                nl = len(sel)
                fl = first[first.index.isin(sel)]
                kl = len(fl)
                rho_l, se_l, _, _ = chance_corrected_reliability(kl, nl, lam0, W_E)
                row[f"p10_{rank}_n"] = nl
                row[f"p10_{rank}_k"] = kl
                row[f"p10_{rank}_rho"] = float(rho_l)
                row[f"p10_{rank}_rho_se"] = float(se_l)
                row[f"p10_{rank}_median_first_ms"] = float(fl.median()) if kl else np.nan
                ne_l = int(np.count_nonzero(np.isin(ti, sel) & (r >= EV[0]) & (r < EV[1])))
                row[f"p10_{rank}_ne"] = ne_l

        # 10-Hz trains: 10 pulses of 2.5 ms, onsets every 100 ms
        tr = opto.index[(opto["kind"] == "train10hz").values & valid].values
        row["train_n"] = len(tr)
        if len(tr):
            m = np.isin(ep, tr)
            r_t, e_t = rel[m], ep[m]
            nbt = cnt(r_t, *BASE)
            row["train_lam0_hz"] = nbt / (len(tr) * W_B)
            for j in range(10):
                lo, hi = 0.1 * j + EV[0], 0.1 * j + EV[1]
                mm = (r_t >= lo) & (r_t < hi)
                row[f"train_k{j + 1}"] = len(np.unique(e_t[mm]))
                row[f"train_ne{j + 1}"] = int(mm.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    files = sorted((ROOT / "data/derived").glob("session_*.pkl.gz"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        dfs = list(ex.map(process, files))
    df = pd.concat(dfs, ignore_index=True)
    man = pd.read_csv(ROOT / "results/cohort/full_28_specimen_manifest.csv")
    df = df.merge(man[["session_id", "specimen_id", "cre_line", "sex", "age_in_days"]], on="session_id", how="left")
    out = ROOT / "results/reanalysis/tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "unit_inference.parquet", index=False)
    print(f"{len(df)} units from {df.session_id.nunique()} sessions -> {out / 'unit_inference.parquet'}")


if __name__ == "__main__":
    main()
