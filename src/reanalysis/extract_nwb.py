"""
extract_nwb.py
==============
Spike-level extraction from Allen Visual Coding Neuropixels NWB files, used by
the spike-based reanalysis (reports/REANALYSIS_RESULTS.md).

For one session this produces a compact, self-contained record:

* ``units``    unit table after the Allen default QC filter
               (isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9, quality == good),
               with probe, peak-channel location, CCF coordinates, shank position
               and waveform metrics.
* ``opto``     every optogenetic epoch (pulse 10 ms / 5 ms, 10-Hz train, cosine ramp).
* ``aligned``  per unit, spike times relative to each epoch onset in
               [ALIGN_PRE, ALIGN_POST] s, with the epoch index.  Epochs overlapping
               an invalid interval for the unit's probe are dropped.
* ``spont``    spike count and exposure in the spontaneous epochs (invalid
               intervals removed): a long-epoch estimate of the spontaneous rate.
* ``zeta``     reference ZETA test (zetapy) on raw spike times for 10-ms pulses,
               window [+1, +9) ms, and on a sham event set shifted by -500 ms.
* ``ccg``      spontaneous-epoch cross-correlograms (+/-30 ms, 0.5-ms bins) from
               each light-driven candidate unit to all units on the same probe
               within 300 um along the shank.

No numbers are synthesised: every value is a function of the NWB file.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.stats as st

logger = logging.getLogger(__name__)

ALIGN_PRE = -0.6
ALIGN_POST = 1.3
EVOKED = (0.001, 0.009)
BASELINE = (-0.500, -0.020)
CCG_HALF = 0.030
CCG_BIN = 0.0005
CCG_MAX_DIST_UM = 300.0
CANDIDATE_P = 1e-3


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _ragged(group, name):
    data = group[name]
    idx = group[f"{name}_index"][:]
    return data, idx


def _invalid_intervals(f) -> list:
    """List of (start, stop, probe_name or None)."""
    if "intervals/invalid_times" not in f:
        return []
    g = f["intervals/invalid_times"]
    starts, stops = g["start_time"][:], g["stop_time"][:]
    tags = _dec(g["tags"][:])
    tidx = g["tags_index"][:]
    out, lo = [], 0
    for s, e, hi in zip(starts, stops, tidx):
        t = tags[lo:hi]
        lo = hi
        probe = next((x for x in t if x.startswith("probe")), None)
        out.append((float(s), float(e), probe))
    return out


def load_session(path: str) -> Dict:
    f = h5py.File(path, "r")
    u = f["units"]
    qc = ((u["isi_violations"][:] < 0.5) & (u["amplitude_cutoff"][:] < 0.1) & (u["presence_ratio"][:] > 0.9)
          & (u["quality"][:] == b"good"))
    unit_ids = u["id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    chan = pd.DataFrame({
        "channel_id": el["id"][:],
        "probe_name": _dec(el["group_name"][:]),
        "probe_id": el["probe_id"][:],
        "structure": _dec(el["location"][:]),
        "probe_vertical_position": el["probe_vertical_position"][:],
        "probe_horizontal_position": el["probe_horizontal_position"][:],
        "ccf_ap": el["x"][:], "ccf_dv": el["y"][:], "ccf_lr": el["z"][:],
    }).set_index("channel_id")

    cols = ["firing_rate", "snr", "isi_violations", "amplitude_cutoff", "presence_ratio",
            "waveform_duration", "waveform_halfwidth", "PT_ratio", "repolarization_slope",
            "recovery_slope", "amplitude", "spread", "peak_channel_id"]
    units = pd.DataFrame({c: u[c][:] for c in cols})
    units.insert(0, "unit_id", unit_ids)
    units = units[qc].reset_index(drop=True)
    units = units.join(chan, on="peak_channel_id")

    # cortical depth along the shank: distance below the highest channel labelled VIS*
    depth = np.full(len(units), np.nan)
    for pname, ch in chan.groupby("probe_name"):
        vis = ch[ch["structure"].str.startswith("VIS")]
        if len(vis) == 0:
            continue
        top = vis["probe_vertical_position"].max()
        m = (units["probe_name"] == pname).values & units["structure"].str.startswith("VIS").values
        depth[m] = top - units.loc[m, "probe_vertical_position"].values
    units["cortical_depth_um"] = depth

    g = f["processing/optotagging/optogenetic_stimulation"]
    opto = pd.DataFrame({
        "start_time": g["start_time"][:], "stop_time": g["stop_time"][:],
        "duration": g["duration"][:], "level": g["level"][:],
        "condition": _dec(g["condition"][:]), "stimulus_name": _dec(g["stimulus_name"][:]),
    }).sort_values("start_time").reset_index(drop=True)
    kind = np.where(opto["stimulus_name"] == "fast_pulses", "train10hz",
           np.where(opto["stimulus_name"] == "raised_cosine", "ramp",
           np.where(np.isclose(opto["duration"], 0.010), "pulse10",
           np.where(np.isclose(opto["duration"], 0.005), "pulse5", "other"))))
    opto["kind"] = kind

    sp = f["intervals/spontaneous_presentations"]
    spont = np.column_stack([sp["start_time"][:], sp["stop_time"][:]])

    subj = f["general/subject"]
    meta = {
        "session_id": int(np.array(f["general/session_id"][()]).item()) if "session_id" in f["general"] else None,
        "genotype": subj["genotype"][()].decode() if isinstance(subj["genotype"][()], bytes) else str(subj["genotype"][()]),
        "sex": subj["sex"][()].decode() if isinstance(subj["sex"][()], bytes) else str(subj["sex"][()]),
        "age": subj["age"][()].decode() if isinstance(subj["age"][()], bytes) else str(subj["age"][()]),
        "subject_id": subj["subject_id"][()].decode() if isinstance(subj["subject_id"][()], bytes) else str(subj["subject_id"][()]),
    }

    st_data, st_idx = _ragged(u, "spike_times")
    starts = np.concatenate([[0], st_idx[:-1]])
    pos = np.nonzero(qc)[0]
    spikes = {int(unit_ids[i]): st_data[starts[i]:st_idx[i]] for i in pos}
    invalid = _invalid_intervals(f)
    f.close()
    return dict(units=units, opto=opto, spont=spont, spikes=spikes, invalid=invalid, meta=meta)


def _interval_mask(t: np.ndarray, intervals) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for s, e in intervals:
        m |= (t >= s) & (t < e)
    return m


def _subtract(intervals: np.ndarray, bad) -> list:
    """Remove bad (s, e) pairs from a list of good intervals."""
    out = [tuple(x) for x in intervals]
    for bs, be in bad:
        nxt = []
        for s, e in out:
            if be <= s or bs >= e:
                nxt.append((s, e))
                continue
            if s < bs:
                nxt.append((s, bs))
            if be < e:
                nxt.append((be, e))
        out = nxt
    return out


def _ccg_counts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Histogram of b - a lags in [-CCG_HALF, CCG_HALF), CCG_BIN bins."""
    nb = int(round(2 * CCG_HALF / CCG_BIN))
    if len(a) == 0 or len(b) == 0:
        return np.zeros(nb, dtype=np.int32)
    lo = np.searchsorted(b, a - CCG_HALF, side="left")
    hi = np.searchsorted(b, a + CCG_HALF, side="left")
    n = hi - lo
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(nb, dtype=np.int32)
    rep_a = np.repeat(a, n)
    offs = np.arange(tot) - np.repeat(np.cumsum(n) - n, n)
    lags = b[np.repeat(lo, n) + offs] - rep_a
    k = np.floor((lags + CCG_HALF) / CCG_BIN).astype(np.int64)
    k = k[(k >= 0) & (k < nb)]
    return np.bincount(k, minlength=nb).astype(np.int32)


def process_session(path: str, session_id: int, zeta_resamples: int = 250) -> Dict:
    from zetapy import zetatest

    S = load_session(path)
    units, opto, spikes = S["units"], S["opto"], S["spikes"]
    onsets = opto["start_time"].values
    block = (onsets.min() - 2.0, opto["stop_time"].max() + 2.0)

    # epochs invalid for each probe
    bad_by_probe: Dict[str, np.ndarray] = {}
    for pname in units["probe_name"].unique():
        bad = np.zeros(len(opto), dtype=bool)
        for s, e, probe in S["invalid"]:
            if probe is None or probe == pname:
                bad |= (opto["start_time"].values + ALIGN_PRE < e) & (opto["start_time"].values + ALIGN_POST > s)
        bad_by_probe[pname] = bad

    aligned: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for uid, pname in zip(units["unit_id"].values, units["probe_name"].values):
        s = spikes[int(uid)]
        lo = np.searchsorted(s, onsets + ALIGN_PRE)
        hi = np.searchsorted(s, onsets + ALIGN_POST)
        bad = bad_by_probe[pname]
        rel, ep = [], []
        for k in range(len(onsets)):
            if bad[k] or hi[k] == lo[k]:
                continue
            rel.append(s[lo[k]:hi[k]] - onsets[k])
            ep.append(np.full(hi[k] - lo[k], k, dtype=np.int16))
        aligned[int(uid)] = (np.concatenate(rel).astype(np.float32) if rel else np.zeros(0, np.float32),
                             np.concatenate(ep) if ep else np.zeros(0, np.int16))
    valid_epochs = {p: ~b for p, b in bad_by_probe.items()}

    # spontaneous-epoch counts (invalid intervals removed, per probe)
    spont_rows = []
    spont_good = {}
    for pname in units["probe_name"].unique():
        bad = [(s, e) for s, e, p in S["invalid"] if p is None or p == pname]
        spont_good[pname] = _subtract(S["spont"], bad)
    for uid, pname in zip(units["unit_id"].values, units["probe_name"].values):
        iv = spont_good[pname]
        dur = float(sum(e - s for s, e in iv))
        s = spikes[int(uid)]
        cnt = int(sum(np.searchsorted(s, e) - np.searchsorted(s, b) for b, e in iv))
        spont_rows.append((int(uid), cnt, dur))
    spont = pd.DataFrame(spont_rows, columns=["unit_id", "spont_count", "spont_duration_s"])

    # reference ZETA on 10-ms pulses (window [+1, +9) ms) and a sham event set at -500 ms
    p10 = opto.index[opto["kind"] == "pulse10"].values
    zrows = []
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    for uid, pname in zip(units["unit_id"].values, units["probe_name"].values):
        ev_idx = p10[valid_epochs[pname][p10]]
        ev = np.sort(onsets[ev_idx]) + EVOKED[0]
        s = spikes[int(uid)]
        s = s[(s > block[0] - 5) & (s < block[1] + 5)]
        res = {}
        for tag, shift in (("zeta", 0.0), ("zeta_sham", -0.5)):
            try:
                out = zetatest(s, ev + shift, max_duration=EVOKED[1] - EVOKED[0],
                               resampling_number=zeta_resamples)
                d = out[1]
                res[f"{tag}_p"] = float(d["zeta_p_value"])
                res[f"{tag}_z"] = float(d["zeta_score"])
                res[f"{tag}_sign"] = float(np.sign(d.get("zeta_deviation", np.nan)))
                res[f"{tag}_latency_s"] = float(d.get("latency_zeta", np.nan)) if d.get("latency_zeta") is not None else np.nan
            except Exception:
                res.update({f"{tag}_p": 1.0, f"{tag}_z": 0.0, f"{tag}_sign": np.nan, f"{tag}_latency_s": np.nan})
        zrows.append(dict(unit_id=int(uid), **res))
    logging.getLogger().setLevel(logging.INFO)
    zeta = pd.DataFrame(zrows)

    # light-driven candidates (exact conditional test, 10-ms pulses) -> spontaneous CCGs
    cand = []
    w_e, w_b = EVOKED[1] - EVOKED[0], BASELINE[1] - BASELINE[0]
    pi0 = w_e / (w_e + w_b)
    for uid, pname in zip(units["unit_id"].values, units["probe_name"].values):
        rel, ep = aligned[int(uid)]
        m = np.isin(ep, p10)
        ne = int(np.sum(m & (rel >= EVOKED[0]) & (rel < EVOKED[1])))
        nb = int(np.sum(m & (rel >= BASELINE[0]) & (rel < BASELINE[1])))
        if ne + nb > 0 and st.binom.sf(ne - 1, ne + nb, pi0) < CANDIDATE_P:
            cand.append(int(uid))
    ccg_rows = []
    nbins = int(round(2 * CCG_HALF / CCG_BIN))
    if cand:
        ui = units.set_index("unit_id")
        for a in cand:
            pa = ui.loc[a, "probe_name"]
            ya = ui.loc[a, "probe_vertical_position"]
            sa = spikes[a]
            sa = sa[_interval_mask(sa, spont_good[pa])]
            part = ui[(ui["probe_name"] == pa) & (np.abs(ui["probe_vertical_position"] - ya) <= CCG_MAX_DIST_UM)].index
            for b in part:
                if b == a:
                    continue
                sb = spikes[int(b)]
                sb = sb[_interval_mask(sb, spont_good[pa])]
                c = _ccg_counts(sa, sb)
                ccg_rows.append((a, int(b), len(sa), len(sb), float(ui.loc[b, "probe_vertical_position"] - ya), c))
    ccg = pd.DataFrame(ccg_rows, columns=["unit_a", "unit_b", "n_a", "n_b", "dy_um", "counts"]) if ccg_rows else \
        pd.DataFrame(columns=["unit_a", "unit_b", "n_a", "n_b", "dy_um", "counts"])
    spont_T = {p: float(sum(e - s for s, e in iv)) for p, iv in spont_good.items()}

    units["session_id"] = session_id
    return dict(session_id=session_id, meta=S["meta"], units=units, opto=opto, aligned=aligned,
                valid_epochs=valid_epochs, spont=spont, zeta=zeta, ccg=ccg, ccg_bins=nbins,
                spont_duration_by_probe=spont_T, candidates=cand)
