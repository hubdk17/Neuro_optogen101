"""
01_matched_ccg.py
=================
Extended CCG reference set for the §4A positive control (reports/PREREGISTRATION_4A.md).

For each driven unit (lfdr < 0.05, onset artifacts excluded) up to 3 matched
non-driven controls are drawn on the same probe:
  * lfdr > 0.5, not driven, not an onset artifact,
  * spontaneous rate within +/-25% of the driven unit's lam_spont_hz,
  * vertical position within +/-100 um,
sampled without replacement within the session (seed 20261006). There is
deliberately NO matching on waveform: waveform is the cell-type signal under test.

Spontaneous-epoch CCGs (data/derived v2 `spont_spikes`) are then computed from
every reference unit (driven and control) to every unit on the same probe
within 300 um, with the v1 machinery unchanged: +/-30 ms, 0.5-ms bins,
Stark-Abeles hollow-Gaussian predictor, causal bins +0.8..+4 ms, anticausal
control bins -4..-0.8 ms, Bonferroni over bins. Pairs need >= 500 spikes in both units.

Outputs
  results/calibration/tables/ccg_matched_controls.csv  matching keys (auditable)
  results/calibration/tables/ccg_pairs_extended.csv    one row per (reference, partner)
  data/derived/ccg_extended_counts.pkl.gz              raw CCG counts (git-ignored)
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
from src.reanalysis.extract_nwb import _ccg_counts, CCG_BIN, CCG_HALF, CCG_MAX_DIST_UM  # noqa: E402
from src.reanalysis.inference import ccg_jitter_test  # noqa: E402

T_OUT = ROOT / "results/calibration/tables"
SEED = 20261006
N_CONTROLS = 3
RATE_TOL = 0.25
POS_TOL_UM = 100.0
MIN_SPIKES = 500


RELAXED_RATE_TOL = 0.50
RELAXED_POS_TOL_UM = 200.0


def match_controls(units: pd.DataFrame) -> pd.DataFrame:
    """units: unit_results rows for one session.

    Tier 'strict' implements the pre-registered rule. Driven units left with no strict
    control get up to 3 'relaxed' controls (rate +/-50%, position +/-200 um, from the
    units not already used). The relaxed tier is a declared deviation from the
    pre-registration, used only in a labelled sensitivity analysis."""
    rng = np.random.default_rng(SEED + int(units.session_id.iloc[0]) % 100000)
    pool = units[(units.lfdr > 0.5) & ~units.driven & ~units.onset_artifact.fillna(False)]
    used = set()
    rows = []
    for _, d in units[units.driven].sort_values("unit_id").iterrows():
        cand = pool[(pool.probe_name == d.probe_name)
                    & (np.abs(pool.lam_spont_hz - d.lam_spont_hz) <= RATE_TOL * d.lam_spont_hz)
                    & (np.abs(pool.probe_vertical_position - d.probe_vertical_position) <= POS_TOL_UM)
                    & ~pool.unit_id.isin(used)]
        n_cand = len(cand)
        pick = cand.sample(n=min(N_CONTROLS, n_cand), random_state=rng.integers(0, 2 ** 31)) if n_cand else cand
        for _, c in pick.iterrows():
            used.add(int(c.unit_id))
            rows.append(dict(session_id=int(d.session_id), driven_unit=int(d.unit_id), control_unit=int(c.unit_id),
                             probe_name=d.probe_name, n_candidates=n_cand,
                             driven_lam_spont_hz=d.lam_spont_hz, control_lam_spont_hz=c.lam_spont_hz,
                             driven_y_um=d.probe_vertical_position, control_y_um=c.probe_vertical_position,
                             driven_waveform_ms=d.waveform_duration, control_waveform_ms=c.waveform_duration,
                             control_lfdr=c.lfdr))
        if n_cand == 0:
            rows.append(dict(session_id=int(d.session_id), driven_unit=int(d.unit_id), control_unit=np.nan,
                             probe_name=d.probe_name, n_candidates=0, driven_lam_spont_hz=d.lam_spont_hz,
                             driven_y_um=d.probe_vertical_position, driven_waveform_ms=d.waveform_duration))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["tier"] = np.where(out.control_unit.notna(), "strict", "none")
    # relaxed tier for driven units with no strict control
    relaxed = []
    for du in out.loc[out.tier == "none", "driven_unit"]:
        d = units.set_index("unit_id").loc[du]
        cand = pool[(pool.probe_name == d.probe_name)
                    & (np.abs(pool.lam_spont_hz - d.lam_spont_hz) <= RELAXED_RATE_TOL * d.lam_spont_hz)
                    & (np.abs(pool.probe_vertical_position - d.probe_vertical_position) <= RELAXED_POS_TOL_UM)
                    & ~pool.unit_id.isin(used)]
        pick = cand.sample(n=min(N_CONTROLS, len(cand)), random_state=rng.integers(0, 2 ** 31)) if len(cand) else cand
        for _, c in pick.iterrows():
            used.add(int(c.unit_id))
            relaxed.append(dict(session_id=int(d.session_id), driven_unit=int(du), control_unit=int(c.unit_id),
                                probe_name=d.probe_name, n_candidates=len(cand),
                                driven_lam_spont_hz=d.lam_spont_hz, control_lam_spont_hz=c.lam_spont_hz,
                                driven_y_um=d.probe_vertical_position, control_y_um=c.probe_vertical_position,
                                driven_waveform_ms=d.waveform_duration, control_waveform_ms=c.waveform_duration,
                                control_lfdr=c.lfdr, tier="relaxed"))
    if relaxed:
        out = pd.concat([out, pd.DataFrame(relaxed)], ignore_index=True)
    return out


def session_ccgs(args):
    sid, refs = args  # refs: list of (unit_id, ref_type, matched_to)
    with gzip.open(ROOT / f"data/derived/session_{sid}.pkl.gz", "rb") as fh:
        R = pickle.load(fh)
    assert R.get("extract_version", 1) >= 2, f"session {sid} needs v2 extraction"
    ui = R["units"].set_index("unit_id")
    sp = R["spont_spikes"]
    rows, counts = [], {}
    for a, rtype, matched in refs:
        pa, ya = ui.loc[a, "probe_name"], ui.loc[a, "probe_vertical_position"]
        sa = sp[a]
        part = ui[(ui.probe_name == pa) & (np.abs(ui.probe_vertical_position - ya) <= CCG_MAX_DIST_UM)].index
        for b in part:
            b = int(b)
            if b == a:
                continue
            sb = sp[b]
            if len(sa) < MIN_SPIKES or len(sb) < MIN_SPIKES:
                continue
            c = _ccg_counts(sa, sb)
            pe, pi, pk, tr = ccg_jitter_test(c, CCG_BIN, CCG_HALF)
            pe_c, pi_c, _, _ = ccg_jitter_test(c, CCG_BIN, CCG_HALF, win=(-0.0040, -0.0008))
            rows.append(dict(session_id=sid, ref_unit=a, ref_type=rtype, matched_to=matched, partner=b,
                             dy_um=float(ui.loc[b, "probe_vertical_position"] - ya), n_a=len(sa), n_b=len(sb),
                             p_exc=pe, p_inh=pi, peak_ratio=pk, trough_ratio=tr,
                             p_exc_anticausal=pe_c, p_inh_anticausal=pi_c))
            counts[(a, b)] = c
    return pd.DataFrame(rows), counts


def main():
    T_OUT.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet")
    match = pd.concat([match_controls(g) for _, g in u.groupby("session_id")], ignore_index=True)
    match.to_csv(T_OUT / "ccg_matched_controls.csv", index=False)
    st_ = match[match.tier == "strict"]
    rl_ = match[match.tier == "relaxed"]
    print(f"driven units: {u.driven.sum()}; strict: {st_.driven_unit.nunique()} units / {len(st_)} controls; "
          f"relaxed: {rl_.driven_unit.nunique()} units / {len(rl_)} controls; "
          f"still unmatched: {u.driven.sum() - st_.driven_unit.nunique() - rl_.driven_unit.nunique()}")

    jobs = []
    for sid, g in u[u.driven].groupby("session_id"):
        refs = [(int(x), "driven", np.nan) for x in g.unit_id]
        m = match[(match.session_id == sid) & match.control_unit.notna()]
        refs += [(int(r.control_unit), "control" if r.tier == "strict" else "control_relaxed", int(r.driven_unit))
                 for r in m.itertuples()]
        jobs.append((int(sid), refs))
    with ProcessPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(session_ccgs, jobs))
    pairs = pd.concat([o[0] for o in out], ignore_index=True)
    meta = u.set_index("unit_id")
    pairs["partner_driven"] = pairs.partner.map(meta.driven).astype(bool)
    pairs["cre_line"] = pairs.ref_unit.map(meta.cre_line)
    pairs["specimen_id"] = pairs.ref_unit.map(meta.specimen_id)
    pairs.to_csv(T_OUT / "ccg_pairs_extended.csv", index=False)
    allc = {}
    for o in out:
        allc.update(o[1])
    with gzip.open(ROOT / "data/derived/ccg_extended_counts.pkl.gz", "wb") as fh:
        pickle.dump(dict(bin_s=CCG_BIN, half_s=CCG_HALF, counts=allc), fh, protocol=4)
    print(pairs.groupby(["cre_line", "ref_type"]).agg(ref_units=("ref_unit", "nunique"), pairs=("partner", "size"),
                                                      inh=("p_inh", lambda x: np.mean(x < 0.01))))


if __name__ == "__main__":
    main()
