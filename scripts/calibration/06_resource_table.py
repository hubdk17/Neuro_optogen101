"""
06_resource_table.py
====================
§4D: per-unit supplementary resource table for all 19,005 units, with a data
dictionary describing every column and its units.

Merges:  results/reanalysis/tables/unit_results.parquet  (per-unit inference)
         results/calibration/tables/4C_unit_criteria.parquet (criterion labels)
         results/calibration/tables/4B_unit_flash_opto.parquet (opto vs flash)
         results/calibration/tables/4A_reference_units.csv (CCG source/target rates)
Writes:  results/calibration/resource/optotagging_units.{csv,parquet}
         results/calibration/resource/DATA_DICTIONARY.md (+ .csv)
Fails if any exported column lacks a dictionary entry.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/calibration/tables"
OUT = ROOT / "results/calibration/resource"

D = {}  # column -> (units, description)


def add(col, units, desc):
    D[col] = (units, desc)


# identity / anatomy
add("unit_id", "-", "Allen ecephys unit id")
add("session_id", "-", "Allen ecephys session id")
add("specimen_id", "-", "Allen specimen (mouse) id; one session per mouse in this cohort")
add("cre_line", "-", "Cre driver line crossed to Ai32 (ChR2(H134R)-EYFP): Pvalb-, Sst- or Vip-IRES-Cre")
add("sex", "-", "sex of the mouse (Allen metadata)")
add("age_in_days", "days", "age at recording (Allen metadata)")
add("probe_name", "-", "Neuropixels probe label within the session (probeA-F)")
add("probe_id", "-", "Allen probe id")
add("structure", "-", "CCF structure acronym of the unit's peak channel")
add("region_group", "-", "coarse region: visual cortex / hippocampal formation / thalamus / midbrain / other")
add("ccf_ap", "um", "CCF anterior-posterior coordinate of the peak channel")
add("ccf_dv", "um", "CCF dorsal-ventral coordinate of the peak channel")
add("ccf_lr", "um", "CCF left-right coordinate of the peak channel")
add("probe_vertical_position", "um", "peak-channel position along the shank (0 = tip)")
add("cortical_depth_um", "um", "distance below the highest VIS* channel on the probe (cortical units only; NaN otherwise)")
# waveform / firing
add("waveform_duration", "ms", "trough-to-peak duration of the mean waveform (Allen metric)")
add("waveform_halfwidth", "ms", "spike half-width (Allen metric)")
add("PT_ratio", "-", "peak-to-trough amplitude ratio (Allen metric)")
add("firing_rate", "Hz", "session-wide mean firing rate (Allen metric)")
add("snr", "-", "waveform signal-to-noise ratio (Allen metric)")
add("lam_spont_hz", "Hz", "spontaneous rate from the session's spontaneous epochs (invalid intervals removed)")
add("spont_duration_s", "s", "valid spontaneous-epoch exposure used for lam_spont_hz")
# pulse blocks
for pfx, name in (("p10", "10-ms single pulses"), ("p5", "5-ms single pulses")):
    add(f"{pfx}_n", "trials", f"valid trials, {name}")
    add(f"{pfx}_nb", "spikes", f"spike count in the baseline window [-480, -20) ms summed over trials, {name}")
    add(f"{pfx}_ne", "spikes", f"spike count in the evoked window [+1, +9) ms summed over trials, {name}")
    add(f"{pfx}_lam0_hz", "Hz", f"baseline rate from [-480, -20) ms of the same trials, {name}")
    add(f"{pfx}_p_exc", "-", f"exact conditional (binomial) test p, evoked > baseline, pi0 = 0.008/0.468, {name}")
    add(f"{pfx}_p_sup", "-", f"exact conditional test p, evoked < baseline, {name}")
    add(f"{pfx}_z", "SD", f"mid-p excitation z-score of the exact test (input to the lfdr), {name}")
    add(f"{pfx}_lrr", "log ratio", f"half-count-corrected log rate ratio evoked/baseline, {name}")
    add(f"{pfx}_lrr_se", "log ratio", f"approximate SE of {pfx}_lrr")
    add(f"{pfx}_k", "trials", f"trials with >= 1 spike in [+1, +9) ms, {name}")
    add(f"{pfx}_rel_raw", "fraction", f"raw trial reliability k/n, {name}")
    add(f"{pfx}_rho", "probability", f"chance-corrected reliability (r - p0)/(1 - p0), {name}")
    add(f"{pfx}_rho_se", "probability", f"SE of {pfx}_rho")
    add(f"{pfx}_p0", "probability", f"chance probability of >= 1 spike in 8 ms at the baseline rate, {name}")
    add(f"{pfx}_p_rel", "-", f"one-sided binomial p for reliability > p0, {name}")
    add(f"{pfx}_median_first_ms", "ms", f"median first-spike latency within [+1, +9) ms (includes spontaneous spikes), {name}")
    add(f"{pfx}_salt_p", "-", f"SALT p (port of Kvitsiani et al. 2013; 8-ms test window, 1-ms bins), {name}")
    add(f"{pfx}_salt_I", "bits", f"SALT information difference, {name}")
    add(f"{pfx}_fit_rho", "probability", f"ML first-spike model: probability of an evoked spike (fitted only if p_exc < 0.01), {name}")
    add(f"{pfx}_fit_delta", "ms", f"ML first-spike model: evoked latency, {name}")
    add(f"{pfx}_fit_sigma", "ms", f"ML first-spike model: evoked jitter (SD), {name}")
    add(f"{pfx}_fit_se_rho", "probability", "Wald SE of fit_rho (under-covers for weak responses, see estimator validation)")
    add(f"{pfx}_fit_se_delta", "ms", "Wald SE of fit_delta")
    add(f"{pfx}_fit_se_logsigma", "log ms", "Wald SE of log(fit_sigma)")
    add(f"{pfx}_fit_converged", "bool", "optimiser convergence flag of the latency-model fit")
for r in ("low", "mid", "high"):
    add(f"p10_{r}_level", "NWB 'level' units", f"light level of the {r} setting in this session (sets differ between sessions)")
    add(f"p10_{r}_n", "trials", f"10-ms trials at the {r} light level")
    add(f"p10_{r}_k", "trials", f"trials with >= 1 evoked spike at the {r} level")
    add(f"p10_{r}_rho", "probability", f"chance-corrected reliability at the {r} level")
    add(f"p10_{r}_rho_se", "probability", f"SE of p10_{r}_rho")
    add(f"p10_{r}_median_first_ms", "ms", f"median first-spike latency at the {r} level")
add("dose_delta_rho", "probability", "p10_high_rho - p10_low_rho")
add("train_n", "trials", "valid 10-Hz train trials (10 x 2.5-ms pulses, 100-ms period)")
add("train_lam0_hz", "Hz", "baseline rate before train trials, [-480, -20) ms")
for j in range(1, 11):
    add(f"train_k{j}", "trials", f"train trials with >= 1 spike in [+1, +9) ms after pulse {j}")
add("train_following", "probability", "mean over the 10 pulses of the chance-corrected per-pulse response")
for w in ("9", "50"):
    lo, hi = ("1", "9") if w == "9" else ("1", "51")
    add(f"zeta{w}_p", "-", f"zetapy ZETA p, 10-ms pulses, window [{lo}, {hi}) ms, seeded (2.5e2 resamples)")
    add(f"zeta{w}_sign", "-", f"sign of the ZETA deviation (+1 excitation, -1 suppression), window [{lo}, {hi}) ms")
# calibration
add("lfdr", "probability", "local false discovery rate of light activation (empirical null from sham windows)")
add("evidence", "probability", "1 - lfdr")
add("q_exact", "-", "Benjamini-Hochberg q of p10_p_exc across all 19,005 units")
add("onset_artifact", "bool", "light-onset artifact: fit_delta < 1.5 ms and fit_sigma < 0.3 ms")
add("driven", "bool", "calibrated light-activated: lfdr < 0.05 and not onset_artifact")
add("direct_like", "bool", "driven and fit_delta < 5 ms and fit_sigma < 1.5 ms (the criterion tested in §4A)")
crit = {"allen_lakunina": ">= 4 of first 5 train pulses with significant per-pulse response, raw reliability >= 0.30, median latency < 8 ms",
        "allen_lakunina_raw": "as allen_lakunina but per-pulse raw probability >= 0.30 at >= 4 of 5 pulses",
        "latency_lt8": "median first-spike latency in [1, 9) ms < 8 ms",
        "rel030_mod2": "raw reliability >= 0.30 and modulation ratio (original definition) > 2",
        "heuristic_full": "original operational rule: rel >= 0.30, latency < 8 ms, modulation > 2, permutation p < 0.05, Cohen's d > 0.1",
        "salt_p01": "SALT p < 0.01", "salt_bh": "SALT BH q < 0.05",
        "zeta9_bh": "ZETA [1, 9) ms BH q < 0.05, positive deviation", "zeta51_bh": "ZETA [1, 51) ms BH q < 0.05, any sign",
        "exact_bh": "exact conditional test BH q < 0.05", "lfdr05": "lfdr < 0.05 (lfdr curve fitted on real data)"}
for k, v in crit.items():
    add(f"label_{k}", "bool", f"criterion label on the real 10-ms pulse trials: {v}")
    add(f"sham_{k}", "bool", f"same criterion evaluated on sham trials placed in spontaneous epochs (should be False)")
add("modulation_ratio_original", "-", "evoked [1,9) ms rate / (baseline [-20,-5) ms rate + 1 Hz), original definition")
add("train_sig_pulses", "pulses", "number of the first 5 train pulses with a significant per-pulse response")
# flash / opto
add("opto_lrr_20_200", "log ratio", "log rate ratio 20-200 ms after 10-ms pulses vs baseline")
add("flash_lrr_20_200", "log ratio", "log rate ratio 20-200 ms after full-field flash onset vs baseline")
add("opto_suppressed", "bool", "suppressed 20-50 or 50-200 ms after 10-ms pulses (exact test, BH q < 0.05; not driven)")
add("flash_suppressed", "bool", "suppressed 20-50 or 50-200 ms after flash onset (same test)")
add("flash_excited", "bool", "excited 20-50 or 50-200 ms after flash onset (same test)")
# CCG
add("ccg_role", "-", "CCG reference role: 'driven', 'control' (strict match), 'control_relaxed', or empty")
add("ccg_n_partners", "pairs", "same-probe partners within 300 um with valid spontaneous CCGs")
add("ccg_source_rate", "fraction", "fraction of partners with a causal short-latency trough (unit inhibits partner)")
add("ccg_target_rate", "fraction", "fraction of partners with an anticausal trough (partner inhibits unit)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(ROOT / "results/reanalysis/tables/unit_results.parquet")
    u["dose_delta_rho"] = u.p10_high_rho - u.p10_low_rho
    p0 = 1 - np.exp(-u.train_lam0_hz * 0.008)
    K = u[[f"train_k{j}" for j in range(1, 11)]].values
    u["train_following"] = np.nanmean((K / u.train_n.values[:, None] - p0.values[:, None]) / (1 - p0.values[:, None]), axis=1)
    u["direct_like"] = u.driven & (u.p10_fit_delta < 5) & (u.p10_fit_sigma < 1.5)
    c = pd.read_parquet(T / "4C_unit_criteria.parquet")
    cc = c[["unit_id"] + [f"crit_real_{k}" for k in crit] + [f"crit_sham_{k}" for k in crit]
           + ["real_mod_ratio", "real_train_sig_pulses"]].copy()
    cc.columns = (["unit_id"] + [f"label_{k}" for k in crit] + [f"sham_{k}" for k in crit]
                  + ["modulation_ratio_original", "train_sig_pulses"])
    f = pd.read_parquet(T / "4B_unit_flash_opto.parquet",
                        columns=["unit_id", "opto_lrr_20_200", "flash_lrr_20_200", "opto_suppressed", "flash_suppressed", "flash_excited"])
    r = pd.read_csv(T / "4A_reference_units.csv")
    r = r.sort_values("ref_type").drop_duplicates("ref_unit")
    r = r.rename(columns={"ref_unit": "unit_id", "ref_type": "ccg_role", "n_partners": "ccg_n_partners",
                          "source_rate": "ccg_source_rate", "target_rate": "ccg_target_rate"})
    r = r[["unit_id", "ccg_role", "ccg_n_partners", "ccg_source_rate", "ccg_target_rate"]]
    base_cols = [k for k in D if k in u.columns]
    out = u[base_cols].merge(cc, on="unit_id", how="left").merge(f, on="unit_id", how="left").merge(r, on="unit_id", how="left")
    order = [k for k in D if k in out.columns]
    missing = [k for k in out.columns if k not in D]
    if missing:
        sys.exit(f"columns without dictionary entries: {missing}")
    out = out[order]
    assert len(out) == len(u) == out.unit_id.nunique()
    out.to_parquet(OUT / "optotagging_units.parquet", index=False)
    out.to_csv(OUT / "optotagging_units.csv", index=False)
    dd = pd.DataFrame([dict(column=k, units=D[k][0], description=D[k][1]) for k in order])
    dd.to_csv(OUT / "DATA_DICTIONARY.csv", index=False)
    with open(OUT / "DATA_DICTIONARY.md", "w") as fh:
        fh.write("# Data dictionary: `optotagging_units.{csv,parquet}`\n\n")
        fh.write(f"{len(out):,} units x {len(order)} columns. One row per unit passing the Allen default QC "
                 "(isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9, quality = good) in the 28 "
                 "Pvalb/Sst/Vip x Ai32 sessions. Every value is computed from NWB spike times by the scripts in "
                 "scripts/reanalysis/ and scripts/calibration/. Windows are relative to light (or flash) onset.\n\n")
        fh.write("| column | units | description |\n|---|---|---|\n")
        for _, x in dd.iterrows():
            fh.write(f"| `{x.column}` | {x.units} | {x.description} |\n")
    print(f"resource table: {len(out)} units x {len(order)} columns -> {OUT}")


if __name__ == "__main__":
    main()
