"""
07_figures.py
=============
Calibration figures C1-C4, drawn only from results/calibration/tables/.
Output: results/calibration/figures/C*.png / .pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/calibration/tables"
F = ROOT / "results/calibration/figures"
F.mkdir(parents=True, exist_ok=True)
COL = {"PV": "#2a78d6", "SST": "#eb6834", "VIP": "#1baf7a"}
INK, INK2, GRID, NULLC = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 300})
NAMES = {"allen_lakunina": "Allen/Lakunina (>=4/5 pulses, rel>=0.3, lat<8)",
         "allen_lakunina_raw": "  variant: raw per-pulse prob.",
         "latency_lt8": "latency < 8 ms alone", "rel030_mod2": "reliability>=0.3 & modulation>2",
         "heuristic_full": "original operational heuristic", "salt_p01": "SALT p<0.01", "salt_bh": "SALT BH q<0.05",
         "zeta9_bh": "ZETA [1,9) ms BH", "zeta51_bh": "ZETA [1,51) ms BH", "exact_bh": "exact test BH*",
         "lfdr05": "lfdr<0.05*"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.png", bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def title(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=10, color=INK)


# C1 criteria calibration
c = pd.read_csv(T / "4C_criteria_calibration.csv")
c["label"] = c.criterion.map(NAMES)
y = np.arange(len(c))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
fpr = c.sham_fpr.clip(lower=1e-5)
ax[0].errorbar(fpr, y, xerr=[fpr - c.sham_fpr_ci_low.clip(lower=1e-5), c.sham_fpr_ci_high.clip(lower=1e-5) - fpr],
               fmt="o", color=INK, ms=5, capsize=2)
ax[0].axvline(0.05, color=INK2, ls="--", lw=1)
ax[0].set_xscale("log")
ax[0].set_yticks(y, c.label, fontsize=8)
ax[0].invert_yaxis()
ax[0].set_xlabel("false-positive rate on sham trials (per unit; 0 plotted at 1e-5)")
title(ax[0], "A  False positives with no light")
for k, (col, lab) in enumerate((("sensitivity", "sensitivity vs calibrated set"), ("precision", "precision vs calibrated set"))):
    a = ax[k + 1]
    v = c[col]
    a.errorbar(v, y, xerr=[v - c[f"{col}_ci_low"], c[f"{col}_ci_high"] - v], fmt="o", color="#2a78d6", ms=5, capsize=2)
    a.set_yticks(y, [""] * len(y))
    a.invert_yaxis()
    a.set_xlim(-0.02, 1.02)
    a.set_xlabel(lab)
    title(a, ["B  Sensitivity", "C  Precision"][k])
ax[2].text(1.0, len(y) - 0.2, "* defines / neighbours the reference set", ha="right", fontsize=7, color=INK2)
save(fig, "C1_criteria_calibration")

# C2 latency/jitter validity
pc = pd.read_csv(T / "4A_positive_control.csv")
ds = pd.read_csv(T / "4A_delta_sigma_tests.csv")
pw = pd.read_csv(T / "4A_power_curve.csv")
ref = pd.read_csv(T / "4A_reference_units.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
p = pc[pc.model.str.contains("strict \\(pre")]
x = np.arange(len(p))
ax[0].bar(x - 0.2, p.driven_rate, width=0.38, color=[COL[m.split()[2].rstrip(",")] for m in p.model], label="driven reference units")
ax[0].bar(x + 0.2, p.control_rate, width=0.38, color=NULLC, label="matched non-driven controls")
ax[0].set_xticks(x, [f"{m.split()[2].rstrip(',')}: OR {o:.2f}\nanimal-level paired p={ap:.2g} ({int(an)} animals)"
                     for m, o, ap, an in zip(p.model, p.odds_ratio, p.animal_t_p, p.animal_n)], fontsize=8)
ax[0].set_ylabel("fraction of partners inhibited")
ax[0].legend(fontsize=8)
ax[0].grid(axis="x", visible=False)
title(ax[0], "A  Positive control (pre-registered matching)")
d = ref[(ref.ref_type == "driven") & ref.cre.isin(["PV", "SST"])].dropna(subset=["p10_fit_delta"])
g = d.groupby(["specimen_id", "cre", "direct_like"]).source_rate.mean().unstack()
for (sp, cre), r in g.iterrows():
    if r.notna().all():
        ax[1].plot([0, 1], [r[False], r[True]], color=COL[cre], marker="o", lw=1.2, alpha=0.8)
ax[1].set_xticks([0, 1], ["indirect-like", "direct-like\n(delta<5 ms, sigma<1.5 ms)"])
ax[1].set_xlim(-0.4, 1.4)
ax[1].set_ylabel("source rate (animal mean)")
row = ds[ds.model.str.startswith("animal-level")].iloc[0]
ax[1].text(0.5, ax[1].get_ylim()[1] * 0.95, f"paired diff {row.mean_diff:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]\n"
           f"p = {row.p:.2f}, {int(row.n_animals)} animals", ha="center", va="top", fontsize=8, color=INK)
ax[1].grid(axis="x", visible=False)
title(ax[1], "B  Direct-like vs indirect-like, per animal")
ax[2].plot(pw.true_difference_direct_minus_indirect, pw.power, marker="o", color=INK, lw=2)
ax[2].axhline(0.8, color=INK2, ls="--", lw=1)
ax[2].set_xlabel("true difference in source rate (direct - indirect)")
ax[2].set_ylabel("power (alpha 0.05)")
title(ax[2], "C  Power of the latency/jitter test")
save(fig, "C2_latency_jitter_validity")

# C3 flash confound
pr = pd.read_csv(T / "4B_region_prevalence_overlap.csv")
regs = ["visual cortex", "thalamus", "hippocampal formation", "midbrain"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
w = 0.13
for i, cre in enumerate(["PV", "SST", "VIP"]):
    s = pr[pr.cre == cre].set_index("region").reindex(regs)
    xs = np.arange(len(regs)) + (i - 1) * 2 * w
    ax[0].bar(xs - w / 2, s.opto_suppressed_frac_animal_mean, width=w, color=COL[cre], label=f"{cre} opto")
    ax[0].bar(xs + w / 2, s.flash_suppressed_frac_animal_mean, width=w, color=COL[cre], alpha=0.35, hatch="//",
              edgecolor="white", label=f"{cre} flash")
ax[0].set_xticks(range(len(regs)), regs, fontsize=8)
ax[0].set_ylabel("fraction suppressed (animal mean)")
ax[0].legend(fontsize=7, ncol=3)
ax[0].grid(axis="x", visible=False)
title(ax[0], "A  Suppression after light pulse vs visual flash")
for i, cre in enumerate(["PV", "SST", "VIP"]):
    s = pr[pr.cre == cre].set_index("region").reindex(regs)
    xs = np.arange(len(regs)) + (i - 1) * 0.25
    ax[1].bar(xs - 0.05, s.frac_opto_suppressed_also_flash_suppressed, width=0.1, color=COL[cre], label=f"{cre}: P(flash-supp | opto-supp)")
    ax[1].bar(xs + 0.05, s.frac_flash_suppressed_among_not_opto, width=0.1, color=NULLC)
ax[1].set_xticks(range(len(regs)), regs, fontsize=8)
ax[1].set_ylabel("fraction also flash-suppressed")
ax[1].legend(fontsize=7)
ax[1].grid(axis="x", visible=False)
title(ax[1], "B  Overlap (grey: among not opto-suppressed)")
cv = pd.read_csv(T / "4B_population_psth_suppressed.csv")
tcols = [k for k in cv.columns if k.startswith("t")]
tt = np.array([float(k[1:]) for k in tcols])
for grp, ls in (("LGd", "-"), ("LP", "--")):
    for stim, colr in (("opto", INK), ("flash", "#2a78d6")):
        s = cv[(cv.group == grp) & (cv.stim == stim)]
        if len(s):
            ax[2].plot(tt, s[tcols].mean().values, ls=ls, color=colr, lw=1.8, label=f"{grp} {stim} ({len(s)} animals)")
ax[2].axhline(1, color=INK2, lw=0.8)
ax[2].axvline(0, color=INK2, lw=0.8)
ax[2].set_xlabel("time from onset (ms)")
ax[2].set_ylabel("rate / baseline (suppressed units)")
ax[2].legend(fontsize=7)
title(ax[2], "C  Thalamic time course: light pulse vs flash")
save(fig, "C3_flash_confound")

# C4 label noise
ln = pd.read_csv(T / "4C_label_noise_performance.csv")
ln = ln[ln.task == "PV vs non-PV"]
tr = pd.read_csv(T / "4C_label_noise_training_sets.csv").set_index("label_source")
y = np.arange(len(ln))
fig, ax = plt.subplots(1, 1, figsize=(9, 3.8))
ax.barh(y + 0.18, ln.own_auc, height=0.34, color=NULLC, label="scored on its own labels (held-out sessions)")
ax.barh(y - 0.18, ln.common_test_auc, height=0.34, color="#2a78d6", label="scored on calibrated units (held-out sessions)")
ax.axvline(0.5, color=INK2, ls="--", lw=1)
ax.set_yticks(y, [f"{s} (n={int(tr.loc[s, 'n_units'])})" for s in ln.label_source], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("AUC, PV vs non-PV from waveform + firing rate")
ax.set_xlim(0.0, 1.0)
fig.text(0.01, -0.02, "AUC < 0.5 with uninformative labels is an artefact of session-held-out CV "
         "(class balance shifts between folds), not anti-learning.", fontsize=7, color=INK2, ha="left", va="top")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", visible=False)
title(ax, "Label-noise propagation into a waveform cell-type classifier")
save(fig, "C4_label_noise")
print("calibration figures written to", F)
