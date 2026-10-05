"""
04_figures.py
=============
Figures for the spike-based reanalysis. Every panel is drawn from tables in
results/reanalysis/tables/ (no analytic curves, no random numbers).
Output: results/reanalysis/figures/R*.png / .pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results/reanalysis/tables"
F = ROOT / "results/reanalysis/figures"
F.mkdir(parents=True, exist_ok=True)
CRES = ["Pvalb-IRES-Cre", "Sst-IRES-Cre", "Vip-IRES-Cre"]
SHORT = {c: c.split("-")[0] for c in CRES}
COL = {"Pvalb-IRES-Cre": "#2a78d6", "Sst-IRES-Cre": "#eb6834", "Vip-IRES-Cre": "#1baf7a"}
INK, INK2, GRID, NULLC = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / f"{name}.png", bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=10, color=INK)


df = pd.read_parquet(T / "unit_results.parquet")
S = json.load(open(ROOT / "results/reanalysis/summary.json"))

# ---------------------------------------------------------------- R1 evidence
lc = pd.read_csv(T / "lfdr_curve.csv")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 0.8, 1.25]})
w = np.diff(lc.z_mid).mean()
ax[0].bar(lc.z_mid, lc.count_real / lc.count_real.sum() / w, width=w * 0.92, color=COL[CRES[0]], alpha=0.85,
          label="observed units (10-ms pulses)")
ax[0].plot(lc.z_mid, S["pi0"] * lc.f0, color=INK, lw=2, label=r"$\pi_0 f_0$: empirical null (sham windows)")
ax[0].set_yscale("log")
ax[0].set_ylim(1e-5, 1)
ax[0].set_xlabel("excitation z (exact conditional test, mid-p)")
ax[0].set_ylabel("density")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=1)
label(ax[0], "A  Observed vs empirical null")
ax[1].plot(lc.z_mid, 1 - lc.lfdr, color=COL[CRES[0]], lw=2)
ax[1].axhline(0.95, color=INK2, lw=1, ls="--")
ax[1].text(lc.z_mid.min() + 0.2, 0.9, "evidence = 0.95 (lfdr = 0.05)", color=INK2, fontsize=8, va="top")
ax[1].set_xlabel("excitation z")
ax[1].set_ylabel("evidence = 1 - lfdr")
label(ax[1], "B  Calibrated evidence score")
mc = pd.read_csv(T / "method_comparison.csv")
mc = mc[~mc.method.str.startswith("repo '")]
y = np.arange(len(mc))
ax[2].barh(y, mc.n_positive, color=NULLC, height=0.7, label="all positives")
ax[2].barh(y, mc.overlap_with_lfdr, color=COL[CRES[0]], height=0.7, label="also lfdr < 0.05")
ax[2].set_yticks(y, mc.method, fontsize=8)
ax[2].invert_yaxis()
for yi, n in zip(y, mc.n_positive):
    ax[2].text(n, yi, f" {n}", va="center", fontsize=8, color=INK)
ax[2].set_xlabel("units called light-activated")
ax[2].legend(loc="lower right", fontsize=8)
ax[2].grid(axis="y", visible=False)
label(ax[2], "C  Method comparison (raw spikes)")
save(fig, "R1_evidence_and_methods")

# ---------------------------------------------------------------- R2 characterisation
d = df[df.driven]
fit = d.dropna(subset=["p10_fit_delta"])
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for c in CRES:
    s = fit[fit.cre_line == c]
    ax[0].scatter(s.p10_fit_delta, s.p10_fit_sigma, s=14, color=COL[c], alpha=0.75, edgecolor="white", lw=0.4,
                  label=f"{SHORT[c]} (n={len(s)})")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"latency $\hat\delta$ (ms)")
ax[0].set_ylabel(r"jitter $\hat\sigma$ (ms)")
ax[0].legend(fontsize=8)
label(ax[0], "A  First-spike model fits")
for c in CRES:
    s = fit[fit.cre_line == c]
    ax[1].scatter(s.p10_fit_delta, s.p10_median_first_ms, s=14, color=COL[c], alpha=0.75, edgecolor="white", lw=0.4)
ax[1].plot([1, 9], [1, 9], color=INK2, lw=1, ls="--")
ax[1].set_xlabel(r"model latency $\hat\delta$ (ms)")
ax[1].set_ylabel("naive median first-spike latency (ms)")
label(ax[1], "B  Naive latency is pulled to 5 ms")
for c in CRES:
    s = d[d.cre_line == c]
    ax[2].scatter(s.p10_rho, s.p5_rho, s=14, color=COL[c], alpha=0.75, edgecolor="white", lw=0.4, label=SHORT[c])
ax[2].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
ax[2].set_xlabel(r"$\hat\rho$, 10-ms pulses")
ax[2].set_ylabel(r"$\hat\rho$, 5-ms pulses (independent trials)")
ax[2].legend(fontsize=8)
label(ax[2], "C  Replication on 5-ms pulses")
wf = pd.read_csv(T / "waveform_narrow_fraction.csv")
x = np.arange(3)
for j, (flag, nm) in enumerate([(False, "not light-activated"), (True, "light-activated")]):
    v = [wf[(wf.cre_line == c) & (wf.driven == flag)]["mean"].values for c in CRES]
    v = [a[0] if len(a) else np.nan for a in v]
    ax[3].bar(x + (j - 0.5) * 0.38, v, width=0.36, color=[COL[c] if flag else NULLC for c in CRES],
              label=nm)
ax[3].set_xticks(x, [SHORT[c] for c in CRES])
ax[3].set_ylabel("fraction narrow-spiking (< 0.4 ms)")
ax[3].legend(fontsize=8, loc="upper right")
ax[3].grid(axis="x", visible=False)
label(ax[3], "D  Waveform check (grey = rest)")
save(fig, "R2_characterisation")

# ---------------------------------------------------------------- R3 yield & suppression per animal
ya = pd.read_csv(T / "yield_per_animal.csv")
df["suppressed_f"] = df.suppressed.astype(float)
sa = df.groupby(["specimen_id", "cre_line"]).suppressed_f.mean().reset_index()
sd = df[df.detectable].groupby(["specimen_id", "cre_line"]).suppressed_f.mean().reset_index()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for k, (tab, col, ttl, yl) in enumerate([(ya, "driven_f", "A  Light-activated yield", "fraction of units (lfdr < 0.05)"),
                                         (sa, "suppressed_f", "B  Suppressed (20-200 ms), all units", "fraction of units"),
                                         (sd, "suppressed_f", "C  Suppressed, units with power", "fraction of units")]):
    for i, c in enumerate(CRES):
        v = tab[tab.cre_line == c][col].values
        jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.25
        ax[k].scatter(i + jit, v, s=26, color=COL[c], edgecolor="white", lw=0.6, zorder=3)
        ax[k].hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
    ax[k].set_xticks(range(3), [SHORT[c] for c in CRES])
    ax[k].set_ylabel(yl)
    ax[k].grid(axis="x", visible=False)
    label(ax[k], ttl)
save(fig, "R3_yield_and_suppression_per_animal")

# ---------------------------------------------------------------- R4 archetypes
ar = pd.read_csv(T / "archetypes.csv").sort_values("n_units", ascending=False)
wins = ["w1", "w2", "w3", "w4", "w5"]
wlab = ["1-9", "11-20", "20-50", "50-200", "200-500"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
greys = plt.cm.Greys(np.linspace(0.35, 0.95, len(ar)))
MARK = ["o", "s", "^", "D", "v", "P", "X", "*"]
for j, ((i, r), gc) in enumerate(zip(ar.iterrows(), greys)):
    yv = [r[f"rate_ratio_{w}"] for w in wins]
    ax[0].plot(range(5), yv, marker=MARK[j % 8], ms=5, lw=2, color=gc,
               label=f"A{int(r.archetype)} (n={int(r.n_units)})")
ax[0].axhline(1, color=INK2, lw=1, ls="--")
ax[0].set_yscale("log")
ax[0].set_xticks(range(5), wlab)
ax[0].set_xlabel("window after 10-ms pulse onset (ms)")
ax[0].set_ylabel("rate / baseline rate")
ax[0].legend(fontsize=7, ncol=2)
label(ax[0], "A  Archetype rate profiles (multinomial mixture)")
bottom = np.zeros(3)
for (i, r), gc in zip(ar.iterrows(), greys):
    v = np.array([r[f"frac_of_{SHORT[c]}_units"] for c in CRES])
    ax[1].bar(range(3), v, bottom=bottom, color=gc, edgecolor="white", lw=1, width=0.6, label=f"A{int(r.archetype)}")
    bottom += v
ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax[1].set_xticks(range(3), [SHORT[c] for c in CRES])
ax[1].set_ylabel("fraction of units")
ax[1].grid(axis="x", visible=False)
label(ax[1], "B  Archetype composition by Cre line")
save(fig, "R4_archetypes")

# ---------------------------------------------------------------- R5 dose & trains
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
lv = ["low", "mid", "high"]
for c in CRES:
    s = d[d.cre_line == c]
    if len(s) < 3:
        continue
    ax[0].plot(range(3), [s[f"p10_{L}_rho"].median() for L in lv], marker="o", lw=2, color=COL[c], label=f"{SHORT[c]} (n={len(s)})")
    ax[1].plot(range(3), [s[f"p10_{L}_median_first_ms"].median() for L in lv], marker="o", lw=2, color=COL[c], label=SHORT[c])
for a_ in ax[:2]:
    a_.set_xticks(range(3), lv)
ax[0].set_xlabel("light level, within-session rank")
ax[0].set_ylabel(r"median chance-corrected $\hat\rho$")
ax[0].legend(fontsize=8)
label(ax[0], "A  Reliability vs light level")
ax[1].set_xlabel("light level, within-session rank")
ax[1].set_ylabel("median first-spike latency (ms)")
label(ax[1], "B  Latency vs light level")
for c in CRES:
    s = d[(d.cre_line == c) & (d.train_n > 0)]
    if len(s) < 3:
        continue
    K = s[[f"train_k{j}" for j in range(1, 11)]].sum().values
    N = s.train_n.sum()
    p0 = np.sum(s.train_n * (1 - np.exp(-s.train_lam0_hz * 0.008))) / N
    pr = K / N
    se = np.sqrt(pr * (1 - pr) / N)
    ax[2].errorbar(range(1, 11), pr, yerr=1.96 * se, marker="o", lw=2, color=COL[c], capsize=2,
                   label=f"{SHORT[c]} ({len(s)} units)")
    ax[2].axhline(p0, color=COL[c], lw=1, ls=":")
ax[2].set_xlabel("pulse in 10-Hz train (dotted: chance level)")
ax[2].set_ylabel("P(spike in +1 to +9 ms), pooled")
ax[2].set_xticks(range(1, 11))
ax[2].legend(fontsize=8)
label(ax[2], "C  10-Hz train, light-activated units")
save(fig, "R5_dose_and_trains")

# ---------------------------------------------------------------- R6 depth & connectivity
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
dep = pd.read_csv(T / "driven_fraction_by_depth.csv")
for c in CRES:
    s = dep[dep.cre_line == c]
    ax[0].plot(range(len(s)), s["mean"], marker="o", lw=2, color=COL[c], label=SHORT[c])
    ticks = s.depth_bin.tolist()
ax[0].set_xticks(range(len(ticks)), [t.replace("(", "").replace("]", "").replace(", ", "-") for t in ticks])
ax[0].set_xlabel("cortical depth along shank (um)")
ax[0].set_ylabel("fraction light-activated")
ax[0].legend(fontsize=8)
label(ax[0], "A  Light-activated units by depth")
cs = pd.read_csv(T / "ccg_summary.csv") if (T / "ccg_summary.csv").exists() else None
if cs is not None and len(cs):
    x = np.arange(len(cs))
    ax[1].bar(x - 0.2, cs.inh_frac, width=0.38, color=[COL.get(c, NULLC) for c in cs.cre_line], label="b after a (+0.8 to +4 ms)")
    ax[1].bar(x + 0.2, cs.inh_frac_anticausal, width=0.38, color=NULLC, label="control: b before a")
    ax[1].set_xticks(x, [f"{c.split('-')[0]}\n{n} pairs, {u} ref. units" for c, n, u in zip(cs.cre_line, cs.pairs, cs.ref_units)])
    ax[1].set_ylabel("fraction of pairs with short-latency trough")
    ax[1].legend(fontsize=8)
    ax[1].grid(axis="x", visible=False)
label(ax[1], "B  Putative inhibition from light-activated units")
save(fig, "R6_depth_and_connectivity")
print("figures written to", F)
