"""
visualization.py
================
Publication-quality figures and diagnostics for optotagging research pipeline.

Figures generated:
- Figure 1: Complete pipeline schematic
- Figure 2: Representative direct, indirect/uncertain, and non-responsive units
- Figure 3: Population feature distributions
- Figure 4: Rule-based versus multifeature model performance
- Figure 5: Session-held-out generalization vs data leakage
- Figure 6: Feature ablation study
- Figure 7: Probability calibration & uncertainty analysis
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "not light responsive",
    "light-responsive / indirect or uncertain",
    "putatively directly optotagged"
]

# Set publication style
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def plot_unit_raster_and_psth(
    aligned_spikes: List[np.ndarray],
    unit_id: int,
    ref_class: str,
    output_path: Optional[str] = None,
    window_s: Tuple[float, float] = (-0.020, 0.030),
    bin_size_s: float = 0.001,
    stim_duration_s: float = 0.010
):
    """
    Generate raster plot and PSTH for a single unit relative to stimulus onset.
    """
    fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(6, 5), sharex=True,
                                             gridspec_kw={"height_ratios": [2.5, 1.5]})
    
    n_trials = len(aligned_spikes)
    w_start, w_stop = window_s
    
    # Stimulus shading: 0 to 10 ms (0.000 to 0.010 s)
    for ax in (ax_raster, ax_psth):
        ax.axvspan(0, stim_duration_s * 1000.0, color="#64B5F6", alpha=0.3, label="10-ms Light Pulse")
        ax.axvline(0, color="#1976D2", linestyle="--", linewidth=1.0)
        ax.axvline(stim_duration_s * 1000.0, color="#1976D2", linestyle="--", linewidth=1.0)
        
    # Raster
    all_spike_times = []
    for trial_idx, spks in enumerate(aligned_spikes):
        if len(spks) > 0:
            spks_ms = spks * 1000.0
            ax_raster.scatter(spks_ms, np.full_like(spks_ms, trial_idx), s=4, color="#212121", alpha=0.8, edgecolors="none")
            all_spike_times.extend(spks)
            
    ax_raster.set_ylabel("Trial Number")
    ax_raster.set_ylim(-1, n_trials)
    ax_raster.set_title(f"Unit {unit_id} ({ref_class})", fontweight="bold")
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    
    # PSTH
    bins = np.arange(w_start, w_stop + bin_size_s, bin_size_s)
    bins_ms = bins * 1000.0
    if len(all_spike_times) > 0:
        counts, _ = np.histogram(all_spike_times, bins=bins)
        # Convert to firing rate (Hz) = counts / (n_trials * bin_size)
        firing_rate = counts / (max(1, n_trials) * bin_size_s)
    else:
        firing_rate = np.zeros(len(bins) - 1)
        
    bin_centers_ms = (bins_ms[:-1] + bins_ms[1:]) / 2.0
    ax_psth.plot(bin_centers_ms, firing_rate, color="#D32F2F", linewidth=1.5)
    ax_psth.fill_between(bin_centers_ms, 0, firing_rate, color="#EF9A9A", alpha=0.5)
    
    # Mark artifact guard zone [+0, +1ms)
    ax_psth.axvspan(0, 1.0, color="gray", alpha=0.25, hatch="//", label="Onset Guard [0, 1ms)")
    
    ax_psth.set_xlabel("Time from Light Onset (ms)")
    ax_psth.set_ylabel("Rate (Hz)")
    ax_psth.set_xlim(w_start * 1000.0, w_stop * 1000.0)
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig


def generate_figure1_pipeline_schematic(output_path: str):
    """
    Generate Figure 1: Pipeline schematic showing data flow, artifact guards,
    multifeature extraction, and session-held-out validation.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis("off")
    
    # Draw schematic blocks
    blocks = [
        {"x": 0.05, "y": 0.70, "w": 0.24, "h": 0.22, "title": "1. Allen Neuropixels Ingestion",
         "items": ["• Visual Coding ecephys NWB", "• Opto stimulation epochs", "• 10-ms pulses (primary)", "• Secondary trains & ramps"]},
        {"x": 0.38, "y": 0.70, "w": 0.24, "h": 0.22, "title": "2. Artifact Control & Alignment",
         "items": ["• Guard: [0, 1ms) & [9, 11ms]", "• Evoked: [+1, +9ms]", "• Sham: [-18, -10ms]", "• Photoelectric spike check"]},
        {"x": 0.71, "y": 0.70, "w": 0.24, "h": 0.22, "title": "3. Multifeature Engineering",
         "items": ["• Median latency & IQR jitter", "• Trial reliability (>=1 spk)", "• Modulation ratio (+1 regularized)", "• Permutation p & Cohen's d"]},
        {"x": 0.05, "y": 0.20, "w": 0.24, "h": 0.24, "title": "4. Operational Labeling",
         "items": ["• Putatively direct", "• Indirect / uncertain", "• Not light responsive", "• Sensitivity sweep (27 combos)"]},
        {"x": 0.38, "y": 0.20, "w": 0.24, "h": 0.24, "title": "5. Leakage-Free Validation",
         "items": ["• Session-held-out (GroupKFold)", "• Specimen-held-out", "• Prevent intra-session leakage", "• Stratified unit baseline"]},
        {"x": 0.71, "y": 0.20, "w": 0.24, "h": 0.24, "title": "6. Uncertainty & Calibration",
         "items": ["• Multiclass probabilities", "• Brier score & ECE", "• Shannon prediction entropy", "• Biological uncertainty bounds"]}
    ]
    
    colors = ["#E3F2FD", "#EDE7F6", "#E8F5E9", "#FFF3E0", "#FCE4EC", "#E0F7FA"]
    border_colors = ["#1976D2", "#512DA8", "#388E3C", "#F57C00", "#C2185B", "#0097A7"]
    
    for i, b in enumerate(blocks):
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=colors[i], edgecolor=border_colors[i], linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(b["x"] + 0.02, b["y"] + b["h"] - 0.04, b["title"],
                fontsize=11, fontweight="bold", color=border_colors[i])
        y_text = b["y"] + b["h"] - 0.08
        for itm in b["items"]:
            ax.text(b["x"] + 0.03, y_text, itm, fontsize=9, color="#212121")
            y_text -= 0.035
            
    # Connect arrows
    arrow_props = dict(arrowstyle="->", lw=2.0, color="#424242")
    # 1 -> 2
    ax.annotate("", xy=(0.38, 0.81), xytext=(0.29, 0.81), arrowprops=arrow_props)
    # 2 -> 3
    ax.annotate("", xy=(0.71, 0.81), xytext=(0.62, 0.81), arrowprops=arrow_props)
    # 3 down to 4
    ax.annotate("", xy=(0.17, 0.44), xytext=(0.83, 0.70),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#757575", connectionstyle="arc3,rad=-0.3"))
    # 4 -> 5
    ax.annotate("", xy=(0.38, 0.32), xytext=(0.29, 0.32), arrowprops=arrow_props)
    # 5 -> 6
    ax.annotate("", xy=(0.71, 0.32), xytext=(0.62, 0.32), arrowprops=arrow_props)
    
    ax.set_title("Figure 1: Multifeature Automated Optotagging Pipeline Architecture",
                 fontsize=13, fontweight="bold", pad=20)
                 
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 1 to %s and %s", output_path, pdf_path)


def generate_figure3_feature_distributions(df: pd.DataFrame, output_path: str):
    """
    Figure 3: Population feature distributions across reference classes:
    Latency, reliability, modulation ratio, baseline vs evoked rate.
    """
    clean_df = df[df["reference_class"].isin(CLASS_NAMES)].copy()
    if clean_df.empty:
        return
        
    palette = {
        "putatively directly optotagged": "#2E7D32",
        "light-responsive / indirect or uncertain": "#F57C00",
        "not light responsive": "#757575"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    
    # 1. Median Latency
    valid_lat = clean_df.dropna(subset=["median_latency_ms"])
    if not valid_lat.empty:
        sns.histplot(data=valid_lat, x="median_latency_ms", hue="reference_class",
                     palette=palette, bins=25, kde=True, ax=axes[0, 0], element="step")
    axes[0, 0].axvline(8.0, color="red", linestyle="--", label="8ms Threshold")
    axes[0, 0].set_title("A. First-Spike Latency Distribution")
    axes[0, 0].set_xlabel("Median Latency (ms)")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8)
    
    # 2. Trial Reliability
    sns.histplot(data=clean_df, x="trial_reliability", hue="reference_class",
                 palette=palette, bins=20, kde=True, ax=axes[0, 1], element="step")
    axes[0, 1].axvline(0.30, color="red", linestyle="--", label="0.30 Threshold")
    axes[0, 1].set_title("B. Trial Reliability Distribution")
    axes[0, 1].set_xlabel("Reliability (P(>=1 evoked spike))")
    axes[0, 1].legend(loc="upper right", frameon=False, fontsize=8)
    
    # 3. Modulation Ratio
    sns.boxplot(data=clean_df, x="reference_class", y="modulation_ratio",
                palette=palette, ax=axes[1, 0], showfliers=False)
    axes[1, 0].axhline(2.0, color="red", linestyle="--", label="2.0 Threshold")
    axes[1, 0].set_title("C. Modulation Ratio by Class")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_xticklabels(["Non-resp", "Indirect", "Direct"], rotation=15)
    axes[1, 0].set_ylabel("Modulation Ratio")
    
    # 4. Baseline vs Evoked Firing Rate
    sns.scatterplot(data=clean_df, x="baseline_rate", y="evoked_rate", hue="reference_class",
                    palette=palette, alpha=0.7, s=35, ax=axes[1, 1])
    max_rate = max(clean_df["baseline_rate"].max(), clean_df["evoked_rate"].max(), 10)
    axes[1, 1].plot([0, max_rate], [0, max_rate], color="black", linestyle=":", label="Unity")
    axes[1, 1].set_title("D. Baseline vs Evoked Firing Rate")
    axes[1, 1].set_xlabel("Baseline Rate (Hz)")
    axes[1, 1].set_ylabel("Evoked Rate (Hz)")
    axes[1, 1].set_xlim(0, max_rate * 1.05)
    axes[1, 1].set_ylim(0, max_rate * 1.05)
    axes[1, 1].legend(loc="lower right", frameon=False, fontsize=8)
    
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 3 to %s", output_path)


def generate_figure4_model_comparisons(ml_summary_df: pd.DataFrame, output_path: str):
    """
    Figure 4: Performance comparison between Rule-Based threshold heuristic
    and Multifeature Machine Learning Classifiers.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    metrics = ["Balanced Accuracy", "Macro F1", "AUROC (OVR)", "AUPRC"]
    available_metrics = [m for m in metrics if m in ml_summary_df.columns]
    
    melted = ml_summary_df.melt(
        id_vars=["Model Classifier", "Feature Set"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model Classifier",
        palette="viridis",
        ax=ax
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation Performance")
    ax.set_title("Figure 4: Rule-Based vs Multifeature Model Performance", fontweight="bold")
    ax.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 4 to %s", output_path)


def generate_figure5_cross_session_generalization(
    random_vs_session_df: pd.DataFrame,
    output_path: str
):
    """
    Figure 5: Session-Held-Out Generalization vs Within-Session Data Leakage.
    Contrasts random unit split (with intra-session pseudo-replication)
    against rigorous Session-Held-Out (GroupKFold) cross-validation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    # 1. Balanced Accuracy
    sns.barplot(
        data=random_vs_session_df,
        x="Model Classifier",
        y="Balanced Accuracy",
        hue="Evaluation Strategy",
        palette=["#E53935", "#1E88E5"],
        ax=ax1
    )
    ax1.set_ylim(0.5, 1.05)
    ax1.set_title("A. Balanced Accuracy Under Leakage Control")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # 2. Brier Score (lower is better calibrated)
    sns.barplot(
        data=random_vs_session_df,
        x="Model Classifier",
        y="Brier Score",
        hue="Evaluation Strategy",
        palette=["#E53935", "#1E88E5"],
        ax=ax2
    )
    ax2.set_title("B. Multi-Class Brier Score (Calibration Error)")
    ax2.set_ylabel("Brier Score (Lower is Better)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 5 to %s", output_path)


def generate_figure6_feature_ablation(ablation_df: pd.DataFrame, output_path: str):
    """
    Figure 6: Progressive Feature Ablation across Models A through F.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    
    sns.lineplot(
        data=ablation_df,
        x="Feature Set",
        y="Macro F1",
        hue="Model Classifier",
        marker="o",
        markersize=8,
        linewidth=2.0,
        ax=ax
    )
    
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Session-Held-Out Macro F1")
    ax.set_xlabel("Feature Ablation Set")
    ax.set_title("Figure 6: Feature Ablation Contribution to Classification", fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="Classifier", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 6 to %s", output_path)


def generate_figure7_calibration_and_uncertainty(oof_df: pd.DataFrame, output_path: str):
    """
    Figure 7: Probability Calibration and Classification Uncertainty.
    - Panel A: Multiclass predicted probability distributions
    - Panel B: Shannon Prediction Entropy by True Class
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    
    # 1. P(direct) distribution across classes
    prob_direct_col = [c for c in oof_df.columns if "prob_" in c and "direct" in c]
    if prob_direct_col:
        p_col = prob_direct_col[0]
        sns.boxplot(
            data=oof_df,
            x="reference_class",
            y=p_col,
            palette={"putatively directly optotagged": "#2E7D32",
                     "light-responsive / indirect or uncertain": "#F57C00",
                     "not light responsive": "#757575"},
            ax=ax1
        )
        ax1.set_ylabel("Predicted P(Direct)")
        ax1.set_xlabel("")
        ax1.set_xticklabels(["Non-resp", "Indirect", "Direct"], rotation=15)
        ax1.set_title("A. Direct Class Probability Assignment")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        
    # 2. Shannon Prediction Entropy
    if "prediction_entropy" in oof_df.columns:
        sns.histplot(
            data=oof_df,
            x="prediction_entropy",
            hue="reference_class",
            element="step",
            bins=20,
            palette={"putatively directly optotagged": "#2E7D32",
                     "light-responsive / indirect or uncertain": "#F57C00",
                     "not light responsive": "#757575"},
            ax=ax2
        )
        ax2.set_xlabel("Prediction Entropy (bits)")
        ax2.set_title("B. Classification Uncertainty Distribution")
        ax2.legend(loc="upper right", frameon=False, fontsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Figure 7 to %s", output_path)
