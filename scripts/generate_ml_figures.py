"""
generate_ml_figures.py
======================
Generates all 8 publication-grade figures required by Section 27 for TCBB:
- Figure ML1: Model-family performance under specimen-held-out validation.
- Figure ML2: Model performance per specimen.
- Figure ML3: Full-feature vs non-defining-feature performance (Label Circularity).
- Figure ML4: Random-unit vs session-held-out vs specimen-held-out performance (Leakage).
- Figure ML5: Graph vs non-graph models.
- Figure ML6: Real graph vs randomized graph (Degree-Preserving Control).
- Figure ML7: Performance versus graph neighborhood size (k = 3, 5, 10).
- Figure ML8: Feature-family ablation across model families.

Saves high-resolution 300 DPI PNG and vector PDF in `results/ml_final/figures/` and `results/figures/`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 1.75,
    "axes.spines.top": False,
    "axes.spines.right": False
})

out_dir = Path("results/ml_final/figures")
out_dir.mkdir(parents=True, exist_ok=True)
global_fig_dir = Path("results/figures")
global_fig_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("results/ml_final")


def fig_ml1_model_family_comparison():
    df = pd.read_csv(data_dir / "model_comparison.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Sort by Balanced Accuracy
    df_sorted = df.sort_values("mean_BA", ascending=True)
    
    palette = sns.color_palette("mako", len(df_sorted))
    y_pos = np.arange(len(df_sorted))
    
    # Plot 1: Balanced Accuracy
    ax1.barh(y_pos, df_sorted["mean_BA"], color=palette, edgecolor="black", alpha=0.85, height=0.65)
    ax1.axvline(0.5, color="#d32f2f", linestyle="--", linewidth=1.5, label="Chance Level (0.50)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace("_", " ").title() for m in df_sorted["model"]])
    ax1.set_xlabel("Mean Balanced Accuracy")
    ax1.set_title("A. Specimen-Held-Out Balanced Accuracy", fontweight="bold", pad=12)
    ax1.set_xlim(0.4, 1.05)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")
    
    for i, v in enumerate(df_sorted["mean_BA"]):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="semibold")
        
    # Plot 2: AUROC and AUPRC
    w = 0.35
    ax2.barh(y_pos + w/2, df_sorted["AUROC"], height=w, label="AUROC", color="#1976d2", alpha=0.85, edgecolor="black")
    ax2.barh(y_pos - w/2, df_sorted["AUPRC"], height=w, label="AUPRC", color="#388e3c", alpha=0.85, edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m.replace("_", " ").title() for m in df_sorted["model"]])
    ax2.set_xlabel("Discrimination Score")
    ax2.set_title("B. Held-Out Discrimination (AUROC vs AUPRC)", fontweight="bold", pad=12)
    ax2.set_xlim(0.0, 1.05)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left")
    
    plt.suptitle("Figure ML1: Model Family Performance under Specimen-Held-Out Validation", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml1_model_family_comparison.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml1_model_family_comparison.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml1_model_family_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML1")


def fig_ml2_per_specimen_performance():
    df = pd.read_csv(data_dir / "specimen_loso_results.csv")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    models = df["model"].unique()
    x = np.arange(len(models))
    w = 0.35
    
    spec_707 = df[df["test_specimen"] == 707296982].sort_values("model")
    spec_739 = df[df["test_specimen"] == 739783171].sort_values("model")
    
    b1 = ax.bar(x - w/2, spec_707["balanced_accuracy"], width=w, label="Specimen 707296982 (N=444, Pos=7 / 1.58%)",
                color="#0288d1", edgecolor="black", alpha=0.85)
    b2 = ax.bar(x + w/2, spec_739["balanced_accuracy"], width=w, label="Specimen 739783171 (N=501, Pos=1 / 0.20%)",
                color="#f57c00", edgecolor="black", alpha=0.85)
                
    ax.axhline(0.5, color="#d32f2f", linestyle="--", linewidth=1.5, label="Chance Level (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in spec_707["model"]], rotation=30, ha="right")
    ax.set_ylabel("Held-Out Balanced Accuracy")
    ax.set_ylim(0.4, 1.08)
    ax.set_title("Figure ML2: Per-Specimen Generalization across 9 Evaluated Architectures", fontweight="bold", pad=14)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml2_per_specimen_performance.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml2_per_specimen_performance.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml2_per_specimen_performance.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML2")


def fig_ml3_label_circularity():
    df = pd.read_csv(data_dir / "label_circularity.csv")
    summary = df.groupby(["model", "setting"]).agg({"balanced_accuracy": "mean", "auroc": "mean"}).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    models = sorted(summary["model"].unique())
    x = np.arange(len(models))
    w = 0.35
    
    set_a = summary[summary["setting"] == "Setting A (Full Features)"].sort_values("model")
    set_b = summary[summary["setting"] == "Setting B (Non-Defining Features)"].sort_values("model")
    
    # Plot A: Balanced Accuracy
    ax1.bar(x - w/2, set_a["balanced_accuracy"], width=w, label="Setting A: Full Features (Heuristic Reconstruction)",
            color="#2e7d32", edgecolor="black", alpha=0.85)
    ax1.bar(x + w/2, set_b["balanced_accuracy"], width=w, label="Setting B: Non-Defining Features (Independent Signal)",
            color="#c62828", edgecolor="black", alpha=0.85)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Chance Level (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=35, ha="right")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_title("A. Balanced Accuracy under Feature Withholding", fontweight="bold", pad=12)
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)
    
    # Plot B: AUROC
    ax2.bar(x - w/2, set_a["auroc"], width=w, label="Setting A (Full)", color="#1565c0", edgecolor="black", alpha=0.85)
    ax2.bar(x + w/2, set_b["auroc"], width=w, label="Setting B (Non-Defining)", color="#e65100", edgecolor="black", alpha=0.85)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Chance Level (0.50)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=35, ha="right")
    ax2.set_ylabel("AUROC")
    ax2.set_title("B. AUROC under Feature Withholding", fontweight="bold", pad=12)
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)
    
    plt.suptitle("Figure ML3: Label-Circularity Audit (Full Heuristic vs Non-Defining Features)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml3_label_circularity.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml3_label_circularity.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml3_label_circularity.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML3")


def fig_ml4_leakage_comparison():
    df = pd.read_csv(data_dir / "leakage_audit.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    models = df["model"].values
    x = np.arange(len(models))
    w = 0.35
    
    # Plot 1: Random Unit vs Specimen LOSO BA
    ax1.bar(x - w/2, df["random_unit_BA"], width=w, label="Random Unit Split (Leakage-Prone)",
            color="#d32f2f", edgecolor="black", alpha=0.85)
    ax1.bar(x + w/2, df["specimen_loso_BA"], width=w, label="Specimen LOSO (Zero-Leakage)",
            color="#1976d2", edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_title("A. Balanced Accuracy across Validation Regimes", fontweight="bold", pad=12)
    ax1.set_ylim(0.4, 1.08)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left")
    
    # Plot 2: Inflation Delta
    colors = ["#f57c00" if v > 0.05 else "#388e3c" for v in df["delta_BA_inflation"]]
    bars = ax2.bar(x, df["delta_BA_inflation"] * 100, width=0.5, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax2.set_ylabel("Performance Inflation (% points)")
    ax2.set_title("B. Quantified Intra-Recording Leakage Inflation (Δ BA)", fontweight="bold", pad=12)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 0.5, f"+{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.suptitle("Figure ML4: Impact of Intra-Recording Data Leakage on Model Validation", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml4_leakage_comparison.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml4_leakage_comparison.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml4_leakage_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML4")


def fig_ml5_graph_vs_nongraph():
    df = pd.read_csv(data_dir / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Classify models into Independent vs Graph
    df["family_type"] = df["model"].apply(lambda m: "Graph Neural Network" if m in ["gcn", "graphsage", "gat"] else "Independent Unit")
    
    palette = {"Independent Unit": "#0288d1", "Graph Neural Network": "#7b1fa2"}
    
    sns.scatterplot(
        data=df, x="mean_BA", y="macro_F1", hue="family_type", size="AUROC",
        sizes=(120, 350), palette=palette, edgecolor="black", linewidth=1.5, alpha=0.9, ax=ax
    )
    
    for _, row in df.iterrows():
        ax.annotate(
            row["model"].replace("_", " ").title(),
            (row["mean_BA"], row["macro_F1"]),
            xytext=(6, 5), textcoords="offset points", fontsize=9, fontweight="semibold"
        )
        
    ax.axvline(0.5, color="#d32f2f", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0.5, color="#d32f2f", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Mean Held-Out Balanced Accuracy")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Figure ML5: Graph Neural Networks vs Independent-Unit Physiological Baselines", fontweight="bold", pad=14)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml5_graph_vs_nongraph.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml5_graph_vs_nongraph.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml5_graph_vs_nongraph.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML5")


def fig_ml6_graph_shuffle_control():
    df = pd.read_csv(data_dir / "graph_shuffle_control.csv")
    summary = df.groupby("graph_topology").agg({"balanced_accuracy": "mean", "auroc": "mean", "auprc": "mean"}).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    topos = summary["graph_topology"].values
    x = np.arange(len(topos))
    
    # Plot 1: Balanced Accuracy
    colors = ["#2e7d32", "#c62828"]
    bars1 = ax1.bar(x, summary["balanced_accuracy"], width=0.5, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Real Spatial Graph\n(k=5, CCF Coordinates)", "Randomized Graph\n(Degree-Preserving Swap)"])
    ax1.set_ylabel("Held-Out Balanced Accuracy")
    ax1.set_title("A. Balanced Accuracy under Topology Shuffling", fontweight="bold", pad=12)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    # Plot 2: AUROC vs AUPRC
    w = 0.3
    ax2.bar(x - w/2, summary["auroc"], width=w, label="AUROC", color="#1565c0", edgecolor="black", alpha=0.85)
    ax2.bar(x + w/2, summary["auprc"], width=w, label="AUPRC", color="#f57c00", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Real Spatial Graph\n(k=5, CCF Coordinates)", "Randomized Graph\n(Degree-Preserving Swap)"])
    ax2.set_ylabel("Score")
    ax2.set_title("B. Discrimination (AUROC vs AUPRC)", fontweight="bold", pad=12)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.suptitle("Figure ML6: Real Neuropixels Geometry vs Randomized Graph Null Control", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml6_graph_shuffle_control.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml6_graph_shuffle_control.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml6_graph_shuffle_control.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML6")


def fig_ml7_graph_neighborhood_k():
    df = pd.read_csv(data_dir / "graph_ablation.csv")
    knn_df = df[df["k_neighbors"].isin([3, 5, 10])].copy()
    
    knn_df["model"] = knn_df["configuration"].apply(lambda c: "GCN" if "GCN" in c else "GraphSAGE")
    summary = knn_df.groupby(["model", "k_neighbors"]).agg({
        "balanced_accuracy": "mean", "macro_f1": "mean", "auroc": "mean", "auprc": "mean"
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Plot 1: Balanced Accuracy vs k
    for m, col, mk in [("GCN", "#7b1fa2", "o"), ("GraphSAGE", "#0288d1", "s")]:
        sub = summary[summary["model"] == m].sort_values("k_neighbors")
        ax1.plot(sub["k_neighbors"], sub["balanced_accuracy"], marker=mk, markersize=8,
                 linewidth=2.2, label=m, color=col)
    ax1.set_xlabel("Neighborhood Size k (Nearest Spatial Neighbors)")
    ax1.set_ylabel("Held-Out Balanced Accuracy")
    ax1.set_title("A. Balanced Accuracy vs k-NN Radius", fontweight="bold", pad=12)
    ax1.set_xticks([3, 5, 10])
    ax1.set_ylim(0.5, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="center right")
    
    # Plot 2: Macro F1 vs k
    for m, col, mk in [("GCN", "#7b1fa2", "o"), ("GraphSAGE", "#0288d1", "s")]:
        sub = summary[summary["model"] == m].sort_values("k_neighbors")
        ax2.plot(sub["k_neighbors"], sub["macro_f1"], marker=mk, markersize=8,
                 linewidth=2.2, label=m, color=col)
    ax2.set_xlabel("Neighborhood Size k (Nearest Spatial Neighbors)")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("B. Macro F1 Score vs k-NN Radius", fontweight="bold", pad=12)
    ax2.set_xticks([3, 5, 10])
    ax2.set_ylim(0.4, 0.8)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center right")
    
    plt.suptitle("Figure ML7: Graph Neural Network Performance as a Function of Spatial Neighborhood Size (k)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml7_graph_neighborhood_k.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml7_graph_neighborhood_k.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml7_graph_neighborhood_k.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML7")


def fig_ml8_feature_family_ablation():
    df = pd.read_csv(data_dir / "feature_ablation.csv")
    summary = df.groupby(["model", "ablated_family"]).agg({"balanced_accuracy": "mean"}).reset_index()
    
    # Compute pivot table
    pivot = summary.pivot(index="model", columns="ablated_family", values="balanced_accuracy")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu_r", cbar_kws={"label": "Held-Out Balanced Accuracy"},
                linewidths=0.5, linecolor="grey", ax=ax)
    ax.set_ylabel("Model Architecture")
    ax.set_xlabel("Ablated Physiological Feature Family")
    ax.set_yticklabels([m.replace("_", " ").title() for m in pivot.index], rotation=0)
    ax.set_title("Figure ML8: Feature-Family Ablation across Diverse Model Families (LOFFO)", fontweight="bold", pad=14)
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ml8_feature_family_ablation.png", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ml8_feature_family_ablation.pdf", bbox_inches="tight")
    plt.savefig(global_fig_dir / "fig_ml8_feature_family_ablation.png", bbox_inches="tight")
    plt.close()
    print("Saved Figure ML8")


if __name__ == "__main__":
    fig_ml1_model_family_comparison()
    fig_ml2_per_specimen_performance()
    fig_ml3_label_circularity()
    fig_ml4_leakage_comparison()
    fig_ml5_graph_vs_nongraph()
    fig_ml6_graph_shuffle_control()
    fig_ml7_graph_neighborhood_k()
    fig_ml8_feature_family_ablation()
    print("All 8 ML figures generated successfully!")
