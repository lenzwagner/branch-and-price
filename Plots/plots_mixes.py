"""
plots_mixes.py
==============
Generates three publication-ready plots from the mixes experiment results:

  1. Grouped Boxplots  – "Performance Gap" analysis (HOM vs. HYB avg. LoS per mix)
  2. Heatmap           – "Churn Penalty" matrix (topology × mix → relative LoS reduction)
  3. Stacked Bar Chart – Workload substitution (therapist vs. AI session share per mix)

Data source: results/mixes/results/results_mixes.xlsx
"""

import os
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "results", "mixes", "results", "results_mixes.xlsx")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "mixes", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colours & style constants (consistent with plots.py palette)
# ---------------------------------------------------------------------------
COLOR_HOM = "#FFC20A"        # yellow – HOM baseline
COLOR_HYB = "#0C7BDC"        # blue   – HYB AI-hybrid
COLOR_HUMAN = "#4A90D9"      # therapist sessions
COLOR_AI = "#E84C3D"         # AI sessions

TOPOLOGY_COLORS = {
    "Sigmoidal": "#9B59B6",
    "Linear":    "#2ECC71",
    "Exponential": "#E67E22",
}

MIX_LABELS = {
    "(0.7, 0.2, 0.1)":  "High-Turnover\n(70 % short-stay)",
    "(0.33, 0.34, 0.33)": "High-Complexity\n(33 % long-stay)",
}
MIX_ORDER = ["High-Turnover\n(70 % short-stay)", "High-Complexity\n(33 % long-stay)"]

LEARN_LABELS = {
    "sigmoid": "Sigmoidal",
    "lin":     "Linear",
    "exp":     "Exponential",
}


# ---------------------------------------------------------------------------
# Helper: _save_fig
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, basename: str) -> None:
    """Save figure as PDF and SVG into OUTPUT_DIR."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{basename}.pdf")
    svg_path = os.path.join(OUTPUT_DIR, f"{basename}.svg")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"  ✓  {basename}.pdf  /  {basename}.svg  →  {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    print(f"Loading data from:\n  {path}")
    df = pd.read_excel(path)

    # Keep only rows with proper severity_mix tuples
    df = df[df["severity_mix"].astype(str).str.startswith("(")].copy()

    # Numeric conversions
    for col in ["focus_avg_los", "post_avg_los",
                 "focus_period_N_human", "focus_period_N_AI",
                 "post_period_N_human",  "post_period_N_AI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived columns
    df["mix_label"] = df["severity_mix"].astype(str).map(MIX_LABELS)

    df["Model"] = df["OnlyHuman"].map({1: "HOM (Baseline)", 0: "HYB (AI-Hybrid)"})

    df["learn_label"] = df["learn_type"].map(LEARN_LABELS)

    # Combined session counts (focus + post period)
    df["N_human_total"] = df["focus_period_N_human"] + df["post_period_N_human"]
    df["N_AI_total"]    = df["focus_period_N_AI"]    + df["post_period_N_AI"]
    df["N_total"]       = df["N_human_total"] + df["N_AI_total"]
    df["AI_share_pct"]  = 100.0 * df["N_AI_total"] / df["N_total"]

    print(f"  Rows loaded: {len(df)}  |  Mixes: {sorted(df['mix_label'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Plot 1 – Grouped Boxplots: Performance Gap
# ---------------------------------------------------------------------------
def plot_performance_gap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Grouped boxplots showing average LoS (focus period) for HOM vs. HYB,
    grouped by patient mix.  Illustrates the widening performance gap in
    high-complexity settings (the 'Learning Dividend').
    """
    plot_df = df[["mix_label", "Model", "focus_avg_los"]].dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.boxplot(
        data=plot_df,
        x="mix_label",
        y="focus_avg_los",
        hue="Model",
        order=MIX_ORDER,
        hue_order=["HOM (Baseline)", "HYB (AI-Hybrid)"],
        palette={"HOM (Baseline)": COLOR_HOM, "HYB (AI-Hybrid)": COLOR_HYB},
        width=0.55,
        linewidth=1.2,
        flierprops=dict(marker="o", markersize=3, linestyle="none", alpha=0.5),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="darkgreen",
                       markeredgecolor="darkgreen", markersize=5),
        ax=ax,
    )

    # Annotate mean gap per mix
    for i, mix in enumerate(MIX_ORDER):
        subset = plot_df[plot_df["mix_label"] == mix]
        hom_mean = subset[subset["Model"] == "HOM (Baseline)"]["focus_avg_los"].mean()
        hyb_mean = subset[subset["Model"] == "HYB (AI-Hybrid)"]["focus_avg_los"].mean()
        gap = hom_mean - hyb_mean
        ax.text(i, ax.get_ylim()[1] * 0.97,
                f"Δ = {gap:.1f} d",
                ha="center", va="top", fontsize=9,
                color="darkgreen", fontweight="bold")

    ax.set_xlabel("Patient Mix", fontsize=12, labelpad=8)
    ax.set_ylabel("Avg. Length of Stay (Days)", fontsize=12)
    ax.set_title(
        "Performance Gap: HOM vs. HYB by Patient Mix\n"
        r"(Performance gap widens in High-Complexity settings → 'Learning Dividend')",
        fontsize=11, pad=12
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_HOM, edgecolor="grey", label="HOM (Baseline)"),
        mpatches.Patch(facecolor=COLOR_HYB, edgecolor="grey", label="HYB (AI-Hybrid)"),
        plt.scatter([], [], marker="D", color="darkgreen", s=30, label="Group Mean"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save:
        _save_fig(fig, "plot1_performance_gap_boxplot")
    return fig


# ---------------------------------------------------------------------------
# Plot 2 – Heatmap: Churn Penalty Matrix
# ---------------------------------------------------------------------------
def plot_churn_penalty_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Heatmap with rows = patient mix, columns = AI topology,
    cell value = relative LoS reduction (%) of HYB vs. HOM.

    Shows the 'Churn Penalty': sigmoidal curves yield little benefit in
    high-turnover settings, while exponential ('plug-and-play') curves
    perform more stably across all mixes.
    """
    # We need HOM baseline per (mix, seed, pttr) and HYB per (mix, learn_type, seed, pttr)
    hom = df[df["OnlyHuman"] == 1][["mix_label", "seed", "pttr", "focus_avg_los"]].copy()
    hom = hom.rename(columns={"focus_avg_los": "hom_los"})

    hyb = df[(df["OnlyHuman"] == 0) & df["learn_label"].notna()][
        ["mix_label", "seed", "pttr", "learn_label", "focus_avg_los"]
    ].copy()
    hyb = hyb.rename(columns={"focus_avg_los": "hyb_los"})

    merged = hyb.merge(hom, on=["mix_label", "seed", "pttr"], how="inner")
    merged["reduction_pct"] = 100.0 * (merged["hom_los"] - merged["hyb_los"]) / merged["hom_los"]

    # Pivot: rows = mix, cols = topology
    pivot = merged.groupby(["mix_label", "learn_label"])["reduction_pct"].mean().reset_index()
    heatmap_df = pivot.pivot(index="mix_label", columns="learn_label", values="reduction_pct")

    # Ensure consistent order
    row_order = [m for m in MIX_ORDER if m in heatmap_df.index]
    col_order = [v for v in ["Exponential", "Linear", "Sigmoidal"] if v in heatmap_df.columns]
    heatmap_df = heatmap_df.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 11, "weight": "bold"},
        cmap="YlGn",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "LoS Reduction vs. Baseline (%)", "shrink": 0.85},
        ax=ax,
        vmin=0,
    )

    ax.set_xlabel("AI Learning Topology", fontsize=12, labelpad=8)
    ax.set_ylabel("Patient Mix", fontsize=12)
    ax.set_title(
        "Churn Penalty Matrix: Relative LoS Reduction by Mix × Topology\n"
        "(Low saturation = 'Churn Penalty'; Exponential topology most robust)",
        fontsize=11, pad=12,
    )

    # Wrap y-tick labels (remove embedded newlines for cleaner look in heatmap)
    ytick_labels = [lbl.get_text().replace("\n", " ") for lbl in ax.get_yticklabels()]
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()

    if save:
        _save_fig(fig, "plot2_churn_penalty_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Plot 3 – Stacked Bar Charts: Workload Substitution
# ---------------------------------------------------------------------------
def plot_workload_substitution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Stacked bar chart showing therapist vs. AI session composition per
    patient mix, split by AI topology (HYB rows only).

    Demonstrates 'Therapist Offloading': in High-Complexity settings the
    system strategically routes long-stay patients to AI, freeing therapist
    capacity for new admissions.
    """
    hyb = df[(df["OnlyHuman"] == 0) & df["learn_label"].notna()].copy()

    # Aggregate mean sessions per (mix, topology)
    agg = (
        hyb.groupby(["mix_label", "learn_label"])[["N_human_total", "N_AI_total"]]
        .mean()
        .reset_index()
    )
    agg["N_total"] = agg["N_human_total"] + agg["N_AI_total"]
    # Normalise to 100 % for share plot
    agg["Human_share"] = 100.0 * agg["N_human_total"] / agg["N_total"]
    agg["AI_share"]    = 100.0 * agg["N_AI_total"]    / agg["N_total"]

    learn_order = ["Exponential", "Linear", "Sigmoidal"]
    mix_order_clean = [m for m in MIX_ORDER if m in agg["mix_label"].values]

    fig, axes = plt.subplots(
        1, len(mix_order_clean),
        figsize=(5.5 * len(mix_order_clean), 5.5),
        sharey=True,
    )
    if len(mix_order_clean) == 1:
        axes = [axes]

    for ax, mix in zip(axes, mix_order_clean):
        subset = agg[agg["mix_label"] == mix].set_index("learn_label").reindex(learn_order)

        x = np.arange(len(learn_order))
        bar_width = 0.55

        bars_human = ax.bar(
            x, subset["Human_share"], bar_width,
            color=COLOR_HUMAN, label="Therapist Sessions",
            edgecolor="white", linewidth=0.8,
        )
        bars_ai = ax.bar(
            x, subset["AI_share"], bar_width,
            bottom=subset["Human_share"],
            color=COLOR_AI, label="AI Sessions",
            edgecolor="white", linewidth=0.8,
        )

        # Annotate AI share inside each bar
        for j, (_, row) in enumerate(subset.iterrows()):
            ai_share = row["AI_share"]
            if not np.isnan(ai_share):
                ax.text(
                    x[j], row["Human_share"] + ai_share / 2.0,
                    f"{ai_share:.1f} %",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(learn_order, fontsize=10)
        ax.set_xlabel("AI Learning Topology", fontsize=11)
        ax.set_ylim(0, 108)
        ax.set_title(mix.replace("\n", " "), fontsize=11, pad=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Session Share (%)", fontsize=12)

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_HUMAN, edgecolor="grey", label="Therapist Sessions"),
        mpatches.Patch(facecolor=COLOR_AI,    edgecolor="grey", label="AI Sessions"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10, frameon=True,
    )
    fig.suptitle(
        "Workload Substitution: Therapist vs. AI Session Share by Mix & Topology\n"
        "(Higher AI share in High-Complexity mix → strategic 'Therapist Offloading')",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save:
        _save_fig(fig, "plot3_workload_substitution_stacked_bar")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    print("\n── Plot 1: Performance Gap Boxplot ──────────────────────────────────")
    fig1 = plot_performance_gap(df, save=True)

    print("\n── Plot 2: Churn Penalty Heatmap ────────────────────────────────────")
    fig2 = plot_churn_penalty_heatmap(df, save=True)

    print("\n── Plot 3: Workload Substitution Stacked Bar ────────────────────────")
    fig3 = plot_workload_substitution(df, save=True)

    print("\nAll plots saved.  Launching interactive view …")
    plt.show()
