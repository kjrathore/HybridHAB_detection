"""
Regression Plots — reads CSVs from reg_train.py output
=======================================================
Run after reg_train.py.

Figures produced:
  Fig 1: R² per species — 2 subplots (spectral only / +ROMS), colored by AC.
          Caption: "Mean R² per species across atmospheric correction pipelines.
                   (A) Spectral features only; (B) spectral + ROMS."

  Fig 2: R² boxplot — per AC, species on x-axis, two boxes (_only vs _roms).
          Caption: "Distribution of R² scores across 50 CV folds per species
                   for {ac}. Dark: spectral only; light: spectral + ROMS."

  Fig 3: R² boxplot — cross-AC comparison per track.
          Caption: "Cross-pipeline R² comparison (50-fold CV) per species ({track})."

  Fig 4: ΔR² heatmap (roms − only) per AC.
          Caption: "Change in macro-averaged R² when ROMS features are added."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_OUT  = Path("ml_outputs/HGB_regr")
CSV_DIR   = BASE_OUT / "csv_results"
PLOTS_DIR = BASE_OUT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ──────────────────────────────────────────────────────────────────────────────
species_df = pd.read_csv(CSV_DIR / "species_comparison.csv")
fold_df    = pd.read_csv(CSV_DIR / "fold_level_results.csv")
global_df  = pd.read_csv(CSV_DIR / "global_comparison.csv")

# Rename Baseline → OC-SAC for display
for df in [species_df, fold_df, global_df]:
    df["ac"] = df["ac"].str.replace("Baseline", "OC-SAC", regex=False)
    df["track"] = df["track"].str.replace("Baseline", "OC-SAC", regex=False)

AC_LIST      = sorted(species_df["ac"].unique())
SPECIES_LIST = sorted(species_df["species"].unique())

SP_LABELS = {
    "Alexandrium_catenella": "Alexandrium\ncatenella",
    "Dinophysis_acuminata":  "Dinophysis\nacuminata",
    "Dinophysis_norvegica":  "Dinophysis\nnorvegia",
    "Karenia":               "Karenia",
    "Pseudo-nitzschia":      "Pseudo-\nnitzschia",
}

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────
FONT = 12
DPI  = 500

RC = {
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.9,
    "axes.grid":         True,
    "grid.color":        '#CCCCCC', #"#E5E5E5",
    "grid.linewidth":    0.7,
    "grid.linestyle":    '-', 
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "text.color":        "#222222",
    "font.family":       "sans-serif",
    "font.size":         FONT,
    "axes.titlesize":    FONT,
    "axes.labelsize":    FONT,
    "legend.fontsize":   FONT - 1,
    "legend.framealpha": 0.9,
    "xtick.labelsize":   FONT - 1,
    "ytick.labelsize":   FONT - 1,
}


# Colors
COLOR_ONLY  = "#909090"   # light gray — spectral + ROMS 
COLOR_ROMS  = "#2C2C2C"   # dark — spectral only

AC_COLORS = {
    "C2RCC":    "#1b6ca8",
    "ACOLITE":  "#c75000",
    "OC-SAC": "#2e8b57",
}

# ──────────────────────────────────────────────────────────────────────────────
# Fig 1: R² per species — spectral only (A) vs +ROMS (B), colored by AC
# ──────────────────────────────────────────────────────────────────────────────
def fig1_r2_bars():
    tracks_map = {"only": "OC model", "roms": "Hybrid model"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    with plt.rc_context(RC):
        for ax, (track_suffix, track_label) in zip(axes, tracks_map.items()):
            sub   = species_df[species_df["track"].str.endswith(f"_{track_suffix}")]
            n_sp  = len(SPECIES_LIST)
            n_ac  = len(AC_LIST)
            x     = np.arange(n_sp)
            w     = 0.22
            offsets = np.linspace(-(n_ac-1)/2, (n_ac-1)/2, n_ac) * (w + 0.03)

            handles = []
            for j, ac in enumerate(AC_LIST):
                color = AC_COLORS[ac]
                rows  = sub[sub["ac"] == ac]
                means = [rows[rows["species"] == sp]["cv_r2_mean"].values
                         for sp in SPECIES_LIST]
                means = [v[0] if len(v) else np.nan for v in means]
                stds  = [rows[rows["species"] == sp]["cv_r2_std"].values
                         for sp in SPECIES_LIST]
                stds  = [v[0] if len(v) else np.nan for v in stds]

                ax.bar(
                    x + offsets[j], means, w,
                    color=color, alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    yerr=stds, capsize=2,
                    error_kw=dict(elinewidth=0.8, ecolor="#555555"),
                )
                handles.append(mpatches.Patch(facecolor=color, alpha=0.85, label=ac))

            ax.axhline(0, color="#999999", lw=0.7, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([SP_LABELS.get(s, s) for s in SPECIES_LIST],
                               fontsize=FONT - 1)
            ax.set_ylim(-0.1, 1.05)
            ax.set_ylabel("R² (log scale)" if track_suffix == "only" else "",
                          fontsize=FONT)
            ax.set_xlabel("Species", fontsize=FONT)
            ax.text(0.03, 0.97, "A" if track_suffix == "only" else "B",
                    transform=ax.transAxes, fontsize=FONT + 3,
                    fontweight="bold", va="top")
            ax.text(0.97, 0.97, track_label,
                    transform=ax.transAxes, fontsize=FONT - 1,
                    va="top", ha="right", color="#444444")

        axes[1].legend(handles=handles, loc="upper right", fontsize=FONT - 1)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig1_r2_per_species.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  Fig 1 saved: fig1_r2_per_species.png")
        print("  Caption: Mean coefficient of determination (R², log scale) per "
              "species across atmospheric correction pipelines (50-fold CV). "
              "(A) Spectral features only; (B) spectral + ROMS oceanographic "
              "features. Error bars indicate ± one standard deviation.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 2: R² boxplot per AC (_only vs _roms)
# ──────────────────────────────────────────────────────────────────────────────
def fig2_r2_boxplot_per_ac():
    for ac in AC_LIST:
        sub    = fold_df[fold_df["ac"] == ac]
        n_sp   = len(SPECIES_LIST)
        x      = np.arange(n_sp)
        width  = 0.32
        gap    = 0.05

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(max(10, n_sp * 1.5), 5))
            bp_handles = []

            for j, (track_sfx, color, label) in enumerate([
                    ("_only", COLOR_ONLY, "OC model"),
                    ("_roms", COLOR_ROMS, "Hybrid model")]):
                track  = f"{ac}{track_sfx}"
                offset = (j - 0.5) * (width + gap)
                data   = [sub[(sub["track"] == track) &
                              (sub["species"] == sp)]["r2"].dropna().values
                           for sp in SPECIES_LIST]
                ax.boxplot(
                    data,
                    positions=x + offset,
                    widths=width,
                    patch_artist=True,
                    notch=False,
                    showfliers=False,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=color, alpha=0.4,
                                    linestyle="none"),
                    medianprops=dict(color="white"),
                    whiskerprops=dict(color=color, linewidth=1.5),
                    capprops=dict(color=color, linewidth=1.5),
                    boxprops=dict(facecolor=color, color='black', alpha=0.85),
                )
                bp_handles.append(
                    mpatches.Patch(fc=color, alpha=0.85, label=label))

            ax.axhline(0, color="#AAAAAA", lw=0.7, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([SP_LABELS.get(s, s) for s in SPECIES_LIST],
                               fontsize=FONT - 1, ha="center")
            ax.set_xlim(-0.6, n_sp - 0.4)
            ax.set_ylim(0.2, 0.7)
            ax.set_ylabel("R² score", fontsize=FONT)
            ax.set_xlabel("Species", fontsize=FONT)
            ax.legend(handles=bp_handles, loc="upper right", framealpha=0.9)

            for xi in x[:-1] + 0.5:
                ax.axvline(xi, color="#DDDDDD", lw=0.7, ls=":")

            fname = f"fig2_{ac}_r2_boxplot.png"
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Fig 2 saved: {fname}")
            print(f"  Caption: Distribution of R² values across 50 cross-validation "
                  f"folds per species for {ac}. R² computed on log₁₀(cells L⁻¹ + 1) "
                  f"scale. Dark boxes: spectral features only; light boxes: spectral "
                  f"+ ROMS oceanographic features.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 3: Cross-AC R² comparison per track
# ──────────────────────────────────────────────────────────────────────────────
def fig3_r2_cross_ac():
    for track_sfx, track_label in [("_only", "OC model"),
                                    ("_roms",  "Hybrid model")]:
        n_sp  = len(SPECIES_LIST)
        n_ac  = len(AC_LIST)
        x     = np.arange(n_sp)
        width = 0.16
        offsets = np.linspace(-(n_ac-1)/2, (n_ac-1)/2, n_ac) * (width + 0.03)

        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(max(10, n_sp * 2.0), 5))
            bp_handles = []

            for j, ac in enumerate(AC_LIST):
                color = AC_COLORS[ac]
                track = f"{ac}{track_sfx}"
                sub   = fold_df[fold_df["track"] == track]
                data  = [sub[sub["species"] == sp]["r2"].dropna().values
                          for sp in SPECIES_LIST]
                ax.boxplot(
                    data,
                    positions=x + offsets[j],
                    widths=width,
                    patch_artist=True,
                    notch=False,
                    showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=1.5),
                    capprops=dict(color=color, linewidth=1.5),
                    boxprops=dict(facecolor=color, color='black', alpha=0.82),
                )
                bp_handles.append(
                    mpatches.Patch(fc=color, alpha=0.82, label=ac))

            ax.axhline(0, color="#AAAAAA", lw=0.7, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([SP_LABELS.get(s, s) for s in SPECIES_LIST],
                               fontsize=FONT - 1, ha="center")
            ax.set_xlim(-0.6, n_sp - 0.4)
            ax.set_ylim(0.2, 0.7)
            ax.set_ylabel("R² score", fontsize=FONT)
            ax.set_xlabel("Species", fontsize=FONT)
            ax.legend(handles=bp_handles, loc="upper right", framealpha=0.9)
            
            for xi in x[:-1] + 0.5:
                ax.axvline(xi, color="#DDDDDD", lw=0.7, ls=":")

            track_tag = track_sfx.strip("_")
            fname = f"fig3_{track_tag}_ac_comparison.png"
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
            plt.close()
            print(f"  Fig 3 saved: {fname}")
            print(f"  Caption: Cross-pipeline comparison of R² distributions "
                  f"(50-fold CV, log scale) per species ({track_label}). "
                  f"Outliers omitted for clarity.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 4: ΔR² heatmap (roms − only) per AC
# ──────────────────────────────────────────────────────────────────────────────
def fig4_delta_heatmap():
    df_only = global_df[global_df["track"].str.endswith("_only")].copy()
    df_roms = global_df[global_df["track"].str.endswith("_roms")].copy()
    merged  = df_only.merge(
        df_roms[["ac", "macro_cv_r2_mean"]],
        on="ac", suffixes=("_only", "_roms"))
    merged["delta_r2"] = (merged["macro_cv_r2_mean_roms"]
                          - merged["macro_cv_r2_mean_only"])
    pivot = merged.set_index("ac")[["delta_r2"]]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4, max(3, len(pivot) * 0.8)))
        sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={"size": FONT},
                    cbar_kws={"shrink": 0.8, "label": "ΔR²"})
        ax.set_ylabel("", fontsize=FONT)
        ax.set_xlabel("", fontsize=FONT)
        ax.set_xticklabels(["ΔR²"], fontsize=FONT)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig4_delta_r2_heatmap.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  Fig 4 saved: fig4_delta_r2_heatmap.png")
        print("  Caption: Change in macro-averaged R² when ROMS oceanographic "
              "features are added to spectral-only models (ROMS − spectral only), "
              "by atmospheric correction pipeline. R² computed on log₁₀(cells L⁻¹ "
              "+ 1) scale.")

# ──────────────────────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────────────────────
print("Generating regression figures...")
print("\n--- Fig 1: R² per species ---")
fig1_r2_bars()

print("\n--- Fig 2: R² boxplot per AC ---")
fig2_r2_boxplot_per_ac()

print("\n--- Fig 3: Cross-AC R² comparison ---")
fig3_r2_cross_ac()

print("\n--- Fig 4: ΔR² heatmap ---")
fig4_delta_heatmap()

print(f"\nAll figures saved to {PLOTS_DIR}/")