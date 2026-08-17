"""
Classification Plots — reads CSVs from clf_train.py output
===========================================================
Run after clf_train.py.

Figures produced:
  Fig 1: Average Precision (AP) per species — 2 subplots (spectral only / +ROMS)
          one panel per AC, all tiers overlaid with color.
          Caption: "Mean average precision (AP) per species across atmospheric
                   correction pipelines at warning (solid) and closure (hatched)
                   thresholds. Left: spectral features only; right: spectral +
                   ROMS oceanographic features."

  Fig 2: F1 boxplot — per AC, species on x-axis, two boxes (_only vs _roms)
          one figure per AC × tier.
          Caption: "Distribution of F1 scores across 50 cross-validation folds
                   per species for {ac} at the {tier} threshold. Dark: spectral
                   only; light: spectral + ROMS."

  Fig 3: F1 boxplot — compare across ACs, one figure per tier × track,
          species on x-axis, three boxes (one per AC).
          Caption: "Cross-AC comparison of F1 score distributions (50-fold CV)
                   per species at the {tier} threshold ({track})."

  Fig 4: ΔF1 heatmap (roms − only) per AC × tier.
          Caption: "Change in macro-averaged F1 score when ROMS oceanographic
                   features are added to spectral-only models, by atmospheric
                   correction pipeline and threshold tier."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths — edit BASE_OUT to match clf_train.py
# ──────────────────────────────────────────────────────────────────────────────
BASE_OUT  = Path("ml_outputs/rf_multispecies_clf")
CSV_DIR   = BASE_OUT / "csv_results"
PLOTS_DIR = BASE_OUT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load CSVs
# ──────────────────────────────────────────────────────────────────────────────
species_df = pd.read_csv(CSV_DIR / "species_comparison.csv")
fold_df    = pd.read_csv(CSV_DIR / "fold_level_results.csv")
global_df  = pd.read_csv(CSV_DIR / "global_comparison.csv")

for df in [species_df, fold_df, global_df]:
    df["ac"]    = df["ac"].str.replace("Baseline", "OC-SAC", regex=False)
    df["track"] = df["track"].str.replace("Baseline", "OC-SAC", regex=False)

AC_LIST      = sorted(species_df["ac"].unique())
SPECIES_LIST = sorted(species_df["species"].unique())
TIERS        = ["warning", "closure"]

# Short species labels for axis ticks
SP_LABELS = {
    "Alexandrium_catenella": "Alexandrium\nCatenella",
    "Dinophysis_acuminata":  "Dinophysis\nAcuminata",
    "Dinophysis_norvegica":  "Dinophysis\nNorvegica",
    "Karenia":               "Karenia",
    "Pseudo-nitzschia":      "Pseudo\nNitzschia",
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
    "grid.color":        "#E5E5E5",
    "grid.linewidth":    0.7,
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
COLOR_ONLY  = "#909090"
COLOR_ROMS  = "#2C2C2C"

AC_COLORS = {
    "C2RCC":    "#1b6ca8",
    "ACOLITE":  "#c75000",
    "OC-SAC": "#2e8b57",
}

# Per-species color palette — 5 distinct colors
SP_COLORS = {
    sp: c for sp, c in zip(SPECIES_LIST, [
        "#1b6ca8",   # blue
        "#c75000",   # orange-red
        "#2e8b57",   # green
        "#7b2d8b",   # purple
        "#b8860b",   # dark gold
    ])
}

# ──────────────────────────────────────────────────────────────────────────────
# Fig 1: Mean PR curves — A = spectral only, B = spectral + ROMS
#         all species on same subplot, colored by species
#         one figure per AC × tier
# ──────────────────────────────────────────────────────────────────────────────
def fig1_pr_curves():
    for ac in AC_LIST:
        for tier in TIERS:
            with plt.rc_context(RC):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

                for ax, (track_sfx, panel_label) in zip(
                        axes, [("_only", "A"), ("_roms", "B")]):
                    track = f"{ac}{track_sfx}"
                    trackfile = track.replace('OC-SAC', 'Baseline')
                    fpath = CSV_DIR / f"pr_curves_{trackfile}_{tier}.parquet"
                    if not fpath.exists():
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=ax.transAxes, color="#999999")
                        continue

                    pr_df = pd.read_parquet(fpath)
                    pr_df["ac"]    = pr_df["ac"].str.replace("Baseline", "OC-SAC", regex=False)
                    pr_df["track"] = pr_df["track"].str.replace("Baseline_only", "OC-SAC_only", regex=False)
                    pr_df["track"] = pr_df["track"].str.replace("Baseline_roms", "OC-SAC_roms", regex=False)
                    for sp in SPECIES_LIST:
                        sp_df = pr_df[pr_df["species"] == sp]
                        if sp_df.empty or sp_df["prec_mean"].isna().all():
                            continue
                        color  = SP_COLORS[sp]
                        recall = sp_df["recall"].values
                        pm     = sp_df["prec_mean"].values
                        ps     = sp_df["prec_std"].values
                        ap_m   = sp_df["ap_mean"].iloc[0]
                        ap_s   = sp_df["ap_std"].iloc[0]
                        if panel_label == "A":
                            ax.plot(recall, pm, color=color, lw=1.8,
                                label=f"{SP_LABELS.get(sp, sp):<12} AP={ap_m:.2f}±{ap_s:.2f}")
                        else:
                            ax.plot(recall, pm, color=color, lw=1.8, marker='o',markevery=20,
                                label=f"{SP_LABELS.get(sp, sp):<12}  AP={ap_m:.2f}±{ap_s:.2f}")
                        ax.fill_between(recall, pm - ps, pm + ps,
                                        color=color, alpha=0.12)

                    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
                    ax.set_xlabel("Recall", fontsize=FONT)
                    if panel_label == "A":
                        ax.set_ylabel("Precision", fontsize=FONT)
                    # ax.text(0.03, 0.97, panel_label,
                    #         transform=ax.transAxes, fontsize=FONT + 3,
                    #         fontweight="bold", va="top")
                    track_label = ("(A) OC model" if track_sfx == "_only"
                                   else "(B) Hybrid model")
                    ax.text(0.97, 0.97, track_label,
                            transform=ax.transAxes, fontsize=FONT - 1,fontweight="bold",
                            va="top", ha="right", color="#444444")
                    ax.legend(loc="lower left", fontsize=FONT - 2,
                              framealpha=0.4, handlelength=1.5, prop={'family': 'monospace'})
                
                fig.tight_layout()
                fname = f"fig1_{ac}_{tier}_pr_curves.png"
                fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
                plt.close()
                print(f"  Fig 1 saved: {fname}")
                print(f"  Caption: Mean precision-recall curves per species for "
                      f"{ac} at the {tier} threshold, averaged across 50 "
                      f"cross-validation folds. Shaded bands indicate ± one "
                      f"standard deviation. Legend entries show mean average "
                      f"precision (AP ± SD). (A) Spectral features only; "
                      f"(B) spectral + ROMS oceanographic features.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 2: F1 boxplot per AC × tier (_only vs _roms)
# ──────────────────────────────────────────────────────────────────────────────
def fig2_f1_boxplot_per_ac():
    YLIM = {"warning": (0.50, 1.01), "closure": (0.0, 1.01)}
    # YLIM = {"warning": (0.0, 1.01), "closure": (0.0, 1.01)}
    for ac in AC_LIST:
        for tier in TIERS:
            sub = fold_df[(fold_df["ac"] == ac) & (fold_df["tier"] == tier)]
            if sub.empty:
                continue

            n_sp   = len(SPECIES_LIST)
            x      = np.arange(n_sp)
            width  = 0.32
            gap    = 0.05

            with plt.rc_context(RC):
                fig, ax = plt.subplots(figsize=(max(10, n_sp * 1.8), 5))

                bp_handles = []
                for j, (track_sfx, color, label) in enumerate([
                        ("_only", COLOR_ONLY, "OC model"),
                        ("_roms", COLOR_ROMS, "Hybrid model")]):
                    track = f"{ac}{track_sfx}"
                    offset = (j - 0.5) * (width + gap)
                    data = [
                        sub[(sub["track"] == track) &
                            (sub["species"] == sp)]["f1"].dropna().values
                        for sp in SPECIES_LIST
                    ]
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
                        medianprops=dict(color="white",  linewidth=2),
                        whiskerprops=dict(color=color, linewidth=1),
                        capprops=dict(color=color, linewidth=1),
                        boxprops=dict(facecolor=color, color=color, alpha=0.85),
                    )
                    bp_handles.append(
                        mpatches.Patch(fc=color, alpha=0.85, label=label))

                ax.set_xticks(x)
                ax.set_xticklabels([SP_LABELS.get(s, s) for s in SPECIES_LIST],
                                   fontsize=FONT - 1, ha="center")
                ax.set_xlim(-0.6, n_sp - 0.4)
                ax.set_ylim(*YLIM[tier])
                ax.set_ylabel("F1 Score", fontsize=FONT)
                ax.set_xlabel("HAB-taxa", fontsize=FONT)
                ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)
                ax.legend(handles=bp_handles, loc="upper right", framealpha=0.9)

                for xi in x[:-1] + 0.5:
                    ax.axvline(xi, color="#888888", lw=1.0, ls=":")

                fig.tight_layout()
                fname = f"fig2_{ac}_{tier}_f1_boxplot.png"
                fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
                plt.close()
                print(f"  Fig 2 saved: {fname}")
                print(f"  Caption: Distribution of F1 scores across 50 cross-validation "
                      f"folds per species for {ac} at the {tier} threshold. "
                      f"Dark boxes: spectral features only; light boxes: spectral + ROMS "
                      f"oceanographic features. Horizontal lines indicate median; "
                      f"whiskers extend to 1.5× IQR.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 3: F1 boxplot — cross-AC comparison, per tier × track
# ──────────────────────────────────────────────────────────────────────────────
def fig3_f1_boxplot_cross_ac():
    for tier in TIERS:
        for track_sfx, track_label in [("_only", "Spectral only"),
                                        ("_roms", "Spectral + ROMS")]:
            n_sp  = len(SPECIES_LIST)
            n_ac  = len(AC_LIST)
            x     = np.arange(n_sp)
            width = 0.16
            offsets = np.linspace(-(n_ac-1)/2, (n_ac-1)/2, n_ac) * (width + 0.03)

            with plt.rc_context(RC):
                fig, ax = plt.subplots(figsize=(max(10, n_sp * 2.0), 5))
                bp_handles = []

                for j, ac in enumerate(AC_LIST):
                    color  = AC_COLORS[ac]
                    track  = f"{ac}{track_sfx}"
                    sub    = fold_df[(fold_df["track"] == track) &
                                     (fold_df["tier"] == tier)]
                    data   = [sub[sub["species"] == sp]["f1"].dropna().values
                               for sp in SPECIES_LIST]
                    ax.boxplot(
                        data,
                        positions=x + offsets[j],
                        widths=width,
                        patch_artist=True,
                        notch=False,
                        showfliers=False,
                        medianprops=dict(color="white", linewidth=1.5),
                        whiskerprops=dict(color="#222222", linewidth=1.5),
                        capprops=dict(color="#222222", linewidth=1),
                        boxprops=dict(facecolor=color, color="#222222",
                                      alpha=0.82, linewidth=1.5),
                    )
                    bp_handles.append(
                        mpatches.Patch(fc=color, alpha=0.82, label=ac))

                ax.set_xticks(x)
                ax.set_xticklabels([SP_LABELS.get(s, s) for s in SPECIES_LIST],
                                   fontsize=FONT - 1, ha="center")
                ax.set_xlim(-0.6, n_sp - 0.4)
                if tier == 'warning':
                    ax.set_ylim(0.38, 1.01)
                else:
                    ax.set_ylim(0.0, 1.01)
                ax.set_ylabel("F1 Score", fontsize=FONT)
                ax.set_xlabel("HAB-taxa", fontsize=FONT)
                ax.legend(handles=bp_handles, loc="lower right", framealpha=0.7)
                ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)
                for xi in x[:-1] + 0.5:
                    ax.axvline(xi, color="#888888", lw=1.0, ls=":")

                track_tag = track_sfx.strip("_")
                fname = f"fig3_{tier}_{track_tag}_ac_comparison.png"
                fig.tight_layout()
                fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
                plt.close()
                print(f"  Fig 3 saved: {fname}")
                print(f"  Caption: Cross-pipeline comparison of F1 score distributions "
                      f"(50-fold CV) per species at the {tier} threshold "
                      f"({track_label}). Boxes show the interquartile range; "
                      f"outliers omitted for clarity.")

# ──────────────────────────────────────────────────────────────────────────────
# Fig 4: ΔF1 heatmap (roms − only)
# ──────────────────────────────────────────────────────────────────────────────
def fig4_delta_heatmap():
    df_only = global_df[global_df["track"].str.endswith("_only")].copy()
    df_roms = global_df[global_df["track"].str.endswith("_roms")].copy()
    merged  = df_only.merge(
        df_roms[["ac", "tier", "cv_f1_macro_mean"]],
        on=["ac", "tier"], suffixes=("_only", "_roms"))
    merged["delta_f1"] = (merged["cv_f1_macro_mean_roms"]
                          - merged["cv_f1_macro_mean_only"])
    pivot = merged.pivot_table(index="ac", columns="tier", values="delta_f1")
    pivot = pivot[TIERS]   # enforce column order

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5,
                    annot_kws={"size": FONT},
                    cbar_kws={"shrink": 0.8, "label": "ΔF1"})
        ax.set_ylabel("", fontsize=FONT)
        ax.set_xlabel("", fontsize=FONT)
        ax.set_xticklabels(["Warning", "Closure"], fontsize=FONT)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig4_delta_f1_heatmap.png",
                    dpi=DPI, bbox_inches="tight")
        plt.close()
        print("  Fig 4 saved: fig4_delta_f1_heatmap.png")
        print("  Caption: Change in macro-averaged F1 score when ROMS oceanographic "
              "features are added to spectral-only models (ROMS − spectral only), "
              "by atmospheric correction pipeline and threshold tier. Positive values "
              "indicate improvement with ROMS; negative values indicate degradation.")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5: Permutation Importance — two subplots (OC model / Hybrid model)
#         per species, one figure per AC × tier
# ──────────────────────────────────────────────────────────────────────────────
IMPORTANCE_DIR = BASE_OUT / "importance"
TOP_N_FEATURES = 10  # features shown per species panel

def fig5_permutation_importance():
    for ac in AC_LIST:
        for tier in TIERS:
            # Load both tracks
            ac_file = ac.replace("OC-SAC", "Baseline")
            path_only = IMPORTANCE_DIR / f"{ac_file}_only_{tier}_perm_importance.csv"
            path_roms = IMPORTANCE_DIR / f"{ac_file}_roms_{tier}_perm_importance.csv"

            if not path_only.exists() or not path_roms.exists():
                print(f"  Fig 5 skipped ({ac} {tier}): importance files not found")
                continue

            df_only = pd.read_csv(path_only).set_index("feature")
            df_roms = pd.read_csv(path_roms).set_index("feature")

            n_sp = len(SPECIES_LIST)
            fig, axes = plt.subplots(
                n_sp, 2,
                figsize=(14, n_sp * 3.2),
                squeeze=False
            )

            with plt.rc_context(RC):
                for row, sp in enumerate(SPECIES_LIST):
                    mean_col = f"{sp}_perm_mean"
                    std_col  = f"{sp}_perm_std"

                    for col, (df, panel_tag, color) in enumerate([
                            (df_only, "(A) OC model",     COLOR_ONLY),
                            (df_roms, "(B) Hybrid model", COLOR_ROMS)]):

                        ax = axes[row][col]

                        if mean_col not in df.columns:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                    transform=ax.transAxes, color="#999999")
                            continue

                        # Top-N by mean importance (positive only)
                        vals = df[mean_col].dropna()
                        vals = vals[vals > 0].nlargest(TOP_N_FEATURES)
                        errs = df.loc[vals.index, std_col] if std_col in df.columns else None

                        y_pos = np.arange(len(vals))
                        ax.barh(y_pos, vals.values, xerr=errs.values if errs is not None else None,
                                color=color, alpha=0.82, height=0.65,
                                error_kw=dict(ecolor="#666666", lw=0.8, capsize=2))

                        clean_labels = [f.removeprefix("A_") for f in vals.index]
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(clean_labels, fontsize=FONT - 2)
                        # ax.set_yticklabels(vals.index, fontsize=FONT - 2)
                        ax.invert_yaxis()
                        ax.set_xlabel("Mean permutation importance", fontsize=FONT - 1)

                        sp_label = SP_LABELS.get(sp, sp).replace("\n", " ")
                        if col == 0:
                            ax.set_ylabel(f"{sp_label}", fontsize=FONT, fontstyle="italic")

                        ax.text(0.98, 0.02, panel_tag,
                                transform=ax.transAxes, fontsize=FONT - 1,
                                fontweight="bold", ha="right", va="bottom",
                                color="#444444")

                fig.tight_layout(h_pad=1.5, w_pad=1.2)
                fname = f"fig5_{ac}_{tier}_perm_importance.png"
                fig.savefig(PLOTS_DIR / fname, dpi=DPI, bbox_inches="tight")
                plt.close()
                print(f"  Fig 5 saved: {fname}")
                print(f"  Caption: Top {TOP_N_FEATURES} permutation-important features "
                      f"per species for {ac} at the {tier} threshold, averaged across "
                      f"50 cross-validation folds. Error bars indicate ± one standard "
                      f"deviation. (A) Ocean colour model; (B) Hybrid model (ocean colour "
                      f"+ ROMS oceanographic features). Only features with positive mean "
                      f"importance are shown.")
# ──────────────────────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────────────────────
print("Generating classification figures...")
print("\n--- Fig 1: Mean PR curves per species ---")
fig1_pr_curves()

print("\n--- Fig 2: F1 boxplot per AC × tier ---")
fig2_f1_boxplot_per_ac()

print("\n--- Fig 3: Cross-AC F1 comparison ---")
fig3_f1_boxplot_cross_ac()

print("\n--- Fig 4: ΔF1 heatmap ---")
fig4_delta_heatmap()

print("\n--- Fig 5: Permutation importance per species ---")
fig5_permutation_importance()

print(f"\nAll figures saved to {PLOTS_DIR}/")