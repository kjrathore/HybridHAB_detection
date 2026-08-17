"""
Spatial CV Appendix Figures — AC Comparison & OC vs Hybrid
=============================================================
Replaces the appendix table with two figures built from
fold_level_results.csv (block-based spatial CV output), showing fold-level
spread rather than only point estimates.

Figure 1: spatial_cv_ac_comparison_warning.png
  Per-fold macro F1 across the three AC pipelines, OC-only model, warning
  threshold. Mirrors the main text's AC comparison figure (fig:f1_acs) but
  for spatial CV.

Figure 2: spatial_cv_oc_vs_hybrid.png
  Per-fold macro F1, OC vs Hybrid, all three AC pipelines shown together
  (not just one), faceted by threshold. This is the figure that carries the
  appendix's actual finding: AC choice converges under the hybrid model at
  the warning threshold but reorders at closure — a single-AC plot would
  hide that.
"""

"""
Spatial CV Appendix Figure — OC vs Hybrid
=============================================
Built from fold_level_results.csv (block-based spatial CV output).

Figure: spatial_cv_oc_vs_hybrid.png
  Per-fold macro F1, OC vs Hybrid, all three AC pipelines shown together
  (not just one), faceted by threshold. Carries the appendix's finding:
  AC choice converges under the hybrid model at the warning threshold but
  reorders at closure — a single-AC plot would hide that.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Config — point at your actual run's output directory
# ──────────────────────────────────────────────────────────────────────────────
FOLD_CSV = Path(
   "ml_outputs/rf_spatial_cv_rf_multiblocking/fold_level_results.csv"
)
OUT_DIR = Path("datasets/GULF_OF_MAINE/ml_outputs/rf_spatial_cv_rf_multiblocking/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AC_LABEL   = {"ACOLITE": "ACOLITE", "Baseline": "OC-SAC", "C2RCC": "C2RCC"}
AC_ORDER   = ["ACOLITE", "OC-SAC", "C2RCC"]
TIER_ORDER = ["warning", "closure"]
TIER_LABEL = {"warning": "Warning threshold", "closure": "Closure threshold"}
MODEL_PALETTE = {"OC": "#909090", "Hybrid": "#2C2C2C"}

sns.set_style("whitegrid")

# ──────────────────────────────────────────────────────────────────────────────
# Load & collapse species-level fold rows to per-fold macro F1 (mean across
# species, NaN-skipping by default in pandas — matches how run_cv computed
# the fold-level macro F1 after the zero-positive-fold exclusion fix)
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(FOLD_CSV)
# filter for blocking type
block_type = 'quadtree_random'#'clustering'#'grid' #'grid' #'quadtree_adaptive', 'quadtree_random'
df = df[df["blocking"]==block_type]
df["ac_label"] = df["ac"].map(AC_LABEL)
df["model"] = df["track"].apply(lambda t: "Hybrid" if t.endswith("_roms") else "OC")


fold_macro = (
    df.groupby(["ac_label", "model", "tier", "fold"])["f1"]
      .mean()
      .reset_index()
      .rename(columns={"f1": "f1_macro"})
)

print(f"Loaded {len(df):,} species-level fold rows -> "
      f"{len(fold_macro):,} per-fold macro F1 rows")

# ──────────────────────────────────────────────────────────────────────────────
# Figure: OC vs Hybrid, all three ACs, faceted by threshold
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, tier in zip(axes, TIER_ORDER):
    sub = fold_macro[fold_macro["tier"] == tier]
    sns.violinplot(data=sub, x="ac_label", y="f1_macro", hue="model",
                    order=AC_ORDER, hue_order=["OC", "Hybrid"],
                    palette=MODEL_PALETTE, split=True, inner="quartile",
                    linecolor='white',
                    cut=0, bw_adjust=1.3, linewidth=2.0, ax=ax)
    ax.set_title(TIER_LABEL[tier])
    ax.set_xlabel("Atmospheric Correction")
    ax.set_ylabel("Taxa-Averaged F1-score" if tier == "warning" else "")
    ax.get_legend().remove()

# Single shared legend above both panels, out of the plot area
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Model", loc="lower center",
           bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)

fig.tight_layout()
fig.savefig(OUT_DIR / f"{block_type}_spatial_cv_oc_vs_hybrid.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'spatial_cv_oc_vs_hybrid.png'}")