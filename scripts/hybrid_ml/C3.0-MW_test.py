"""
Mann-Whitney U test (+ rank-biserial correlation effect size) for two comparisons:

  1. _only vs _roms  — does adding ROMS features improve performance?
     Output: mann_whitney_only_vs_roms.csv

  2. AC method pairwise — do atmospheric correction pipelines differ?
     Compares same-track-type (both _only or both _roms) across AC methods.
     Output: mann_whitney_ac_comparison.csv

Input:  hybrid_ml/rf_outputs_multispecies_clf/csv_results/fold_level_results.csv
"""

import pandas as pd
from itertools import combinations
from scipy.stats import mannwhitneyu
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = Path(
    "ml_outputs/rf_multispecies_clf/csv_results/fold_level_results.csv"
)
OUTPUT_ROMS = Path("ml_outputs/rf_multispecies_clf/csv_results/mann_whitney_only_vs_roms.csv")
OUTPUT_AC   = Path("ml_outputs/rf_multispecies_clf/csv_results/mann_whitney_ac_comparison.csv")
METRICS     = ["f1", "prauc"]
ALPHA       = 0.05
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")


def mw_test(a, b):
    """Return (U, p, rbc, effect_label) for two 1-D arrays."""
    stat, pval = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1 - (2 * stat) / (len(a) * len(b))
    abs_r = abs(rbc)
    effect = "negligible" if abs_r < 0.1 else \
             "small"      if abs_r < 0.3 else \
             "medium"     if abs_r < 0.5 else "large"
    return stat, pval, rbc, effect

# Identify _only and _roms tracks per AC method
# track names follow pattern: {AC}_only  /  {AC}_roms
records = []

for ac in df["ac"].unique():
    sub = df[df["ac"] == ac]

    only_tracks = sub[sub["track"].str.endswith("_only")]["track"].unique()
    roms_tracks  = sub[sub["track"].str.endswith("_roms")]["track"].unique()

    if not len(only_tracks) or not len(roms_tracks):
        print(f"  [{ac}] missing _only or _roms track — skipping")
        continue

    # Expect one of each; take first if multiple exist
    only_track = only_tracks[0]
    roms_track  = roms_tracks[0]

    for tier in sub["tier"].unique():
        for species in sub["species"].unique():
            for metric in METRICS:
                a = sub[(sub["track"] == only_track) &
                        (sub["tier"]    == tier)    &
                        (sub["species"] == species)][metric].dropna().values

                b = sub[(sub["track"] == roms_track) &
                        (sub["tier"]    == tier)     &
                        (sub["species"] == species)][metric].dropna().values

                if len(a) < 2 or len(b) < 2:
                    continue

                stat, pval, rbc, effect = mw_test(a, b)
                # rbc > 0  →  roms tends to rank higher than only

                records.append(dict(
                    ac          = ac,
                    tier        = tier,
                    species     = species,
                    metric      = metric,
                    only_track  = only_track,
                    roms_track  = roms_track,
                    n_only      = len(a),
                    n_roms      = len(b),
                    median_only = round(float(a.mean()), 5),
                    median_roms = round(float(b.mean()), 5),
                    delta       = round(float(b.mean() - a.mean()), 5),  # roms − only
                    U_stat      = round(stat, 3),
                    p_value     = round(pval, 6),
                    significant = pval < ALPHA,
                    direction   = "roms_better" if b.mean() > a.mean() else "only_better",
                    rbc         = round(rbc, 4),   # rank-biserial correlation
                    effect_size = effect,
                ))

results_roms = pd.DataFrame(records)

# ── Save _only vs _roms ───────────────────────────────────────────────────────
results_roms.to_csv(OUTPUT_ROMS, index=False)
print(f"\nSaved {len(results_roms)} comparisons → {OUTPUT_ROMS}")

# ── Console summary (_only vs _roms) ─────────────────────────────────────────
print("\n" + "=" * 80)
print("ROMS vs ONLY — SIGNIFICANT DIFFERENCES  (p < 0.05)")
print("=" * 80)
sig = results_roms[results_roms["significant"]].sort_values(["ac", "metric", "species"])
if sig.empty:
    print("  None found.")
else:
    for _, r in sig.iterrows():
        print(
            f"  {r.ac:10s} | {r.tier:10s} | {r.species:30s} | {r.metric:6s} | "
            f"Δ={r.delta:+.4f}  p={r.p_value:.4f}  rbc={r.rbc:+.3f} ({r.effect_size})  [{r.direction}]"
        )

print("\n" + "=" * 80)
print("ROMS vs ONLY — SUMMARY BY AC × METRIC")
print("=" * 80)
summary = (
    results_roms.groupby(["ac", "metric"])
    .agg(
        n_tests      =("p_value", "count"),
        n_sig        =("significant", "sum"),
        n_roms_better=("direction", lambda x: (x == "roms_better").sum()),
        n_only_better=("direction", lambda x: (x == "only_better").sum()),
        mean_delta   =("delta", "mean"),
        mean_rbc     =("rbc", "mean"),
    )
    .reset_index()
)
print(summary.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AC METHOD PAIRWISE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
# For each track_type (_only / _roms), tier, species, and metric,
# compare every pair of AC methods.

print("\n\n" + "=" * 80)
print("ATMOSPHERIC CORRECTION PAIRWISE COMPARISON")
print("=" * 80)

ac_records = []
ac_methods = df["ac"].unique().tolist()

# Track types to compare separately (keeps the two experiments independent)
track_types = {"_only": "only", "_roms": "roms"}

for track_suffix, track_label in track_types.items():
    sub_tt = df[df["track"].str.endswith(track_suffix)]

    for tier in sub_tt["tier"].unique():
        for species in sub_tt["species"].unique():
            for metric in METRICS:
                for ac_a, ac_b in combinations(ac_methods, 2):

                    def get_vals(ac):
                        return sub_tt[
                            (sub_tt["ac"]      == ac)     &
                            (sub_tt["tier"]    == tier)   &
                            (sub_tt["species"] == species)
                        ][metric].dropna().values

                    a = get_vals(ac_a)
                    b = get_vals(ac_b)

                    if len(a) < 2 or len(b) < 2:
                        continue

                    stat, pval, rbc, effect = mw_test(a, b)
                    # rbc > 0  →  ac_b tends to rank higher than ac_a

                    ac_records.append(dict(
                        track_type  = track_label,
                        tier        = tier,
                        species     = species,
                        metric      = metric,
                        ac_a        = ac_a,
                        ac_b        = ac_b,
                        n_a         = len(a),
                        n_b         = len(b),
                        mean_a      = round(float(a.mean()), 5),
                        mean_b      = round(float(b.mean()), 5),
                        delta       = round(float(b.mean() - a.mean()), 5),  # ac_b − ac_a
                        U_stat      = round(stat, 3),
                        p_value     = round(pval, 6),
                        significant = pval < ALPHA,
                        direction   = f"{ac_b}_better" if b.mean() > a.mean() else f"{ac_a}_better",
                        rbc         = round(rbc, 4),
                        effect_size = effect,
                    ))

results_ac = pd.DataFrame(ac_records)

# ── Save AC comparison ────────────────────────────────────────────────────────
results_ac.to_csv(OUTPUT_AC, index=False)
print(f"Saved {len(results_ac)} comparisons → {OUTPUT_AC}")

# ── Console summary (AC) ──────────────────────────────────────────────────────
sig_ac = results_ac[results_ac["significant"]].sort_values(
    ["track_type", "metric", "species", "ac_a", "ac_b"]
)
print(f"\nSignificant AC differences: {len(sig_ac)} / {len(results_ac)}")
if not sig_ac.empty:
    for _, r in sig_ac.iterrows():
        print(
            f"  [{r.track_type:4s}] {r.tier:10s} | {r.species:30s} | {r.metric:6s} | "
            f"{r.ac_a} vs {r.ac_b}  Δ={r.delta:+.4f}  p={r.p_value:.4f}  "
            f"rbc={r.rbc:+.3f} ({r.effect_size})  [{r.direction}]"
        )

print("\n" + "=" * 80)
print("AC COMPARISON — SUMMARY BY TRACK TYPE × PAIR × METRIC")
print("=" * 80)
ac_summary = (
    results_ac.groupby(["track_type", "ac_a", "ac_b", "metric"])
    .agg(
        n_tests =("p_value", "count"),
        n_sig   =("significant", "sum"),
        mean_delta=("delta", "mean"),
        mean_rbc  =("rbc", "mean"),
    )
    .reset_index()
)
print(ac_summary.to_string(index=False))