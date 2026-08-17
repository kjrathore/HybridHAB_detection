"""
HAB Multi-Label Classification — Model A: Binary Relevance RF
==============================================================
MultiOutputClassifier(RandomForestClassifier) — one binary RF per species.

Six target species with literature-grounded fixed thresholds:
  Species                  Warning (cells/L)   Closure (cells/L)
  ──────────────────────── ──────────────────  ─────────────────
  Alexandrium_catenella           100                  300
  Dinophysis_acuminata            200                  500
  Dinophysis_norvegica            200                  500
  Karenia                       1,000                5,000
  Pseudo-nitzschia             2,000               13,000
  Margalefidinium              1,000                6,000

Three atmospheric correction pipelines × ±ROMS = six tracks:
  C2RCC_only  | C2RCC_roms
  ACOLITE_only | ACOLITE_roms
  Baseline_only| Baseline_roms

CV design
----------
  50-fold ShuffleSplit (80/20).  No separate held-out test split.
  Final model fit on ALL data for Gini importance only.
  PR curves drawn from the last CV fold.

Metrics (CV mean ± std ± median, per species and macro):
  F1, Precision, Recall, PR-AUC (average precision)
  Multi-label: Hamming loss, subset accuracy, Jaccard

CSV layout (comparison-friendly)
----------------------------------
  global_comparison.csv
      One row per (ac, track, tier).  Sorted so _only/_roms are adjacent.
      Macro F1, PR-AUC, Hamming, Jaccard — mean, median, std.

  species_comparison.csv
      One row per (ac, track, tier, species).
      cv_f1_mean/median/std, cv_prauc_mean/median/std, n_positive.

  roms_delta_summary.csv
      One row per (ac, tier, species).
      delta_f1 and delta_prauc = roms − only.  Sorted by delta_f1 desc.

  fold_level_results.csv
      One row per (ac, track, tier, species, fold).
      Raw f1 and prauc per fold — use directly for boxplots.

Plots  (flat directory, prefixed)
----------------------------------
  plots/{ac}_{tier}_f1_boxplot.png   ← grouped boxplot _only vs _roms per species
  plots/{ac}_{tier}_pr_curves.png    ← PR curves: rows=species, cols=tracks
  csv_results/roms_delta_heatmap.png ← ΔF1 macro heatmap (AC × tier)

Importance  (flat directory)
------------------------------
  importance/{ac}_{track}_{tier}_perm_importance.csv
      Rows = features, columns = {species}_perm_mean / {species}_perm_std.
      Sorted by mean importance averaged across species.

Outputs → hybrid_ml/rf_outputs_multilabel_BR/
  csv_results/
    global_comparison.csv
    species_comparison.csv
    roms_delta_summary.csv
    fold_level_results.csv
  importance/
    {ac}_{track}_{tier}_perm_importance.csv
"""


import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, jaccard_score,
    average_precision_score, precision_recall_curve,
    recall_score, precision_score,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

print("=" * 80)
print("HAB MULTI-LABEL CLF — BINARY RELEVANCE RF (CV-ONLY)")
print("=" * 80)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DATA = Path(
    "processed_data"
)
COMPOSITE_PATH = (
    BASE_DATA / "sentinel_3_L2/composite_features/composite_features.parquet"
)

BASE_OUT = Path("datasets/GULF_OF_MAINE/ml_outputs/rf_multispecies_clf_fixed")
CSV_DIR  = BASE_OUT / "csv_results"
IMP_DIR  = BASE_OUT / "importance"

for d in [CSV_DIR, IMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Species & thresholds
# ──────────────────────────────────────────────────────────────────────────────
SPECIES_THRESHOLDS = {
    "Alexandrium_catenella": {"warning":    100, "closure":    300},
    "Dinophysis_acuminata":  {"warning":    200, "closure":    500},
    "Dinophysis_norvegica":  {"warning":    200, "closure":    500},
    "Karenia":               {"warning":  1_000, "closure":  5_000},
    "Pseudo-nitzschia":      {"warning":  2_000, "closure": 13_000},
}
SPECIES_LIST    = list(SPECIES_THRESHOLDS.keys())
THRESHOLD_TIERS = ["warning", "closure"]

# ──────────────────────────────────────────────────────────────────────────────
# Feature sets
# ──────────────────────────────────────────────────────────────────────────────
BAND_NUM_TO_WL = {
    "rrs_2": "rrs_412", "rrs_4": "rrs_490", "rrs_6": "rrs_560",
    "rrs_8": "rrs_665", "rrs_9": "rrs_673", "rrs_10": "rrs_682",
    "rrs_11": "rrs_709", "rrs_12": "rrs_754", "rrs_17": "rrs_865",
}
RRS_SUFFIXES = list(BAND_NUM_TO_WL.values())
INDEX_SUFFIXES = [
    "FLH_681", "FLH_665", "FLHmax", "GLH", "BLH",
    "MCI", "RBD", "DINI", "EBI", "GBI", "KBBI",
    "NDNI", "NDCI", "NDWI",
    "RedEdge_Ratio", "CI", "BlueGreen_Ratio",
    "Green_Red_Ratio", "Blue_Red_Ratio", "Red_NIR_Ratio",
    "Fluorescence_Peak",
]
SPECTRAL_SUFFIXES = RRS_SUFFIXES + INDEX_SUFFIXES

AC_PREFIXES = {"C2RCC": "C", "ACOLITE": "A", "Baseline": "B"}

ROMS_FEATURES = [
    "roms_temp_sur", "roms_salt_sur", "roms_zeta",
    "roms_u_sur", "roms_v_sur",
    "roms_Uwind", "roms_Vwind", "roms_Pair",
    "roms_current_speed", "roms_current_dir",
    "roms_wind_speed", "roms_wind_dir",
]
TEMPORAL_FEATURES = ["doy_sin", "doy_cos"]
# filter on this dataset name for fixed vs ship data.
TEST_DATASETS = [
    'mvco', 'gsodock', 'harpswell',  'fiddlers', 'mdibl',
  'ecoa', 'oceanalliance',  'radbot_mvco', 'radbot_ios', 'radbot_jeffreys_basin',
#  'NESLTER_broadscale', 'azmp', 'gom'
 ]

N_CV  = 50
N_EST = 100
MIN_BLOOM_SAMPLES = 10

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
# def resolve_features(prefix, df):
#     candidates = [f"{prefix}_{s}" for s in SPECTRAL_SUFFIXES]
#     return [c for c in candidates if c in df.columns and df[c].notna().any()]

def resolve_features(prefix, df):
    candidates = [f"{prefix}_{s}" for s in SPECTRAL_SUFFIXES]
    spectral = [c for c in candidates if c in df.columns and df[c].notna().any()]
    temporal = [f for f in TEMPORAL_FEATURES if f in df.columns and df[f].notna().any()]
    return spectral + temporal

def resolve_roms(df):
    return [f for f in ROMS_FEATURES if f in df.columns and df[f].notna().any()]

def build_label_matrix(df, tier):
    cols = {}
    for sp in SPECIES_LIST:
        if sp not in df.columns:
            cols[sp] = pd.Series(np.nan, index=df.index)
        else:
            cols[sp] = (df[sp] > SPECIES_THRESHOLDS[sp][tier]).astype(int)
    Y_df = pd.DataFrame(cols, index=df.index)
    valid_rows = Y_df.notna().all(axis=1)
    return Y_df[valid_rows].astype(int), valid_rows

def print_label_stats(Y_df, tier, prefix="  "):
    n = len(Y_df)
    print(f"{prefix}Label statistics at '{tier}' threshold:")
    for sp in Y_df.columns:
        n_pos = Y_df[sp].sum()
        print(f"{prefix}  {sp:<30} bloom: {n_pos:>5,} / {n:,}  ({n_pos/n*100:.1f}%)")
    multi     = (Y_df.sum(axis=1) > 1).sum()
    any_bloom = (Y_df.sum(axis=1) > 0).sum()
    print(f"{prefix}  Rows ≥1 label: {any_bloom:,} ({any_bloom/n*100:.1f}%)  "
          f"≥2 labels: {multi:,} ({multi/n*100:.1f}%)")

def _get_proba_matrix(model, X):
    return np.column_stack([est.predict_proba(X)[:, 1]
                            for est in model.estimators_])

def run_cv(X, Y, dataset_series):
    ss = ShuffleSplit(n_splits=N_CV, test_size=0.20, random_state=RANDOM_STATE)
    # ss = StratifiedShuffleSplit(n_splits=N_CV, test_size=0.20, random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(
        n_estimators=N_EST, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    model_template = MultiOutputClassifier(base_rf, n_jobs=-1)

    fold_f1_macro    = []
    fold_hamming     = []
    fold_subset_acc  = []
    fold_jaccard     = []
    fold_prauc_macro = []
    fold_f1_sp    = {sp: [] for sp in SPECIES_LIST}
    fold_prauc_sp = {sp: [] for sp in SPECIES_LIST}
    fold_recall_sp    = {sp: [] for sp in SPECIES_LIST}
    fold_precision_sp = {sp: [] for sp in SPECIES_LIST}
    last_fold = None

    last_fold = None

    RECALL_GRID = np.linspace(0, 1, 100)
    fold_prec_sp = {sp: [] for sp in SPECIES_LIST}   # interpolated per fold

    for train_idx, test_idx in ss.split(X, Y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]

        # ── filter test to target datasets only ──
        test_mask = dataset_series.iloc[test_idx].isin(TEST_DATASETS)
        X_te = X_te[test_mask]
        Y_te = Y_te[test_mask]
        if len(X_te) == 0:
            continue

        if any(Y_tr[sp].nunique() < 2 for sp in Y_tr.columns):
            continue

        model = copy.deepcopy(model_template)
        model.fit(X_tr, Y_tr)
        Y_pred  = pd.DataFrame(model.predict(X_te),
                               columns=Y_te.columns, index=Y_te.index)
        Y_proba = _get_proba_matrix(model, X_te)
        last_fold = (Y_te, Y_pred, Y_proba, model)

        fold_f1_macro.append(
            f1_score(Y_te, Y_pred, average="macro", zero_division=0))
        fold_hamming.append(hamming_loss(Y_te, Y_pred))
        fold_subset_acc.append(accuracy_score(Y_te, Y_pred))
        fold_jaccard.append(
            jaccard_score(Y_te, Y_pred, average="macro", zero_division=0))

        label_prauc = []
        for i, sp in enumerate(SPECIES_LIST):
            y_true_sp = Y_te[sp].values
            fold_f1_sp[sp].append(
                f1_score(y_true_sp, Y_pred[sp].values, zero_division=0))
            fold_recall_sp[sp].append(
                recall_score(y_true_sp, Y_pred[sp].values, zero_division=0))
            fold_precision_sp[sp].append(
                precision_score(y_true_sp, Y_pred[sp].values, zero_division=0))
            if y_true_sp.sum() == 0 or y_true_sp.sum() == len(y_true_sp):
                fold_prauc_sp[sp].append(np.nan)
                continue
            ap = average_precision_score(y_true_sp, Y_proba[:, i])
            fold_prauc_sp[sp].append(ap)
            label_prauc.append(ap)
            prec, rec, _ = precision_recall_curve(y_true_sp, Y_proba[:, i])
            prec_interp = np.interp(RECALL_GRID, rec[::-1], prec[::-1])
            fold_prec_sp[sp].append(prec_interp)

        fold_prauc_macro.append(np.nanmean(label_prauc) if label_prauc else np.nan)

    def stats(vals):
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "std":    float(np.nanstd(arr)),
            "_folds": vals,
        }

    cv_out = {
        "f1_macro":    stats(fold_f1_macro),
        "hamming":     stats(fold_hamming),
        "subset_acc":  stats(fold_subset_acc),
        "jaccard":     stats(fold_jaccard),
        "prauc_macro": stats(fold_prauc_macro),
        "f1_per_label":        {sp: stats(fold_f1_sp[sp])        for sp in SPECIES_LIST},
        "prauc_per_label":     {sp: stats(fold_prauc_sp[sp])     for sp in SPECIES_LIST},
        "recall_per_label":    {sp: stats(fold_recall_sp[sp])    for sp in SPECIES_LIST},
        "precision_per_label": {sp: stats(fold_precision_sp[sp]) for sp in SPECIES_LIST},
        "n_folds_used": len(fold_f1_macro),
        # Mean PR curve per species across folds
        "pr_curves": {
            sp: {
                "recall":    RECALL_GRID,
                "prec_mean": np.nanmean(fold_prec_sp[sp], axis=0)
                             if fold_prec_sp[sp] else np.full(100, np.nan),
                "prec_std":  np.nanstd(fold_prec_sp[sp], axis=0)
                             if fold_prec_sp[sp] else np.full(100, np.nan),
            }
            for sp in SPECIES_LIST
        },
    }
    return cv_out, last_fold

def round_floats(df, decimals=5):
    fc = df.select_dtypes(include="float").columns
    df[fc] = df[fc].round(decimals)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Load & filter
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nLoading: {COMPOSITE_PATH}")
df_raw = pd.read_parquet(COMPOSITE_PATH)
print(f"  Raw shape: {df_raw.shape[0]:,} × {df_raw.shape[1]}")

# Filter based on fixed vs ship
# df_raw = df_raw[df_raw["dashboardIdName"].isin(DATASETS)]

if "delta_days" in df_raw.columns:
    df_raw = df_raw[df_raw["delta_days"] == 0].copy()
    print(f"  After delta_days==0: {df_raw.shape[0]:,} rows")

df_raw["date"] = pd.to_datetime(df_raw["obs_date"])

# ── Seasonal features ────────────────────────────────────────────────────────
df_raw["doy_sin"] = np.sin(2 * np.pi * df_raw["date"].dt.dayofyear / 365)
df_raw["doy_cos"] = np.cos(2 * np.pi * df_raw["date"].dt.dayofyear / 365)

c2rcc_rename = {f"C_{s}": f"C_{d}" for s, d in BAND_NUM_TO_WL.items()
                if f"C_{s}" in df_raw.columns}
if c2rcc_rename:
    df_raw = df_raw.rename(columns=c2rcc_rename)

if "location_id" in df_raw.columns and "drqnp" in df_raw["location_id"].values:
    df_raw = df_raw[df_raw["location_id"] != "drqnp"].copy()

# Common observation set
print("\n" + "=" * 60)
print("COMMON OBSERVATIONS")
print("=" * 60)
ac_valid_masks = {}
for ac, pfx in AC_PREFIXES.items():
    col = f"{pfx}_rrs_412"
    if col not in df_raw.columns:
        fb = [c for c in df_raw.columns
              if c.startswith(f"{pfx}_rrs_") and df_raw[c].notna().any()]
        col = fb[0] if fb else None
    if col:
        ac_valid_masks[ac] = df_raw[col].notna()
        print(f"  {ac:10s}: {ac_valid_masks[ac].sum():,} rows")
    else:
        ac_valid_masks[ac] = pd.Series(False, index=df_raw.index)

df_common = df_raw[pd.concat(ac_valid_masks.values(), axis=1).all(axis=1)].copy()
print(f"  Common rows (all ACs): {len(df_common):,}")

roms_ok = "roms_available" in df_common.columns
df_common_roms = (df_common[df_common["roms_available"] == True].copy()
                  if roms_ok else df_common.copy())
print(f"  Common rows with ROMS: {len(df_common_roms):,}")

# Median imputation
print("\n  Applying median imputation...")
all_spec_cols = [c for ac, pfx in AC_PREFIXES.items()
                 for c in resolve_features(pfx, df_common)]
impute_cols = list(set(all_spec_cols + resolve_roms(df_common_roms)))

for col in impute_cols:
    if col in df_common.columns:
        n = df_common[col].isna().sum()
        if n > 0:
            df_common[col] = df_common[col].fillna(df_common[col].median())
            print(f"    {col:<40} filled {n:,} NaNs")
    if col in df_common_roms.columns:
        n = df_common_roms[col].isna().sum()
        if n > 0:
            df_common_roms[col] = df_common_roms[col].fillna(
                df_common_roms[col].median())
print("  Imputation complete.")

# Feature audit
print("\n  Building feature audit CSV...")
all_feat_cols = sorted(set(all_spec_cols + resolve_roms(df_common_roms)))
track_feature_map = {}
for ac, pfx in AC_PREFIXES.items():
    spec = resolve_features(pfx, df_common)
    roms = resolve_roms(df_common_roms)
    track_feature_map[f"{ac}_only"] = spec
    track_feature_map[f"{ac}_roms"] = spec + roms

audit_rows = []
for feat in all_feat_cols:
    row = {"feature": feat}
    for track, feat_list in track_feature_map.items():
        if feat not in feat_list:
            row[track] = None
        else:
            df_src = df_common_roms if "_roms" in track else df_common
            row[track] = int(df_src[feat].notna().sum()) if feat in df_src.columns else 0
    audit_rows.append(row)
pd.DataFrame(audit_rows).to_csv(CSV_DIR / "feature_audit.csv", index=False)
print(f"  feature_audit.csv saved")

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
global_rows  = []
species_rows = []
fold_rows    = []

for ac_name, ac_prefix in AC_PREFIXES.items():

    spec_only  = resolve_features(ac_prefix, df_common)
    spec_roms  = resolve_features(ac_prefix, df_common_roms)
    roms_feats = resolve_roms(df_common_roms)

    print(f"\n{'═'*70}")
    print(f"  AC: {ac_name}  |  spectral: {len(spec_only)}  |  ROMS: {len(roms_feats)}")
    print(f"{'═'*70}")

    if not spec_only:
        print(f"  ⚠ No spectral features — skipping.")
        continue

    tracks = {
        f"{ac_name}_only": {"df": df_common.copy(), "features": spec_only},
        f"{ac_name}_roms": {"df": df_common_roms.copy() if spec_roms else pd.DataFrame(),
                            "features": spec_roms + roms_feats},
    }

    for track_key, tcfg in tracks.items():
        df_in    = tcfg["df"]
        features = tcfg["features"]

        if df_in.empty or not features:
            print(f"\n  Track {track_key}: no data, skipping.")
            continue

        print(f"\n  ── Track: {track_key}  |  n={len(df_in):,}  |  features={len(features)}")

        for tier in THRESHOLD_TIERS:
            print(f"\n    Tier: {tier.upper()}")
            Y_df, valid_rows = build_label_matrix(df_in, tier)
            X_all = df_in[valid_rows][features].copy()

            n_total = len(X_all)
            if n_total < 50:
                print(f"    ⚠ Only {n_total} rows — skipping.")
                continue

            print_label_stats(Y_df, tier, prefix="    ")

            skipped = [sp for sp in SPECIES_LIST if Y_df[sp].sum() < MIN_BLOOM_SAMPLES]
            if skipped:
                print(f"    ⚠ <{MIN_BLOOM_SAMPLES} positives (unreliable): {skipped}")

            print(f"    Running {N_CV}-fold CV...")
            # cv_out, last_fold = run_cv(X_all, Y_df)
            cv_out, last_fold = run_cv(X_all, Y_df, df_in["dashboardIdName"].loc[X_all.index])
            Y_te_last, Y_pred_last, Y_proba_last, model_last = last_fold

            print(f"    Macro F1    : {cv_out['f1_macro']['mean']:.3f} "
                  f"± {cv_out['f1_macro']['std']:.3f}  "
                  f"(median={cv_out['f1_macro']['median']:.3f})")
            print(f"    Macro PR-AUC: {cv_out['prauc_macro']['mean']:.3f} "
                  f"± {cv_out['prauc_macro']['std']:.3f}  "
                  f"(median={cv_out['prauc_macro']['median']:.3f})")
            print(f"    Hamming: {cv_out['hamming']['mean']:.4f}  "
                  f"Jaccard: {cv_out['jaccard']['mean']:.3f}  "
                  f"(folds: {cv_out['n_folds_used']})")
            for sp in SPECIES_LIST:
                print(f"      {sp:<30} "
                      f"F1={cv_out['f1_per_label'][sp]['mean']:.3f}±"
                      f"{cv_out['f1_per_label'][sp]['std']:.3f}  "
                      f"AP={cv_out['prauc_per_label'][sp]['mean']:.3f}±"
                      f"{cv_out['prauc_per_label'][sp]['std']:.3f}")

            # Final model → permutation importance
            final_model = copy.deepcopy(
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=N_EST, min_samples_split=5, min_samples_leaf=2,
                        class_weight="balanced", random_state=RANDOM_STATE,
                    ), n_jobs=-1))
            final_model.fit(X_all, Y_df)

            # Permutation importance — one CSV per track, all species as columns
            perm_rows_imp = [{"feature": f} for f in features]
            for i, sp in enumerate(SPECIES_LIST):
                est   = model_last.estimators_[i]
                valid = Y_te_last[sp].notna()
                if valid.sum() < 5:
                    for row in perm_rows_imp:
                        row[f"{sp}_perm_mean"] = np.nan
                        row[f"{sp}_perm_std"]  = np.nan
                    continue
                perm = permutation_importance(
                    est, X_all.loc[Y_te_last[valid].index],
                    Y_te_last.loc[valid, sp],
                    n_repeats=20, random_state=RANDOM_STATE,
                    n_jobs=-1, scoring="f1",
                )
                for row, pm, ps in zip(perm_rows_imp,
                                       perm.importances_mean,
                                       perm.importances_std):
                    row[f"{sp}_perm_mean"] = round(float(pm), 5)
                    row[f"{sp}_perm_std"]  = round(float(ps), 5)

            perm_df = pd.DataFrame(perm_rows_imp)
            mean_cols = [c for c in perm_df.columns if c.endswith("_perm_mean")]
            perm_df["_avg"] = perm_df[mean_cols].mean(axis=1)
            perm_df = perm_df.sort_values("_avg", ascending=False).drop(columns="_avg")
            perm_df.to_csv(
                IMP_DIR / f"{track_key}_{tier}_perm_importance.csv", index=False)
            print(f"    ✓ Importance saved")

            # Save mean PR curves to parquet for plotting
            pr_rows = []
            for sp in SPECIES_LIST:
                curves = cv_out["pr_curves"][sp]
                for r, pm, ps in zip(curves["recall"],
                                     curves["prec_mean"],
                                     curves["prec_std"]):
                    pr_rows.append({
                        "ac": ac_name, "track": track_key, "tier": tier,
                        "species": sp,
                        "recall": round(float(r), 4),
                        "prec_mean": round(float(pm), 5),
                        "prec_std":  round(float(ps), 5),
                        "ap_mean":   round(cv_out["prauc_per_label"][sp]["mean"], 5),
                        "ap_std":    round(cv_out["prauc_per_label"][sp]["std"],  5),
                    })
            pd.DataFrame(pr_rows).to_parquet(
                CSV_DIR / f"pr_curves_{track_key}_{tier}.parquet", index=False)
            print(f"    ✓ PR curves saved")

            # Accumulate rows
            for sp in SPECIES_LIST:
                species_rows.append({
                    "ac": ac_name, "track": track_key, "tier": tier,
                    "species": sp,
                    "threshold":     SPECIES_THRESHOLDS[sp][tier],
                    "n_positive":    int(Y_df[sp].sum()),
                    "cv_f1_mean":    cv_out["f1_per_label"][sp]["mean"],
                    "cv_f1_median":  cv_out["f1_per_label"][sp]["median"],
                    "cv_f1_std":     cv_out["f1_per_label"][sp]["std"],
                    "cv_prauc_mean":   cv_out["prauc_per_label"][sp]["mean"],
                    "cv_prauc_median": cv_out["prauc_per_label"][sp]["median"],
                    "cv_prauc_std":    cv_out["prauc_per_label"][sp]["std"],
                    "cv_recall_mean":   cv_out["recall_per_label"][sp]["mean"],
                    "cv_recall_median": cv_out["recall_per_label"][sp]["median"],
                    "cv_recall_std":    cv_out["recall_per_label"][sp]["std"],
                    "cv_precision_mean":   cv_out["precision_per_label"][sp]["mean"],
                    "cv_precision_median": cv_out["precision_per_label"][sp]["median"],
                    "cv_precision_std":    cv_out["precision_per_label"][sp]["std"],
                })
                for fold_i, (f1_v, prauc_v, recall_v, precision_v) in enumerate(zip(
                        cv_out["f1_per_label"][sp]["_folds"],
                        cv_out["prauc_per_label"][sp]["_folds"],
                        cv_out["recall_per_label"][sp]["_folds"],
                        cv_out["precision_per_label"][sp]["_folds"])):
                    fold_rows.append({
                        "ac": ac_name, "track": track_key, "tier": tier,
                        "species": sp, "fold": fold_i,
                        "f1": f1_v, "prauc": prauc_v,
                        "recall": recall_v, "precision": precision_v,
                    })

            global_rows.append({
                "ac": ac_name, "track": track_key, "tier": tier,
                "n_samples":             n_total,
                "n_features":            len(features),
                "n_folds_used":          cv_out["n_folds_used"],
                "cv_f1_macro_mean":      cv_out["f1_macro"]["mean"],
                "cv_f1_macro_median":    cv_out["f1_macro"]["median"],
                "cv_f1_macro_std":       cv_out["f1_macro"]["std"],
                "cv_prauc_macro_mean":   cv_out["prauc_macro"]["mean"],
                "cv_prauc_macro_median": cv_out["prauc_macro"]["median"],
                "cv_prauc_macro_std":    cv_out["prauc_macro"]["std"],
                "cv_hamming_mean":       cv_out["hamming"]["mean"],
                "cv_hamming_std":        cv_out["hamming"]["std"],
                "cv_subset_acc_mean":    cv_out["subset_acc"]["mean"],
                "cv_subset_acc_std":     cv_out["subset_acc"]["std"],
                "cv_jaccard_mean":       cv_out["jaccard"]["mean"],
                "cv_jaccard_std":        cv_out["jaccard"]["std"],
            })

    print(f"\n  ✓ {ac_name} complete")

# ──────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

global_df  = pd.DataFrame(global_rows)
species_df = pd.DataFrame(species_rows)
fold_df    = pd.DataFrame(fold_rows)

global_df = round_floats(global_df.sort_values(
    ["tier", "ac", "track"]).reset_index(drop=True))
global_df.to_csv(CSV_DIR / "global_comparison.csv", index=False)
print(f"  global_comparison.csv  ({len(global_df)} rows)")

species_df = round_floats(species_df.sort_values(
    ["tier", "ac", "species", "track"]).reset_index(drop=True))
species_df.to_csv(CSV_DIR / "species_comparison.csv", index=False)
print(f"  species_comparison.csv ({len(species_df)} rows)")

if not species_df.empty:
    df_only = species_df[species_df["track"].str.endswith("_only")]
    df_roms = species_df[species_df["track"].str.endswith("_roms")]
    key = ["ac", "tier", "species"]
    merged = df_only.merge(
        df_roms[key + ["cv_f1_mean", "cv_f1_std",
                       "cv_prauc_mean", "cv_prauc_std"]],
        on=key, suffixes=("_only", "_roms"),
    )
    merged["delta_f1"]    = merged["cv_f1_mean_roms"]    - merged["cv_f1_mean_only"]
    merged["delta_prauc"] = merged["cv_prauc_mean_roms"] - merged["cv_prauc_mean_only"]
    delta_df = merged[
        key + ["n_positive", "threshold",
               "cv_f1_mean_only", "cv_f1_std_only",
               "cv_f1_mean_roms", "cv_f1_std_roms", "delta_f1",
               "cv_prauc_mean_only", "cv_prauc_std_only",
               "cv_prauc_mean_roms", "cv_prauc_std_roms", "delta_prauc"]
    ].sort_values(["tier", "ac", "delta_f1"], ascending=[True, True, False])
    round_floats(delta_df).to_csv(CSV_DIR / "roms_delta_summary.csv", index=False)
    print(f"  roms_delta_summary.csv")

if not species_df.empty:
    df_only = species_df[species_df["track"].str.endswith("_only")]
    df_roms = species_df[species_df["track"].str.endswith("_roms")]
    key = ["ac", "tier", "species"]
    recall_merged = df_only.merge(
        df_roms[key + ["cv_recall_mean", "cv_recall_std"]],
        on=key, suffixes=("_only", "_roms"),
    )
    recall_merged["delta_recall"] = (
        recall_merged["cv_recall_mean_roms"] - recall_merged["cv_recall_mean_only"]
    )
    recall_merged["extra_blooms_per_100"] = (
        recall_merged["delta_recall"] * 100
    ).round(1)
    recall_summary = recall_merged[
        key + ["n_positive", "threshold",
               "cv_recall_mean_only", "cv_recall_mean_roms",
               "delta_recall", "extra_blooms_per_100"]
    ].sort_values(["tier", "ac", "extra_blooms_per_100"], ascending=[True, True, False])
    round_floats(recall_summary).to_csv(CSV_DIR / "recall_gain_summary.csv", index=False)
    print(f"  recall_gain_summary.csv")

fold_df = round_floats(fold_df.sort_values(
    ["tier", "ac", "species", "track", "fold"]).reset_index(drop=True))
fold_df.to_csv(CSV_DIR / "fold_level_results.csv", index=False)
print(f"  fold_level_results.csv ({len(fold_df)} rows)")

# Console summary
print("\n" + "=" * 80)
print("MACRO CV F1 SUMMARY")
print("=" * 80)
if not global_df.empty:
    pivot = global_df.pivot_table(
        index=["ac", "tier"], columns="track",
        values="cv_f1_macro_mean", aggfunc="first").round(3)
    print(pivot.to_string())
    print("\n  ROMS delta:")
    for tier in THRESHOLD_TIERS:
        for ac in AC_PREFIXES:
            sub = global_df[(global_df["ac"] == ac) & (global_df["tier"] == tier)]
            only = sub[sub["track"].str.endswith("_only")]["cv_f1_macro_mean"].values
            roms = sub[sub["track"].str.endswith("_roms")]["cv_f1_macro_mean"].values
            if len(only) and len(roms):
                print(f"    {ac:10s} {tier:7s}  ΔF1={roms[0]-only[0]:+.3f}")

print("\nDONE — run clf_plots.py to generate figures.")