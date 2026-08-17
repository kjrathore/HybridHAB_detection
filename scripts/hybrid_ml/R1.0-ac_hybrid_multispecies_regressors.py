"""
HAB Multi-Output Regression — Model B: Bloom-Concentration Regression
=======================================================================
Predicts log10(cells/L + 1) per HAB species on bloom-only rows
(concentration > species warning threshold).

Regressor: RandomForestRegressor wrapped in MultiOutputRegressor
           (one estimator per species).

Six tracks:
  C2RCC_only  | C2RCC_roms
  ACOLITE_only | ACOLITE_roms
  Baseline_only| Baseline_roms

CV design
----------
  50-fold ShuffleSplit (80/20).  No separate held-out test split —
  CV mean/std IS the reported performance.  Final model is fit on ALL
  bloom rows and used only for feature importance (Gini).
  Scatter and residual plots come from the LAST CV fold.

CSV layout (designed for cross-AC / _only-vs-_roms comparison)
----------------------------------------------------------------
  global_comparison.csv
      One row per (ac, track, model).
      Sorted so C2RCC_only and C2RCC_roms sit adjacent — delta is one
      row subtraction.
      Columns: macro_cv_r2_mean, macro_cv_r2_std, macro_cv_rmse_mean,
               macro_cv_mae_mean, n_bloom_rows, n_features, n_species
      Note: RMSE and MAE in cells/L (back-transformed); R² on log scale.

  species_comparison.csv
      One row per (ac, track, model, species).
      Columns: cv_r2_mean/std, cv_rmse_mean/std, cv_mae_mean/std,
               n_bloom, n_folds
      Note: RMSE and MAE in cells/L; R² on log scale.

  model_comparison_wide.csv
      Pivot: index=(ac, species), columns={model}_{track} → cv_r2_mean.
      One glance shows RF_C2RCC_only vs RF_C2RCC_roms vs GBR_C2RCC_only…
      for every species.

  roms_delta_summary.csv
      One row per (ac, model, species).
      Columns: cv_r2_mean_only, cv_r2_mean_roms, delta_r2,
               cv_rmse_mean_only, cv_rmse_mean_roms, delta_rmse.
      Sorted descending by delta_r2 — biggest ROMS gains at the top.

Plots  (flat directory, prefixed, publication gray theme)
----------------------------------------------------------
  plots/{ac}_r2_boxplot.png    ← grouped R² boxplot _only vs _roms per species
  plots/{ac}_scatter.png       ← obs vs pred grid: rows=species, cols=tracks
  csv_results/roms_delta_heatmap.png

Importance  (flat directory)
------------------------------
  importance/{ac}_{track}_perm_importance.csv
      Rows = features, columns = {species}_perm_mean / {species}_perm_std.
      Sorted by mean importance averaged across species.

Outputs → hybrid_ml/rf_outputs_multilabel_regression/
  csv_results/
    global_comparison.csv
    species_comparison.csv
    roms_delta_summary.csv
    fold_level_results.csv
    model_comparison_wide.csv
  importance/
    {ac}_{track}_perm_importance.csv
"""

import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

print("=" * 80)
print("HAB MULTI-OUTPUT REGRESSION — RF (BLOOM-ONLY, CV-ONLY)")
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
BASE_OUT = Path("ml_outputs/HGB_regr")
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
SPECIES_LIST = list(SPECIES_THRESHOLDS.keys())

# REGRESSOR = MultiOutputRegressor(
#     RandomForestRegressor(
#         n_estimators=200, min_samples_split=5, min_samples_leaf=2,
#         max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1,
#     ), n_jobs=1,
# )
REGRESSOR = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=10, random_state=RANDOM_STATE,
    ), n_jobs=1,
)
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

N_CV = 50
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

def reg_metrics(y_true, y_pred):
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    r2   = r2_score(y_true_arr, y_pred_arr)
    true_cells = 10**y_true_arr - 1
    pred_cells = 10**y_pred_arr - 1
    rmse = np.sqrt(mean_squared_error(true_cells, pred_cells))
    mae  = mean_absolute_error(true_cells, pred_cells)
    bias = float(np.mean(y_pred_arr - y_true_arr))
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias}

def build_bloom_arrays(df, active_species, feat_cols):
    warn = pd.Series({sp: SPECIES_THRESHOLDS[sp]["warning"] for sp in active_species})
    bloom_mask = (df[active_species] > warn).any(axis=1)
    X = df.loc[bloom_mask, feat_cols].copy()
    Y = pd.DataFrame(index=X.index)
    for sp in active_species:
        sp_bloom = df.loc[X.index, sp] > SPECIES_THRESHOLDS[sp]["warning"]
        Y[sp] = np.where(sp_bloom, np.log10(df.loc[X.index, sp] + 1), np.nan)
    return X, Y

def run_cv(X, Y):
    ss = ShuffleSplit(n_splits=N_CV, test_size=0.20, random_state=RANDOM_STATE)
    fold_r2   = {sp: [] for sp in Y.columns}
    fold_rmse = {sp: [] for sp in Y.columns}
    fold_mae  = {sp: [] for sp in Y.columns}
    last_fold = None

    for train_idx, test_idx in ss.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]

        model = copy.deepcopy(REGRESSOR)
        model.fit(X_tr, Y_tr.fillna(Y_tr.mean()))
        Y_pred = pd.DataFrame(model.predict(X_te),
                              columns=Y.columns, index=Y_te.index)
        last_fold = (Y_te, Y_pred, model)

        for sp in Y.columns:
            valid = Y_te[sp].notna()
            if valid.sum() < 5:
                continue
            m = reg_metrics(Y_te.loc[valid, sp], Y_pred.loc[valid, sp])
            fold_r2[sp].append(m["r2"])
            fold_rmse[sp].append(m["rmse"])
            fold_mae[sp].append(m["mae"])

    cv_out = {}
    for sp in Y.columns:
        cv_out[sp] = {
            "cv_r2_mean":     np.nanmean(fold_r2[sp]),
            "cv_r2_median":   np.nanmedian(fold_r2[sp]),
            "cv_r2_std":      np.nanstd(fold_r2[sp]),
            "cv_rmse_mean":   np.nanmean(fold_rmse[sp]),
            "cv_rmse_median": np.nanmedian(fold_rmse[sp]),
            "cv_rmse_std":    np.nanstd(fold_rmse[sp]),
            "cv_mae_mean":    np.nanmean(fold_mae[sp]),
            "cv_mae_median":  np.nanmedian(fold_mae[sp]),
            "cv_mae_std":     np.nanstd(fold_mae[sp]),
            "n_folds":        len(fold_r2[sp]),
            "_fold_r2":       fold_r2[sp],
            "_fold_rmse":     fold_rmse[sp],
            "_fold_mae":      fold_mae[sp],
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
print(f"  Raw: {df_raw.shape[0]:,} × {df_raw.shape[1]}")

if "delta_days" in df_raw.columns:
    df_raw = df_raw[df_raw["delta_days"] == 0].copy()

df_raw["date"] = pd.to_datetime(df_raw["obs_date"])

c2rcc_rename = {f"C_{s}": f"C_{d}" for s, d in BAND_NUM_TO_WL.items()
                if f"C_{s}" in df_raw.columns}
if c2rcc_rename:
    df_raw = df_raw.rename(columns=c2rcc_rename)

if "location_id" in df_raw.columns and "drqnp" in df_raw["location_id"].values:
    df_raw = df_raw[df_raw["location_id"] != "drqnp"].copy()

# Common observation set
ac_valid_masks = {}
for ac, pfx in AC_PREFIXES.items():
    col = f"{pfx}_rrs_412"
    if col not in df_raw.columns:
        fb = [c for c in df_raw.columns
              if c.startswith(f"{pfx}_rrs_") and df_raw[c].notna().any()]
        col = fb[0] if fb else None
    ac_valid_masks[ac] = (df_raw[col].notna() if col
                          else pd.Series(False, index=df_raw.index))

df_common = df_raw[pd.concat(ac_valid_masks.values(), axis=1).all(axis=1)].copy()
print(f"  Common rows (all ACs): {len(df_common):,}")

roms_ok = "roms_available" in df_common.columns
df_common_roms = (df_common[df_common["roms_available"] == True].copy()
                  if roms_ok else df_common.copy())
print(f"  Common rows with ROMS: {len(df_common_roms):,}")

# Median imputation
print("  Applying median imputation...")
all_spec_cols = [c for ac, pfx in AC_PREFIXES.items()
                 for c in resolve_features(pfx, df_common)]
impute_cols = list(set(all_spec_cols + resolve_roms(df_common_roms)))
for col in impute_cols:
    if col in df_common.columns:
        n = df_common[col].isna().sum()
        if n > 0:
            df_common[col] = df_common[col].fillna(df_common[col].median())
    if col in df_common_roms.columns:
        n = df_common_roms[col].isna().sum()
        if n > 0:
            df_common_roms[col] = df_common_roms[col].fillna(
                df_common_roms[col].median())
print("  Imputation complete.")

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
        df_in     = tcfg["df"]
        feat_cols = tcfg["features"]

        if df_in.empty or not feat_cols:
            print(f"\n  Track {track_key}: no data, skipping.")
            continue

        print(f"\n  ── Track: {track_key}  |  n_all={len(df_in):,}")

        active_species = []
        for sp in SPECIES_LIST:
            if sp not in df_in.columns:
                continue
            n = (df_in[sp] > SPECIES_THRESHOLDS[sp]["warning"]).sum()
            if n >= MIN_BLOOM_SAMPLES:
                active_species.append(sp)
            else:
                print(f"    ⚠ {sp}: {n} bloom rows — skipping")

        if not active_species:
            print(f"    ⚠ No species qualify — skipping track.")
            continue

        X, Y = build_bloom_arrays(df_in, active_species, feat_cols)
        print(f"    Bloom-union rows: {len(X):,}")
        for sp in active_species:
            vals = Y[sp].dropna()
            print(f"      {sp:<30} n={len(vals):,}  "
                  f"log10=[{vals.min():.2f},{vals.max():.2f}]  "
                  f"mean={vals.mean():.2f}")

        if len(X) < 30:
            print(f"    ⚠ Too few rows — skipping.")
            continue

        print(f"    Running {N_CV}-fold CV...")
        cv_out, (Y_te_last, Y_pred_last, model_last) = run_cv(X, Y)

        for sp in active_species:
            print(f"      {sp:<30} "
                  f"R²={cv_out[sp]['cv_r2_mean']:.3f}±{cv_out[sp]['cv_r2_std']:.3f}  "
                  f"RMSE={cv_out[sp]['cv_rmse_mean']:.0f} cells/L")

        macro_r2   = np.nanmean([cv_out[sp]["cv_r2_mean"]   for sp in active_species])
        macro_rmse = np.nanmean([cv_out[sp]["cv_rmse_mean"] for sp in active_species])
        macro_mae  = np.nanmean([cv_out[sp]["cv_mae_mean"]  for sp in active_species])
        macro_r2_std = np.nanmean([cv_out[sp]["cv_r2_std"] for sp in active_species])
        macro_r2_med = np.nanmean([cv_out[sp]["cv_r2_median"] for sp in active_species])
        macro_rmse_med = np.nanmean([cv_out[sp]["cv_rmse_median"] for sp in active_species])
        macro_mae_med  = np.nanmean([cv_out[sp]["cv_mae_median"]  for sp in active_species])
        print(f"    Macro R²={macro_r2:.3f}  RMSE={macro_rmse:.0f} cells/L  MAE={macro_mae:.0f} cells/L")

        # Final model → permutation importance
        final_model = copy.deepcopy(REGRESSOR)
        final_model.fit(X, Y.fillna(Y.mean()))

        perm_rows_imp = [{"feature": f} for f in feat_cols]
        for sp, est in zip(active_species, model_last.estimators_):
            valid = Y_te_last[sp].notna()
            if valid.sum() < 5:
                for row in perm_rows_imp:
                    row[f"{sp}_perm_mean"] = np.nan
                    row[f"{sp}_perm_std"]  = np.nan
                continue
            perm = permutation_importance(
                est, X.loc[Y_te_last[valid].index],
                Y_te_last.loc[valid, sp],
                n_repeats=20, random_state=RANDOM_STATE,
                n_jobs=-1, scoring="r2",
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
        perm_df.to_csv(IMP_DIR / f"{track_key}_perm_importance.csv", index=False)
        print(f"    ✓ Importance saved")

        # Accumulate rows — also save last-fold predictions for scatter plots
        for sp in active_species:
            species_rows.append({
                "ac": ac_name, "track": track_key,
                "species": sp,
                "warning_thr":    SPECIES_THRESHOLDS[sp]["warning"],
                "n_bloom":        int(Y[sp].notna().sum()),
                "cv_r2_mean":     cv_out[sp]["cv_r2_mean"],
                "cv_r2_median":   cv_out[sp]["cv_r2_median"],
                "cv_r2_std":      cv_out[sp]["cv_r2_std"],
                "cv_rmse_mean":   cv_out[sp]["cv_rmse_mean"],
                "cv_rmse_median": cv_out[sp]["cv_rmse_median"],
                "cv_rmse_std":    cv_out[sp]["cv_rmse_std"],
                "cv_mae_mean":    cv_out[sp]["cv_mae_mean"],
                "cv_mae_median":  cv_out[sp]["cv_mae_median"],
                "cv_mae_std":     cv_out[sp]["cv_mae_std"],
                "n_folds":        cv_out[sp]["n_folds"],
            })
            for fold_i, (r2_v, rmse_v, mae_v) in enumerate(zip(
                    cv_out[sp]["_fold_r2"],
                    cv_out[sp]["_fold_rmse"],
                    cv_out[sp]["_fold_mae"])):
                fold_rows.append({
                    "ac": ac_name, "track": track_key,
                    "species": sp, "fold": fold_i,
                    "r2": r2_v, "rmse": rmse_v, "mae": mae_v,
                })

        global_rows.append({
            "ac": ac_name, "track": track_key,
            "n_bloom_rows":         len(X),
            "n_features":           len(feat_cols),
            "n_species":            len(active_species),
            "macro_cv_r2_mean":     macro_r2,
            "macro_cv_r2_median":   macro_r2_med,
            "macro_cv_r2_std":      macro_r2_std,
            "macro_cv_rmse_mean":   macro_rmse,
            "macro_cv_rmse_median": macro_rmse_med,
            "macro_cv_mae_mean":    macro_mae,
            "macro_cv_mae_median":  macro_mae_med,
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
    ["ac", "track"]).reset_index(drop=True))
global_df.to_csv(CSV_DIR / "global_comparison.csv", index=False)
print(f"  global_comparison.csv  ({len(global_df)} rows)")

species_df = round_floats(species_df.sort_values(
    ["ac", "species", "track"]).reset_index(drop=True))
species_df.to_csv(CSV_DIR / "species_comparison.csv", index=False)
print(f"  species_comparison.csv ({len(species_df)} rows)")

fold_df = round_floats(fold_df.sort_values(
    ["ac", "species", "track", "fold"]).reset_index(drop=True))
fold_df.to_csv(CSV_DIR / "fold_level_results.csv", index=False)
print(f"  fold_level_results.csv ({len(fold_df)} rows)")

if not species_df.empty:
    wide = species_df.pivot_table(
        index=["ac", "species"], columns="track",
        values="cv_r2_mean", aggfunc="first").round(5).reset_index()
    wide.to_csv(CSV_DIR / "model_comparison_wide.csv", index=False)
    print(f"  model_comparison_wide.csv")

    df_only = species_df[species_df["track"].str.endswith("_only")]
    df_roms = species_df[species_df["track"].str.endswith("_roms")]
    key = ["ac", "species"]
    merged = df_only.merge(
        df_roms[key + ["cv_r2_mean", "cv_rmse_mean", "cv_mae_mean",
                       "cv_r2_std", "cv_rmse_std"]],
        on=key, suffixes=("_only", "_roms"),
    )
    merged["delta_r2"]   = merged["cv_r2_mean_roms"]  - merged["cv_r2_mean_only"]
    merged["delta_rmse"] = merged["cv_rmse_mean_roms"] - merged["cv_rmse_mean_only"]
    delta_df = merged[
        key + ["n_bloom", "warning_thr",
               "cv_r2_mean_only", "cv_r2_std_only",
               "cv_r2_mean_roms", "cv_r2_std_roms", "delta_r2",
               "cv_rmse_mean_only", "cv_rmse_std_only",
               "cv_rmse_mean_roms", "cv_rmse_std_roms", "delta_rmse"]
    ].sort_values(["ac", "delta_r2"], ascending=[True, False])
    round_floats(delta_df).to_csv(CSV_DIR / "roms_delta_summary.csv", index=False)
    print(f"  roms_delta_summary.csv")

# Console summary
print("\n" + "=" * 80)
print("MACRO CV R² SUMMARY")
print("=" * 80)
if not global_df.empty:
    pivot = global_df.pivot_table(
        index="track", values="macro_cv_r2_mean", aggfunc="first").round(3)
    print(pivot.to_string())
    print("\n  ROMS delta R²:")
    for ac in AC_PREFIXES:
        sub = global_df[global_df["ac"] == ac]
        only = sub[sub["track"].str.endswith("_only")]["macro_cv_r2_mean"].values
        roms = sub[sub["track"].str.endswith("_roms")]["macro_cv_r2_mean"].values
        if len(only) and len(roms):
            print(f"    {ac:10s}  ΔR²={roms[0]-only[0]:+.3f}")

print("\nDONE — run reg_plots.py to generate figures.")