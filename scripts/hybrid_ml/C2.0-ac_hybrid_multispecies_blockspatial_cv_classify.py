"""
HAB Multi-Label Classification — Model A: Binary Relevance RF / ANN
====================================================================
Metric-block spatial cross-validation, rebuilt around leak-free spatial
partitioning principles (see geospatialmachinelearning.com, "Reducing
Spatial Leakage in Model Training").

What changed vs. the lat/lon-rounding version
-----------------------------------------------
  - Coordinates are projected to a metric CRS (UTM 19N) before any
    blocking. Degree rounding produces unequal cell sizes (lon cells
    shrink with cos(latitude)); metres do not.
  - Block size is derived from a Moran's I sweep over candidate lag
    distances (block size = 2x the radius where I decays toward zero),
    not picked arbitrarily. Toggle with ENABLE_MORAN_SWEEP; falls back
    to BLOCK_SIZE_M_FALLBACK if libpysal/esda are unavailable.
  - Sparse blocks (<MIN_BLOCK_OBS) are dropped before CV rather than
    silently inflating the spatial-group count.
  - Every track/tier now runs both spatial (GroupKFold on block_id) and
    random (plain KFold) CV, and reports the ratio as a leakage gate —
    a sustained large gap means the old approach's scores were inflated
    by spatial proximity, not genuine generalisation.
  - The previous "sgkf" was named as if stratified but was plain
    GroupKFold; that's removed here (strat_labels is no longer
    computed or passed to the splitter).

Six target species, three AC pipelines x +-ROMS = six tracks, same as
before. CSV/plot output conventions unchanged in spirit, written under
a new output root so old results aren't overwritten.
"""

import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Transformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, hamming_loss, accuracy_score, jaccard_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────────
# Model selection
# ──────────────────────────────────────────────────────────────────────────────
MODEL_TYPE = "RF"   # "RF" or "ANN"
N_EST = 100
ANN_HIDDEN_LAYERS  = (100, 50)
ANN_MAX_ITER       = 500
ANN_ALPHA          = 1e-3
ANN_EARLY_STOPPING = True

# ──────────────────────────────────────────────────────────────────────────────
# Spatial blocking config
# ──────────────────────────────────────────────────────────────────────────────
UTM_EPSG = 32619                 # UTM zone 19N — Gulf of Maine (~66W-72W)
ENABLE_MORAN_SWEEP = True        # set False if libpysal/esda unavailable
MORAN_CANDIDATE_RADII_M = [2_000, 5_000, 10_000, 20_000, 35_000, 50_000]
MORAN_TARGET_COL = "bloom_index" # log10(max species conc + 1), built below
MORAN_SAMPLE_N = 2_000           # subsample for tractable distance-band weights
MORAN_DECAY_THRESH = 0.1         # |I| below this = "decayed"
BLOCK_SIZE_M_FALLBACK = 10_000   # used if Moran sweep disabled/unavailable
MIN_BLOCK_OBS = 5                # drop blocks with fewer obs than this
RUN_RANDOM_BASELINE = True       # doubles CV cost; needed for the leakage gate
LEAKAGE_RATIO_GATE = 0.85        # spatial_f1 / random_f1 below this -> warn

N_CV = 10
MIN_BLOOM_SAMPLES = 5

print("=" * 80)
print(f"HAB MULTI-LABEL CLF — BINARY RELEVANCE {MODEL_TYPE} (METRIC-BLOCK SPATIAL CV)")
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
BASE_OUT = Path(f"datasets/GULF_OF_MAINE/ml_outputs/rf_spatial_cv_{MODEL_TYPE.lower()}")
CSV_DIR  = BASE_OUT / "csv_results"
IMP_DIR  = BASE_OUT / "importance"
DIAG_DIR = BASE_OUT / "diagnostics"
for d in [CSV_DIR, IMP_DIR, DIAG_DIR]:
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

DATASETS = ['mvco', 'gsodock', 'harpswell',  'fiddlers', 'mdibl',
  'ecoa', 'oceanalliance',  'radbot_mvco', 'radbot_ios', 'radbot_jeffreys_basin',
 'NESLTER_broadscale', 'azmp', 'gom'
 ]

# ──────────────────────────────────────────────────────────────────────────────
# Spatial blocking helpers
# ──────────────────────────────────────────────────────────────────────────────
def project_to_meters(df, lat_col="latitude", lon_col="longitude", epsg=UTM_EPSG):
    """Project lon/lat (EPSG:4326) to a metric UTM CRS. All distance-based
    operations (block size, future buffer zones) must use these columns,
    never raw degrees."""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_m, y_m = transformer.transform(df[lon_col].values, df[lat_col].values)
    df = df.copy()
    df["x_m"] = x_m
    df["y_m"] = y_m
    return df


def moran_sweep(df, target_col, candidate_radii_m, x_col="x_m", y_col="y_m",
                 sample_n=MORAN_SAMPLE_N, decay_thresh=MORAN_DECAY_THRESH):
    """Sweep Moran's I across candidate distance bands to locate the spatial
    correlation range. Returns the smallest radius where |I| drops below
    decay_thresh, or the largest candidate if I never decays in range.
    Subsamples for tractable distance-band weight construction."""
    try:
        import libpysal
        from esda.moran import Moran
    except ImportError:
        print("  ⚠ libpysal/esda not installed — skipping Moran's I sweep, "
              f"using BLOCK_SIZE_M_FALLBACK={BLOCK_SIZE_M_FALLBACK:,} m.")
        return None

    sub = df.dropna(subset=[target_col, x_col, y_col])
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=RANDOM_STATE)

    results = []
    for radius_m in candidate_radii_m:
        try:
            w = libpysal.weights.DistanceBand.from_array(
                sub[[x_col, y_col]].values, threshold=radius_m, silence_warnings=True)
            mi = Moran(sub[target_col].values, w)
            results.append((radius_m, mi.I, mi.p_sim))
            print(f"    radius {radius_m:>7,} m  ->  Moran's I = {mi.I:.3f}  p = {mi.p_sim:.4f}")
        except Exception as e:
            print(f"    radius {radius_m:>7,} m  ->  failed ({e})")
            results.append((radius_m, np.nan, np.nan))

    res_df = pd.DataFrame(results, columns=["radius_m", "moran_i", "p_value"])
    res_df.to_csv(DIAG_DIR / "moran_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res_df["radius_m"], res_df["moran_i"], marker="o")
    ax.axhline(decay_thresh, color="grey", linestyle="--", linewidth=1)
    ax.axhline(-decay_thresh, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag distance (m)")
    ax.set_ylabel("Moran's I")
    ax.set_title(f"Spatial autocorrelation of {target_col} vs. lag distance")
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "moran_sweep.png", dpi=200)
    plt.close(fig)

    decayed = res_df[res_df["moran_i"].abs() < decay_thresh]
    chosen_range = (decayed["radius_m"].min() if not decayed.empty
                     else res_df["radius_m"].max())
    print(f"  -> correlation range ~= {chosen_range:,.0f} m, "
          f"block size set to 2x = {2 * chosen_range:,.0f} m")
    return float(chosen_range)


def assign_metric_blocks(df, block_size_m, x_col="x_m", y_col="y_m"):
    bx = np.floor(df[x_col] / block_size_m).astype(int)
    by = np.floor(df[y_col] / block_size_m).astype(int)
    df = df.copy()
    df["block_id"] = bx.astype(str) + "_" + by.astype(str)
    return df


def merge_sparse_blocks(df, min_obs=MIN_BLOCK_OBS, block_col="block_id",
                         x_col="x_m", y_col="y_m", block_size_m=None,
                         max_merge_factor=3.0):
    """Merge sparse blocks (<min_obs) into their nearest sufficiently-large
    neighbor by centroid distance, instead of discarding observations.
    Only drops observations as a last resort, for blocks with no eligible
    neighbor within max_merge_factor x block_size_m — i.e. genuinely
    isolated points, not a data-loss decision."""
    from scipy.spatial import cKDTree

    df = df.copy()
    counts = df[block_col].value_counts()
    sparse = counts[counts < min_obs].index
    large  = counts[counts >= min_obs].index

    if len(sparse) == 0:
        print("  No sparse blocks to merge.")
        return df
    if len(large) == 0:
        print("  ⚠ No blocks meet min_obs anywhere — cannot merge, leaving unchanged.")
        return df

    centroids = df.groupby(block_col)[[x_col, y_col]].mean()
    anchor_ids = list(large)
    tree = cKDTree(centroids.loc[anchor_ids].values)
    max_dist = max_merge_factor * block_size_m if block_size_m else np.inf

    remap = {}
    n_merged = 0
    for bid in sparse:
        dist, idx = tree.query(centroids.loc[bid].values)
        if dist <= max_dist:
            remap[bid] = anchor_ids[idx]
            n_merged += 1

    df[block_col] = df[block_col].map(lambda b: remap.get(b, b))

    still_sparse = df[block_col].value_counts()
    still_sparse = still_sparse[still_sparse < min_obs].index
    n_isolated_rows = int(df[block_col].isin(still_sparse).sum())
    if n_isolated_rows:
        df = df[~df[block_col].isin(still_sparse)].copy()
        print(f"  Merged {n_merged:,} sparse blocks into their nearest neighbor "
              f"(within {max_dist:,.0f} m). Dropped {n_isolated_rows:,} obs in "
              f"{len(still_sparse):,} isolated blocks with no eligible neighbor.")
    else:
        print(f"  Merged {n_merged:,} sparse blocks into their nearest neighbor "
              f"(within {max_dist:,.0f} m). No observations dropped.")
    return df


def verify_no_block_split(groups, fold_assignment):
    """Confirms GroupKFold never split a block across folds. This is a
    grouping-integrity check, not a buffer-distance check — adjacent
    blocks can still be geographically close across a fold boundary."""
    check = pd.DataFrame({"block": groups, "fold": fold_assignment})
    n_split = (check.groupby("block")["fold"].nunique() > 1).sum()
    if n_split:
        print(f"  ⚠ {n_split} blocks split across folds — grouping not respected.")
    else:
        print(f"  ✓ all {check['block'].nunique():,} blocks confined to a single fold")
    return n_split == 0


def plot_spatial_blocks(df, output_path, lat_col="latitude", lon_col="longitude",
                         block_col="block_id", figsize=(10, 8)):
    blocks = df[block_col].values
    unique_blocks = np.unique(blocks)
    rng = np.random.RandomState(0)
    palette_idx = rng.permutation(len(unique_blocks)) % 20
    block_to_color = dict(zip(unique_blocks, palette_idx))
    color_idx = np.array([block_to_color[b] for b in blocks])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[lon_col], df[lat_col], c=color_idx, cmap="tab20",
               s=8, alpha=0.7, edgecolors="none", rasterized=True)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Metric spatial blocks (n={len(unique_blocks):,}, {len(df):,} obs)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved {output_path}")


def plot_cv_fold_assignment(df, groups, output_path, n_splits=N_CV,
                             lat_col="latitude", lon_col="longitude", figsize=(10, 8)):
    gkf = GroupKFold(n_splits=n_splits)
    fold_assignment = np.full(len(df), -1)
    for fold_i, (_, test_idx) in enumerate(gkf.split(df, groups=groups)):
        fold_assignment[test_idx] = fold_i

    verify_no_block_split(groups, fold_assignment)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(df[lon_col], df[lat_col], c=fold_assignment, cmap="tab10",
                    s=8, alpha=0.7, edgecolors="none", rasterized=True)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"GroupKFold test-fold assignment, {n_splits} folds (metric blocks)")
    fig.colorbar(sc, ax=ax, ticks=range(n_splits), label="Test fold index")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved {output_path}")
    return fold_assignment

# ──────────────────────────────────────────────────────────────────────────────
# Model helpers (unchanged logic from the original script)
# ──────────────────────────────────────────────────────────────────────────────
def build_model_template():
    if MODEL_TYPE == "RF":
        base = RandomForestClassifier(
            n_estimators=N_EST, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    elif MODEL_TYPE == "ANN":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=ANN_HIDDEN_LAYERS,
                max_iter=ANN_MAX_ITER,
                alpha=ANN_ALPHA,
                early_stopping=ANN_EARLY_STOPPING,
                random_state=RANDOM_STATE,
            )),
        ])
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE!r} (expected 'RF' or 'ANN')")
    return MultiOutputClassifier(base, n_jobs=-1)


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
    print(f"{prefix}  Rows >=1 label: {any_bloom:,} ({any_bloom/n*100:.1f}%)  "
          f">=2 labels: {multi:,} ({multi/n*100:.1f}%)")


def _get_proba_matrix(model, X):
    return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])


def round_floats(df, decimals=5):
    fc = df.select_dtypes(include="float").columns
    df[fc] = df[fc].round(decimals)
    return df


def make_strat_label(Y, n_splits, other_thresh=None):
    """Bit-pack the multi-label Y (N x n_classes) into a single integer per
    row so StratifiedGroupKFold (which expects 1-D y) can approximately
    balance label combinations across folds. Combinations rarer than
    other_thresh are collapsed into a single catch-all bucket so the
    stratifier never encounters a stratum too small to split."""
    if other_thresh is None:
        other_thresh = n_splits
    powers = 2 ** np.arange(Y.shape[1])
    combo = Y.values @ powers
    vals, counts = np.unique(combo, return_counts=True)
    rare = vals[counts < other_thresh]
    strat = combo.copy()
    strat[np.isin(strat, rare)] = -1
    return strat.astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# CV runner — spatial (groups given) or random (groups=None) baseline
# ──────────────────────────────────────────────────────────────────────────────
def run_cv(X, Y, groups=None, n_splits=N_CV):
    if groups is not None:
        n_groups = len(np.unique(groups))
        strat_labels = make_strat_label(Y, n_splits=n_splits)
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            split_iter = list(splitter.split(X, strat_labels, groups))
        except ValueError as e:
            print(f"    ⚠ StratifiedGroupKFold failed ({e}) — "
                  f"falling back to plain GroupKFold.")
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = list(splitter.split(X, Y, groups=groups))
    else:
        n_groups = None
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(X, Y))

    model_template = build_model_template()

    fold_f1_macro, fold_hamming, fold_subset_acc = [], [], []
    fold_jaccard, fold_prauc_macro = [], []
    fold_f1_sp      = {sp: [] for sp in SPECIES_LIST}
    fold_prauc_sp   = {sp: [] for sp in SPECIES_LIST}
    fold_n_pos_sp   = {sp: [] for sp in SPECIES_LIST}
    fold_n_zero_positive_species  = []
    fold_n_excluded_train_species = []
    last_fold = None

    RECALL_GRID = np.linspace(0, 1, 100)
    fold_prec_sp = {sp: [] for sp in SPECIES_LIST}

    for train_idx, test_idx in split_iter:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]

        # Only species with both classes present in the training fold can be
        # fit. Previously the whole fold was dropped if ANY one species was
        # degenerate; that throws away the other 4 species' valid signal and
        # silently shrinks n_folds_used. Now only the affected species is
        # excluded from this fold (model fit + scoring), not the whole fold.
        valid_species = [sp for sp in SPECIES_LIST if Y_tr[sp].nunique() >= 2]
        n_excluded_train = len(SPECIES_LIST) - len(valid_species)
        if not valid_species:
            continue

        model = copy.deepcopy(model_template)
        model.fit(X_tr, Y_tr[valid_species])
        Y_pred  = pd.DataFrame(model.predict(X_te),
                               columns=valid_species, index=Y_te.index)
        Y_proba = _get_proba_matrix(model, X_te)  # columns aligned to valid_species order
        last_fold = (Y_te, Y_pred, Y_proba, model, valid_species)

        fold_n_excluded_train_species.append(n_excluded_train)
        fold_hamming.append(hamming_loss(Y_te[valid_species], Y_pred))
        fold_subset_acc.append(accuracy_score(Y_te[valid_species], Y_pred))
        fold_jaccard.append(
            jaccard_score(Y_te[valid_species], Y_pred, average="macro", zero_division=0))

        label_prauc = []
        label_f1 = []
        n_zero_pos = 0
        for j, sp in enumerate(valid_species):
            y_true_sp = Y_te[sp].values
            fold_n_pos_sp[sp].append(int(y_true_sp.sum()))
            # No true positives in this fold's test set: F1 is undefined here,
            # not zero. Scoring it 0 would penalise a correct "no bloom"
            # prediction as a failure. Exclude from this fold's macro average
            # instead, consistent with how PR-AUC already handles this case.
            if y_true_sp.sum() == 0:
                n_zero_pos += 1
                fold_f1_sp[sp].append(np.nan)
                fold_prauc_sp[sp].append(np.nan)
                continue
            f1_sp = f1_score(y_true_sp, Y_pred[sp].values, zero_division=0)
            fold_f1_sp[sp].append(f1_sp)
            label_f1.append(f1_sp)
            if y_true_sp.sum() == len(y_true_sp):
                fold_prauc_sp[sp].append(np.nan)
                continue
            ap = average_precision_score(y_true_sp, Y_proba[:, j])
            fold_prauc_sp[sp].append(ap)
            label_prauc.append(ap)
            prec, rec, _ = precision_recall_curve(y_true_sp, Y_proba[:, j])
            prec_interp = np.interp(RECALL_GRID, rec[::-1], prec[::-1])
            fold_prec_sp[sp].append(prec_interp)

        # Species excluded from training this fold get no score at all
        # (not 0, not a test-side NaN) — they were never fit.
        for sp in SPECIES_LIST:
            if sp not in valid_species:
                fold_f1_sp[sp].append(np.nan)
                fold_prauc_sp[sp].append(np.nan)
                fold_n_pos_sp[sp].append(int(Y_te[sp].sum()))

        fold_n_zero_positive_species.append(n_zero_pos)
        fold_f1_macro.append(np.nanmean(label_f1) if label_f1 else np.nan)
        fold_prauc_macro.append(np.nanmean(label_prauc) if label_prauc else np.nan)

    def stats(vals):
        arr = np.array(vals, dtype=float)
        return {
            "mean":   float(np.nanmean(arr)) if len(arr) else np.nan,
            "median": float(np.nanmedian(arr)) if len(arr) else np.nan,
            "std":    float(np.nanstd(arr)) if len(arr) else np.nan,
            "_folds": vals,
        }

    cv_out = {
        "f1_macro":    stats(fold_f1_macro),
        "hamming":     stats(fold_hamming),
        "subset_acc":  stats(fold_subset_acc),
        "jaccard":     stats(fold_jaccard),
        "prauc_macro": stats(fold_prauc_macro),
        "f1_per_label":    {sp: stats(fold_f1_sp[sp])    for sp in SPECIES_LIST},
        "prauc_per_label": {sp: stats(fold_prauc_sp[sp]) for sp in SPECIES_LIST},
        "n_positive_per_label": fold_n_pos_sp,
        "n_folds_used":    len(fold_f1_macro),
        "n_spatial_groups": n_groups,
        "avg_zero_positive_species_per_fold": float(np.mean(fold_n_zero_positive_species))
                                                if fold_n_zero_positive_species else np.nan,
        "avg_train_excluded_species_per_fold": float(np.mean(fold_n_excluded_train_species))
                                                 if fold_n_excluded_train_species else np.nan,
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

# ──────────────────────────────────────────────────────────────────────────────
# Load & filter
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nLoading: {COMPOSITE_PATH}")
df_raw = pd.read_parquet(COMPOSITE_PATH)
print(f"  Raw shape: {df_raw.shape[0]:,} x {df_raw.shape[1]}")

df_raw = df_raw[df_raw["dashboardIdName"].isin(DATASETS)]

if "delta_days" in df_raw.columns:
    df_raw = df_raw[df_raw["delta_days"] == 0].copy()
    print(f"  After delta_days==0: {df_raw.shape[0]:,} rows")

df_raw["date"] = pd.to_datetime(df_raw["obs_date"])
df_raw["doy_sin"] = np.sin(2 * np.pi * df_raw["date"].dt.dayofyear / 365)
df_raw["doy_cos"] = np.cos(2 * np.pi * df_raw["date"].dt.dayofyear / 365)

c2rcc_rename = {f"C_{s}": f"C_{d}" for s, d in BAND_NUM_TO_WL.items()
                if f"C_{s}" in df_raw.columns}
if c2rcc_rename:
    df_raw = df_raw.rename(columns=c2rcc_rename)

if "location_id" in df_raw.columns and "drqnp" in df_raw["location_id"].values:
    df_raw = df_raw[df_raw["location_id"] != "drqnp"].copy()

assert "latitude" in df_raw.columns and "longitude" in df_raw.columns, (
    "df_raw must contain 'latitude' and 'longitude' columns for spatial CV."
)

# ── Metric-block spatial grouping ────────────────────────────────────────────
print("\n" + "=" * 60)
print("METRIC SPATIAL BLOCKING")
print("=" * 60)

df_raw = project_to_meters(df_raw)

present_species = [sp for sp in SPECIES_LIST if sp in df_raw.columns]
df_raw[MORAN_TARGET_COL] = np.log10(
    df_raw[present_species].fillna(0).max(axis=1) + 1
)

if ENABLE_MORAN_SWEEP:
    print("  Running Moran's I sweep to calibrate block size...")
    corr_range_m = moran_sweep(df_raw, MORAN_TARGET_COL, MORAN_CANDIDATE_RADII_M)
    BLOCK_SIZE_M = 2 * corr_range_m if corr_range_m is not None else BLOCK_SIZE_M_FALLBACK
else:
    print(f"  Moran's I sweep disabled — using BLOCK_SIZE_M_FALLBACK={BLOCK_SIZE_M_FALLBACK:,} m")
    BLOCK_SIZE_M = BLOCK_SIZE_M_FALLBACK

print(f"  Final block size: {BLOCK_SIZE_M:,.0f} m")
df_raw = assign_metric_blocks(df_raw, BLOCK_SIZE_M)
df_raw = merge_sparse_blocks(df_raw, block_size_m=BLOCK_SIZE_M)

n_raw_blocks = df_raw["block_id"].nunique()
print(f"  Spatial blocks (all rows): {n_raw_blocks:,}")

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
print(f"  Spatial blocks (common): {df_common['block_id'].nunique():,}")

roms_ok = "roms_available" in df_common.columns
df_common_roms = (df_common[df_common["roms_available"] == True].copy()
                  if roms_ok else df_common.copy())
print(f"  Common rows with ROMS: {len(df_common_roms):,}")
print(f"  Spatial blocks (ROMS): {df_common_roms['block_id'].nunique():,}")

# Diagnostic plots — generated once on the full common+ROMS dataset, since
# block structure doesn't depend on AC or tier.
print("\n  Generating spatial diagnostic plots...")
plot_spatial_blocks(df_common_roms, DIAG_DIR / "spatial_blocks_map.png")
plot_cv_fold_assignment(
    df_common_roms, df_common_roms["block_id"].values,
    DIAG_DIR / "groupkfold_fold_assignment_map.png", n_splits=N_CV)

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
            df_common_roms[col] = df_common_roms[col].fillna(df_common_roms[col].median())
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

    print(f"\n{'='*70}")
    print(f"  AC: {ac_name}  |  spectral: {len(spec_only)}  |  ROMS: {len(roms_feats)}")
    print(f"{'='*70}")

    if not spec_only:
        print(f"  No spectral features — skipping.")
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

        print(f"\n  -- Track: {track_key}  |  n={len(df_in):,}  |  features={len(features)}")

        for tier in THRESHOLD_TIERS:
            print(f"\n    Tier: {tier.upper()}")
            Y_df, valid_rows = build_label_matrix(df_in, tier)
            X_all = df_in[valid_rows][features].copy()
            groups_all = df_in[valid_rows]["block_id"].values
            n_groups_cv = len(np.unique(groups_all))

            n_total = len(X_all)
            if n_total < 50:
                print(f"    Only {n_total} rows — skipping.")
                continue

            print_label_stats(Y_df, tier, prefix="    ")
            print(f"    Spatial blocks available: {n_groups_cv:,}")
            if n_groups_cv < 10:
                print(f"    ⚠ Only {n_groups_cv} spatial blocks — spatial CV may be unreliable.")

            skipped = [sp for sp in SPECIES_LIST if Y_df[sp].sum() < MIN_BLOOM_SAMPLES]
            if skipped:
                print(f"    ⚠ <{MIN_BLOOM_SAMPLES} positives (unreliable): {skipped}")

            print(f"    Running {N_CV}-fold spatial (block) CV ({MODEL_TYPE})...")
            cv_spatial, last_fold = run_cv(X_all, Y_df, groups=groups_all)

            cv_random = None
            if RUN_RANDOM_BASELINE:
                print(f"    Running {N_CV}-fold random CV baseline ({MODEL_TYPE})...")
                cv_random, _ = run_cv(X_all, Y_df, groups=None)

            print(f"    Spatial  macro F1 : {cv_spatial['f1_macro']['mean']:.3f} "
                  f"± {cv_spatial['f1_macro']['std']:.3f}")
            print(f"    Avg species/fold with zero test-set positives: "
                  f"{cv_spatial['avg_zero_positive_species_per_fold']:.1f} / {len(SPECIES_LIST)}  "
                  f"(excluded from that fold's macro F1, not scored as 0)")
            print(f"    Avg species/fold excluded from training (no class variance): "
                  f"{cv_spatial['avg_train_excluded_species_per_fold']:.1f} / {len(SPECIES_LIST)}")
            if cv_random:
                print(f"    Random   macro F1 : {cv_random['f1_macro']['mean']:.3f} "
                      f"± {cv_random['f1_macro']['std']:.3f}")
                ratio = (cv_spatial['f1_macro']['mean'] / cv_random['f1_macro']['mean']
                         if cv_random['f1_macro']['mean'] else np.nan)
                gate_pass = ratio >= LEAKAGE_RATIO_GATE
                print(f"    Leakage ratio (spatial/random): {ratio:.3f}  "
                      f"{'OK' if gate_pass else f'BELOW GATE ({LEAKAGE_RATIO_GATE})'}")
            else:
                ratio, gate_pass = np.nan, None

            for sp in SPECIES_LIST:
                print(f"      {sp:<30} "
                      f"F1={cv_spatial['f1_per_label'][sp]['mean']:.3f}±"
                      f"{cv_spatial['f1_per_label'][sp]['std']:.3f}  "
                      f"AP={cv_spatial['prauc_per_label'][sp]['mean']:.3f}±"
                      f"{cv_spatial['prauc_per_label'][sp]['std']:.3f}")

            # Final model on all data -> permutation importance
            final_model = build_model_template()
            final_model.fit(X_all, Y_df)

            Y_te_last, Y_pred_last, Y_proba_last, model_last, valid_species_last = last_fold
            perm_rows_imp = [{"feature": f} for f in features]
            for sp in SPECIES_LIST:
                if sp not in valid_species_last:
                    for row in perm_rows_imp:
                        row[f"{sp}_perm_mean"] = np.nan
                        row[f"{sp}_perm_std"]  = np.nan
                    continue
                est   = model_last.estimators_[valid_species_last.index(sp)]
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
                for row, pm, ps in zip(perm_rows_imp, perm.importances_mean, perm.importances_std):
                    row[f"{sp}_perm_mean"] = round(float(pm), 5)
                    row[f"{sp}_perm_std"]  = round(float(ps), 5)

            perm_df = pd.DataFrame(perm_rows_imp)
            mean_cols = [c for c in perm_df.columns if c.endswith("_perm_mean")]
            perm_df["_avg"] = perm_df[mean_cols].mean(axis=1)
            perm_df = perm_df.sort_values("_avg", ascending=False).drop(columns="_avg")
            perm_df.to_csv(IMP_DIR / f"{track_key}_{tier}_perm_importance.csv", index=False)
            print(f"    Importance saved")

            pr_rows = []
            for sp in SPECIES_LIST:
                curves = cv_spatial["pr_curves"][sp]
                for r, pm, ps in zip(curves["recall"], curves["prec_mean"], curves["prec_std"]):
                    pr_rows.append({
                        "ac": ac_name, "track": track_key, "tier": tier, "species": sp,
                        "recall": round(float(r), 4),
                        "prec_mean": round(float(pm), 5),
                        "prec_std":  round(float(ps), 5),
                        "ap_mean":   round(cv_spatial["prauc_per_label"][sp]["mean"], 5),
                        "ap_std":    round(cv_spatial["prauc_per_label"][sp]["std"],  5),
                    })
            pd.DataFrame(pr_rows).to_parquet(
                CSV_DIR / f"pr_curves_{track_key}_{tier}.parquet", index=False)
            print(f"    PR curves saved")

            for sp in SPECIES_LIST:
                species_rows.append({
                    "ac": ac_name, "track": track_key, "tier": tier, "species": sp,
                    "threshold":     SPECIES_THRESHOLDS[sp][tier],
                    "n_positive":    int(Y_df[sp].sum()),
                    "n_spatial_groups": n_groups_cv,
                    "cv_f1_mean":    cv_spatial["f1_per_label"][sp]["mean"],
                    "cv_f1_median":  cv_spatial["f1_per_label"][sp]["median"],
                    "cv_f1_std":     cv_spatial["f1_per_label"][sp]["std"],
                    "cv_prauc_mean":   cv_spatial["prauc_per_label"][sp]["mean"],
                    "cv_prauc_median": cv_spatial["prauc_per_label"][sp]["median"],
                    "cv_prauc_std":    cv_spatial["prauc_per_label"][sp]["std"],
                })
                for fold_i, (f1_v, prauc_v, n_pos_v) in enumerate(zip(
                        cv_spatial["f1_per_label"][sp]["_folds"],
                        cv_spatial["prauc_per_label"][sp]["_folds"],
                        cv_spatial["n_positive_per_label"][sp])):
                    fold_rows.append({
                        "ac": ac_name, "track": track_key, "tier": tier,
                        "species": sp, "fold": fold_i,
                        "n_positive": n_pos_v, "f1": f1_v, "prauc": prauc_v,
                    })

            global_rows.append({
                "ac": ac_name, "track": track_key, "tier": tier,
                "n_samples":             n_total,
                "n_features":            len(features),
                "n_folds_used":          cv_spatial["n_folds_used"],
                "n_spatial_groups":      cv_spatial["n_spatial_groups"],
                "block_size_m":          BLOCK_SIZE_M,
                "avg_zero_pos_species_per_fold": cv_spatial["avg_zero_positive_species_per_fold"],
                "avg_train_excluded_species_per_fold": cv_spatial["avg_train_excluded_species_per_fold"],
                "cv_f1_macro_mean":      cv_spatial["f1_macro"]["mean"],
                "cv_f1_macro_median":    cv_spatial["f1_macro"]["median"],
                "cv_f1_macro_std":       cv_spatial["f1_macro"]["std"],
                "cv_prauc_macro_mean":   cv_spatial["prauc_macro"]["mean"],
                "cv_prauc_macro_median": cv_spatial["prauc_macro"]["median"],
                "cv_prauc_macro_std":    cv_spatial["prauc_macro"]["std"],
                "cv_hamming_mean":       cv_spatial["hamming"]["mean"],
                "cv_hamming_std":        cv_spatial["hamming"]["std"],
                "cv_subset_acc_mean":    cv_spatial["subset_acc"]["mean"],
                "cv_subset_acc_std":     cv_spatial["subset_acc"]["std"],
                "cv_jaccard_mean":       cv_spatial["jaccard"]["mean"],
                "cv_jaccard_std":        cv_spatial["jaccard"]["std"],
                "random_f1_macro_mean":  cv_random["f1_macro"]["mean"] if cv_random else np.nan,
                "random_f1_macro_std":   cv_random["f1_macro"]["std"] if cv_random else np.nan,
                "leakage_ratio":         ratio,
                "leakage_gate_pass":     gate_pass,
            })

    print(f"\n  {ac_name} complete")

# ──────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

global_df  = pd.DataFrame(global_rows)
species_df = pd.DataFrame(species_rows)
fold_df    = pd.DataFrame(fold_rows)

global_df = round_floats(global_df.sort_values(["tier", "ac", "track"]).reset_index(drop=True))
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
        df_roms[key + ["cv_f1_mean", "cv_f1_std", "cv_prauc_mean", "cv_prauc_std"]],
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

fold_df = round_floats(fold_df.sort_values(
    ["tier", "ac", "species", "track", "fold"]).reset_index(drop=True))
fold_df.to_csv(CSV_DIR / "fold_level_results.csv", index=False)
print(f"  fold_level_results.csv ({len(fold_df)} rows)")

print("\n" + "=" * 80)
print("MACRO CV F1 SUMMARY")
print("=" * 80)
if not global_df.empty:
    pivot = global_df.pivot_table(
        index=["ac", "tier"], columns="track",
        values="cv_f1_macro_mean", aggfunc="first").round(3)
    print(pivot.to_string())

    n_gate_fail = (global_df["leakage_gate_pass"] == False).sum()
    if n_gate_fail:
        print(f"\n  ⚠ {n_gate_fail} track/tier combinations fell below the "
              f"leakage gate ({LEAKAGE_RATIO_GATE}) — spatial F1 dropped "
              f"substantially relative to random-split F1. See "
              f"global_comparison.csv -> leakage_ratio.")

print(f"\nDONE ({MODEL_TYPE}) — run clf_plots.py to generate comparison figures.")