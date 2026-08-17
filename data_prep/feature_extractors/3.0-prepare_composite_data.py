"""
build_composite_features.py
============================
Merge all feature sets into a single composite DataFrame.

Sources
-------
1. ROMS    (R) : ifcb_roms_features.parquet        (53,575 rows – multi delta_days)  ← BASE
2. C2RCC   (C) : c2rcc_features_mvc.parquet        (21,660 rows)
3. ACOLITE (A) : acolite_features_mvc.parquet      (21,660 rows)
4. BAC     (B) : bac_features.parquet              (14,831 rows)

Merge key : (latitude, longitude, obs_date)

Species note
------------
Species columns originate from the same IFCB source but may differ slightly
between datasets due to different temporal aggregation windows. They are NOT
used as merge keys. After merging, species cols from satellite sources are
dropped in favour of the ROMS base copy (which comes directly from the IFCB
parquet with no additional aggregation).

Output
------
composite_features.parquet  – wide-format, one row per (lat, lon, date)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

BASE = Path("datasets/GULF_OF_MAINE/processed_data")

PATHS = {
    "roms":    BASE / "roms_features/ifcb_roms_features.parquet", #need not to run again.
    "c2rcc":   BASE / "sentinel_3_L2_past3days/c2rcc_features/c2rcc_features_mvc.parquet",
    "acolite": BASE / "sentinel_3_L2_past3days/acolite_features/acolite_features_mvc.parquet",
    "bac":     BASE / "sentinel_3_L2_past3days/bac_features/sh_s3_features.parquet",
}

OUT_PATH = BASE / "sentinel_3_L2_past3days/composite_features/composite_features.parquet"

# Column prefix per source (applied to overlapping satellite/index cols)
SUFFIX = {
    "acolite": "A",
    "bac":     "B",   
    "c2rcc":   "C",
    "roms":    "R",
}

# Merge key
MERGE_KEY = ["latitude", "longitude", "obs_date"]
# MERGE_KEY = ["location_id", "obs_date"]

# Species columns – kept from ROMS base only; dropped from satellite sources after merge
SPECIES_COLS = [
    "Alexandrium_catenella",
    "Dinophysis_acuminata",
    "Dinophysis_norvegica",
    "Karenia",
    "Pseudo-nitzschia",
    # "Margalefidinium",
    # "Mesodinium",
    # "Pseudo-nitzschia",
    # "Tripos_furca",
    # "Tripos_fusus",
    # "Tripos_muelleri",
]

# Columns to exclude from satellite sources (already in ROMS base or not needed)
SAT_DROP = set(MERGE_KEY + SPECIES_COLS + [
    "location_id", "dataset_id", "dataset_name",
    "dashboardIdName", "depth", "cruise", "sample_type", "data_available",
])

# BAC-unique L2 bio-optical products (not present in other sources – no prefix needed)
BAC_L2_COLS = ["CHL_OC4ME", "CHL_NN", "TSM_NN"]#, "ADG443_NN", "KD490_M07"]

# ROMS numeric features
ROMS_NUMERIC_COLS = [
    "roms_temp_sur", "roms_salt_sur", "roms_zeta",
    "roms_u_sur", "roms_v_sur",
    "roms_Uwind", "roms_Vwind", "roms_Pair",
    "roms_current_speed", "roms_current_dir",
    "roms_wind_speed", "roms_wind_dir",
    "roms_grid_dist_km",
]
ROMS_META_COLS = ["roms_date", "roms_source", "roms_file",
                  "roms_fhour", "roms_cycle", "roms_eta", "roms_xi"]

# ── ROMS filtering mode ──────────────────────────────────────────────────────
# "exact"     : keep only rows where delta_days == 0              [DEFAULT]
# "aggregate" : mean of all delta_days per (lat, lon, obs_date)
ROMS_MODE = "exact"
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# HELPERS
# =============================================================================

def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PATHS[name])
    print(f"  [{SUFFIX[name]}] {name:<7s}  {len(df):,} rows × {df.shape[1]} cols")
    return df


def round_key(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df = df.copy()
    df["latitude"]  = df["latitude"].round(decimals)
    df["longitude"] = df["longitude"].round(decimals)
    return df


def prefix_cols(df: pd.DataFrame, cols: list, src: str) -> pd.DataFrame:
    """Rename cols → {SUFFIX[src]}_{col}."""
    return df.rename(columns={c: f"{SUFFIX[src]}_{c}" for c in cols if c in df.columns})


def check_species_agreement(df_a: pd.DataFrame, df_b: pd.DataFrame,
                             label_a: str, label_b: str, tol: float = 1e-3):
    """
    Inner-join on MERGE_KEY and compare SPECIES_COLS values.
    Informational only – mismatches expected due to different aggregation windows.
    """
    sp_a = [c for c in SPECIES_COLS if c in df_a.columns]
    sp_b = [c for c in SPECIES_COLS if c in df_b.columns]
    common_sp = list(set(sp_a) & set(sp_b))
    if not common_sp:
        print(f"  No species cols to compare between [{label_a}] and [{label_b}]")
        return

    merged = (
        df_a[MERGE_KEY + common_sp].drop_duplicates(MERGE_KEY)
        .merge(
            df_b[MERGE_KEY + common_sp].drop_duplicates(MERGE_KEY),
            on=MERGE_KEY, suffixes=("_a", "_b"),
        )
    )
    print(f"\n  Species check  [{label_a}] vs [{label_b}]  –  {len(merged):,} overlapping rows")
    for sp in SPECIES_COLS:
        ca, cb = f"{sp}_a", f"{sp}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        both = merged[[ca, cb]].dropna()
        if both.empty:
            continue
        diff  = (both[ca] - both[cb]).abs()
        n_mis = int((diff > tol).sum())
        flag  = "✓" if n_mis == 0 else f"~  {n_mis} rows differ (aggregation window)"
        print(f"    {sp:<30s}  max_diff={diff.max():.2e}  {flag}")


def slim_sat(df: pd.DataFrame, src: str, extra_keep: list = None) -> pd.DataFrame:
    """
    Strip shared/metadata cols from a satellite DataFrame, prefix the rest.
    extra_keep: cols to retain unprefixed (e.g. BAC_L2_COLS).
    """
    extra_keep = extra_keep or []
    sat_cols = [c for c in df.columns if c not in SAT_DROP and c not in extra_keep]
    out = df[MERGE_KEY + extra_keep + sat_cols].copy()
    return prefix_cols(out, sat_cols, src)


# =============================================================================
# LOAD
# =============================================================================
print("=" * 70)
print("Loading feature sets")
print("=" * 70)

# current round = 3
roms    = round_key(load("roms"))
c2rcc   = round_key(load("c2rcc"))
acolite = round_key(load("acolite"))
bac     = round_key(load("bac"))


# =============================================================================
# ROMS FILTERING BLOCK  (builds the base)
# =============================================================================
print("\n" + "=" * 70)
print(f"ROMS pre-processing  (mode = '{ROMS_MODE}')")
print("=" * 70)

roms_numeric_avail = [c for c in ROMS_NUMERIC_COLS if c in roms.columns]
roms_meta_avail    = [c for c in ROMS_META_COLS    if c in roms.columns]

if ROMS_MODE == "exact":
    # ── Option 1: same-day match only ────────────────────────────────────────
    base = roms[roms["delta_days"] == 0].copy()
    print(f"  Kept delta_days == 0 : {len(base):,} / {len(roms):,} rows")

elif ROMS_MODE == "aggregate":
    # ── Option 2: mean over all delta_days per (lat, lon, obs_date) ──────────
    # Species + IFCB metadata: take from delta_days == 0 (or nearest)
    ifcb_cols = SPECIES_COLS + ["dataset_id", "dataset_name", "dashboardIdName",
                                 "depth", "cruise", "sample_type", "location_id",
                                 "delta_days"]
    ifcb_cols = [c for c in ifcb_cols if c in roms.columns]

    roms_agg_num = (
        roms.groupby(MERGE_KEY, sort=False)[roms_numeric_avail]
        .mean()
        .reset_index()
    )
    roms_ifcb_nearest = (
        roms.sort_values("delta_days")
        .groupby(MERGE_KEY, sort=False)[ifcb_cols + roms_meta_avail]
        .first()
        .reset_index()
    )
    base = roms_agg_num.merge(roms_ifcb_nearest, on=MERGE_KEY, how="left")
    print(f"  Aggregated (mean) {len(roms):,} → {len(base):,} rows")

else:
    raise ValueError(f"Unknown ROMS_MODE: '{ROMS_MODE}'. Use 'exact' or 'aggregate'.")

print(f"  Base rows for merge: {len(base):,}")


# =============================================================================
# STEP 1 : base (ROMS)  ⊕  C2RCC
# =============================================================================
print("\n" + "=" * 70)
print("Step 1 : [R] ROMS base  ⊕  [C] C2RCC")
print("=" * 70)

check_species_agreement(base, c2rcc, "ROMS", "C2RCC")

c2rcc_slim = slim_sat(c2rcc, "c2rcc")

composite = base.merge(c2rcc_slim, on=MERGE_KEY, how="left") #was left earlier

# Drop any species cols brought in from c2rcc (keep ROMS copy)
dup_sp = [f"{sp}_x" for sp in SPECIES_COLS if f"{sp}_x" in composite.columns] + \
         [f"{sp}_y" for sp in SPECIES_COLS if f"{sp}_y" in composite.columns]
composite = composite.drop(columns=dup_sp, errors="ignore")

n_C = composite[f"{SUFFIX['c2rcc']}_rrs_8"].notna().sum()
print(f"\n  [C] matched : {n_C:,} / {len(composite):,} rows")


# =============================================================================
# STEP 2 : + ACOLITE
# =============================================================================
print("\n" + "=" * 70)
print("Step 2 : + [A] ACOLITE")
print("=" * 70)

check_species_agreement(base, acolite, "ROMS", "ACOLITE")

acolite_slim = slim_sat(acolite, "acolite")

composite = composite.merge(acolite_slim, on=MERGE_KEY, how="left")

n_A = composite[f"{SUFFIX['acolite']}_rrs_412"].notna().sum()
print(f"\n  [A] matched : {n_A:,} / {len(composite):,} rows")


# =============================================================================
# STEP 3 : + BAC
# =============================================================================
print("\n" + "=" * 70)
print("Step 3 : + [B] BAC")
print("=" * 70)

check_species_agreement(base, bac, "ROMS", "BAC")

# BAC extra keep cols (unprefixed L2 products + provenance)
bac_extra = [c for c in BAC_L2_COLS + ["loc_key", "date_start", "date_end"]
             if c in bac.columns]
bac_slim = slim_sat(bac, "bac", extra_keep=bac_extra)

composite = composite.merge(bac_slim, on=MERGE_KEY, how="left")

n_B = composite["CHL_NN"].notna().sum()
print(f"\n  [B] matched : {n_B:,} / {len(composite):,} rows")


# =============================================================================
# CLEANUP
# =============================================================================
# Drop any accidental duplicate columns
dup_cols = [c for c in composite.columns if c.endswith("_dup") or
            (c.endswith("_x") or c.endswith("_y"))]
if dup_cols:
    print(f"\n  Dropping {len(dup_cols)} residual dup cols: {dup_cols}")
    composite = composite.drop(columns=dup_cols, errors="ignore")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("COMPOSITE SUMMARY")
print("=" * 70)
print(f"  Shape            : {composite.shape}")
print(f"  Unique dates     : {composite['obs_date'].nunique():,}")
if "dashboardIdName" in composite.columns:
    print(f"  Unique locations : {composite['dashboardIdName'].nunique():,}")

print()
avail_checks = {
    "[R] ROMS ocean. data ": "roms_temp_sur",
    "[C] C2RCC sat. data  ": f"{SUFFIX['c2rcc']}_rrs_8",
    "[A] ACOLITE sat. data": f"{SUFFIX['acolite']}_rrs_412",
    "[B] BAC L2 products  ": "CHL_NN",
}
print("  Data availability per source:")
for label, col in avail_checks.items():
    if col in composite.columns:
        n   = composite[col].notna().sum()
        pct = 100 * n / len(composite)
        print(f"    {label}: {n:>6,}  ({pct:.1f}%)")

print("\n  Bloom presence (>0 cells/L) per species:")
for sp in SPECIES_COLS:
    if sp in composite.columns:
        valid = composite[sp].notna().sum()
        n     = (composite[sp] > 0).sum()
        pct   = 100 * n / valid if valid else 0
        print(f"    {sp:<30s}: {n:>5,}  ({pct:.1f}% of {valid:,} non-null)")

print("\n  Column groups by source prefix:")
for src, sfx in SUFFIX.items():
    pfx_cols = [c for c in composite.columns if c.startswith(f"{sfx}_")]
    print(f"    [{sfx}] {src:<7s}: {len(pfx_cols)} cols")
base_cols = [c for c in composite.columns
             if not any(c.startswith(f"{s}_") for s in SUFFIX.values())]
print(f"    [–] base/shared: {len(base_cols)} cols"
      f"  ({', '.join(base_cols[:6])}{'...' if len(base_cols) > 6 else ''})")


# =============================================================================
# SAVE
# =============================================================================
composite.to_parquet(OUT_PATH, index=False)
print(f"\n✓ Saved → {OUT_PATH}")
print(f"  Memory : {composite.memory_usage(deep=True).sum() / 1e6:.1f} MB")