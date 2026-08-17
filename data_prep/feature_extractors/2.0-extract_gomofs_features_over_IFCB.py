"""
Extract ROMS Features for IFCB Matchup Data  (Hybrid local + OPeNDAP)
======================================================================
Output format: LONG — one row per (observation × lag).
Each observation is identified by (obs_date, latitude, longitude).
Lags: [-2, -1, 0, +1, +2] days relative to obs_date.
Lag rows where the ROMS file is missing are dropped entirely.

Supported local filename formats
---------------------------------
Format A (older):  nos.gomofs.2ds.n001.20200620.t12z.nc
                   parts[0]=nos  parts[1]=gomofs  parts[2]=2ds
                   parts[3]=n001  parts[4]=YYYYMMDD  parts[5]=t12z

Format B (newer):  gomofs.t12z.20240911.2ds.n001.nc
                   parts[0]=gomofs  parts[1]=t12z  parts[2]=YYYYMMDD
                   parts[3]=2ds  parts[4]=n001

Both formats are detected automatically.  All local files are treated as
fhour=n001 (no selection needed locally).  The --gomofs-fhour argument
only affects OPeNDAP URL construction.

Pipeline:
    1. Load grid from first local file (one-time, no network)
    2. Check spatial overlap for all obs (KDTree, vectorised)
    3. Fetch data per date:
         - Local only  : if local_roms_dir is provided
         - OPeNDAP only: if local_roms_dir is not provided
       Skip obs before ROMS_YEAR_MIN entirely.

Performance:
    - Grid cells deduplicated per date → each unique (eta, xi) loaded ONCE
    - ds[vars].isel(...).load() batches all variable fetches in ONE round-trip
    - Results broadcast back to all matching rows via vectorised concat

Usage:
    python extract_roms_for_ifcb.py \
        [--ifcb-path   datasets/.../habhub_IFCB_GOM.parquet] \
        [--output      datasets/.../ifcb_roms_features.parquet] \
        [--local-dir   datasets/.../gomofs/]          # optional; uses OPeNDAP if omitted
        [--gomofs-fhour n006]                         # only used for OPeNDAP URLs
"""

from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Directory that holds local GoMOFS NetCDF files.
    # If set, ALL data is sourced locally (no OPeNDAP).
    # If None, ALL data is sourced via OPeNDAP.
    local_roms_dir: Optional[Path] = Path("datasets/GULF_OF_MAINE/raw_data/gomofs/")

    # Skip all obs before this year
    roms_year_min: int = 2018

    # --- OPeNDAP settings (only used when local_roms_dir is None) -----------
    gomofs_product: str  = "2ds"
    gomofs_fhour: str    = "n001"   # forecast hour for OPeNDAP URL construction
    gomofs_cycle: str    = "t12z"

    thredds_base: str = (
        "https://www.ncei.noaa.gov/thredds/dodsC/model-gomofs-files"
    )

    # Pause between OPeNDAP file opens (seconds)
    opendap_sleep_s: float = 0.5
    # -------------------------------------------------------------------------

    # Lags relative to obs_date (days). Negative = before obs, positive = after.
    lag_days: list[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])

    # Bounding box pre-filter
    lon_min: float = -72.0
    lon_max: float = -60.0
    lat_min: float =  38.0
    lat_max: float =  48.0

    # Max distance to nearest ocean cell (deg). 0.02 ~ 2.2 km ~ 3 cells at 700 m
    max_grid_dist_deg: float = 0.02

    roms_variables: list[str] = field(default_factory=lambda: [
        "temp_sur",   # Sea surface temperature  (°C)
        "salt_sur",   # Sea surface salinity      (PSU)
        "zeta",       # Sea surface height        (m)
        "u_sur",      # Surface eastward current  (m/s)
        "v_sur",      # Surface northward current (m/s)
        "Uwind",      # Surface eastward wind     (m/s)
        "Vwind",      # Surface northward wind    (m/s)
        "Pair",       # Air pressure              (mbar)
    ])


CFG = Config()


# ============================================================================
# FILENAME PARSING  (two formats)
# ============================================================================

# Pre-compiled pattern for YYYYMMDD
_DATE_RE = re.compile(r"^\d{8}$")


def _parse_local_filename(filename: str) -> dict:
    """
    Parse metadata from a GoMOFS local filename.

    Format A: nos.gomofs.2ds.n001.20200620.t12z.nc
    Format B: gomofs.t12z.20240911.2ds.n001.nc

    Returns dict with keys: date_str, product, fhour, cycle
    Returns empty dict on parse failure.
    """
    stem  = Path(filename).stem          # strip .nc
    parts = stem.split(".")

    # ── Format A: nos.gomofs.<product>.<fhour>.<YYYYMMDD>.<cycle> ──────────
    # Minimum 6 parts; parts[0]=="nos", parts[1]=="gomofs"
    if len(parts) >= 6 and parts[0] == "nos" and parts[1] == "gomofs":
        date_str = parts[4]
        if _DATE_RE.match(date_str):
            return {
                "date_str": date_str,
                "product":  parts[2],
                "fhour":    parts[3],
                "cycle":    parts[5] if len(parts) > 5 else "unknown",
                "format":   "A",
            }

    # ── Format B: gomofs.<cycle>.<YYYYMMDD>.<product>.<fhour> ──────────────
    # Minimum 5 parts; parts[0]=="gomofs"
    if len(parts) >= 5 and parts[0] == "gomofs":
        date_str = parts[2]
        if _DATE_RE.match(date_str):
            return {
                "date_str": date_str,
                "product":  parts[3] if len(parts) > 3 else "unknown",
                "fhour":    parts[4] if len(parts) > 4 else "unknown",
                "cycle":    parts[1],
                "format":   "B",
            }

    return {}


def _build_local_index(local_dir: Path) -> dict[str, Path]:
    """
    Scan *local_dir* for GoMOFS NetCDF files and build a date → Path index.
    Supports both filename formats.  If multiple files match the same date,
    the last one (alphabetically) wins — in practice this is not an issue
    because local files are all n001.
    """
    index: dict[str, Path] = {}
    nc_files = sorted(local_dir.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(
            f"No *.nc files found in local_roms_dir: {local_dir}"
        )

    skipped = 0
    for path in nc_files:
        meta = _parse_local_filename(path.name)
        if not meta:
            skipped += 1
            continue
        index[meta["date_str"]] = path

    print(f"  Local index: {len(index):,} dates  "
          f"({skipped} files skipped — unrecognised format)")
    if index:
        dates_sorted = sorted(index)
        print(f"  Date range : {dates_sorted[0]} → {dates_sorted[-1]}")

    return index


# ============================================================================
# HELPERS
# ============================================================================

@contextmanager
def suppress_clib_stderr():
    """Redirect C-lib stderr to /dev/null (suppresses NetCDF4 noise)."""
    devnull_fd    = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stderr_fd)


def _nan_record(cfg: Config) -> dict:
    """Return a dict of NaN-valued ROMS feature columns."""
    d = {f"roms_{v}": np.nan for v in cfg.roms_variables}
    d.update({
        "roms_current_speed": np.nan,
        "roms_current_dir":   np.nan,
        "roms_wind_speed":    np.nan,
        "roms_wind_dir":      np.nan,
    })
    return d


# ============================================================================
# STEP 1 — Grid initialisation (local, one-time)
# ============================================================================

def init_grid(cfg: Config) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load lat_rho / lon_rho / h from a GoMOFS file (local preferred).
    Returns:
        grid_tree  : cKDTree over ocean cells
        ocean_idx  : (N, 2) array of (eta, xi) indices for ocean cells
        lat_ocean  : latitudes  of ocean cells
        lon_ocean  : longitudes of ocean cells
    """
    print("Loading GoMOFS grid ...")

    if cfg.local_roms_dir is not None:
        nc_files = sorted(cfg.local_roms_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No local GoMOFS files found in {cfg.local_roms_dir}"
            )
        src_path = nc_files[0]
        print(f"  Source (local): {src_path.name}")
        with suppress_clib_stderr():
            ds = xr.open_dataset(src_path, engine="netcdf4", decode_times=False)
    else:
        # Fall back to OPeNDAP for a recent date just to grab the grid
        # (only reached when running in pure OPeNDAP mode)
        today = datetime.utcnow()
        url   = _build_opendap_url(today - timedelta(days=7), cfg)
        print(f"  Source (OPeNDAP): {url.split('/')[-1]}")
        with suppress_clib_stderr():
            ds = xr.open_dataset(url, engine="netcdf4", decode_times=False)

    lon_rho = ds["lon_rho"].values
    lat_rho = ds["lat_rho"].values
    h       = ds["h"].values if "h" in ds else None
    ds.close()

    domain_mask = (h > 0) & np.isfinite(h) if h is not None else np.isfinite(lat_rho)
    ocean_idx   = np.argwhere(domain_mask)
    grid_tree   = cKDTree(
        np.column_stack([lat_rho[domain_mask], lon_rho[domain_mask]])
    )

    print(f"  Grid shape  : {lon_rho.shape}")
    print(f"  Ocean cells : {domain_mask.sum():,} / {domain_mask.size:,}")
    print(f"  Lon range   : [{lon_rho.min():.2f}, {lon_rho.max():.2f}]")
    print(f"  Lat range   : [{lat_rho.min():.2f}, {lat_rho.max():.2f}]")

    return grid_tree, ocean_idx, lat_rho[domain_mask], lon_rho[domain_mask]


# ============================================================================
# STEP 2 — Spatial overlap (vectorised KDTree, no I/O)
# ============================================================================

def check_spatial_overlap(
    obs_df: pd.DataFrame,
    grid_tree: cKDTree,
    ocean_idx: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """
    Vectorised KDTree query for all observations.
    Adds: roms_eta, roms_xi, roms_grid_dist_km, _in_domain (internal flag).
    """
    print("\nChecking spatial overlap ...")

    lats = obs_df["latitude"].values
    lons = obs_df["longitude"].values
    n    = len(obs_df)

    in_bbox = (
        (lats >= cfg.lat_min) & (lats <= cfg.lat_max) &
        (lons >= cfg.lon_min) & (lons <= cfg.lon_max)
    )
    bbox_idx = np.where(in_bbox)[0]

    eta_col   = np.full(n, -1,     dtype=int)
    xi_col    = np.full(n, -1,     dtype=int)
    dist_col  = np.full(n, np.nan, dtype=float)
    in_domain = np.zeros(n, dtype=bool)

    if len(bbox_idx):
        coords      = np.column_stack([lats[bbox_idx], lons[bbox_idx]])
        dists, kids = grid_tree.query(coords, k=1)
        eta_col[bbox_idx]   = ocean_idx[kids, 0]
        xi_col[bbox_idx]    = ocean_idx[kids, 1]
        dist_col[bbox_idx]  = dists * 111.0
        in_domain[bbox_idx] = dists <= cfg.max_grid_dist_deg

    obs_df = obs_df.copy()
    obs_df["roms_eta"]          = eta_col
    obs_df["roms_xi"]           = xi_col
    obs_df["roms_grid_dist_km"] = dist_col
    obs_df["_in_domain"]        = in_domain

    n_in  = int(in_domain.sum())
    n_out = n - n_in
    print(f"  In domain    : {n_in:,} ({n_in/n*100:.1f}%)")
    print(f"  Out of domain: {n_out:,} ({n_out/n*100:.1f}%)")

    return obs_df


# ============================================================================
# STEP 3 — File access
# ============================================================================

def _build_opendap_url(date: datetime, cfg: Config) -> str:
    fname = (
        f"nos.gomofs.{cfg.gomofs_product}.{cfg.gomofs_fhour}"
        f".{date:%Y%m%d}.{cfg.gomofs_cycle}.nc"
    )
    return f"{cfg.thredds_base}/{date:%Y}/{date:%m}/{fname}"


def open_roms_file(
    date: datetime,
    cfg: Config,
    local_index: Optional[dict[str, Path]],
) -> tuple[Optional[xr.Dataset], str, Optional[str], Optional[str], Optional[str]]:
    """
    Open the GoMOFS file for *date*.

    If local_index is provided (local mode), only the index is consulted —
    no OPeNDAP fallback.  If local_index is None (OPeNDAP mode), the THREDDS
    server is queried directly.

    Returns: (ds, src_type, filename, fhour, cycle)
    src_type in {"local", "opendap", "missing", "error"}
    """
    date_str = date.strftime("%Y%m%d")

    # ── LOCAL MODE ───────────────────────────────────────────────────────────
    if local_index is not None:
        path = local_index.get(date_str)
        if path is None:
            return None, "missing", None, None, None
        try:
            with suppress_clib_stderr():
                ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
            meta = _parse_local_filename(path.name)
            return ds, "local", path.name, meta.get("fhour"), meta.get("cycle")
        except Exception as exc:
            tqdm.write(f"  [!] Local open failed {path.name}: {exc}")
            return None, "error", path.name, None, None

    # ── OPeNDAP MODE ────────────────────────────────────────────────────────
    url   = _build_opendap_url(date, cfg)
    fname = url.split("/")[-1]
    try:
        with suppress_clib_stderr():
            ds = xr.open_dataset(url, engine="netcdf4", decode_times=False)
        meta = _parse_local_filename(fname)
        if cfg.opendap_sleep_s > 0:
            time.sleep(cfg.opendap_sleep_s)
        return (
            ds, "opendap", fname,
            meta.get("fhour", cfg.gomofs_fhour),
            meta.get("cycle", cfg.gomofs_cycle),
        )
    except Exception as exc:
        tqdm.write(f"  [!] OPeNDAP open failed {fname}: {exc}")
        if cfg.opendap_sleep_s > 0:
            time.sleep(cfg.opendap_sleep_s)
        return None, "missing", fname, None, None


# ============================================================================
# STEP 4 — Batch extraction
# ============================================================================

def extract_cells_batch(
    ds: xr.Dataset,
    unique_cells: np.ndarray,
    cfg: Config,
) -> dict[tuple[int, int], dict]:
    """
    Load all ROMS variables for every unique (eta, xi) in ONE network round-trip.
    Returns: dict mapping (eta, xi) → feature dict
    """
    etas = unique_cells[:, 0]
    xis  = unique_cells[:, 1]

    eta_min, eta_max = int(etas.min()), int(etas.max())
    xi_min,  xi_max  = int(xis.min()),  int(xis.max())

    dim_slices: dict = {
        "ocean_time": 0,
        "eta_rho": slice(eta_min, eta_max + 1),
        "xi_rho":  slice(xi_min,  xi_max  + 1),
        "eta_u":   slice(eta_min, eta_max + 1),
        "xi_u":    slice(xi_min,  xi_max  + 1),
        "eta_v":   slice(eta_min, eta_max + 1),
        "xi_v":    slice(xi_min,  xi_max  + 1),
    }

    avail_vars = [v for v in cfg.roms_variables if v in ds]
    indexers   = {k: v for k, v in dim_slices.items() if k in ds.dims}

    try:
        patch = ds[avail_vars].isel(indexers).load()
    except Exception as exc:
        tqdm.write(f"  [!] Batch load failed: {exc}")
        return {(int(e), int(x)): _nan_record(cfg) for e, x in unique_cells}

    cell_cache: dict[tuple[int, int], dict] = {}

    for eta, xi in unique_cells:
        eta, xi   = int(eta), int(xi)
        local_eta = eta - eta_min
        local_xi  = xi  - xi_min

        features: dict = {f"roms_{v}": np.nan for v in cfg.roms_variables}

        for var in avail_vars:
            var_dims = patch[var].dims
            local_indexers: dict = {}
            if "eta_rho" in var_dims: local_indexers["eta_rho"] = local_eta
            if "xi_rho"  in var_dims: local_indexers["xi_rho"]  = local_xi
            if "eta_u"   in var_dims: local_indexers["eta_u"]   = local_eta
            if "xi_u"    in var_dims: local_indexers["xi_u"]    = local_xi
            if "eta_v"   in var_dims: local_indexers["eta_v"]   = local_eta
            if "xi_v"    in var_dims: local_indexers["xi_v"]    = local_xi
            try:
                val = float(patch[var].isel(local_indexers).values)
            except Exception:
                val = np.nan
            features[f"roms_{var}"] = val

        u_c = features.get("roms_u_sur", np.nan)
        v_c = features.get("roms_v_sur", np.nan)
        if np.isfinite(u_c) and np.isfinite(v_c):
            features["roms_current_speed"] = float(np.hypot(u_c, v_c))
            features["roms_current_dir"]   = float(np.degrees(np.arctan2(v_c, u_c)))
        else:
            features["roms_current_speed"] = np.nan
            features["roms_current_dir"]   = np.nan

        u_w = features.get("roms_Uwind", np.nan)
        v_w = features.get("roms_Vwind", np.nan)
        if np.isfinite(u_w) and np.isfinite(v_w):
            features["roms_wind_speed"] = float(np.hypot(u_w, v_w))
            features["roms_wind_dir"]   = float(np.degrees(np.arctan2(v_w, u_w)))
        else:
            features["roms_wind_speed"] = np.nan
            features["roms_wind_dir"]   = np.nan

        cell_cache[(eta, xi)] = features

    return cell_cache


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(obs_df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:

    # ------------------------------------------------------------------
    # Build local file index (once) if using local mode
    # ------------------------------------------------------------------
    local_index: Optional[dict[str, Path]] = None
    if cfg.local_roms_dir is not None:
        print(f"\nBuilding local file index from: {cfg.local_roms_dir}")
        local_index = _build_local_index(cfg.local_roms_dir)
        mode_label  = "LOCAL"
    else:
        mode_label = "OPeNDAP"
    print(f"  Data source mode: {mode_label}\n")

    # ------------------------------------------------------------------
    # Step 1: Grid (one-time)
    # ------------------------------------------------------------------
    grid_tree, ocean_idx, _, _ = init_grid(cfg)

    # ------------------------------------------------------------------
    # Step 2: Spatial overlap (one-time, vectorised)
    # ------------------------------------------------------------------
    obs_df = check_spatial_overlap(obs_df, grid_tree, ocean_idx, cfg)

    # ------------------------------------------------------------------
    # Pre-flag skipped rows
    # ------------------------------------------------------------------
    obs_df["_date_dt"] = pd.to_datetime(obs_df["obs_date"]).dt.normalize()
    year               = pd.to_datetime(obs_df["obs_date"]).dt.year

    skip_year  = year < cfg.roms_year_min
    out_domain = ~obs_df["_in_domain"]
    fetch_mask = obs_df["_in_domain"] & ~skip_year
    fetch_df   = obs_df[fetch_mask]

    n_total      = len(obs_df)
    n_fetch      = fetch_mask.sum()
    n_skip_year  = skip_year.sum()
    n_out_domain = out_domain.sum()

    print(f"\nSkipping {n_skip_year:,} obs before {cfg.roms_year_min}")
    print(f"Fetching data for {n_fetch:,} in-domain obs "
          f"({n_out_domain:,} out-of-domain skipped)")
    print(f"Lags: {cfg.lag_days}  →  up to {len(cfg.lag_days)} rows per obs\n")

    # ------------------------------------------------------------------
    # Step 3+4: Per-date, per-lag fetch
    # Output is built as a list of flat dicts (long format).
    # One dict = one (obs, lag) row.
    # Rows where the ROMS file is missing are dropped entirely.
    # ------------------------------------------------------------------

    obs_passthrough_cols = [
        c for c in obs_df.columns if c not in ("_in_domain", "_date_dt")
    ]

    output_rows: list[dict] = []

    stats = {
        "ok":         0,
        "no_file":    0,   # lag row dropped (missing file)
        "out_domain": n_out_domain,
        "skip_year":  n_skip_year,
    }

    for date_val, group in tqdm(
        fetch_df.groupby("_date_dt"), desc="Dates", unit="day"
    ):
        date_dt      = pd.Timestamp(date_val).to_pydatetime()
        unique_cells = (
            group[["roms_eta", "roms_xi"]]
            .drop_duplicates()
            .values.astype(int)
        )

        # Cache: lag_day → cell_cache dict
        lag_cache: dict[int, Optional[dict[tuple[int, int], dict]]] = {}
        lag_meta:  dict[int, Optional[dict]] = {}

        for lag in cfg.lag_days:
            lag_date = date_dt + timedelta(days=lag)
            ds, src_type, fname, fhour, cycle = open_roms_file(
                lag_date, cfg, local_index
            )

            if ds is None:
                lag_cache[lag] = None
                lag_meta[lag]  = None
                stats["no_file"] += len(group)
                continue

            try:
                lag_cache[lag] = extract_cells_batch(ds, unique_cells, cfg)
            except Exception as exc:
                tqdm.write(
                    f"  [!] extract_cells_batch failed lag={lag:+d} {fname}: {exc}"
                )
                lag_cache[lag] = None
                lag_meta[lag]  = None
                stats["no_file"] += len(group)
            else:
                lag_meta[lag] = {
                    "roms_source": src_type,
                    "roms_file":   fname,
                    "roms_fhour":  fhour,
                    "roms_cycle":  cycle,
                    "roms_date":   lag_date.strftime("%Y-%m-%d"),
                }
            finally:
                if ds is not None:
                    ds.close()

        # Build output rows — one per (obs row × lag)
        for idx, row in group.iterrows():
            key = (int(row["roms_eta"]), int(row["roms_xi"]))

            for lag in cfg.lag_days:
                cell_cache = lag_cache.get(lag)
                if cell_cache is None:
                    continue   # missing file → drop this lag row

                features = cell_cache.get(key)
                if features is None:
                    continue   # cell not extracted → drop

                out_row = {c: row[c] for c in obs_passthrough_cols}
                out_row["delta_days"] = lag
                out_row.update(lag_meta[lag])
                out_row.update(features)

                output_rows.append(out_row)
                stats["ok"] += 1

    # ------------------------------------------------------------------
    # Assemble output dataframe
    # ------------------------------------------------------------------
    if not output_rows:
        print("\nWARNING: No output rows produced.")
        return pd.DataFrame()

    result_df = pd.DataFrame(output_rows)

    # Tidy column order: obs identity → delta_days → roms meta → roms features
    id_cols      = ["obs_date", "latitude", "longitude"]
    lag_col      = ["delta_days"]
    meta_cols    = [
        "roms_date", "roms_source", "roms_file", "roms_fhour",
        "roms_cycle", "roms_eta", "roms_xi", "roms_grid_dist_km",
    ]
    feature_cols = [
        c for c in result_df.columns
        if c.startswith("roms_") and c not in meta_cols
    ]
    other_cols = [
        c for c in result_df.columns
        if c not in id_cols + lag_col + meta_cols + feature_cols
    ]

    ordered = id_cols + lag_col + other_cols + meta_cols + feature_cols
    ordered = [c for c in ordered if c in result_df.columns]
    result_df = result_df[ordered]

    result_df["delta_days"] = result_df["delta_days"].astype("int8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_obs_extracted = result_df.groupby(["obs_date", "latitude", "longitude"]).ngroups

    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY  (mode: {mode_label})")
    print(f"{'='*60}")
    print(f"  Input observations   : {n_total:>6,}")
    print(f"  Out of domain        : {stats['out_domain']:>6,}")
    print(f"  Year < {cfg.roms_year_min}           : {stats['skip_year']:>6,}")
    print(f"  Eligible obs         : {n_fetch:>6,}")
    print(f"  Output rows (obs×lag): {stats['ok']:>6,}")
    print(f"  Unique obs with data : {n_obs_extracted:>6,}")
    print(f"  Lag rows dropped     : {stats['no_file']:>6,}  (missing file)")
    print(f"\n  Lags present in output:")
    for lag, cnt in result_df["delta_days"].value_counts().sort_index().items():
        print(f"    lag {lag:+d} : {cnt:,} rows")
    ok_local   = (result_df["roms_source"] == "local").sum()
    ok_opendap = (result_df["roms_source"] == "opendap").sum()
    print(f"\n  Source: local={ok_local:,}  opendap={ok_opendap:,}")
    print(f"{'='*60}\n")

    return result_df


# ============================================================================
# IFCB LOADING
# ============================================================================

CONC_COLS = [
    "Alexandrium_catenella", "Dinophysis_acuminata", "Dinophysis_norvegica",
    "Karenia", "Margalefidinium", "Mesodinium", "Pseudo-nitzschia",
    "Tripos_furca", "Tripos_fusus", "Tripos_muelleri",
]

STATIONS_REMOVE = ["nauset", "jamestown", "lombos"]

STATION_COORDS = {
    "harpswell": (43.781, -69.975),
    "fiddlers":  (41.645, -70.675),
    "gsodock":   (41.570, -71.410),
    "mvco":      (41.325, -70.566),
    "mdibl":     (44.440, -68.205),
}


def load_ifcb(path: Path) -> pd.DataFrame:
    """
    Load and preprocess IFCB parquet into daily aggregated obs_df.
    Output columns include: location_id, obs_date, latitude, longitude,
    dashboardIdName, and all concentration columns.
    """
    print(f"\nLoading IFCB: {path}")
    df = pd.read_parquet(path)
    print(f"  Raw rows: {len(df):,}")

    df["datetime"] = pd.to_datetime(df["date"])

    # 1. Remove excluded stations
    df = df[~df["dashboardIdName"].isin(STATIONS_REMOVE)].reset_index(drop=True)
    print(f"  After station filter: {len(df):,}")

    # 2. Fix lat/lon for fixed stations
    for name, (lat, lon) in STATION_COORDS.items():
        mask = df["dashboardIdName"] == name
        df.loc[mask, ["latitude", "longitude"]] = lat, lon
        uniq = df.loc[mask, "location_id"].unique()
        if len(uniq):
            df.loc[mask, "location_id"] = uniq[0]

    df.sort_values("datetime", inplace=True)
    df["obs_date"] = df["datetime"].dt.date

    # 3. Daily aggregation per (location_id, obs_date)
    agg_dict: dict = {
        "latitude":        "first",
        "longitude":       "first",
        "dataset_id":      "first",
        "dataset_name":    "first",
        "dashboardIdName": "first",
    }
    # 95th-percentile for concentrations: preserves bloom spikiness,
    # filters single-point outliers
    conc_present = [c for c in CONC_COLS if c in df.columns]
    for c in conc_present:
        agg_dict[c] = lambda x: x.quantile(0.95)

    for col in ["depth", "cruise", "sample_type"]:
        if col in df.columns:
            agg_dict[col] = "first"

    df = df.groupby(["location_id", "obs_date"], as_index=False).agg(agg_dict)
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.strftime("%Y%m%d")

    print(f"  After daily aggregation: {len(df):,} rows")
    print(f"  Unique locations : {df['location_id'].nunique()}")
    print(f"  Unique dates     : {df['obs_date'].nunique()}  "
          f"({df['obs_date'].min()} to {df['obs_date'].max()})")

    return df


# ============================================================================
# CLI
# ============================================================================

IFCB_FILE = (
    "datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_IFCB_GOM.parquet"
)
OUTPUT_FILE = (
    "datasets/GULF_OF_MAINE/processed_data/roms_features/ifcb_roms_features.parquet"
)
DEFAULT_LOCAL_DIR = (
    "datasets/GULF_OF_MAINE/raw_data/gomofs/"
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract GoMOFS ROMS features for IFCB observations "
            "(long format, multi-lag)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ifcb-path", default=IFCB_FILE,
        help="Path to IFCB parquet file.",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--local-dir", default=DEFAULT_LOCAL_DIR,
        help=(
            "Directory containing local GoMOFS NetCDF files. "
            "If provided, ALL data is sourced locally (no OPeNDAP). "
            "Pass --local-dir '' to force OPeNDAP-only mode."
        ),
    )
    parser.add_argument(
        "--gomofs-fhour", default=CFG.gomofs_fhour,
        help="Forecast hour for OPeNDAP URL construction (e.g. n001, n006). "
             "Ignored when --local-dir is set.",
    )
    parser.add_argument(
        "--opendap-sleep", type=float, default=CFG.opendap_sleep_s,
        help="Sleep (s) between OPeNDAP file opens. Ignored in local mode.",
    )
    parser.add_argument(
        "--year-min", type=int, default=CFG.roms_year_min,
        help="Skip obs before this year.",
    )
    parser.add_argument(
        "--lags", type=int, nargs="+", default=CFG.lag_days,
        help="Lag days relative to obs_date.",
    )
    args = parser.parse_args()

    # Apply CLI overrides to config
    CFG.gomofs_fhour    = args.gomofs_fhour
    CFG.opendap_sleep_s = args.opendap_sleep
    CFG.roms_year_min   = args.year_min
    CFG.lag_days        = sorted(args.lags)

    # Local dir: empty string → OPeNDAP-only mode
    if args.local_dir:
        CFG.local_roms_dir = Path(args.local_dir)
    else:
        CFG.local_roms_dir = None

    ifcb_path = Path(args.ifcb_path)
    output    = Path(args.output)

    if not ifcb_path.exists():
        raise FileNotFoundError(f"IFCB parquet not found: {ifcb_path}")

    obs_df    = load_ifcb(ifcb_path)
    result_df = run(obs_df, CFG)

    if len(result_df):
        output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output, index=False)
        print(f"Saved → {output}")
        print(f"  Rows: {len(result_df):,}  Cols: {len(result_df.columns)}")

        roms_cols = sorted(c for c in result_df.columns if c.startswith("roms_"))
        print(f"\nROMS columns ({len(roms_cols)}):")
        for c in roms_cols:
            print(f"  {c}")


if __name__ == "__main__":
    main()