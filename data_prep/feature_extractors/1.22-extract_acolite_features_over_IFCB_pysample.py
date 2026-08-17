"""
ACOLITE Feature Extraction for IFCB Matchups — Pixel-Level MVC Edition
=======================================================================

PSEUDO-ALGORITHM
================

SETUP
-----
1. Load IFCB parquet → filter excluded stations → fix coordinates → 
   aggregate to daily rows (95th percentile for concentrations).
2. Build scene index: {YYYYMMDD: [path, ...]} from ACOLITE output dirs.

PER OBS-DATE (parallelised across dates via ProcessPoolExecutor/spawn)
----------------------------------------------------------------------
For each obs_date:
    Collect all scenes within ±TEMPORAL_WINDOW_DAYS.

    For each IFCB location on that date:

        A. DEFINE TARGET GRID (once per location)
           - Fixed n×n regular grid centred on (lat, lon) at ~300 m resolution.
           - Same grid reused for every scene → guarantees pixel alignment.

        B. RESAMPLE EACH SCENE ONTO TARGET GRID
           For each scene:
             i.  Build SwathDefinition from 2-D lat/lon arrays.
             ii. Stack all rrs bands into [swath_rows, swath_cols, n_bands].
             iii.Single kd_tree.resample_nearest call for the full band stack
                 → gridded_stack: [n_rows, n_cols, n_bands].
             iv. Resample l2_flags the same way (separate call, same geometry).
             v.  Apply pixel-level flag mask:
                   invalid = (gridded_flags != 0)
                   gridded_stack[invalid, :] = NaN
                 (Flag is pixel-level; one flag value covers all bands.)
             vi. Also NaN-out any residual NaN/non-finite band values.
             Result: [n_rows, n_cols, n_bands] with NaN at invalid pixels.

        C. PIXEL-LEVEL MAXIMUM-VALUE COMPOSITE (MVC)
           - Stack valid scene arrays: [n_scenes, n_rows, n_cols, n_bands].
           - nanmax across scene axis (axis=0)
             → mvc: [n_rows, n_cols, n_bands]
           - A pixel that was cloudy in all scenes remains NaN.
           - A pixel clear in at least one scene gets that scene's Rrs value.
           - Because the flag is pixel-level, spectral consistency is preserved:
             either all bands at a pixel are valid (from one scene) or NaN.

        D. COMPUTE SPECTRAL INDICES PER PIXEL
           - For each valid pixel in mvc, compute all indices using the n_bands
             values at that pixel → indices: [n_rows, n_cols, n_indices].
           - Invalid (NaN) pixels propagate NaN into indices automatically.

        E. SPATIAL SUMMARY
           - nanmedian over the n×n grid for each band and each index
             → one scalar per feature.
           - n_valid_pixels = count of pixels where ALL bands are finite in mvc.
           - pct_valid_pixels = n_valid_pixels / (n_rows * n_cols) * 100.
           - Skip location if n_valid_pixels < MIN_VALID_PIXELS.

        F. RECORD
           - Emit one flat dict per location with band medians + index medians
             + coverage metadata.
           - No sat_date / delta_days stored (values may span multiple scenes).

MERGE & SAVE
------------
- Merge feature records back to IFCB df on [obs_date, location_id].
- Save to acolite_features_mvc.parquet.

KEY DESIGN DECISIONS
====================
- pyresample kd_tree: one kdtree build per scene covers all bands (efficient).
- Flag mask is pixel-level → no per-band masking logic needed.
- nanmax is the MVC operator → maximises spatial coverage across the window.
- Indices computed on MVC pixel arrays (not on scalar medians) → spectrally
  consistent per pixel.
- Spatial median is the only aggregation over values → robust to outliers.
- spawn multiprocessing → each worker gets a clean HDF5/NetCDF4 context.

Band mapping (ACOLITE rrs_* → OLCI equivalent)
-----------------------------------------------
  rrs_412 → B02 (412.5 nm)       rrs_443 → B03 (442.5 nm)
  rrs_490 → B04 (490 nm)         rrs_510 → B05 (510 nm)
  rrs_560 → B06 (560 nm)         rrs_620 → B07 (620 nm)
  rrs_665 → B08 (665 nm)         rrs_674 → B09 (673.75 nm)
  rrs_682 → B10 proxy (681.25 nm) rrs_709 → B11 (708.75 nm)
  rrs_754 → B12 (753.75 nm)      rrs_865 → B17 (865 nm)
"""

import logging
import multiprocessing
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = Path("/home/server/pi/homes/rathorek/projects/HHAB_ROMS/datasets/GULF_OF_MAINE")
ACOLITE_DIR = BASE_DIR / "processed_data" / "sentinel_3_acolite"
IFCB_FILE   = BASE_DIR / "raw_data/IFCB/habhub_IFCB_GOM.parquet"    
OUTPUT_DIR  = BASE_DIR / "processed_data" / "sentinel_3_L2_past3days" / "acolite_features"   # pysample

TEMPORAL_WINDOW_DAYS = 3       # ±N days around each IFCB observation
SPATIAL_BUFFER_DEG   = 0.02   # ~2 km buffer radius
GRID_RESOLUTION_DEG  = 0.003   # ~300 m — matches OLCI native resolution
MIN_VALID_PIXELS     = 3       # minimum valid pixels in MVC to emit a record
RESAMPLE_RADIUS_M    = 400     # kd_tree radius of influence in metres
N_WORKERS            = 8

BAND_MAP = {
    "rrs_412": 411.50,
    "rrs_443": 442.50,
    "rrs_490": 490.00,
    "rrs_510": 510.00,
    "rrs_560": 559.00,
    "rrs_620": 620.00,
    "rrs_665": 664.60,
    "rrs_674": 673.75,
    "rrs_682": 681.25,
    "rrs_709": 708.75,
    "rrs_754": 753.75,
    "rrs_865": 864.90,
}

STATION_UPDATES = {
    "harpswell": (43.781, -69.975),
    "fiddlers":  (41.645, -70.675),
    "gsodock":   (41.570, -71.410),
    "mvco":      (41.325, -70.566),
    "mdibl":     (44.440, -68.205),
}
STATIONS_EXCLUDE = ["nauset", "jamestown", "lombos"]

CONC_COLS = [
    "Alexandrium_catenella", "Dinophysis_acuminata", "Dinophysis_norvegica",
    "Karenia", "Margalefidinium", "Mesodinium", "Pseudo-nitzschia",
    "Tripos_furca", "Tripos_fusus", "Tripos_muelleri",
]

GOMOFS_POLYGON = Polygon([(-70, 38), (-60, 42), (-64, 45), (-73, 43), (-70, 38)])



# =============================================================================
# SCENE INDEX
# =============================================================================

_DIR_DATE_RE = re.compile(r"OL_1_EFR____(\d{8})T\d{6}")


def _scene_date(dir_path: Path) -> str | None:
    m = _DIR_DATE_RE.search(dir_path.name)
    return m.group(1) if m else None


def _find_l2w(scene_dir: Path) -> Path | None:
    candidates = sorted(scene_dir.glob("*_L2W.nc"))
    return candidates[0] if candidates else None


def build_scene_index(acolite_dir: Path) -> dict:
    """Return {YYYYMMDD: [str(path), ...]}."""
    index: dict[str, list] = {}
    for scene_dir in sorted(acolite_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        date_str = _scene_date(scene_dir)
        if date_str is None:
            continue
        nc_path = _find_l2w(scene_dir)
        if nc_path is None:
            continue
        index.setdefault(date_str, []).append(str(nc_path))
    if not index:
        log.warning("No ACOLITE scene directories found in %s", acolite_dir)
    return index


def candidate_dates(obs_ts: pd.Timestamp, window: int) -> list[str]:
    """All YYYYMMDD strings within -window days."""
    return [
        (obs_ts + timedelta(days=d)).strftime("%Y%m%d")
        for d in range(-window,  1)
    ]


# =============================================================================
# WORKER
# =============================================================================

def _process_date_worker(task: dict) -> list:
    """
    Spawned subprocess worker.
    Imports pyresample and xarray locally for HDF5/NetCDF4 safety.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import xarray as xr
    from pyresample import geometry as prg, kd_tree

    obs_date_str   = task["obs_date_str"]
    locations      = task["locations"]
    available      = task["available"]        # [{"date": str, "path": str}, ...]
    spatial_buffer = task["spatial_buffer"]   # degrees
    grid_res       = task["grid_res"]         # degrees
    min_valid_px   = task["min_valid_px"]
    resample_radius= task["resample_radius"]  # metres
    band_names     = task["band_names"]       # ordered list
    EPS            = 1e-10

    n_bands = len(band_names)

    # ------------------------------------------------------------------
    # Spectral index computation — operates on [n_pixels, n_bands] arrays
    # where axis-1 order matches band_names.
    # Returns dict {index_name: 1-D array of length n_pixels}
    # ------------------------------------------------------------------
    def _compute_indices(px):
        """
        px: dict {band_name: 1-D float32 array of valid pixel values}
        Returns dict {index_name: 1-D array}
        NaN bands propagate NaN into dependent indices automatically.
        """
        def g(name):
            return px.get(name, np.full(next(iter(px.values())).shape, np.nan))

        b412 = g("rrs_412"); b490 = g("rrs_490"); b560 = g("rrs_560")
        b665 = g("rrs_665"); b674 = g("rrs_674"); b682 = g("rrs_682")
        b709 = g("rrs_709"); b754 = g("rrs_754"); b865 = g("rrs_865")

        s681  = (681.25 - 673.75) / (708.75 - 673.75)
        s665  = (665.00 - 673.75) / (708.75 - 673.75)
        glh_s = (560.0  - 490.0)  / (665.0  - 490.0)
        blh_s = (490.0  - 412.5)  / (560.0  - 412.5)
        mci_s = (708.75 - 665.0)  / (753.75 - 665.0)

        flh_681 = b682 - (b674 + (b709 - b674) * s681)
        flh_665 = b665 - (b674 + (b709 - b674) * s665)
        flhmax  = flh_681 - flh_665
        glh     = b560 - (b490 + (b665 - b490) * glh_s)
        blh     = b490 - (b412 + (b560 - b412) * blh_s)
        mci     = b709 - (b665 + (b754 - b665) * mci_s)

        return {
            "FLH_681":           flh_681,
            "FLH_665":           flh_665,
            "FLHmax":            flhmax,
            "GLH":               glh,
            "BLH":               blh,
            "MCI":               mci,
            "RBD":               b682 - b665,
            "DINI":              flhmax / (glh * b490 + EPS),
            "EBI":               flhmax * blh,
            "GBI":               flh_681 * glh * 1e6,
            "KBBI":              (b682 - b665) / (b682 + b665 + EPS),
            "NDNI":              (b709 - b490) / (b709 + b490 + EPS),
            "NDCI":              (b709 - b665) / (b709 + b665 + EPS),
            "NDWI":              (b560 - b865) / (b560 + b865 + EPS),
            "RedEdge_Ratio":     b709 / (b665 + EPS),
            "CI":                b665 / (b560 + EPS),
            "BlueGreen_Ratio":   b490 / (b560 + EPS),
            "Green_Red_Ratio":   b560 / (b665 + EPS),
            "Blue_Red_Ratio":    b490 / (b665 + EPS),
            "Red_NIR_Ratio":     b665 / (b754 + EPS),
            "Fluorescence_Peak": b682,
        }

    # ------------------------------------------------------------------
    # Build fixed target AreaDefinition for a given station location.
    # Called once per location — reused across all scenes.
    # ------------------------------------------------------------------
    def _make_target_area(lat, lon, buf, res):
        """
        Regular lon/lat grid centred on (lat, lon).
        n_cols = n_rows = ceil(2*buf / res) + 1  (odd → centred pixel)
        area_extent = (lon_min, lat_min, lon_max, lat_max)
        """
        n = int(np.ceil(2 * buf / res)) + 1
        return prg.AreaDefinition(
            area_id="ifcb_box",
            description="local IFCB buffer grid",
            proj_id="longlat",
            projection={"proj": "longlat", "datum": "WGS84"},
            width=n,
            height=n,
            area_extent=(
                lon - buf,  # lon_min
                lat - buf,  # lat_min
                lon + buf,  # lon_max
                lat + buf,  # lat_max
            ),
        )

    # ------------------------------------------------------------------
    # Resample one scene onto the target grid.
    # Returns gridded_stack [n_rows, n_cols, n_bands] with NaN at
    # invalid pixels (flag != 0 or fill).
    # Returns None if the scene has no overlap with the target area.
    # ------------------------------------------------------------------
    def _resample_scene(ds, target_area):
        """
        Single kd_tree call for the full band stack + flags.
        Flag mask is pixel-level → broadcasts across all bands.
        """
        lat2d = ds["lat"].values.astype(np.float64)
        lon2d = ds["lon"].values.astype(np.float64)

        # Quick bbox pre-check — skip scene if it can't overlap target
        ta_ext  = target_area.area_extent  # (lon_min, lat_min, lon_max, lat_max)
        if (lon2d.max() < ta_ext[0] or lon2d.min() > ta_ext[2] or
                lat2d.max() < ta_ext[1] or lat2d.min() > ta_ext[3]):
            return None

        swath = prg.SwathDefinition(lons=lon2d, lats=lat2d)

        # --- Band stack: [swath_rows, swath_cols, n_bands] ---
        band_arrays = []
        for bname in band_names:
            arr = (ds[bname].values.astype(np.float32)
                   if bname in ds
                   else np.full(lat2d.shape, np.nan, dtype=np.float32))
            band_arrays.append(arr)
        band_stack = np.stack(band_arrays, axis=-1)  # [r, c, n_bands]

        # --- Resample band stack (one kdtree build) ---
        gridded_stack = kd_tree.resample_nearest(
            swath, band_stack,
            target_area,
            radius_of_influence=resample_radius,
            fill_value=np.nan,
            nprocs=1,
        )  # [n_rows, n_cols, n_bands]

        # --- Resample flags ---
        if "l2_flags" in ds:
            flags_raw = ds["l2_flags"].values.astype(np.float32)
        else:
            flags_raw = np.zeros(lat2d.shape, dtype=np.float32)

        gridded_flags = kd_tree.resample_nearest(
            swath, flags_raw,
            target_area,
            radius_of_influence=resample_radius,
            fill_value=-1.0,   # sentinel → treated as invalid
            nprocs=1,
        )  # [n_rows, n_cols]

        # --- Pixel-level flag mask (broadcasts across all bands) ---
        # flag != 0 → invalid pixel → NaN all bands at that pixel
        invalid = (gridded_flags != 0)          # [n_rows, n_cols]
        gridded_stack[invalid, :] = np.nan      # [n_rows, n_cols, n_bands]

        # Safety: NaN any residual non-finite values
        gridded_stack[~np.isfinite(gridded_stack)] = np.nan

        return gridded_stack   # [n_rows, n_cols, n_bands]

    # ------------------------------------------------------------------
    # Main worker loop
    # ------------------------------------------------------------------
    records = []

    # Pre-open all datasets for this date's scenes (avoids re-opening per location)
    open_datasets = {}
    for scene in available:
        try:
            open_datasets[scene["path"]] = xr.open_dataset(scene["path"])
        except Exception as exc:
            import sys
            print(f"[{obs_date_str}] Cannot open {scene['path']}: {exc}",
                  file=sys.stderr)

    if not open_datasets:
        return records

    try:
        for loc in locations:
            loc_id = str(loc["location_id"])
            lat    = float(loc["latitude"])
            lon    = float(loc["longitude"])

            # A. Fixed target grid for this location
            target_area = _make_target_area(lat, lon, spatial_buffer, grid_res)
            n_rows = target_area.height
            n_cols = target_area.width

            # B. Resample each scene onto target grid
            scene_arrays = []   # list of [n_rows, n_cols, n_bands]

            for scene in available:
                ds = open_datasets.get(scene["path"])
                if ds is None:
                    continue
                try:
                    gridded = _resample_scene(ds, target_area)
                except Exception as exc:
                    import sys
                    print(f"[{obs_date_str}] loc={loc_id} resample error "
                          f"{scene['path']}: {exc}", file=sys.stderr)
                    continue

                if gridded is not None:
                    scene_arrays.append(gridded)

            if not scene_arrays:
                continue

            # C. Pixel-level MVC: nanmax across scenes
            # Stack: [n_scenes, n_rows, n_cols, n_bands]
            stack = np.stack(scene_arrays, axis=0)
            mvc   = np.nanmax(stack, axis=0)   # [n_rows, n_cols, n_bands]

            # Count valid pixels: pixel is valid if ALL bands are finite
            # all_bands_valid = np.all(np.isfinite(mvc), axis=-1)  # [n_rows, n_cols]
            any_valid = np.any(np.isfinite(mvc), axis=-1)
            n_valid   = int(any_valid.sum())
            n_total   = n_rows * n_cols
            pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0

            if n_valid < min_valid_px:
                continue

            # D. Compute spectral indices per pixel on MVC arrays
            # Extract valid pixels: [n_valid_pixels, n_bands]
            valid_pixels = mvc[any_valid, :]   # [n_valid, n_bands]

            px = {bname: valid_pixels[:, i] for i, bname in enumerate(band_names)}
            indices = _compute_indices(px)  # {name: [n_valid] array}

            # E. Spatial summary: nanmedian over valid pixels
            record = {
                "location_id":      loc_id,
                "obs_date":         obs_date_str,
                "n_valid_pixels":   n_valid,
                "pct_valid_pixels": round(pct_valid, 1),
                "n_scenes":         len(scene_arrays),
            }

            # Band medians
            for i, bname in enumerate(band_names):
                col = valid_pixels[:, i]
                record[bname] = float(np.nanmedian(col))

            # Index medians
            for idx_name, arr in indices.items():
                finite = arr[np.isfinite(arr)]
                record[idx_name] = float(np.nanmedian(finite)) if len(finite) > 0 \
                    else float("nan")

            records.append(record)

    finally:
        # Close all datasets
        for ds in open_datasets.values():
            try:
                ds.close()
            except Exception:
                pass

    return records



def load_ifcb(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df[df["datetime"].dt.year > 2017].reset_index(drop=True)
    df = df[~df["dashboardIdName"].isin(STATIONS_EXCLUDE)].reset_index(drop=True)

    for name, (lat, lon) in STATION_UPDATES.items():
        mask = df["dashboardIdName"] == name
        df.loc[mask, ["latitude", "longitude"]] = lat, lon
        uniq = df.loc[mask, "location_id"].unique()
        df.loc[mask, "location_id"] = uniq[0]

    df.sort_values("datetime", inplace=True)
    df["obs_date"] = df["datetime"].dt.strftime("%Y%m%d")

    agg_dict = {
        "latitude": "first", "longitude": "first",
        "dataset_id": "first", "dataset_name": "first",
        "dashboardIdName": "first",
    }
    for c in CONC_COLS:
        if c in df.columns:
            agg_dict[c] = lambda x: x.quantile(0.95)
    for col in ["depth", "cruise", "sample_type"]:
        if col in df.columns:
            agg_dict[col] = "first"

    df = df.dropna(subset=["obs_date"]).reset_index(drop=True)

    df = (
        df.groupby(["location_id", "obs_date"], as_index=False)
        .agg(agg_dict)
        .sort_values("obs_date")
        .reset_index(drop=True)
    )
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.strftime("%Y%m%d")
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )
    dfgom = gdf[gdf.within(GOMOFS_POLYGON)].copy()
    dfgom = dfgom.sort_values("obs_date").reset_index(drop=True)

    log.info(
        "IFCB loaded: %d rows | %d locations | %d dates | %s → %s",
        len(dfgom), dfgom["location_id"].nunique(), dfgom["obs_date"].nunique(),
        dfgom["obs_date"].min(), dfgom["obs_date"].max(),
    )
    log.info("Stations: %s", sorted(dfgom["dashboardIdName"].dropna().unique().tolist()))
    return dfgom

# =============================================================================
# MAIN
# =============================================================================

def extract_features(
    ifcb_file:       Path  = IFCB_FILE,
    acolite_dir:     Path  = ACOLITE_DIR,
    output_dir:      Path  = OUTPUT_DIR,
    temporal_window: int   = TEMPORAL_WINDOW_DAYS,
    spatial_buffer:  float = SPATIAL_BUFFER_DEG,
    grid_res:        float = GRID_RESOLUTION_DEG,
    n_workers:       int   = N_WORKERS,
) -> pd.DataFrame:

    t_start    = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("ACOLITE FEATURE EXTRACTION  [Pixel-Level MVC via pyresample]")
    log.info("=" * 70)
    log.info("ACOLITE dir      : %s", acolite_dir)
    log.info("Temporal window  : +/-%d day(s)", temporal_window)
    log.info("Spatial buffer   : %.4f deg (~%.1f km)", spatial_buffer, spatial_buffer * 111)
    log.info("Grid resolution  : %.4f deg (~%.0f m)", grid_res, grid_res * 111_000)
    log.info("Resample radius  : %d m", RESAMPLE_RADIUS_M)
    log.info("Min valid pixels : %d  (all bands finite in MVC)", MIN_VALID_PIXELS)
    log.info("MVC operator     : nanmax across scenes per pixel per band")
    log.info("Spatial summary  : nanmedian over valid MVC pixels")
    log.info("Workers          : %d  (spawn)", n_workers)
    log.info("-" * 70)

    # ------------------------------------------------------------------
    # 1. Load & preprocess IFCB
    # ------------------------------------------------------------------
    df = load_ifcb(ifcb_file)
    
    # ------------------------------------------------------------------
    # 2. Scene index
    # ------------------------------------------------------------------
    scene_index = build_scene_index(Path(acolite_dir))
    scene_dates = sorted(scene_index.keys())
    n_scenes    = sum(len(v) for v in scene_index.values())

    log.info("-" * 70)
    log.info("ACOLITE scenes : %d files across %d dates  (%s to %s)",
             n_scenes, len(scene_index),
             scene_dates[0] if scene_dates else "?",
             scene_dates[-1] if scene_dates else "?")
    log.info("Exact-date overlap with IFCB: %d / %d obs dates",
             len(set(df["obs_date"].unique()) & set(scene_index.keys())),
             df["obs_date"].nunique())

    # ------------------------------------------------------------------
    # 3. Build tasks
    # ------------------------------------------------------------------
    tasks         = []
    skipped_dates = 0
    band_names    = list(BAND_MAP.keys())

    for obs_date_str, grp in df.groupby("obs_date"):
        obs_ts    = pd.Timestamp(obs_date_str)
        available = [
            {"date": d, "path": path}
            for d in candidate_dates(obs_ts, temporal_window)
            for path in scene_index.get(d, [])
        ]
        if not available:
            skipped_dates += 1
            continue

        locations = (
            grp.drop_duplicates(subset=["location_id"])
               [["location_id", "latitude", "longitude"]]
               .to_dict(orient="records")
        )
        tasks.append({
            "obs_date_str":   obs_date_str,
            "locations":      locations,
            "available":      available,
            "spatial_buffer": spatial_buffer,
            "grid_res":       grid_res,
            "min_valid_px":   MIN_VALID_PIXELS,
            "resample_radius": RESAMPLE_RADIUS_M,
            "band_names":     band_names,
        })

    log.info("Tasks: %d dates with scenes  |  %d dates skipped (no scene in window)",
             len(tasks), skipped_dates)
    log.info("-" * 70)

    # ------------------------------------------------------------------
    # 4. Parallel extraction
    # ------------------------------------------------------------------
    all_records   = []
    n_done        = 0
    n_total       = len(tasks)
    progress_step = max(1, n_total // 20)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(_process_date_worker, t): t["obs_date_str"]
            for t in tasks
        }
        for future in as_completed(futures):
            n_done += 1
            try:
                all_records.extend(future.result())
            except Exception as exc:
                log.error("[%s] Worker exception: %s", futures[future], exc)

            if n_done % progress_step == 0 or n_done == n_total:
                elapsed = (datetime.now() - t_start).total_seconds()
                rate    = n_done / elapsed if elapsed > 0 else 0
                eta_s   = (n_total - n_done) / rate if rate > 0 else 0
                log.info(
                    "  %d / %d (%.0f%%) | %d records | %.1f dates/s | ETA ~%.0fs",
                    n_done, n_total, 100.0 * n_done / n_total,
                    len(all_records), rate, eta_s,
                )

    # ------------------------------------------------------------------
    # 5. Merge and save
    # ------------------------------------------------------------------
    log.info("-" * 70)
    log.info("Extraction complete: %d records", len(all_records))

    if not all_records:
        log.warning("No records extracted.")
        return pd.DataFrame()

    features_df = pd.DataFrame(all_records)
    features_df["n_valid_pixels"]   = features_df["n_valid_pixels"].astype("uint16")
    features_df["pct_valid_pixels"] = features_df["pct_valid_pixels"].astype("float32")
    features_df["n_scenes"]         = features_df["n_scenes"].astype("uint8")

    df["location_id"]          = df["location_id"].astype(str)
    features_df["location_id"] = features_df["location_id"].astype(str)

    df_out = df.merge(features_df, on=["obs_date", "location_id"], how="left")
    df_out["data_available"] = df_out["rrs_665"].notna()

    n_with = int(df_out["data_available"].sum())
    n_tot  = len(df_out)

    log.info("=" * 70)
    log.info("MATCHUP SUMMARY")
    log.info("  Total IFCB rows          : %d", n_tot)
    log.info("  Rows with satellite data : %d (%.1f%%)", n_with, 100.0 * n_with / n_tot)
    log.info("  Rows without             : %d", n_tot - n_with)

    valid = df_out[df_out["data_available"]]
    if len(valid):
        log.info("  Valid pixel stats (spatial MVC coverage):")
        log.info("    n_valid_pixels : mean=%.1f  median=%.1f  min=%d  max=%d",
                 valid["n_valid_pixels"].mean(), valid["n_valid_pixels"].median(),
                 valid["n_valid_pixels"].min(),  valid["n_valid_pixels"].max())
        log.info("    n_scenes used  : mean=%.1f  min=%d  max=%d",
                 valid["n_scenes"].mean(),
                 valid["n_scenes"].min(), valid["n_scenes"].max())
        log.info("  Band medians (spatial nanmedian of MVC pixels):")
        for bname, wl in BAND_MAP.items():
            col = valid.get(bname, pd.Series(dtype=float)).dropna()
            if len(col):
                log.info("    %-10s (%6.2f nm): median=%+.5f  [%.5f, %.5f]",
                         bname, wl, col.median(), col.min(), col.max())
        log.info("  Key index medians:")
        for idx in ["FLH_681", "FLHmax", "MCI", "NDCI", "KBBI", "DINI", "GLH"]:
            col = valid.get(idx, pd.Series(dtype=float)).dropna()
            if len(col):
                log.info("    %-20s median = %+.5f", idx, col.median())

    out_path = output_dir / "acolite_features_mvc.parquet"
    df_out.to_parquet(out_path, index=False)
    log.info("-" * 70)
    log.info("Saved --> %s", out_path)
    log.info("Total runtime: %.1f s", (datetime.now() - t_start).total_seconds())
    log.info("=" * 70)

    return df_out


if __name__ == "__main__":
    extract_features()