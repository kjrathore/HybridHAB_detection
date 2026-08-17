"""
C2RCC Mosaic Feature Extraction for IFCB Matchups — Pixel-Level MVC Edition
============================================================================

PSEUDO-ALGORITHM
================

SETUP
-----
1. Load IFCB parquet → filter excluded stations → fix coordinates →
   aggregate to daily rows (95th percentile for concentrations).
2. Build mosaic index: {YYYYMMDD: path} from mosaic directory.

PER OBS-DATE (parallelised across dates via ProcessPoolExecutor/spawn)
----------------------------------------------------------------------
For each obs_date:
    Collect ALL mosaic files within ±TEMPORAL_WINDOW_DAYS.

    For each IFCB location on that date:

        A. EXTRACT BBOX FROM EACH MOSAIC
           For each mosaic in window:
             i.  Slice lat/lon bbox from the regular grid.
                 (lat is DESCENDING → slice(lat+buf, lat-buf))
             ii. Load all rrs bands → [n_lat, n_lon, n_bands] float32.
             iii.Apply pixel-level validity mask:
                   Primary  : quality_flags (WATER bit) & c2rcc_flags (Valid_PE bit)
                   Fallback : all rrs bands > 0 (mosaics store 0.0 for invalid)
                 Set invalid pixels to NaN across ALL bands (pixel-level).
             iv. Result: [n_lat, n_lon, n_bands] with NaN at invalid pixels.

        B. PIXEL-LEVEL MAXIMUM-VALUE COMPOSITE (MVC)
           - Mosaics share the SAME regular lat/lon grid → no resampling needed.
           - Stack valid scene arrays: [n_mosaics, n_lat, n_lon, n_bands].
           - nanmax across mosaic axis (axis=0)
             → mvc: [n_lat, n_lon, n_bands]
           - Pixel clear in at least one mosaic → gets that mosaic's Rrs.
           - Pixel cloudy/invalid in ALL mosaics → remains NaN.

        C. COMPUTE SPECTRAL INDICES PER PIXEL
           - Flatten mvc to valid pixels: [n_valid, n_bands]
             (valid = ALL bands finite after MVC)
           - Compute all indices per pixel → [n_valid, n_indices].
           - NaN pixels propagate NaN into indices automatically.

        D. SPATIAL SUMMARY
           - nanmedian over valid pixels for each band and each index
             → one scalar per feature.
           - n_valid_pixels = count of pixels with ALL bands finite in mvc.
           - pct_valid_pixels = n_valid / (n_lat * n_lon) * 100.
           - n_mosaics_used = number of mosaics that had bbox overlap.
           - Skip location if n_valid_pixels < MIN_VALID_PIXELS.

        E. IOPs
           - Same MVC stack approach for IOP variables.
           - IOPs masked independently: iop > 0 and finite (no flag dependency).
           - nanmedian over valid pixels.

        F. RECORD
           - Emit one flat dict per location.
           - No sat_date / delta_days (values may span multiple mosaics).

MERGE & SAVE
------------
- Merge feature records back to IFCB df on [obs_date, location_id].
- Save to c2rcc_features_mvc.parquet.

KEY DESIGN DECISIONS
====================
- Regular grid: C2RCC mosaics are pre-projected → pixel alignment guaranteed
  across dates → no resampling required (unlike ACOLITE swath data).
- nanmax is the MVC operator → maximises spatial coverage across window.
- Flag mask is pixel-level → set NaN across all bands simultaneously.
- Indices computed on MVC pixel arrays (not scalar medians) → spectrally
  consistent per pixel.
- Spatial median is the only aggregation over values → robust to outliers.
- spawn multiprocessing → each worker gets a clean HDF5/NetCDF4 context.

Mosaic file naming: S3_OLCI_GOM_YYYYMMDD_mosaic.nc
Band mapping (rrs_N → OLCI band N):
  rrs_2  → B02 (412.5 nm)     rrs_4  → B04 (490 nm)
  rrs_6  → B06 (560 nm)       rrs_8  → B08 (665 nm)
  rrs_9  → B09 (673.75 nm)    rrs_10 → B10 (681.25 nm)
  rrs_11 → B11 (708.75 nm)    rrs_12 → B12 (753.75 nm)
  rrs_17 → B17 (865 nm)

Coordinate layout (confirmed by diagnostic):
  - lat: DESCENDING (45.0 → 36.003, step -0.003 deg)
  - lon: ascending  (-76.0 → -66.004, step +0.003 deg)
  - Spatial slice must use slice(lat+buf, lat-buf) for lat axis.

Valid-pixel masking:
  - No fill values in mosaics (all pixels have values).
  - Masked/invalid ocean pixels stored as 0.0, not NaN or fill.
  - Primary mask: quality_flags WATER bit & c2rcc_flags Valid_PE bit.
  - Fallback: all rrs bands > 0.
"""

import logging
import multiprocessing
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon

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

MOSAIC_DIR = Path(
    "/data3/GULF_OF_MAINE/processed_data/sentinel_3_c2rcc_mosaics/sentinel_3_c2rcc_mosaics"
)
IFCB_FILE = (
    "datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_IFCB_GOM.parquet"
)
OUTPUT_DIR = Path(
    "datasets/GULF_OF_MAINE/processed_data/sentinel_3_L2_past3days/c2rcc_features"   #MVC
)

TEMPORAL_WINDOW_DAYS = 3      # past N days around each IFCB observation
SPATIAL_BUFFER_DEG   = 0.02  # ~2 km
MIN_VALID_PIXELS     = 3      # minimum valid pixels required in MVC
N_WORKERS            = 6

EPS = 1e-10

IOP_VARS  = ["kd489", "kdmin", "iop_adet", "iop_agelb",
             "iop_apig", "iop_bpart", "iop_bwit"]
BAND_COLS = ["rrs_2", "rrs_4", "rrs_6", "rrs_8", "rrs_9",
             "rrs_10", "rrs_11", "rrs_12", "rrs_17"]
BAND_WL   = {
    "rrs_2":  412.5,  "rrs_4":  490.0,  "rrs_6":  560.0,
    "rrs_8":  665.0,  "rrs_9":  673.75, "rrs_10": 681.25,
    "rrs_11": 708.75, "rrs_12": 753.75, "rrs_17": 865.0,
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
# HELPERS
# =============================================================================

def mosaic_date(path: Path) -> str | None:
    m = re.search(r"(\d{8})_mosaic", path.name)
    return m.group(1) if m else None


def build_mosaic_index(mosaic_dir: Path) -> dict:
    """Return {YYYYMMDD: str(Path)}."""
    index = {}
    for p in sorted(mosaic_dir.glob("S3_OLCI_GOM_*_mosaic.nc")):
        d = mosaic_date(p)
        if d:
            index[d] = str(p)
    if not index:
        log.warning("No mosaic files found in %s", mosaic_dir)
    return index


def candidate_dates(obs_ts: pd.Timestamp, window: int) -> list:
    """All YYYYMMDD strings within ±window days."""
    return [
        (obs_ts + timedelta(days=d)).strftime("%Y%m%d")
        for d in range(-window, 1)
    ]


# =============================================================================
# WORKER
# =============================================================================

def _process_date_worker(task: dict) -> list:
    """
    Spawned subprocess worker.
    xarray imported locally for HDF5/NetCDF4 safety.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import xarray as xr

    obs_date_str   = task["obs_date_str"]
    locations      = task["locations"]
    available      = task["available"]      # [(date_str, path_str), ...]
    spatial_buffer = task["spatial_buffer"]
    min_valid_px   = task["min_valid_px"]

    band_cols = ["rrs_2", "rrs_4", "rrs_6", "rrs_8", "rrs_9",
                 "rrs_10", "rrs_11", "rrs_12", "rrs_17"]
    iop_vars  = ["kd489", "kdmin", "iop_adet", "iop_agelb",
                 "iop_apig", "iop_bpart", "iop_bwit"]
    eps = 1e-10

    # ------------------------------------------------------------------
    # Spectral indices — operates on [n_valid_pixels] 1-D arrays
    # ------------------------------------------------------------------
    def _compute_indices(b02, b04, b06, b08, b09, b10, b11, b12, b17):
        s681  = (681.25 - 673.75) / (708.75 - 673.75)
        s665  = (665.00 - 673.75) / (708.75 - 673.75)
        glh_s = (560.0  - 490.0)  / (665.0  - 490.0)
        blh_s = (490.0  - 412.5)  / (560.0  - 412.5)
        mci_s = (708.75 - 665.0)  / (753.75 - 665.0)

        flh_681 = b10 - (b09 + (b11 - b09) * s681)
        flh_665 = b08 - (b09 + (b11 - b09) * s665)
        flhmax  = flh_681 - flh_665
        glh     = b06 - (b04 + (b08 - b04) * glh_s)
        blh     = b04 - (b02 + (b06 - b02) * blh_s)
        mci     = b11 - (b08 + (b12 - b08) * mci_s)

        return {
            "FLH_681":           flh_681,
            "FLH_665":           flh_665,
            "FLHmax":            flhmax,
            "GLH":               glh,
            "BLH":               blh,
            "MCI":               mci,
            "RBD":               b10 - b08,
            "DINI":              flhmax / (glh * b04 + eps),
            "EBI":               flhmax * blh,
            "GBI":               flh_681 * glh * 1e6,
            "KBBI":              (b10 - b08) / (b10 + b08 + eps),
            "NDNI":              (b11 - b04) / (b11 + b04 + eps),
            "NDCI":              (b11 - b08) / (b11 + b08 + eps),
            "NDWI":              (b06 - b17) / (b06 + b17 + eps),
            "RedEdge_Ratio":     b11 / (b08 + eps),
            "CI":                b08 / (b06 + eps),
            "BlueGreen_Ratio":   b04 / (b06 + eps),
            "Green_Red_Ratio":   b06 / (b08 + eps),
            "Blue_Red_Ratio":    b04 / (b08 + eps),
            "Red_NIR_Ratio":     b08 / (b12 + eps),
            "Fluorescence_Peak": b10,
        }

    # ------------------------------------------------------------------
    # Extract bbox from one mosaic → [n_lat, n_lon, n_bands] with NaN
    # at invalid pixels. Returns None if bbox is empty.
    # ------------------------------------------------------------------
    def _extract_bbox(ds, lat, lon, buf):
        # Descending lat → slice(high, low); ascending lon → slice(low, high)
        sub = ds.sel(
            lat=slice(lat + buf, lat - buf),
            lon=slice(lon - buf, lon + buf),
        )
        n_lat = sub.sizes.get("lat", 0)
        n_lon = sub.sizes.get("lon", 0)
        if n_lat == 0 or n_lon == 0:
            return None, None

        # Load all rrs bands → [n_lat, n_lon, n_bands]
        band_arrays = []
        for var in band_cols:
            if var not in sub:
                return None, None
            band_arrays.append(sub[var].values.astype(np.float32))
        bbox_stack = np.stack(band_arrays, axis=-1)  # [n_lat, n_lon, n_bands]

        # ------------------------------------------------------------------
        # Pixel-level validity mask
        # Primary: quality_flags (WATER=bit1) & c2rcc_flags (Valid_PE=bit31)
        # Fallback: all rrs bands > 0 (mosaics use 0.0 for invalid pixels)
        # ------------------------------------------------------------------
        WATER_MASK    = 2
        VALID_PE_MASK = 2147483648  # 1 << 31

        valid = None
        if "quality_flags" in sub and "c2rcc_flags" in sub:
            qf = sub["quality_flags"].values.astype(np.int64)
            cf = sub["c2rcc_flags"].values.astype(np.int64)
            flag_valid = ((qf & WATER_MASK) != 0) & ((cf & VALID_PE_MASK) != 0)
            if flag_valid.any():
                valid = flag_valid  # [n_lat, n_lon] bool

        if valid is None:
            # Fallback: pixel valid only if ALL bands > 0
            valid = np.all(bbox_stack > 0, axis=-1)  # [n_lat, n_lon]

        # Apply pixel-level mask: invalid pixels → NaN across ALL bands
        bbox_stack[~valid, :] = np.nan   # broadcasts over band axis

        # Load IOP stack separately — same spatial mask applied
        iop_arrays = []
        for iop in iop_vars:
            if iop in sub:
                arr = sub[iop].values.astype(np.float32)
                # IOP-specific validity: > 0 AND within rrs-valid pixels
                iop_arr = np.where(valid & (arr > 0) & np.isfinite(arr),
                                   arr, np.nan)
            else:
                iop_arr = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
            iop_arrays.append(iop_arr)
        iop_stack = np.stack(iop_arrays, axis=-1)  # [n_lat, n_lon, n_iops]

        return bbox_stack, iop_stack   # [n_lat, n_lon, n_bands/n_iops]

    # ------------------------------------------------------------------
    # Pre-open all datasets once (shared across locations on this date)
    # ------------------------------------------------------------------
    open_datasets = {}
    for date_str, path_str in available:
        try:
            open_datasets[(date_str, path_str)] = xr.open_dataset(path_str)
        except Exception as exc:
            import sys
            print(f"[{obs_date_str}] Cannot open {path_str}: {exc}",
                  file=sys.stderr)

    records = []

    try:
        for loc in locations:
            loc_id = str(loc["location_id"])
            lat    = float(loc["latitude"])
            lon    = float(loc["longitude"])

            # A. Extract bbox from every mosaic in the window
            band_scene_list = []   # list of [n_lat, n_lon, n_bands]
            iop_scene_list  = []   # list of [n_lat, n_lon, n_iops]
            n_mosaics_used  = 0

            for (date_str, path_str), ds in open_datasets.items():
                try:
                    bbox_bands, bbox_iops = _extract_bbox(
                        ds, lat, lon, spatial_buffer
                    )
                except Exception as exc:
                    import sys
                    print(f"[{obs_date_str}] loc={loc_id} bbox error: {exc}",
                          file=sys.stderr)
                    continue

                if bbox_bands is not None:
                    band_scene_list.append(bbox_bands)
                    iop_scene_list.append(bbox_iops)
                    n_mosaics_used += 1

            if not band_scene_list:
                continue

            # B. Pixel-level MVC: nanmax across mosaics
            # Stack: [n_mosaics, n_lat, n_lon, n_bands]
            band_stack = np.stack(band_scene_list, axis=0)
            mvc_bands  = np.nanmax(band_stack, axis=0)  # [n_lat, n_lon, n_bands]

            iop_stack  = np.stack(iop_scene_list, axis=0)
            mvc_iops   = np.nanmax(iop_stack, axis=0)   # [n_lat, n_lon, n_iops]

            # C. Valid pixel mask: ALL bands finite after MVC
            any_valid = np.any(np.isfinite(mvc_bands), axis=-1)
            n_valid   = int(any_valid.sum())
            n_total   = mvc_bands.shape[0] * mvc_bands.shape[1]
            pct_valid = 100.0 * n_valid / n_total if n_total > 0 else 0.0

            if n_valid < min_valid_px:
                continue

            # D. Flatten to valid pixels → [n_valid, n_bands]
            valid_bands = mvc_bands[any_valid, :]   # [n_valid, n_bands]  may contain NaN

            b02 = valid_bands[:, 0]; b04 = valid_bands[:, 1]
            b06 = valid_bands[:, 2]; b08 = valid_bands[:, 3]
            b09 = valid_bands[:, 4]; b10 = valid_bands[:, 5]
            b11 = valid_bands[:, 6]; b12 = valid_bands[:, 7]
            b17 = valid_bands[:, 8]

            # E. Spectral indices per pixel
            indices = _compute_indices(b02, b04, b06, b08, b09,
                                       b10, b11, b12, b17)

            # F. Spatial summary
            record = {
                "location_id":      loc_id,
                "obs_date":         obs_date_str,
                "n_valid_pixels":   n_valid,
                "pct_valid_pixels": round(pct_valid, 1),
                "n_mosaics_used":   n_mosaics_used,
            }

            # Band medians
            for i, var in enumerate(band_cols):
                record[var] = float(np.nanmedian(valid_bands[:, i]))

            # Index medians
            for idx_name, arr in indices.items():
                finite = arr[np.isfinite(arr)]
                record[idx_name] = (float(np.nanmedian(finite))
                                    if len(finite) > 0 else float("nan"))

            # IOP medians (use any_valid mask on mvc_iops)
            valid_iops = mvc_iops[any_valid, :]  # [n_valid, n_iops]
            for j, iop in enumerate(iop_vars):
                col    = valid_iops[:, j]
                finite = col[np.isfinite(col)]
                record[iop] = (float(np.nanmedian(finite))
                               if len(finite) > 0 else float("nan"))

            records.append(record)

    finally:
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
    ifcb_file:       str   = IFCB_FILE,
    mosaic_dir:      Path  = MOSAIC_DIR,
    output_dir:      Path  = OUTPUT_DIR,
    temporal_window: int   = TEMPORAL_WINDOW_DAYS,
    spatial_buffer:  float = SPATIAL_BUFFER_DEG,
    n_workers:       int   = N_WORKERS,
) -> pd.DataFrame:

    t_start    = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("C2RCC MOSAIC FEATURE EXTRACTION  [Pixel-Level MVC]")
    log.info("=" * 70)
    log.info("Input file       : %s", ifcb_file)
    log.info("Mosaic dir       : %s", mosaic_dir)
    log.info("Output dir       : %s", output_dir)
    log.info("Temporal window  : -%d day(s)", temporal_window)
    log.info("Spatial buffer   : %.4f deg (~%.1f km)", spatial_buffer, spatial_buffer * 111)
    log.info("Min valid px     : %d  (all bands finite in MVC)", MIN_VALID_PIXELS)
    log.info("MVC operator     : nanmax across mosaics per pixel per band")
    log.info("Spatial summary  : nanmedian over valid MVC pixels")
    log.info("Workers          : %d processes (spawn)", n_workers)
    log.info("-" * 70)

    # ------------------------------------------------------------------
    # 1. Load & preprocess IFCB
    # ------------------------------------------------------------------
    df = load_ifcb(ifcb_file)
    
    # ------------------------------------------------------------------
    # 2. Build mosaic index
    # ------------------------------------------------------------------
    mosaic_index = build_mosaic_index(Path(mosaic_dir))
    mosaic_dates = sorted(mosaic_index.keys())

    log.info("-" * 70)
    log.info("Mosaic files found : %d", len(mosaic_index))
    if mosaic_dates:
        log.info("  Date range       : %s to %s", mosaic_dates[0], mosaic_dates[-1])

    overlap = set(df["obs_date"].unique()) & set(mosaic_index.keys())
    log.info("  Exact-date overlap with IFCB: %d / %d observation dates",
             len(overlap), df["obs_date"].nunique())

    # ------------------------------------------------------------------
    # 3. Build tasks — ALL mosaics in window passed per task
    # ------------------------------------------------------------------
    tasks         = []
    skipped_dates = 0

    for obs_date_str, grp in df.groupby("obs_date"):
        obs_ts    = pd.Timestamp(obs_date_str)
        available = [
            (d, mosaic_index[d])
            for d in candidate_dates(obs_ts, temporal_window)
            if d in mosaic_index
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
            "available":      available,   # ALL dates in window, not closest-first
            "spatial_buffer": spatial_buffer,
            "min_valid_px":   MIN_VALID_PIXELS,
        })

    log.info("  Dates with >=1 mosaic in window : %d", len(tasks))
    log.info("  Dates with no mosaic (skipped)  : %d", skipped_dates)
    log.info("-" * 70)
    log.info("Submitting %d tasks to %d worker process(es)...", len(tasks), n_workers)

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
            pool.submit(_process_date_worker, task): task["obs_date_str"]
            for task in tasks
        }

        for future in as_completed(futures):
            obs_date_str = futures[future]
            n_done      += 1

            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                log.error("[%s] Worker raised an exception: %s", obs_date_str, exc)

            if n_done % progress_step == 0 or n_done == n_total:
                elapsed = (datetime.now() - t_start).total_seconds()
                rate    = n_done / elapsed if elapsed > 0 else 0
                eta_s   = (n_total - n_done) / rate if rate > 0 else 0
                log.info(
                    "  Progress: %d / %d dates (%.0f%%) | "
                    "%d records so far | %.1f dates/s | ETA ~%.0fs",
                    n_done, n_total,
                    100.0 * n_done / n_total,
                    len(all_records), rate, eta_s,
                )

    # ------------------------------------------------------------------
    # 5. Assemble and save
    # ------------------------------------------------------------------
    log.info("-" * 70)
    log.info("Extraction complete. Total feature records: %d", len(all_records))

    if not all_records:
        log.warning("No records extracted.")
        return pd.DataFrame()

    features_df = pd.DataFrame(all_records)
    features_df["n_valid_pixels"]   = features_df["n_valid_pixels"].astype("uint16")
    features_df["pct_valid_pixels"] = features_df["pct_valid_pixels"].astype("float32")
    features_df["n_mosaics_used"]   = features_df["n_mosaics_used"].astype("uint8")

    df["location_id"]          = df["location_id"].astype(str)
    features_df["location_id"] = features_df["location_id"].astype(str)

    df_out = df.merge(features_df, on=["obs_date", "location_id"], how="left")
    df_out["data_available"] = df_out["rrs_8"].notna()

    n_with   = int(df_out["data_available"].sum())
    n_total_ = len(df_out)

    log.info("=" * 70)
    log.info("MATCHUP SUMMARY")
    log.info("=" * 70)
    log.info("  Total IFCB rows          : %d", n_total_)
    log.info("  Rows with satellite data : %d (%.1f%%)", n_with, 100.0 * n_with / n_total_)
    log.info("  Rows without             : %d", n_total_ - n_with)

    valid_rows = df_out[df_out["data_available"]]
    if len(valid_rows) > 0:
        log.info("  n_mosaics_used : mean=%.1f  min=%d  max=%d",
                 valid_rows["n_mosaics_used"].mean(),
                 valid_rows["n_mosaics_used"].min(),
                 valid_rows["n_mosaics_used"].max())
        log.info("  Valid px : mean=%.1f  median=%.1f  min=%d  max=%d",
                 valid_rows["n_valid_pixels"].mean(),
                 valid_rows["n_valid_pixels"].median(),
                 valid_rows["n_valid_pixels"].min(),
                 valid_rows["n_valid_pixels"].max())
        log.info("  Band medians (MVC spatial nanmedian):")
        for var, wl in BAND_WL.items():
            col = valid_rows[var].dropna()
            if len(col):
                log.info("    %-6s (%6.2f nm): median=%+.5f  range=[%.5f, %.5f]",
                         var, wl, col.median(), col.min(), col.max())
        log.info("  Key index medians:")
        for idx in ["FLH_681", "FLHmax", "MCI", "NDCI", "KBBI", "DINI", "GLH"]:
            col = valid_rows[idx].dropna()
            if len(col):
                log.info("    %-20s median = %+.5f", idx, col.median())
        log.info("  IOP medians:")
        for iop in IOP_VARS:
            col = valid_rows[iop].dropna()
            if len(col):
                log.info("    %-20s median = %+.5f", iop, col.median())

    out_path = output_dir / "c2rcc_features_mvc.parquet"
    df_out.to_parquet(out_path, index=False)
    elapsed_total = (datetime.now() - t_start).total_seconds()
    log.info("-" * 70)
    log.info("Saved --> %s", out_path)
    log.info("Total runtime: %.1f s", elapsed_total)
    log.info("=" * 70)

    return df_out


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    extract_features()