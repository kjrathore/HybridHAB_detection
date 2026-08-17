"""
extract_sh_s3_features.py
==========================
Extract Sentinel-3 OLCI L2 spectral features for IFCB matchups
using Sentinel Hub CDSE Process API.

WHY SENTINEL HUB OVER PLANETARY COMPUTER:
  - Native spatial subset: specify bbox → server returns only those pixels
  - No geo-coordinate file to load (~200s per scene on PC → eliminated)
  - All 9 bands returned in one request (~3s per station)
  - dataMask replaces WQSF for valid-pixel filtering (simpler, adequate)

RATE LIMITS (Copernicus General User):
  - 10,000 PU / month  (each 12x12 px / 9-band request ≈ 0.0017 PU → not a constraint)
  - ~30 requests / minute (General User estimate; enforced via MIN_REQUEST_INTERVAL)
  - No Batch API access → synchronous Process API only
  - 429 response → honour Retry-After header, then retry

PU COST PER REQUEST:
  (15*11 px / 262144) * (9 bands / 3) * 1.0 (float32) ≈ 0.0019 PU
  Full run (~2671 dates * ~8 stations * ~1.5 attempts) ≈ 32,000 requests ≈ 61 PU total
  Well within 10,000 PU/month budget.

PSEUDO-ALGORITHM:
  For each obs_date (chronological):
    For each (lat, lon) station on that date:
      1. Build time window: [obs_date - TEMPORAL_WINDOW_DAYS, obs_date]
      2. ONE request: TILE mosaicking over the full window
         - evalscript iterates ALL acquisitions across all days in window
         - Per pixel: picks first sample where dataMask=1 AND B08>0
         - updateOutputMetadata: records which dates contributed (for logging)
      3. If valid pixels >= MIN_VALID_PIXELS → compute indices, record result
      Rate-limit: enforce MIN_REQUEST_INTERVAL between requests
      Retry on HTTP 429 with Retry-After backoff
    Save checkpoint every CHECKPOINT_EVERY dates

PU COST NOTE:
  1 request per station per date (not 5).
  ~2671 dates * ~8 stations = ~21k requests ≈ 40 PU total.
  Well within 10,000 PU/month budget.
  delta_days logged as min/max of contributing scene dates (low priority).

Run: python extract_sh_s3_features.py
Requires: pip install sentinelhub
Credentials: export SH_CLIENT_ID=... SH_CLIENT_SECRET=...
"""

import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from shapely.geometry import Polygon

import numpy as np
import pandas as pd
import geopandas as gpd
from sentinelhub import (
    BBox, CRS, DataCollection, MimeType, SHConfig,
    SentinelHubRequest, bbox_to_dimensions,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

IFCB_FILE = (
    "datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_IFCB_GOM.parquet"
)
OUTPUT_DIR = Path(
    "datasets/GULF_OF_MAINE/processed_data/sentinel_3_L2_past3days/bac_features"
)


CDSE_URL   = "https://sh.dataspace.copernicus.eu"
TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COLLECTION = "sentinel-3-olci-l2"

TEMPORAL_WINDOW_DAYS = 3   # query window: obs_date ± 2 days in one request
SPATIAL_BUFFER_DEG = 0.02      # ±0.02° → ~14×14 px at 300m, enough for 12x12 valid core
MIN_VALID_PIXELS   = 3         # minimum valid (non-masked) pixels to accept a matchup
CHECKPOINT_EVERY   = 100        # save parquet every N dates

# Rate limiting — General User: stay well under ~30 req/min
# 1 request per 2.5s = 24 req/min, leaving headroom for retries
MIN_REQUEST_INTERVAL = 0.1     # seconds between requests
MAX_RETRIES          = 2
RETRY_BACKOFF_BASE   = 3      # seconds, doubles each retry

EPS = 1e-10

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
# Evalscript output band order (15 bands total):
#   0-8:  optical Rrs (max-B08 acquisition)
#   9-13: derived products (max over window)
#   14:   validity flag
OPTICAL_BANDS = ["rrs_2", "rrs_4", "rrs_6", "rrs_8", "rrs_9",
                 "rrs_10", "rrs_11", "rrs_12", "rrs_17"]
DERIVED_BANDS = ["CHL_OC4ME", "CHL_NN", "TSM_NN", "ADG443_NN", "KD490_M07"]
BAND_NAMES    = OPTICAL_BANDS  # used for spectral index computation


# =============================================================================
# SENTINEL HUB CONFIG + COLLECTION
# =============================================================================

def get_config() -> SHConfig:
    config = SHConfig()
    config.sh_client_id     = 'sh-cc8ce4d0-a106-49cf-b613-205dddfbb2fc' #os.environ.get("SH_CLIENT_ID", "")
    config.sh_client_secret = 'YBlL8YufyBsfqalp8ZpzhfRbEnjYkU4r'  #os.environ.get("SH_CLIENT_SECRET", "")
    config.sh_base_url      = CDSE_URL
    config.sh_token_url     = TOKEN_URL

    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.\n"
            "Get credentials: https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings"
        )
    return config


# Define L2 collection — not in sentinelhub-py defaults, needs manual definition
S3_OLCI_L2 = DataCollection.define(
    name        = "SENTINEL3_OLCI_L2_CDSE",
    api_id      = COLLECTION,
    catalog_id  = COLLECTION,
    service_url = CDSE_URL,
)

# Evalscript: single TILE request covering obs_date - TEMPORAL_WINDOW_DAYS.
# Output bands (15 total):
#   0-8  : optical Rrs (B02,B04,B06,B08,B09,B10,B11,B12,B17) — pixel with max B08
#   9-13 : derived products (CHL_OC4ME,CHL_NN,TSM_NN,ADG443_NN,KD490_M07) — max over window
#   14   : validity flag (1.0 = valid optical retrieval found, 0 = none)
#
# Optical strategy: pick the acquisition with highest B08 among valid pixels.
# B08 (665nm red) is the strongest bloom signal band — max B08 selects the
# most bloom-relevant scene in the window, not just the most recent.
# Overhead vs first-valid: one extra comparison per sample — negligible.
#
# Derived strategy: independent max per band across all valid acquisitions.
# These are log10-scaled (CHL_NN, TSM_NN etc.) so max = peak bloom signal.
EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02","B04","B06","B08","B09","B10","B11","B12","B17",
                    "CHL_OC4ME","CHL_NN","TSM_NN","ADG443_NN","KD490_M07",
                    "dataMask"],
        }],
        output: { bands: 15, sampleType: "FLOAT32" },
        mosaicking: "TILE"
    };
}
function evaluatePixel(samples) {
    // --- Optical: find acquisition with max B08 among valid pixels ---
    var bestB08 = -1;
    var bestIdx = -1;
    for (var i = 0; i < samples.length; i++) {
        var s = samples[i];
        if (s.dataMask === 1 && s.B08 > 0 && s.B08 > bestB08) {
            bestB08 = s.B08;
            bestIdx = i;
        }
    }

    // --- Derived: max over all valid acquisitions ---
    var maxCHL_OC4ME = 0, maxCHL_NN = 0, maxTSM_NN = 0;
    var maxADG443_NN = 0, maxKD490_M07 = 0;
    for (var j = 0; j < samples.length; j++) {
        var d = samples[j];
        if (d.dataMask !== 1) continue;
        if (d.CHL_OC4ME > maxCHL_OC4ME) maxCHL_OC4ME = d.CHL_OC4ME;
        if (d.CHL_NN    > maxCHL_NN)    maxCHL_NN    = d.CHL_NN;
        if (d.TSM_NN    > maxTSM_NN)    maxTSM_NN    = d.TSM_NN;
        if (d.ADG443_NN > maxADG443_NN) maxADG443_NN = d.ADG443_NN;
        if (d.KD490_M07 > maxKD490_M07) maxKD490_M07 = d.KD490_M07;
    }

    if (bestIdx === -1) {
        // No valid optical retrieval in window
        return [0,0,0,0,0,0,0,0,0, maxCHL_OC4ME,maxCHL_NN,maxTSM_NN,maxADG443_NN,maxKD490_M07, 0];
    }

    var b = samples[bestIdx];
    return [
        b.B02, b.B04, b.B06, b.B08, b.B09, b.B10, b.B11, b.B12, b.B17,
        maxCHL_OC4ME, maxCHL_NN, maxTSM_NN, maxADG443_NN, maxKD490_M07,
        1.0
    ];
}

"""


# =============================================================================
# STEP 1: LOAD + PREPROCESS IFCB
# =============================================================================

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
    # df["loc_key"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    
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
# STEP 2: SPECTRAL INDICES
# Computed pixel-by-pixel on 1D arrays before median aggregation
# =============================================================================

def compute_indices(b02, b04, b06, b08, b09, b10, b11, b12, b17) -> dict:
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
        "DINI":              flhmax / (glh * b04 + EPS),
        "EBI":               flhmax * blh,
        "GBI":               flh_681 * glh * 1e6,
        "KBBI":              (b10 - b08) / (b10 + b08 + EPS),
        "NDNI":              (b11 - b04) / (b11 + b04 + EPS),
        "NDCI":              (b11 - b08) / (b11 + b08 + EPS),
        "NDWI":              (b06 - b17) / (b06 + b17 + EPS),
        "RedEdge_Ratio":     b11 / (b08 + EPS),
        "CI":                b08 / (b06 + EPS),
        "BlueGreen_Ratio":   b04 / (b06 + EPS),
        "Green_Red_Ratio":   b06 / (b08 + EPS),
        "Blue_Red_Ratio":    b04 / (b08 + EPS),
        "Red_NIR_Ratio":     b08 / (b12 + EPS),
        "Fluorescence_Peak": b10,
    }


# =============================================================================
# STEP 3: RATE-LIMITED REQUEST WRAPPER
# =============================================================================

class RateLimiter:
    """Enforces minimum interval between SH Process API calls."""
    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        self._min_interval = min_interval
        self._last_call    = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        gap     = self._min_interval - elapsed
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.time()


def fetch_patch(
    lat: float, lon: float,
    date_start: str, date_end: str,
    config: SHConfig,
    limiter: RateLimiter,
) -> np.ndarray | None:
    """
    Fetch a spatial patch centred on (lat, lon) over [date_start, date_end].
    TILE mosaicking: evalscript picks best valid pixel across all acquisitions.
    Returns array of shape (H, W, 15) or None on failure.
    """
    bbox = BBox(
        bbox=[lon - SPATIAL_BUFFER_DEG, lat - SPATIAL_BUFFER_DEG,
              lon + SPATIAL_BUFFER_DEG, lat + SPATIAL_BUFFER_DEG],
        crs=CRS.WGS84,
    )
    size = bbox_to_dimensions(bbox, resolution=300)

    request = SentinelHubRequest(
        evalscript = EVALSCRIPT,
        input_data = [
            SentinelHubRequest.input_data(
                data_collection = S3_OLCI_L2,
                time_interval   = (date_start, date_end),
                other_args      = {"dataFilter": {"mosaickingOrder": "mostRecent"}},
            )
        ],
        responses = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox      = bbox,
        size      = size,
        config    = config,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            data = request.get_data()
            if not data or data[0] is None:
                return None
            return data[0]
        except Exception as e:
            err_str = str(e)
            retry_after = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            if "429" in err_str or "Too Many Requests" in err_str:
                log.warning("  [429] rate limited  attempt=%d/%d  waiting=%.0fs",
                            attempt, MAX_RETRIES, retry_after)
            elif attempt == MAX_RETRIES:
                log.warning("  [fetch] failed after %d attempts: %s", MAX_RETRIES, e)
                return None
            else:
                log.debug("  [fetch] attempt %d/%d failed: %s  retrying in %.0fs",
                          attempt, MAX_RETRIES, e, retry_after)
            time.sleep(retry_after)

    return None


# =============================================================================
# STEP 4: EXTRACT ONE (station, date) MATCHUP
# =============================================================================

def extract_matchup(
    lat: float, lon: float, location_id: str,
    obs_date_str: str,
    config: SHConfig,
    limiter: RateLimiter,
) -> dict | None:
    """
    Single request covering obs_date - TEMPORAL_WINDOW_DAYS.
    TILE evalscript picks best valid pixel per location across all acquisitions.
    """
    obs_ts     = pd.Timestamp(obs_date_str)
    date_start = (obs_ts - timedelta(days=TEMPORAL_WINDOW_DAYS)).strftime("%Y-%m-%d")
    date_end   = (obs_ts).strftime("%Y-%m-%d")

    arr = fetch_patch(lat, lon, date_start, date_end, config, limiter)

    if arr is None:
        log.debug("  [%s] loc=%s: fetch returned None", obs_date_str, location_id)
        return None

    # Band 14 = validity flag: 1.0 where optical retrieval was found, else 0
    # Derived bands (9-13) may have values even when optical is invalid (separate max)
    valid_mask  = arr[:, :, 14] > 0   # pixels with valid optical retrieval
    n_valid     = int(valid_mask.sum())
    n_total     = valid_mask.size
    n_any_data  = int((arr[:, :, 14] >= 0).sum())  # total pixels returned

    log.debug("  [%s] loc=%s: %d/%d optical-valid pixels",
              obs_date_str, location_id, n_valid, n_total)

    record = {
        "location_id":     location_id,
        "obs_date":    obs_date_str,
        "date_start":  date_start,
        "date_end":    date_end,
    }

    # --- Derived bands: median over valid pixels (max already done in evalscript) ---
    for i, name in enumerate(DERIVED_BANDS):
        band_idx = 9 + i
        vals = arr[:, :, band_idx].ravel().astype(np.float64)
        vals = vals[(vals > 0) & np.isfinite(vals)]
        record[name] = float(np.median(vals)) if len(vals) > 0 else np.nan

    # --- Optical bands: require MIN_VALID_PIXELS ---
    if n_valid < MIN_VALID_PIXELS:
        log.debug("  [%s] loc=%s: insufficient optical pixels (%d < %d) — derived only",
                  obs_date_str, location_id, n_valid, MIN_VALID_PIXELS)
        # Still return record with derived values if any exist
        has_derived = any(not np.isnan(v) for k, v in record.items() if k in DERIVED_BANDS)
        record["n_valid_pixels"]   = n_valid
        record["pct_valid_pixels"] = round(100.0 * n_valid / n_total, 1)
        return record if has_derived else None

    # Build a joint mask: pixels where ALL optical bands are > 0 and finite
    # This ensures all band arrays have identical shape for index computation
    joint_mask = valid_mask.copy()
    for i in range(len(OPTICAL_BANDS)):
        band = arr[:, :, i]
        joint_mask = joint_mask & (band > 0) & np.isfinite(band)

    min_px = int(joint_mask.sum())
    if min_px < MIN_VALID_PIXELS:
        log.debug("  [%s] loc=%s: joint band mask left only %d px", obs_date_str, location_id, min_px)
        return None

    band_arrays = {}
    for i, name in enumerate(OPTICAL_BANDS):
        band_arrays[name] = arr[:, :, i][joint_mask].astype(np.float64)

    # Median per optical band
    for name, v in band_arrays.items():
        record[name] = float(np.median(v))
    record["n_valid_pixels"]   = min_px
    record["pct_valid_pixels"] = round(100.0 * min_px / n_total, 1)

    # Spectral indices (pixel-by-pixel then median)
    idx = compute_indices(
        band_arrays["rrs_2"],  band_arrays["rrs_4"],  band_arrays["rrs_6"],
        band_arrays["rrs_8"],  band_arrays["rrs_9"],  band_arrays["rrs_10"],
        band_arrays["rrs_11"], band_arrays["rrs_12"], band_arrays["rrs_17"],
    )
    for name, arr_idx in idx.items():
        finite       = arr_idx[np.isfinite(arr_idx)]
        record[name] = float(np.median(finite)) if len(finite) > 0 else np.nan

    log.debug("  [%s] loc=%s: SUCCESS  rrs_8=%.5f  CHL_NN=%.3f  FLH_681=%.5f",
              obs_date_str, location_id,
              record.get("rrs_8", np.nan), record.get("CHL_NN", np.nan), record.get("FLH_681", np.nan))
    return record


# =============================================================================
# MAIN
# =============================================================================

def run():
    t0 = datetime.now()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df      = load_ifcb(IFCB_FILE)
    config  = get_config()
    limiter = RateLimiter(MIN_REQUEST_INTERVAL)

    obs_dates = sorted(df["obs_date"].unique())
    n_total   = len(obs_dates)
    log.info("Processing %d unique observation dates...", n_total)
    log.info(
        "Rate limit: %.1fs between requests → max %.0f req/min",
        MIN_REQUEST_INTERVAL, 60 / MIN_REQUEST_INTERVAL,
    )

    all_records  = []
    n_requests   = 0
    n_satisfied  = 0
    n_unsatisfied = 0

    for i, obs_date_str in enumerate(obs_dates):
        day_df    = df[df["obs_date"] == obs_date_str]
        locations = (
            day_df
            .drop_duplicates("location_id")
            [["location_id", "latitude", "longitude", "dashboardIdName"]]
            .to_dict(orient="records")
        )

        if not locations:
            continue

        log.info("[%s] %d location(s)", obs_date_str, len(locations))

        for loc in locations:
            location_id = loc["location_id"]
            lat     = float(loc["latitude"])
            lon     = float(loc["longitude"])

            # Count requests: up to len(TEMPORAL_DELTAS) per station
            # (actual count may be fewer if early delta succeeds)
            record = extract_matchup(lat, lon, location_id, obs_date_str, config, limiter)
            n_requests += 1   # approximate; exact count inside extract_matchup

            if record is not None:
                all_records.append(record)
                n_satisfied += 1
                log.info("  ✓ %s  n_px=%d  rrs_8=%.5f  CHL_NN=%.3f",
                         location_id, record["n_valid_pixels"],
                         record.get("rrs_8", np.nan), record.get("CHL_NN", np.nan))
            else:
                n_unsatisfied += 1
                log.info("  ✗ %s  no valid data in -%dd window", location_id, TEMPORAL_WINDOW_DAYS)

        # Progress + checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == n_total:
            elapsed  = (datetime.now() - t0).total_seconds()
            rate     = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s    = (n_total - i - 1) / rate if rate > 0 else 0
            match_rt = n_satisfied / (n_satisfied + n_unsatisfied) * 100 if (n_satisfied + n_unsatisfied) else 0
            log.info(
                "=== %d/%d dates | %d records | match=%.1f%% | %.2f dates/s | ETA ~%dm%02ds ===",
                i + 1, n_total, len(all_records), match_rt,
                rate, int(eta_s // 60), int(eta_s % 60),
            )
            if all_records:
                ckpt = OUTPUT_DIR / "sh_s3_features_checkpoint.parquet"
                pd.DataFrame(all_records).to_parquet(ckpt, index=False)
                log.info("Checkpoint → %s", ckpt)

    log.info("Extraction done: %d records | %d satisfied | %d unsatisfied",
             len(all_records), n_satisfied, n_unsatisfied)

    if not all_records:
        log.warning("No records extracted — check credentials, date range, and station coords")
        return

    feat_df = pd.DataFrame(all_records)

    # Merge back to IFCB by (obs_date, location_id)
    df_out = df.merge(feat_df, on=["obs_date", "location_id"], how="left")
    df_out["data_available"] = df_out["rrs_8"].notna()

    df_out.rename(columns={'rrs_2': 'rrs_412', 'rrs_4': 'rrs_490',
                'rrs_6': 'rrs_560','rrs_8': 'rrs_665',
                'rrs_9': 'rrs_673', 'rrs_10': 'rrs_682',
                'rrs_11': 'rrs_709', 'rrs_12': 'rrs_754',
                'rrs_17': 'rrs_865',}, inplace=True)

    n_with = int(df_out["data_available"].sum())
    log.info("Matchup: %d / %d rows have satellite data (%.1f%%)",
             n_with, len(df_out), 100 * n_with / len(df_out))

    summary = (
        df_out.groupby("dashboardIdName")["data_available"]
        .agg(total="count", matched="sum")
        .assign(pct=lambda x: (100 * x["matched"] / x["total"]).round(1))
        .sort_values("matched", ascending=False)
    )
    log.info("Per-station summary:\n%s", summary.to_string())

    out_path = OUTPUT_DIR / "sh_s3_features.parquet"
    df_out.to_parquet(out_path, index=False)
    log.info("Saved → %s  (%.1fs total)", out_path, (datetime.now() - t0).total_seconds())


if __name__ == "__main__":
    run()