import subprocess
import os
import zipfile
import shutil
import csv
import logging
from pathlib import Path
from datetime import datetime

import netCDF4 as nc
import numpy as np
from shapely.geometry import Polygon
from shapely.vectorized import contains

# ── Config ────────────────────────────────────────────────────────────────────
SCENE_LIST   = "datasets/GULF_OF_MAINE/raw_data/IFCB/gomofs_sentinel3_efr_all.txt"
SENTINEL_DIR = Path("datasets/GULF_OF_MAINE/raw_data/sentinel_3")
TEMP_DIR     = Path("datasets/GULF_OF_MAINE/raw_data/sentinel_3/temp_extracted22")
OUTPUT_DIR   = Path("datasets/GULF_OF_MAINE/processed_data/sentinel_3_c2rcc")
LOG_DIR      = Path("datasets/GULF_OF_MAINE/logs")
GRAPH_XML    = "datasets/GULF_OF_MAINE/processed_data/sentinel_3_c2rcc/c2rcc.xml"
GPT_CMD      = "gpt"

GPT_CACHE    = "6G"
GPT_TILES    = "6"
GPT_TILESIZE = "1024"

CLOUD_THRESHOLD = 0.50   # skip if >50% cloud cover in AOI

# Subset selection
START_IDX    = 5327
MAX_SCENES   = 6791   #1141 #2446 #3957 #5326 #6791

# Gulf of Maine polygon
GOM_POLYGON = Polygon([
    (-70.0, 38.0), (-61.0, 41.0),
    (-64.0, 46.0), (-73.0, 43.0),
    (-70.0, 38.0)
])
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOG_DIR / f"c2rcc_run_{timestamp}.log"
    csv_file  = LOG_DIR / f"c2rcc_run_{timestamp}.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return log_file, csv_file


def init_csv(csv_file):
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scene", "satellite", "sensing_time",
            "cloud_frac_pct", "n_pixels_aoi", "n_valid", "n_cloud",
            "status", "output_file", "note"
        ])


def append_csv(csv_file, row: dict):
    with open(csv_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scene", "satellite", "sensing_time",
            "cloud_frac_pct", "n_pixels_aoi", "n_valid", "n_cloud",
            "status", "output_file", "note"
        ])
        w.writerow(row)


def parse_scene_meta(scene_name):
    parts = scene_name.split("_")
    satellite    = parts[0]          # S3A / S3B
    sensing_time = parts[7]          # 20240101T135348
    return satellite, sensing_time


def read_scene_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def find_scene(sentinel_dir, scene_name):
    sen3     = sentinel_dir / scene_name
    zip_path = sentinel_dir / (scene_name.replace(".SEN3", ".SEN3.zip"))
    if sen3.exists():
        return sen3, "dir"
    elif zip_path.exists():
        return zip_path, "zip"
    return None, None


def extract_zip(zip_path, temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)
    sen3_dirs = list(temp_dir.glob("*.SEN3"))
    if not sen3_dirs:
        raise FileNotFoundError(f"No .SEN3 found after extracting {zip_path.name}")
    return sen3_dirs[0]


def check_efr_cloud_fraction(sen3_dir: Path, polygon: Polygon,
                              threshold: float = 0.5) -> dict:
    qf_path     = sen3_dir / "qualityFlags.nc"
    coords_path = sen3_dir / "geo_coordinates.nc"

    if not qf_path.exists() or not coords_path.exists():
        raise FileNotFoundError(f"Missing qualityFlags.nc or geo_coordinates.nc in {sen3_dir}")

    with nc.Dataset(coords_path) as geo:
        lat = geo["latitude"][:]   # (rows, cols), scaled int → float via scale_factor
        lon = geo["longitude"][:]

    # Handle masked arrays
    lat = np.ma.filled(lat, np.nan).astype(np.float64)
    lon = np.ma.filled(lon, np.nan).astype(np.float64)

    with nc.Dataset(qf_path) as qf:
        flags = qf["quality_flags"][:]  # uint32 bitmask

    flags = np.ma.filled(flags, 0).astype(np.uint32)

    # Mask to AOI
    in_aoi = contains(polygon, lon, lat)  # bool (rows, cols)

    if in_aoi.sum() == 0:
        return {
            "cloud_frac_aoi": None,
            "n_pixels_aoi":   0,
            "n_valid":        0,
            "n_cloud":        0,
            "usable":         False,
        }

    flags_aoi = flags[in_aoi]

    CLOUD           = np.uint32(1 << 10)
    CLOUD_AMBIGUOUS = np.uint32(1 << 11)
    INVALID         = np.uint32(1 << 19)

    cloud_mask = (flags_aoi & (CLOUD | CLOUD_AMBIGUOUS)) > 0
    valid_mask = (flags_aoi & INVALID) == 0

    n_valid = int(valid_mask.sum())
    n_cloud = int((cloud_mask & valid_mask).sum())
    cloud_frac = float(n_cloud / n_valid) if n_valid > 0 else 1.0

    return {
        "cloud_frac_aoi": round(cloud_frac * 100, 1),
        "n_pixels_aoi":   int(in_aoi.sum()),
        "n_valid":        n_valid,
        "n_cloud":        n_cloud,
        "usable":         cloud_frac < threshold,
    }


def get_output_path(sensing_time, satellite, output_dir):
    return output_dir / f"{satellite}_{sensing_time}.nc"


def run_gpt(graph_xml, source_product, output_path):
    cmd = [
        GPT_CMD, graph_xml,
        f"{source_product}",
        f"-Poutput={output_path}",
        "-c", GPT_CACHE,
        "-q", GPT_TILES,
        f"-J-Dsnap.tileSize={GPT_TILESIZE}",
    ]
    logging.info(f"GPT cmd: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def main():
    log_file, csv_file = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    init_csv(csv_file)

    logging.info(f"Log file : {log_file}")
    logging.info(f"CSV file : {csv_file}")

    scenes = read_scene_list(SCENE_LIST)
    scenes = scenes[START_IDX:]
    if MAX_SCENES is not None:
        scenes = scenes[:MAX_SCENES]

    logging.info(f"Total scenes to evaluate: {len(scenes)}")

    counts = {"success": 0, "skipped_cloud": 0, "skipped_exists": 0,
              "failed_gpt": 0, "failed_other": 0}

    for i, scene_name in enumerate(scenes):
        satellite, sensing_time = parse_scene_meta(scene_name)
        output_path = get_output_path(sensing_time, satellite, OUTPUT_DIR)

        logging.info(f"[{i+1}/{len(scenes)}] {scene_name}")

        csv_row = {
            "scene":          scene_name,
            "satellite":      satellite,
            "sensing_time":   sensing_time,
            "cloud_frac_pct": None,
            "n_pixels_aoi":   None,
            "n_valid":        None,
            "n_cloud":        None,
            "status":         None,
            "output_file":    None,
            "note":           "",
        }

        # ── Already processed ─────────────────────────────────────────────
        if output_path.exists():
            logging.info(f"  ✓ Already exists, skipping")
            csv_row.update(status="skipped_exists", output_file=output_path.name)
            append_csv(csv_file, csv_row)
            counts["skipped_exists"] += 1
            continue

        # ── Find scene ────────────────────────────────────────────────────
        scene_path, scene_type = find_scene(SENTINEL_DIR, scene_name)
        if scene_path is None:
            logging.warning(f"  ✗ Not found in {SENTINEL_DIR}")
            csv_row.update(status="failed", note="scene not found on disk")
            append_csv(csv_file, csv_row)
            counts["failed_other"] += 1
            continue

        # ── Extract zip if needed ─────────────────────────────────────────
        extracted_path = None
        if scene_type == "zip":
            logging.info(f"  Extracting zip...")
            try:
                extracted_path = extract_zip(scene_path, TEMP_DIR)
                sen3_dir = extracted_path
            except Exception as e:
                logging.error(f"  ✗ Extraction failed: {e}")
                csv_row.update(status="failed", note=f"zip extraction error: {e}")
                append_csv(csv_file, csv_row)
                counts["failed_other"] += 1
                continue
        else:
            sen3_dir = scene_path

        # ── Cloud check ───────────────────────────────────────────────────
        try:
            cloud = check_efr_cloud_fraction(sen3_dir, GOM_POLYGON, CLOUD_THRESHOLD)
            csv_row.update(
                cloud_frac_pct = cloud["cloud_frac_aoi"],
                n_pixels_aoi   = cloud["n_pixels_aoi"],
                n_valid        = cloud["n_valid"],
                n_cloud        = cloud["n_cloud"],
            )
            logging.info(
                f"  Cloud: {cloud['cloud_frac_aoi']}%  "
                f"(valid={cloud['n_valid']}, cloud={cloud['n_cloud']}, "
                f"aoi_pixels={cloud['n_pixels_aoi']})"
            )
        except Exception as e:
            logging.error(f"  ✗ Cloud check failed: {e}")
            csv_row.update(status="failed", note=f"cloud check error: {e}")
            append_csv(csv_file, csv_row)
            if extracted_path and extracted_path.exists():
                shutil.rmtree(extracted_path)
            counts["failed_other"] += 1
            continue

        # ── Skip if too cloudy ────────────────────────────────────────────
        if not cloud["usable"]:
            logging.info(f"  ✗ Skipping — cloud cover {cloud['cloud_frac_aoi']}% > {CLOUD_THRESHOLD*100}%")
            csv_row.update(status="skipped_cloud",
                           note=f"cloud {cloud['cloud_frac_aoi']}% > threshold")
            append_csv(csv_file, csv_row)
            if extracted_path and extracted_path.exists():
                shutil.rmtree(extracted_path)
            counts["skipped_cloud"] += 1
            continue

        # ── Run GPT ───────────────────────────────────────────────────────
        ok = run_gpt(GRAPH_XML, sen3_dir, output_path)

        # ── Cleanup temp ──────────────────────────────────────────────────
        if extracted_path and extracted_path.exists():
            shutil.rmtree(extracted_path)
            logging.info(f"  Cleaned temp: {extracted_path.name}")

        if ok:
            logging.info(f"  ✓ Done: {output_path.name}")
            csv_row.update(status="success", output_file=output_path.name)
            counts["success"] += 1
        else:
            logging.error(f"  ✗ GPT failed")
            csv_row.update(status="failed_gpt", note="gpt non-zero exit")
            counts["failed_gpt"] += 1

        append_csv(csv_file, csv_row)

    # ── Summary ───────────────────────────────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info(f"Success       : {counts['success']}")
    logging.info(f"Skipped cloud : {counts['skipped_cloud']}")
    logging.info(f"Skipped exists: {counts['skipped_exists']}")
    logging.info(f"Failed GPT    : {counts['failed_gpt']}")
    logging.info(f"Failed other  : {counts['failed_other']}")
    logging.info(f"CSV log       : {csv_file}")


if __name__ == "__main__":
    main()