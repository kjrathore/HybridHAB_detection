"""
ACOLITE Processor for Sentinel-3 OLCI Data
Gulf of Maine / Generic coastal HAB application

Features:
- Resume-aware: skips already-processed scenes
- Reads zips directly (no pre-extraction needed)
- Minimal settings matching proven working config
- Outputs L2W NetCDF with rrs_* bands + quality flags only
- Priority-region mode: filters inputs against a known file list
- Extracts zips to a dedicated temp dir to avoid clutter

Usage:
    python acolite_processor.py                        # uses GOM.yaml
    python acolite_processor.py --config GOM.yaml
    python acolite_processor.py --dry-run              # show what would be processed
    python acolite_processor.py --workers 4            # parallel (careful with RAM)
"""

import os
import sys
import argparse
import yaml
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple

# Headless mode for remote servers
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_priority_filelist(filelist_path: Path) -> set:
    """
    Load the set of scene stems from a text file (one filename/path per line).
    Lines are stripped of whitespace and empty lines / comments are ignored.
    Matching is done on stem only (no extension, no directory), so entries like:
        S3A_OL_1_EFR____20170101T150049_20170101T150349_....SEN3
        S3A_OL_1_EFR____20170101T150049_20170101T150349_....SEN3.zip
        /full/path/to/S3A_OL_1_EFR____.SEN3.zip
    all resolve to the same bare scene ID used by scene_id_from_zip().
    """
    if not filelist_path.exists():
        print(f"ERROR: Priority file list not found: {filelist_path}")
        sys.exit(1)

    stems = set()
    with open(filelist_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Take just the filename portion if a full path was given
            name = Path(line).name
            # Strip .zip then .SEN3 to arrive at the bare scene ID
            if name.endswith(".SEN3.zip"):
                name = name[: -len(".SEN3.zip")]
            elif name.endswith(".zip"):
                name = name[: -len(".zip")]
            if name.endswith(".SEN3"):
                name = name[: -len(".SEN3")]
            stems.add(name)

    print(f"  Priority file list: {len(stems)} scene(s) loaded from {filelist_path}")
    return stems


def find_zip_inputs(input_dir: Path) -> List[Path]:
    """Return sorted list of all .SEN3.zip files in input_dir."""
    zips = sorted(input_dir.glob("*.SEN3.zip"))
    if not zips:
        # also accept plain .zip with S3 prefix
        zips = sorted(input_dir.glob("S3*.zip"))
    return zips


def scene_id_from_zip(zip_path: Path) -> str:
    """Extract scene identifier (stem without .SEN3.zip)."""
    name = zip_path.name
    # Strip .zip then .SEN3 if present
    if name.endswith(".SEN3.zip"):
        return name[: -len(".SEN3.zip")]
    return zip_path.stem


def short_scene_id(scene_id: str) -> str:
    """
    Extract the short datetime prefix used by Jack's processor as output dir name.
    e.g. 'S3A_OL_1_EFR____20170101T150049_20170101T150349_...'
      -> 'S3A_OL_1_EFR____20170101T150049'
    Splits on '_' and stops after the first 15-char timestamp token (YYYYMMDDTHHmmss).
    """
    parts = scene_id.split("_")
    result = []
    for part in parts:
        result.append(part)
        if len(part) == 15 and part[0].isdigit() and "T" in part:
            break
    return "_".join(result)


def is_processed(scene_id: str, output_dir: Path) -> bool:
    """
    Check if this scene already has a valid L2W output.
    Handles two dir naming conventions:
      - Full scene ID : S3A_OL_1_EFR____20170101T150049_20170101T150349_...
      - Short prefix  : S3A_OL_1_EFR____20170101T150049  (Jack's processor)
    """
    # Full scene ID subdir (new runs)
    if (output_dir / scene_id).is_dir():
        if list((output_dir / scene_id).glob("*_L2W.nc")):
            return True

    # Short datetime-prefix subdir (existing processed files from Jack's processor)
    short = short_scene_id(scene_id)
    if (output_dir / short).is_dir():
        if list((output_dir / short).glob("*_L2W.nc")):
            return True

    # Flat output dir fallback
    if list(output_dir.glob(f"*{short}*_L2W.nc")):
        return True

    return False


def make_settings(
    zip_path: Path,
    output_dir: Path,
    temp_extract_dir: Path,
    scene_id: str,
    bbox: dict,
    utm_zone: str,
) -> Path:
    """
    Write a minimal ACOLITE settings file for one scene.
    Parameters mirror the proven working run.

    temp_extract_dir: ACOLITE will unzip the input here (instead of inside
    the output dir), keeping output_dir clean and allowing a shared scratch
    location that can be wiped independently.
    """
    scene_out = output_dir / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    # Ensure the shared temp extraction dir exists
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    limit = f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"

    lines = [
        f"## ACOLITE settings — generated {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"## Scene: {scene_id}",
        "",
        # --- I/O ---
        f"inputfile={zip_path.absolute()}",
        f"output={scene_out.absolute()}",
        # Direct ACOLITE to extract the zip here rather than next to the output
        f"input_extract_folder={temp_extract_dir.absolute()}",
        "",
        # # --- Region ---
        # f"limit={limit}",
        # "",
        # --- Atmospheric correction (fixed AOT — proven to work) ---
        "dsf_aot_estimate=fixed",
        "fixed_aot550=0.1",
        "",
        # --- Output products: Rrs only + required quality flags ---
        # rrs_* = remote sensing reflectance (above water, per-band)
        # Note: l2_flags is always written by ACOLITE; listing it is optional
        "l2w_parameters=rrs_*,l2_flags",
        "",
        # --- Masking (minimal — mirror the working run) ---
        "l2w_mask=True",
        "l2w_mask_negative_rhow=True",
        "l2w_mask_cirrus=True",
        "l2w_mask_high_toa=True",
        "l2w_mask_threshold=0.0215",       # ACOLITE default
        "",
        # --- Suppress all intermediate NetCDF writes (biggest time saver) ---
        "l1r_export_netcdf=False",
        "l2r_export_netcdf=False",
        "l1r_export_geotiff=False",
        "l2r_export_geotiff=False",
        "l2w_export_geotiff=False",
        "l1r_delete_netcdf=True",          # clean up if somehow written
        "l2r_delete_netcdf=True",
        "",
        # --- Extracted zip cleanup ---
        "extract_inputfile=True",          # ACOLITE must unzip to read
        "delete_extracted_input=True",     # delete .SEN3 dir after processing
        "",
        # --- Reproject L2W only, skip L1R/L2R reprojection ---
        f"output_projection={utm_zone}",
        "reproject_outputs=L2W",
        "reproject_before_ac=False",
        "",

        # --- Misc ---
        "merge_tiles=False",
        "merge_zones=False",
        "verbosity=5",
    ]

    settings_path = scene_out / "acolite_settings.txt"
    settings_path.write_text("\n".join(lines) + "\n")
    return settings_path


# ---------------------------------------------------------------------------
# ACOLITE runner
# ---------------------------------------------------------------------------

def find_acolite_script(acolite_dir: Path) -> Optional[Path]:
    candidates = [
        acolite_dir / "acolite_run.py",
        acolite_dir / "acolite" / "acolite_run.py",
        acolite_dir / "launch_acolite.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_one_scene(
    zip_path: Path,
    output_dir: Path,
    temp_extract_dir: Path,
    scene_id: str,
    bbox: dict,
    utm_zone: str,
    acolite_dir: Path,
) -> Tuple[str, bool, str]:
    """
    Process a single scene. Returns (scene_id, success, message).
    Designed to be safe to call from a subprocess worker.
    """
    settings_path = make_settings(
        zip_path, output_dir, temp_extract_dir, scene_id, bbox, utm_zone
    )

    acolite_script = find_acolite_script(acolite_dir)
    if acolite_script is None:
        return scene_id, False, f"ACOLITE script not found in {acolite_dir}"

    # --- Method 1: direct Python import (fastest, same process) ---
    try:
        sys.path.insert(0, str(acolite_dir))
        import acolite as ac

        run_fn = None
        for attr in ("acolite_run", "run"):
            if hasattr(ac, attr):
                run_fn = getattr(ac, attr)
                break
        if run_fn is None and hasattr(ac, "acolite"):
            run_fn = getattr(ac.acolite, "acolite_run", None)

        if run_fn:
            run_fn(settings=str(settings_path))
            return scene_id, True, "Method 1 (import)"
    except Exception as e:
        pass  # fall through to subprocess

    # --- Method 2: subprocess ---
    try:
        env = os.environ.copy()
        env.update({"MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen"})

        if "acolite_run.py" in str(acolite_script) or "launch_acolite.py" in str(acolite_script):
            rel = acolite_script.relative_to(acolite_dir)
            cmd = ["python", str(rel)]
            cwd = str(acolite_dir)
        else:
            cmd = ["python", str(acolite_script)]
            cwd = None

        if "launch_acolite.py" in str(acolite_script):
            cmd += ["--cli"]
        cmd += ["--settings", str(settings_path)]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=7200
        )
        if result.returncode == 0:
            return scene_id, True, "Method 2 (subprocess)"
        else:
            return scene_id, False, result.stderr[-500:] if result.stderr else "unknown error"
    except Exception as e:
        return scene_id, False, str(e)


# ---------------------------------------------------------------------------
# Main batch processor
# ---------------------------------------------------------------------------

def process_batch(config_path: str, dry_run: bool = False, workers: int = 1):
    cfg = load_config(config_path)

    region = cfg["region"]
    bbox = region["bbox"]
    utm_zone = region["utm_zone"]

    ac_cfg = cfg["acolite"]
    input_dir = Path(ac_cfg["input_dir"])
    output_dir = Path(ac_cfg["output_dir"])
    acolite_dir = Path(ac_cfg["installation_dir"])

    # -----------------------------------------------------------------------
    # Priority file list — only process scenes listed in this file
    # -----------------------------------------------------------------------
    PRIORITY_FILELIST = Path(
        "/home/server/pi/homes/rathorek/projects/HHAB_ROMS/datasets/"
        "GULF_OF_MAINE/raw_data/IFCB/gomofs_sentinel3_efr_all.txt"
    )
    priority_stems = load_priority_filelist(PRIORITY_FILELIST)

    # -----------------------------------------------------------------------
    # Dedicated temp extraction dir — keeps output_dir clean
    # -----------------------------------------------------------------------
    TEMP_EXTRACT_DIR = Path(
        "datasets/GULF_OF_MAINE/raw_data/sentinel_3/temp_acolite"
    ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ACOLITE BATCH PROCESSOR — {region['name']}")
    print(f"{'='*70}")
    print(f"  Input        : {input_dir}")
    print(f"  Output       : {output_dir}")
    print(f"  Temp extract : {TEMP_EXTRACT_DIR}")
    print(f"  ACOLITE      : {acolite_dir}")
    print(f"  BBox         : S={bbox['south']} W={bbox['west']} N={bbox['north']} E={bbox['east']}")
    print(f"  CRS          : {utm_zone}")
    print(f"{'='*70}\n")

    # --- Validate ACOLITE installation ---
    if not acolite_dir.exists():
        print(f"ERROR: ACOLITE directory not found: {acolite_dir}")
        sys.exit(1)

    if find_acolite_script(acolite_dir) is None and not dry_run:
        print(f"ERROR: No ACOLITE run script found in {acolite_dir}")
        sys.exit(1)

    # --- Discover inputs ---
    all_zips = find_zip_inputs(input_dir)
    if not all_zips:
        print(f"No .SEN3.zip files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(all_zips)} zip file(s) in input directory.")

    # --- Filter 1: keep only scenes in the priority list ---
    priority_zips = [
        zp for zp in all_zips
        if scene_id_from_zip(zp) in priority_stems
    ]
    excluded = len(all_zips) - len(priority_zips)
    print(f"  In priority list  : {len(priority_zips)}  (excluded {excluded} not in list)")

    if not priority_zips:
        print("\nNo input zips matched the priority file list. Nothing to do.")
        return

    # --- Filter 2: skip already processed ---
    to_process = []
    skipped = []

    for zp in priority_zips:
        sid = scene_id_from_zip(zp)
        if is_processed(sid, output_dir):
            skipped.append(sid)
        else:
            to_process.append((zp, sid))

    print(f"  Already processed : {len(skipped)}")
    print(f"  To process        : {len(to_process)}")

    if not to_process:
        print("\nAll priority scenes already processed. Nothing to do.")
        return

    if dry_run:
        print("\n--- DRY RUN: scenes that would be processed ---")
        for zp, sid in to_process:
            print(f"  {sid}")
        return

    # --- Process ---
    print(f"\nStarting processing ({'parallel x' + str(workers) if workers > 1 else 'sequential'})...\n")

    success_list, fail_list = [], []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {
                exe.submit(
                    run_one_scene,
                    zp, output_dir, TEMP_EXTRACT_DIR, sid, bbox, utm_zone, acolite_dir
                ): sid
                for zp, sid in to_process
            }
            for fut in as_completed(futures):
                sid, ok, msg = fut.result()
                if ok:
                    success_list.append(sid)
                    print(f"  ✓ {sid}  ({msg})")
                else:
                    fail_list.append(sid)
                    print(f"  ✗ {sid}")
                    print(f"    {msg}")
    else:
        for i, (zp, sid) in enumerate(to_process, 1):
            print(f"[{i}/{len(to_process)}] {sid}")
            _, ok, msg = run_one_scene(
                zp, output_dir, TEMP_EXTRACT_DIR, sid, bbox, utm_zone, acolite_dir
            )
            if ok:
                success_list.append(sid)
                print(f"  ✓ done ({msg})\n")
            else:
                fail_list.append(sid)
                print(f"  ✗ failed: {msg}\n")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Succeeded : {len(success_list)}")
    print(f"  Failed    : {len(fail_list)}")
    if fail_list:
        print("\n  Failed scenes:")
        for s in fail_list:
            print(f"    - {s}")

    # Count output files
    nc_files = list(output_dir.rglob("*_L2W.nc"))
    print(f"\n  Total L2W.nc files in output dir: {len(nc_files)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ACOLITE batch processor for Sentinel-3 OLCI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="datasets/GOM.yaml", help="YAML config file (default: GOM.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed, don't run")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel workers (default: 1 sequential). Each needs ~4GB RAM.",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    process_batch(args.config, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()