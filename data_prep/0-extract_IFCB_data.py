#!/usr/bin/env python3
"""
HABHub IFCB Dataset Extractor
Author: Kunal Rathore

Fetches year-by-year to avoid 502 gateway errors on large date ranges.
Each year saved as intermediate parquet, then merged and aggregated at the end.

Stage 1 — IFCB Dashboard /api/export_metadata/{dashboard_id_name}
Stage 2 — HABHub API     /api/v1/ifcb-datasets/{id}/
Output  — one row per (location_id, date), species as columns (daily mean)
"""

import os
import time
import requests
import pandas as pd
from io import StringIO

# ============================================================================
# CONFIG
# ============================================================================

HABHUB_API  = "https://habhub-api.whoi.edu/api/v1"
IFCB_DASH   = "https://habon-ifcb.whoi.edu"  #for meta data

START_YEAR  = 2017
END_YEAR    = 2025

REQUEST_DELAY = 1
SCRATCH_DIR   = "datasets/GULF_OF_MAINE/raw_data/IFCB/"
OUTPUT_FILE   = "datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_IFCB_GOM.parquet"

BBOX = {"west": -76.0, "east": -60.0, "south": 36.0, "north": 46.0}

# ============================================================================
# HELPERS
# ============================================================================

def api_get(url: str, params: dict, retries: int = 3, timeout: int = 120) -> requests.Response | None:
    """GET with retry on 5xx errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"    error: {e} (attempt {attempt+1}/{retries})")
        time.sleep(2 ** attempt)
    return None


def fetch_bin_metadata(dashboard_id_name: str) -> pd.DataFrame:
    """Fetch full bin metadata CSV (all years) from IFCB Dashboard."""
    url = f"{IFCB_DASH}/api/export_metadata/{dashboard_id_name}"
    r = api_get(url, params={"start_date": f"{START_YEAR}-01-01"})
    if r is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(r.content.decode("utf-8")), low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        if "skip" in df.columns:
            df = df[df["skip"] != 1]
        keep = ["pid", "sample_time", "latitude", "longitude",
                "depth", "ml_analyzed", "cruise", "sample_type"]
        df = df[[c for c in keep if c in df.columns]].copy()
        for col in ["depth", "ml_analyzed", "latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"    metadata: {len(df):,} bins  |  "
              f"depth: {df['depth'].notna().sum():,} / {len(df):,}")
        return df
    except Exception as e:
        print(f"    ! CSV parse error: {e}")
        return pd.DataFrame()


def get_datasets() -> list[dict]:
    url = f"{HABHUB_API}/ifcb-datasets/"
    r = api_get(url, params={
        "start_date": f"{START_YEAR}-01-01", "end_date": f"{END_YEAR}-12-31",
        "seasonal": "false", "exclude_month_range": "false", "smoothing_factor": "4",
    })
    r.raise_for_status()
    features = r.json()["features"]

    df = pd.json_normalize(features)
    print(f"\nAll available datasets ({len(features)} total):")
    print(df[['id', 'properties.name', 'properties.dashboardIdName']].to_string(index=False))

    return features  # return all, no whitelist filter


def fetch_year_timeseries(dataset_id, year: int) -> dict | None:
    url = f"{HABHUB_API}/ifcb-datasets/{dataset_id}/"
    params = {
        "start_date":          f"{year}-01-01",
        "end_date":            f"{year}-12-31",
        "seasonal":            "false",
        "exclude_month_range": "false",
        "smoothing_factor":    "1",
    }
    r = api_get(url, params=params, timeout=120)
    return r.json() if r else None


def parse_concentrations_wide(ts_response: dict) -> pd.DataFrame:
    timeseries = ts_response.get("properties", {}).get("timeseriesData", [])
    species_data: dict[str, dict] = {}
    bin_meta: dict[str, dict] = {}

    for species_entry in timeseries:
        species = species_entry.get("species", "")
        conc_map = {}
        for sample in species_entry.get("data", []):
            bin_pid = sample.get("binPid", "")
            for m in sample.get("metrics", []):
                if m.get("metricId") == "cell_concentration":
                    conc_map[bin_pid] = m.get("value")
            if bin_pid not in bin_meta:
                bin_meta[bin_pid] = {
                    "sample_time": sample.get("sampleTime"),
                    "latitude":    sample.get("latitude"),
                    "longitude":   sample.get("longitude"),
                }
        if conc_map:
            species_data[species] = conc_map

    if not bin_meta:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(bin_meta, orient="index")
    df.index.name = "bin_pid"
    df = df.reset_index()
    for species, conc_map in species_data.items():
        df[species] = pd.to_numeric(df["bin_pid"].map(conc_map), errors="coerce")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs("habhub_data", exist_ok=True)

    datasets = get_datasets()

    # Pre-fetch all metadata CSVs once (covers all years)
    print("\nFetching bin metadata (all years) ...")
    meta_cache: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        dash_id = ds.get("properties", {}).get("dashboardIdName", "")
        if dash_id and dash_id not in meta_cache:
            print(f"  {dash_id}")
            meta_cache[dash_id] = fetch_bin_metadata(dash_id)

    # Year-by-year extraction
    for year in range(START_YEAR, END_YEAR + 1):
        out_path = f"{SCRATCH_DIR}/{year}.parquet"
        if os.path.exists(out_path):
            print(f"\n[{year}] already exists, skipping")
            continue

        print(f"\n{'='*50}\n  Year {year}\n{'='*50}")
        year_frames = []

        for i, ds in enumerate(datasets, 1):
            ds_id    = ds["id"]
            ds_props = ds.get("properties", {})
            ds_name  = ds_props.get("name", str(ds_id))
            dash_id  = ds_props.get("dashboardIdName", "")

            print(f"  [{i}/{len(datasets)}] {ds_name}")

            ts = fetch_year_timeseries(ds_id, year)
            if ts is None:
                print("    ! no response, skipping")
                time.sleep(REQUEST_DELAY)
                continue

            conc_df = parse_concentrations_wide(ts)
            if conc_df.empty:
                print("    - no data")
                time.sleep(REQUEST_DELAY)
                continue

            # Per-dataset cell concentration count
            conc_cols_now = [c for c in conc_df.columns
                             if c not in {"bin_pid", "sample_time", "latitude", "longitude"}]
            total_conc = conc_df[conc_cols_now].notna().sum().sum()
            print(f"    {len(conc_df):,} bins  |  cell_concentration values: {total_conc:,}")

            meta_df = meta_cache.get(dash_id, pd.DataFrame())
            if not meta_df.empty:
                conc_df = conc_df.drop(columns=["sample_time", "latitude", "longitude"],
                                       errors="ignore")
                merged = conc_df.merge(
                    meta_df.rename(columns={"pid": "bin_pid"}),
                    on="bin_pid", how="left"
                )
            else:
                merged = conc_df.copy()

            merged["dataset_id"]      = str(ds_id)
            merged["dataset_name"]    = ds_name
            merged["dashboardIdName"] = dash_id
            year_frames.append(merged)
            time.sleep(REQUEST_DELAY)

        if not year_frames:
            print(f"  No data for {year}")
            continue

        yr_df = pd.concat(year_frames, ignore_index=True)
        yr_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"  Saved {len(yr_df):,} bins -> {out_path}")

    # -------------------------------------------------------------------------
    # Merge all years
    # -------------------------------------------------------------------------
    print("\nMerging yearly files ...")
    parts = sorted([f for f in os.listdir(SCRATCH_DIR) if f.endswith(".parquet")])
    if not parts:
        print("No yearly files found.")
        return

    df = pd.concat(
        [pd.read_parquet(f"{SCRATCH_DIR}/{f}") for f in parts],
        ignore_index=True
    )
    print(f"  {len(df):,} total bins across {len(parts)} years")

    # Type cleanup
    df["sample_time"] = pd.to_datetime(df["sample_time"], utc=True, errors="coerce")
    df["latitude"]    = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"]   = pd.to_numeric(df["longitude"], errors="coerce")
    df["depth"]       = pd.to_numeric(df["depth"],     errors="coerce")
    df["dataset_id"]  = df["dataset_id"].astype(str)
    df["date"] = df["sample_time"].copy()
    # Bbox filter
    n_before = len(df)
    df = df[
        df["latitude"].between(BBOX["south"], BBOX["north"]) &
        df["longitude"].between(BBOX["west"], BBOX["east"])
    ]
    print(f"  Bbox: {n_before:,} -> {len(df):,} ({n_before - len(df):,} dropped)")

    # Location ID
    df["location_id"] = (
        df["latitude"].round(4).astype(str) + "_" + df["longitude"].round(4).astype(str)
    ).apply(lambda s: format(abs(hash(s)) % 10**8, "08d"))

    # Identify species columns
    non_conc = {"bin_pid", "location_id", "dataset_id", "dataset_name", "dashboardIdName",
                "sample_time",  "date", "latitude", "longitude", "depth",
                "ml_analyzed", "cruise", "sample_type"}
    conc_cols = sorted([c for c in df.columns if c not in non_conc])
    

    # Daily mean aggregation
    n_bins = len(df)
    # df["date"] = df["sample_time"].dt.date
    # lets skip aggrgation as of now.
    # agg_dict = {
    #     "latitude":        "first",
    #     "longitude":       "first",
    #     "dataset_id":      "first",
    #     "dataset_name":    "first",
    #     "dashboardIdName": "first",
    #     **{c: "mean" for c in conc_cols},
    # }
    # for col in ["depth", "cruise", "sample_type"]:
    #     if col in df.columns:
    #         agg_dict[col] = "first"

    # df = df.groupby(["location_id", "date"], as_index=False).agg(agg_dict)
    # df["date"] = pd.to_datetime(df["sample_time"])

    # Final column order
    id_cols   = ["location_id", "date", "dataset_id", "dataset_name", "dashboardIdName"]
    meta_cols = [c for c in ["latitude", "longitude", "depth", "cruise", "sample_type"]
                 if c in df.columns]
    df = df[id_cols + meta_cols + conc_cols]

    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy", index=False)
    df.to_csv(OUTPUT_FILE.replace(".parquet",".csv"),index=False)
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Bins (raw)   : {n_bins:,}")
    print(f"  Daily obs    : {len(df):,}  (after mean aggregation)")
    print(f"  Locations    : {df['location_id'].nunique():,} unique lat/lon points")
    print(f"  Datasets     : {df['dataset_id'].nunique()}")
    print(f"  Species cols : {len(conc_cols)}  ->  {conc_cols}")
    print(f"  Date range   : {df['date'].min()}  ->  {df['date'].max()}")
    print(f"  Depth        : {df['depth'].notna().sum():,} / {len(df):,} daily obs")
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    print(f"  File size    : {size_mb:.2f} MB  ->  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


'''

Merging yearly files ...
  377,594 total bins across 10 years
  Bbox: 377,594 -> 349,501 (28,093 dropped)

============================================================
DONE
============================================================
  Bins (raw)   : 349,501
  Daily obs    : 349,501  (after mean aggregation)
  Locations    : 14,435 unique lat/lon points
  Datasets     : 17
  Species cols : 10  ->  ['Alexandrium_catenella', 'Dinophysis_acuminata', 'Dinophysis_norvegica', 'Karenia', 'Margalefidinium', 'Mesodinium', 'Pseudo-nitzschia', 'Tripos_furca', 'Tripos_fusus', 'Tripos_muelleri']
  Date range   : 2017-01-01 00:00:00  ->  2025-12-20 00:00:00
  Depth        : 176,305 / 349,501 daily obs
  File size    : 3.42 MB  ->  datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_ifcb_daily.parquet

'''