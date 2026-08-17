"""
GoMOFS 2DS Nowcast URL Generator
Filters IFCB dates to Gulf of Maine polygon, generates wget-ready URL list.

Output: gomofs_urls.txt  (one URL per line)
Run downloads with:
    wget -i gomofs_urls.txt -P <output_dir> -c --retry-connrefused -q --show-progress
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon

# =============================================================================
# CONFIG
# =============================================================================

IFCB_PARQUET = Path(
    "datasets/GULF_OF_MAINE/raw_data/IFCB/habhub_IFCB_GOM.parquet"
)
OUTPUT_DIR = Path(
    "/home/server/pi/homes/rathorek/projects/HHAB_ROMS/datasets/"
    "GULF_OF_MAINE/raw_data/gomofs"
)
URL_FILE = Path("datasets/GULF_OF_MAINE/raw_data/IFCB/gomofs_urls_fallback.txt")

BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/"
    "access/gulf-of-maine-operational-forecast-system-gomofs/"
)

# Gulf of Maine polygon (counter-clockwise: SW -> SE -> NE -> NW -> SW)
GOM_POLYGON = Polygon([(-70.0, 38.0), (-62.0, 42.0), (-64.0, 46.0), (-73.0, 43.0), (-70.0, 38.0)])

NOWCAST_HOUR = "n001"
TIME_CYCLE    = "t12z"

# =============================================================================
# LOAD AND FILTER IFCB TO GULF OF MAINE POLYGON
# =============================================================================

df = pd.read_parquet(IFCB_PARQUET)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.year > 2023]

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

dfgom = gdf[gdf.within(GOM_POLYGON)].copy()
dfgom.sort_values("date", inplace=True)

print(f"Total IFCB rows        : {len(df)}")
print(f"Inside GOM polygon     : {len(dfgom)}")
print(f"Unique dates (GOM)     : {dfgom['date'].nunique()}")

target_dates = sorted(dfgom["date"].dt.normalize().unique())

print(f"Date range             : {target_dates[0].date()} to {target_dates[-1].date()}")

# =============================================================================
# BUILD URLs  (primary + fallback naming)
# =============================================================================
# Primary  : nos.gomofs.2ds.{hour}.{YYYYMMDD}.{cycle}.nc
# Fallback : gomofs.2ds.{hour}.{YYYYMMDD}.{cycle}.nc

urls = []
for date in target_dates:
    year  = date.strftime("%Y")
    month = date.strftime("%m")
    dstr  = date.strftime("%Y%m%d")
    prefix = f"{BASE_URL}{year}/{month}/"

    # primary = f"nos.gomofs.2ds.{NOWCAST_HOUR}.{dstr}.{TIME_CYCLE}.nc"
    # urls.append(prefix + primary)
    # gomofs.t12z.20240919.2ds.n001.nc
    fallback = f"gomofs.{TIME_CYCLE}.{dstr}.2ds.{NOWCAST_HOUR}.nc"   
    urls.append(prefix + fallback)

# Write URL file
URL_FILE.write_text("\n".join(urls) + "\n")
print(f"\nURLs written           : {len(urls)}")
print(f"URL file               : {URL_FILE.resolve()}")

# =============================================================================
# PRINT WGET COMMAND
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("Run downloads with:")
print("="*70)
print(f"""
# Download all GoMOFS 2DS nowcast files:
wget -i {URL_FILE} \\
     -P {OUTPUT_DIR} \\
     -c --retry-connrefused \\
     --tries=5 --wait=2 \\
     -q --show-progress

# If some primary filenames (nos.gomofs.*) return 404, try fallback names:
# Replace 'nos.gomofs' with 'gomofs' in the URL file and re-run wget
# (wget skips already-downloaded files with -c flag)
sed 's|nos\\.gomofs\\.|gomofs\\.|g' {URL_FILE} > gomofs_urls_fallback.txt
wget -i gomofs_urls_fallback.txt \\
     -P {OUTPUT_DIR} \\
     -c --retry-connrefused \\
     --tries=5 --wait=2 \\
     -q --show-progress \\
     --no-clobber
""")


'''
primary file naming 2020 to   2024-08
fallback file       2024-08  to 2025-12

'''