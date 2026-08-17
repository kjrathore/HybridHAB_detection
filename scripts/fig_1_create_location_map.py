"""
Create publication-quality figure showing Sentinel-3 image and IFCB observation locations

Two-panel figure:
1. Sentinel-3 OLCI composite image over study region
2. IFCB observation site locations (fixed stations vs. ship-based)

Usage:
    python create_ifcb_location_map.py --ifcb-data path/to/ifcb_observations.csv
    
    # Or specify Sentinel-3 image explicitly:
    python create_ifcb_location_map.py --ifcb-data ifcb.csv --s3-image path/to/image.tif
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import argparse
import yaml
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

# Try to import rasterio for geospatial imagery
try:
    import rasterio
    from rasterio.plot import show as rio_show
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("⚠ rasterio not available - Sentinel-3 image will not be displayed")


def load_config(config_path="s3_olci_config.yaml"):
    """Load region configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_best_s3_image(acolite_output_dir):
    """
    Find the best available Sentinel-3 image for display
    Prioritizes: RGB composites > single-date mosaics > individual scenes
    """
    acolite_dir = Path(acolite_output_dir)
    
    # Search locations in order of preference
    search_paths = [
        acolite_dir / "mosaics" / "all_scenes_*",
        acolite_dir / "mosaics" / "monthly_*",
        acolite_dir / "mosaics" / "weekly_*",
        acolite_dir / "mosaics" / "daily",
        acolite_dir,
    ]
    
    for search_path in search_paths:
        if '*' in str(search_path):
            # Glob pattern
            candidates = list(Path(str(search_path).split('*')[0]).parent.glob(
                Path(str(search_path).split('*')[0]).name + '*'
            ))
            for candidate in candidates:
                if candidate.is_dir():
                    tif_files = list(candidate.glob('*.tif'))
                    if tif_files:
                        return tif_files[0]
        else:
            # Direct path
            tif_files = list(Path(search_path).glob('**/*.tif'))
            if tif_files:
                # Prefer files with RGB or composite in name
                rgb_files = [f for f in tif_files if 'rgb' in f.name.lower() or 'composite' in f.name.lower()]
                if rgb_files:
                    return rgb_files[0]
                return tif_files[0]
    
    return None


def plot_s3_image(ax, image_path, bbox, title="Sentinel-3 OLCI"):
    """Plot Sentinel-3 image on cartopy axis"""
    
    if not RASTERIO_AVAILABLE or image_path is None:
        # Just show basemap
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5, zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3, zorder=0)
        ax.text(0.5, 0.5, 'Sentinel-3 Image\n(not available)', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=12, alpha=0.5)
    else:
        print(f"Loading Sentinel-3 image: {image_path.name}")
        
        with rasterio.open(image_path) as src:
            # Read data
            if src.count >= 3:
                # Multi-band: create RGB composite
                # Try bands that might be RGB or true color
                rgb = np.stack([src.read(i) for i in [1, 2, 3]])
                rgb = np.moveaxis(rgb, 0, -1)
                
                # Mask invalid values
                rgb = np.ma.masked_invalid(rgb)
                
                # Normalize for display (clip to 98th percentile to handle outliers)
                for i in range(3):
                    band = rgb[:, :, i]
                    valid_data = band[~band.mask] if np.ma.is_masked(band) else band
                    if len(valid_data) > 0:
                        vmin, vmax = np.percentile(valid_data, [2, 98])
                        rgb[:, :, i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
                
            else:
                # Single band: use colormap
                data = src.read(1)
                data = np.ma.masked_invalid(data)
                
                # Normalize
                valid_data = data[~data.mask] if np.ma.is_masked(data) else data
                if len(valid_data) > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                    rgb = plt.cm.viridis((data - vmin) / (vmax - vmin))[:, :, :3]
            
            # Get image extent
            extent = [src.bounds.left, src.bounds.right, 
                     src.bounds.bottom, src.bounds.top]
            
            # Plot image
            ax.imshow(rgb, extent=extent, transform=ccrs.PlateCarree(), 
                     origin='upper', zorder=1, alpha=0.8)
    
    # Add coastline and features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle=':', zorder=3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, 
                     linestyle='--', color='gray', zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def plot_ifcb_locations(ax, ifcb_data, bbox, fixed_threshold=100):
    """Plot IFCB observation locations on cartopy axis"""
    
    # Count observations per location
    location_counts = ifcb_data.groupby('location_id').size()
    location_info = ifcb_data.groupby('location_id').agg({
        'latitude': 'first',
        'longitude': 'first',
        'location_name': 'first',
        'location_type': 'first'
    })
    location_info['n_obs'] = location_counts
    location_info['is_fixed'] = location_counts > fixed_threshold
    
    # Add basemap features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.6, zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle=':', zorder=3)
    
    # Plot ship-based observations first (so fixed stations appear on top)
    ship_sites = location_info[~location_info['is_fixed']]
    if len(ship_sites) > 0:
        ax.scatter(ship_sites['longitude'], ship_sites['latitude'],
                  s=50, c='steelblue', marker='o', 
                  edgecolors='navy', linewidth=1.0,
                  transform=ccrs.PlateCarree(), 
                  label=f'Ship-based (n={len(ship_sites):,})',
                  zorder=4, alpha=0.7)
    
    # Plot fixed stations
    fixed_sites = location_info[location_info['is_fixed']]
    if len(fixed_sites) > 0:
        ax.scatter(fixed_sites['longitude'], fixed_sites['latitude'],
                  s=120, c='crimson', marker='^', 
                  edgecolors='darkred', linewidth=1.5,
                  transform=ccrs.PlateCarree(), 
                  label=f'Fixed Stations (n={len(fixed_sites):,})',
                  zorder=5, alpha=0.9)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5,
                     linestyle='--', color='gray', zorder=6)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
             edgecolor='black', fancybox=False)
    
    ax.set_title('IFCB Observation Locations', fontsize=12, fontweight='bold', pad=10)
    
    # Print summary
    print(f"\nIFCB Location Summary:")
    print(f"  Total locations: {len(location_info):,}")
    print(f"  Fixed stations (>{fixed_threshold} obs): {len(fixed_sites):,}")
    print(f"  Ship-based (≤{fixed_threshold} obs): {len(ship_sites):,}")
    print(f"  Total observations: {location_info['n_obs'].sum():,}")


def create_figure(ifcb_data_path, config_path="s3_olci_config.yaml", 
                 s3_image_path=None, output_dir="figures", fixed_threshold=100):
    """
    Create two-panel figure with Sentinel-3 image and IFCB locations
    
    Parameters:
    -----------
    ifcb_data_path : str or Path
        Path to CSV file with IFCB observations
    config_path : str
        Path to YAML configuration file
    s3_image_path : str or Path, optional
        Explicit path to Sentinel-3 image. If None, will search for best available.
    output_dir : str
        Output directory for figure
    fixed_threshold : int
        Minimum observations to classify as fixed station (default: 100)
    """
    
    # Load configuration
    print("="*80)
    print("CREATING IFCB LOCATION MAP")
    print("="*80)
    
    config = load_config(config_path)
    bbox = config['region']['bbox']
    region_name = config['region']['name']
    region_desc = config['region']['description']
    
    # Study area bounds
    lon_min, lon_max = bbox['west'], bbox['east']
    lat_min, lat_max = bbox['south'], bbox['north']
    
    print(f"Region: {region_desc}")
    print(f"Bounds: [{lat_min:.2f}°N-{lat_max:.2f}°N, {lon_min:.2f}°W-{lon_max:.2f}°W]")
    
    # Load IFCB data
    print(f"\nLoading IFCB data: {ifcb_data_path}")
    ifcb_data = pd.read_csv(ifcb_data_path)
    print(f"  Loaded {len(ifcb_data):,} observations")
    
    # Find or load Sentinel-3 image
    if s3_image_path is None:
        print("\nSearching for Sentinel-3 image...")
        s3_image_path = find_best_s3_image(config['acolite']['output_dir'])
        if s3_image_path:
            print(f"  Found: {s3_image_path}")
        else:
            print("  No Sentinel-3 image found - will show basemap only")
    else:
        s3_image_path = Path(s3_image_path)
        if not s3_image_path.exists():
            print(f"⚠ Specified image not found: {s3_image_path}")
            s3_image_path = None
    
    # Create figure
    print("\nCreating figure...")
    fig = plt.figure(figsize=(18, 8))
    
    # Subplot 1: Sentinel-3 image
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    plot_s3_image(ax1, s3_image_path, bbox, title="Sentinel-3 OLCI Composite")
    
    # Subplot 2: IFCB locations
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    plot_ifcb_locations(ax2, ifcb_data, bbox, fixed_threshold=fixed_threshold)
    
    # Overall title
    fig.suptitle(f'HAB Monitoring Study Area: {region_desc}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    output_file = output_path / f'{region_name}_ifcb_locations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {output_file}")
    
    # Also save as PDF for publication
    output_file_pdf = output_path / f'{region_name}_ifcb_locations.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {output_file_pdf}")
    
    print("="*80)
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create figure showing Sentinel-3 imagery and IFCB observation locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with IFCB data file
  python create_ifcb_location_map.py --ifcb-data ifcb_observations.csv
  
  # Specify Sentinel-3 image explicitly
  python create_ifcb_location_map.py --ifcb-data ifcb.csv --s3-image mosaic.tif
  
  # Use custom config and change fixed station threshold
  python create_ifcb_location_map.py --ifcb-data ifcb.csv --config my_config.yaml --threshold 50
        """
    )
    
    parser.add_argument(
        '--ifcb-data',
        type=str,
        default='data/whohab_GOM_cell_conc_mean_2017-25.csv',
        help='Path to CSV file with IFCB observations (required columns: datetime, latitude, longitude, location_id, location_name, location_type)'
    )
    
    parser.add_argument(
        '--s3-image',
        type=str,
        help='Path to Sentinel-3 image (GeoTIFF). If not provided, will search for best available in ACOLITE outputs'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='s3_olci_config.yaml',
        help='Path to configuration file (default: s3_olci_config.yaml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures',
        help='Output directory for figure (default: figures/)'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=100,
        help='Minimum observations to classify as fixed station (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        return
    
    # Check if IFCB data exists
    if not Path(args.ifcb_data).exists():
        print(f"✗ IFCB data file not found: {args.ifcb_data}")
        return
    
    # Create figure
    create_figure(
        ifcb_data_path=args.ifcb_data,
        config_path=args.config,
        s3_image_path=args.s3_image,
        output_dir=args.output_dir,
        fixed_threshold=args.threshold
    )


if __name__ == "__main__":
    main()


    