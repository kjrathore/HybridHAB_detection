#!/usr/bin/env python3
"""
Alexandrium catenella Abundance Prediction Using Satellite Data
Extracts spatial statistics (median & maximum) from Sentinel-2 bands
and builds linear regression models for relative abundance prediction.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Install required netCDF libraries
import subprocess
import sys

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Configuration
# Configuration
MOSAIC_DIR = Path('sentinel2/sentinel2_data/mosaics_14day')
STATIONS_FILE = 'chile/stations.csv'
ABUNDANCE_FILE = 'chile/rel_abundance.csv'
BUFFER_RADIUS_M = 200
MAX_TIME_DIFF_DAYS = 7
OUTPUT_DIR = Path('regr_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
N_JOBS = 8  # Number of parallel workers


# Band names in the netCDF files
BANDS = ['rrs_B1', 'rrs_B2', 'rrs_B3', 'rrs_B4', 'rrs_B5', 'rrs_B6', 'rrs_B7', 'rrs_B8A']


def load_stations():
    """Load station coordinates and metadata."""
    stations = pd.read_csv(STATIONS_FILE)
    print(f"Loaded {len(stations)} stations")
    return stations


def load_abundance_data():
    """Load and process Alexandrium catenella relative abundance data."""
    df = pd.read_csv(ABUNDANCE_FILE)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Melt to long format
    id_vars = ['Date']
    value_vars = [col for col in df.columns if col != 'Date']
    
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars,
                      var_name='station_code', value_name='rel_abundance')
    
    print(f"Loaded {len(df_long)} abundance observations")
    print(f"Date range: {df_long['Date'].min()} to {df_long['Date'].max()}")
    
    return df_long


def parse_mosaic_date(filename):
    """Extract date from mosaic filename."""
    # Expected format: Mosaic_YYYYMMDD_YYYYMMDD.nc
    parts = filename.stem.split('_')
    if len(parts) >= 2:
        try:
            start_date = datetime.strptime(parts[1], '%Y%m%d')
            return start_date
        except:
            return None
    return None


def calculate_pixel_distance(lat, lon, ds):
    """
    Calculate distance from point to all pixels in dataset.
    Returns distance array in meters (approximate).
    """
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Approximate conversion to meters (111km per degree lat, adjusted for lon)
    lat_m = (lat_grid - lat) * 111000
    lon_m = (lon_grid - lon) * 111000 * np.cos(np.radians(lat))
    
    distance = np.sqrt(lat_m**2 + lon_m**2)
    
    return distance


def extract_spatial_stats(ds, lat, lon, radius_m=200):
    """
    Extract median and maximum values from bands within radius of point.
    """
    stats = {}
    
    # Calculate distance to all pixels
    distance = calculate_pixel_distance(lat, lon, ds)
    
    # Create mask for pixels within radius
    mask = distance <= radius_m
    
    if not np.any(mask):
        print(f"  Warning: No pixels found within {radius_m}m of ({lat}, {lon})")
        return None
    
    # Extract statistics for each band
    for band in BANDS:
        if band not in ds:
            continue
            
        band_data = ds[band].values
        
        # Get values within radius
        values = band_data[mask]
        
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            stats[f'{band}_median'] = np.nan
            stats[f'{band}_max'] = np.nan
        else:
            stats[f'{band}_median'] = np.median(values)
            stats[f'{band}_max'] = np.max(values)
    
    stats['n_pixels'] = np.sum(mask)
    stats['n_valid_pixels'] = len(values) if len(values) > 0 else 0
    
    return stats


def match_mosaic_to_date(target_date, mosaic_files, window_days=14):
    """
    Find the mosaic file that best matches the target date.
    """
    best_mosaic = None
    min_diff = float('inf')
    
    for mosaic_file in mosaic_files:
        mosaic_date = parse_mosaic_date(mosaic_file)
        if mosaic_date is None:
            continue
        
        diff = abs((target_date - mosaic_date).days)
        
        if diff <= window_days and diff < min_diff:
            min_diff = diff
            best_mosaic = mosaic_file
    
    return best_mosaic, min_diff if best_mosaic else None


def process_single_observation(row_data, mosaic_files):
    """
    Process a single observation - designed for parallel execution.
    """
    station_code = row_data['station_code']
    station_name = row_data['station_name']
    date = row_data['Date']
    lat = row_data['latitude']
    lon = row_data['longitude']
    rel_abundance = row_data['rel_abundance']
    
    # Find matching mosaic
    mosaic_file, time_diff = match_mosaic_to_date(date, mosaic_files)
    
    if mosaic_file is None:
        return None
    
    # Skip if time difference is too large
    if time_diff > MAX_TIME_DIFF_DAYS:
        return None
    
    try:
        # Load mosaic - try different engines
        try:
            ds = xr.open_dataset(mosaic_file, engine='netcdf4')
        except:
            try:
                ds = xr.open_dataset(mosaic_file, engine='h5netcdf')
            except:
                ds = xr.open_dataset(mosaic_file)
        
        # Extract spatial statistics
        stats = extract_spatial_stats(ds, lat, lon, BUFFER_RADIUS_M)
        
        ds.close()
        
        if stats is None:
            return None
        
        # Combine all information
        result = {
            'station_code': station_code,
            'station_name': station_name,
            'date': date,
            'latitude': lat,
            'longitude': lon,
            'rel_abundance': rel_abundance,
            'mosaic_file': mosaic_file.name,
            'time_diff_days': time_diff,
            **stats
        }
        
        return result
        
    except Exception as e:
        # Silently skip errors in parallel processing
        return None


def extract_features_for_all_stations():
    """
    Extract spatial features for all stations and dates using parallel processing.
    Uses all observed relative abundance values (including zeros).
    """
    # Load data
    stations = load_stations()
    abundance = load_abundance_data()
    
    # Get all mosaic files
    mosaic_files = sorted(MOSAIC_DIR.glob('Mosaic_*.nc'))
    print(f"Found {len(mosaic_files)} mosaic files")
    
    if len(mosaic_files) == 0:
        print(f"ERROR: No mosaic files found in {MOSAIC_DIR}")
        return None
    
    # Merge stations with abundance data
    data = abundance.merge(stations, on='station_code', how='left')
    
    print(f"Using all {len(data)} observations (including zeros)")
    print(f"Parallel processing with {N_JOBS} workers")
    
    # Convert to list of dictionaries for parallel processing
    rows_list = data.to_dict('records')
    
    # Create partial function with mosaic_files
    process_func = partial(process_single_observation, mosaic_files=mosaic_files)
    
    # Process in parallel with progress bar
    print(f"\nProcessing {len(rows_list)} station-date combinations...")
    
    results = []
    with Pool(processes=N_JOBS) as pool:
        # Use imap instead of map to show progress
        for result in tqdm(pool.imap(process_func, rows_list), 
                          total=len(rows_list),
                          desc="Extracting features",
                          unit="obs"):
            if result is not None:
                results.append(result)
    
    if len(results) == 0:
        print("ERROR: No valid results extracted")
        return None
    
    df_features = pd.DataFrame(results)
    
    print(f"\nSuccessfully extracted features for {len(df_features)} observations")
    print(f"Skipped: {len(rows_list) - len(df_features)} observations (no matching mosaic or outside time window)")
    print(f"Abundance range: [{df_features['rel_abundance'].min():.0f}, {df_features['rel_abundance'].max():.0f}]")
    print(f"Zero abundance samples: {(df_features['rel_abundance'] == 0).sum()}")
    print(f"Non-zero abundance samples: {(df_features['rel_abundance'] > 0).sum()}")
    print(f"Time difference filter: {df_features['time_diff_days'].max():.1f} days max")
    print(f"Features per band: median and maximum")
    print(f"Total features: {len([c for c in df_features.columns if 'rrs_' in c])}")
    
    return df_features
    print(f"Abundance range: [{df_features['rel_abundance'].min():.0f}, {df_features['rel_abundance'].max():.0f}]")
    print(f"Zero abundance samples: {(df_features['rel_abundance'] == 0).sum()}")
    print(f"Non-zero abundance samples: {(df_features['rel_abundance'] > 0).sum()}")
    print(f"Time difference filter: {df_features['time_diff_days'].max():.1f} days max")
    print(f"Features per band: median and maximum")
    print(f"Total features: {len([c for c in df_features.columns if 'rrs_' in c])}")
    
    return df_features


def prepare_regression_data(df, stat_type='median'):
    """
    Prepare feature matrix and target for regression.
    stat_type: 'median' or 'max' (for band features only)
    Target: observed rel_abundance
    """
    # Select feature columns based on stat_type
    feature_cols = [c for c in df.columns if f'rrs_B' in c and stat_type in c]
    
    # Target is always the observed rel_abundance
    target_col = 'rel_abundance'
    
    # Remove rows with missing features or target
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"\n{stat_type.upper()} band features → Observed rel_abundance:")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, median: {np.median(y):.2f}")
    
    return X, y, feature_cols, df_clean


def evaluate_linear_regression(X, y, feature_names, stat_type):
    """
    Train and evaluate linear regression model with plots.
    stat_type: 'median' or 'max' (refers to band features)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\nLinear Regression Results:")
    print("-" * 60)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    
    # Feature coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nTop 5 features by coefficient magnitude:")
    for _, row in coef_df.head(5).iterrows():
        print(f"  {row['feature']:20s}: {row['coefficient']:+.6f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Predicted vs Actual (Train)
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_pred_train, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
    lim = [min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())]
    ax1.plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Abundance', fontsize=11)
    ax1.set_ylabel('Predicted Abundance', fontsize=11)
    ax1.set_title(f'Training Set (R² = {train_r2:.4f})', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual (Test)
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5, color='orange')
    lim = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    ax2.plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Abundance', fontsize=11)
    ax2.set_ylabel('Predicted Abundance', fontsize=11)
    ax2.set_title(f'Test Set (R² = {test_r2:.4f})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals_test = y_test - y_pred_test
    ax3.scatter(y_pred_test, residuals_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5, color='green')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Abundance', fontsize=11)
    ax3.set_ylabel('Residuals', fontsize=11)
    ax3.set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Coefficients
    ax4 = axes[1, 1]
    top_n = min(10, len(coef_df))
    top_coefs = coef_df.head(top_n)
    colors = ['red' if x < 0 else 'blue' for x in top_coefs['coefficient']]
    ax4.barh(range(top_n), top_coefs['coefficient'], color=colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_coefs['feature'], fontsize=9)
    ax4.set_xlabel('Coefficient Value', fontsize=11)
    ax4.set_title(f'Top {top_n} Feature Coefficients', fontsize=12, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', lw=1)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Linear Regression: {stat_type.upper()} Band Features → Observed Abundance',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    plot_file = OUTPUT_DIR / f'regression_plot_{stat_type}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    plt.close()
    
    # Return results
    results = {
        'stat_type': stat_type,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_samples': len(X),
        'n_features': len(feature_names)
    }
    
    return results, model, coef_df


def main():
    """Main execution function."""
    print("="*80)
    print("Alexandrium catenella Abundance Prediction - Linear Regression")
    print("="*80)
    print(f"Configuration:")
    print(f"  Buffer radius: {BUFFER_RADIUS_M}m")
    print(f"  Max time difference: {MAX_TIME_DIFF_DAYS} days")
    
    # Extract features
    print("\n" + "="*80)
    print("1. EXTRACTING SPATIAL FEATURES")
    print("="*80)
    df_features = extract_features_for_all_stations()
    
    if df_features is None:
        print("Failed to extract features. Exiting.")
        return
    
    # Save feature dataframe
    feature_file = OUTPUT_DIR / 'alexandrium_features.csv'
    df_features.to_csv(feature_file, index=False)
    print(f"\nFeatures saved to: {feature_file}")
    
    # Model 1: MEDIAN band features → Observed abundance
    print("\n" + "="*80)
    print("2. LINEAR REGRESSION - MEDIAN BAND FEATURES")
    print("="*80)
    X_med, y_med, features_med, df_med = prepare_regression_data(
        df_features, stat_type='median'
    )
    results_med, model_med, coef_med = evaluate_linear_regression(
        X_med, y_med, features_med, 'median'
    )
    
    # Save median results
    results_med_df = pd.DataFrame([results_med])
    results_med_file = OUTPUT_DIR / 'model_results_median.csv'
    results_med_df.to_csv(results_med_file, index=False)
    
    coef_med_file = OUTPUT_DIR / 'coefficients_median.csv'
    coef_med.to_csv(coef_med_file, index=False)
    
    # Model 2: MAXIMUM band features → Observed abundance
    print("\n" + "="*80)
    print("3. LINEAR REGRESSION - MAXIMUM BAND FEATURES")
    print("="*80)
    X_max, y_max, features_max, df_max = prepare_regression_data(
        df_features, stat_type='max'
    )
    results_max, model_max, coef_max = evaluate_linear_regression(
        X_max, y_max, features_max, 'max'
    )
    
    # Save maximum results
    results_max_df = pd.DataFrame([results_max])
    results_max_file = OUTPUT_DIR / 'model_results_max.csv'
    results_max_df.to_csv(results_max_file, index=False)
    
    coef_max_file = OUTPUT_DIR / 'coefficients_max.csv'
    coef_max.to_csv(coef_max_file, index=False)
    
    print("\n" + "="*80)
    print("4. SUMMARY")
    print("="*80)
    print(f"\nMedian Band Features Model:")
    print(f"  Test R²:  {results_med['test_r2']:.4f}")
    print(f"  Test RMSE: {results_med['test_rmse']:.4f}")
    print(f"  Samples:   {results_med['n_samples']}")
    
    print(f"\nMaximum Band Features Model:")
    print(f"  Test R²:  {results_max['test_r2']:.4f}")
    print(f"  Test RMSE: {results_max['test_rmse']:.4f}")
    print(f"  Samples:   {results_max['n_samples']}")
    
    # Determine better model
    if results_med['test_r2'] > results_max['test_r2']:
        print(f"\n→ MEDIAN band features perform better (R² = {results_med['test_r2']:.4f})")
    else:
        print(f"\n→ MAXIMUM band features perform better (R² = {results_max['test_r2']:.4f})")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"Features:              {feature_file}")
    print(f"\nMedian Band Model:")
    print(f"  Results:             {results_med_file}")
    print(f"  Coefficients:        {coef_med_file}")
    print(f"  Plot:                {OUTPUT_DIR}/regression_plot_median.png")
    print(f"\nMaximum Band Model:")
    print(f"  Results:             {results_max_file}")
    print(f"  Coefficients:        {coef_max_file}")
    print(f"  Plot:                {OUTPUT_DIR}/regression_plot_max.png")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()