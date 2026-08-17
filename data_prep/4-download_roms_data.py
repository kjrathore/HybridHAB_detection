"""
Gulf of Maine Operational Forecast System (GoMOFS) Data Downloader
Downloads ROMS model output from NOAA NCEI THREDDS server

Data source: https://www.ncei.noaa.gov/oa/prod-model/
Operational model: Gulf of Maine Operational Forecast System (GoMOFS)

Note: Data availability starts from ~2018
      File structure before Sept 10, 2024
"""

import os
import yaml
from pathlib import Path
import requests
from datetime import datetime, timedelta
import argparse
import time
from urllib.parse import urljoin


class GoMOFSDownloader:
    def __init__(self, config_path="gomofs_config.yaml"):
        """
        Initialize GoMOFS downloader from config file
        
        Parameters:
        -----------
        config_path : str
            Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract configuration
        self.region_name = self.config['region']['name']
        self.region_description = self.config['region']['description']
        
        # Use download date range from config
        self.start_date = datetime.strptime(
            self.config['download']['date_range']['start'], 
            '%Y-%m-%d'
        )
        self.end_date = datetime.strptime(
            self.config['download']['date_range']['end'], 
            '%Y-%m-%d'
        )
        
        # GoMOFS specific configuration
        gomofs_config = self.config.get('gomofs', {})
        
        # Product types to download (can be multiple)
        self.product_types = gomofs_config.get('product_types', ['2ds'])
        
        # Time cycles to download (can be multiple)
        self.time_cycles = gomofs_config.get('time_cycles', ['t12z'])
        
        # Forecast hours to download (can be multiple)
        # Examples: ['n001', 'n003', 'n006'] or ['f024', 'f048']
        self.forecast_hours = gomofs_config.get('forecast_hours', ['n001'])
        
        # Station types (for stations product only)
        self.station_types = gomofs_config.get('station_types', ['nowcast', 'forecast'])
        
        # Create output directory
        self.output_dir = Path(gomofs_config.get('output_dir', './gomofs_data'))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # NOAA NCEI actual URL structure (not THREDDS)
        # Note: THREDDS server URL is different and may not work for direct downloads
        # Use this prod-model URL for reliable access
        # self.base_url = "https://www.ncei.noaa.gov/oa/prod-model/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/gulf-of-maine-operational-forecast-system-gomofs/"
        self.base_url = "https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/gulf-of-maine-operational-forecast-system-gomofs/"
        
        print("\n" + "="*80)
        print(f"GoMOFS DATA DOWNLOADER: {self.region_description}")
        print("="*80)
        print(f"Region: {self.region_name}")
        print(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Product Types: {', '.join(self.product_types)}")
        print(f"Time Cycles: {', '.join(self.time_cycles)}")
        print(f"Forecast Hours: {', '.join(self.forecast_hours)}")
        if 'stations' in self.product_types:
            print(f"Station Types: {', '.join(self.station_types)}")
        print("="*80 + "\n")
        
        # Data availability note
        if self.start_date.year < 2018:
            print("⚠  WARNING: GoMOFS data availability starts from ~2018")
            print(f"  Requested start date: {self.start_date.date()}")
            print(f"  Files before 2018 may not exist\n")
    
    def construct_filename(self, date, product_type, time_cycle, forecast_hour=None, station_type=None):
        """
        Construct GoMOFS filename for a given date
        
        Parameters:
        -----------
        date : datetime
            Date for the file
        product_type : str
            Product type: '2ds', 'fields', or 'stations'
        time_cycle : str
            Time cycle: 't00z', 't06z', 't12z', 't18z'
        forecast_hour : str, optional
            Forecast hour for 2ds/fields: 'n001'-'n006' or 'f001'-'f072'
        station_type : str, optional
            For stations only: 'nowcast' or 'forecast'
        
        Returns:
        --------
        str : Filename
        
        Examples (before Sept 10, 2024):
        ---------
        nos.gomofs.2ds.n001.20180601.t12z.nc
        nos.gomofs.fields.f024.20180601.t12z.nc
        nos.gomofs.stations.nowcast.20180601.t12z.nc
        nos.gomofs.stations.forecast.20180601.t12z.nc
        """
        date_str = date.strftime('%Y%m%d')
        
        if product_type in ['2ds', 'fields']:
            # Format: nos.gomofs.{2ds|fields}.{n|f}HHH.YYYYMMDD.tCCz.nc
            if forecast_hour is None:
                raise ValueError(f"forecast_hour required for {product_type}")
            filename = f"nos.gomofs.{product_type}.{forecast_hour}.{date_str}.{time_cycle}.nc"
        
        elif product_type == 'stations':
            # Format: nos.gomofs.stations.{nowcast|forecast}.YYYYMMDD.tCCz.nc
            if station_type is None:
                raise ValueError("station_type required for stations")
            filename = f"nos.gomofs.stations.{station_type}.{date_str}.{time_cycle}.nc"
        
        else:
            raise ValueError(f"Unknown product type: {product_type}")
        
        return filename
    
    def construct_url(self, date, product_type, time_cycle, forecast_hour=None, station_type=None):
        """
        Construct full download URL for a given date
        
        Parameters:
        -----------
        date : datetime
            Date for the file
        product_type : str
            Product type: '2ds', 'fields', or 'stations'
        time_cycle : str
            Time cycle: 't00z', 't06z', 't12z', 't18z'
        forecast_hour : str, optional
            Forecast hour for 2ds/fields
        station_type : str, optional
            For stations only: 'nowcast' or 'forecast'
        
        Returns:
        --------
        str : Full download URL
        """
        filename = self.construct_filename(date, product_type, time_cycle, forecast_hour, station_type)
        
        # Directory structure: YYYY/MM/
        year_dir = date.strftime('%Y')
        month_dir = date.strftime('%m')
        
        # Full URL: base_url/YYYY/MM/filename
        # Example: .../gomofs/2018/01/nos.gomofs.2ds.n001.20180101.t12z.nc
        url = f"{self.base_url}{year_dir}/{month_dir}/{filename}"
        
        return url
    
    def check_file_exists(self, url, timeout=10):
        """
        Check if a file exists on the server using HEAD request
        
        Parameters:
        -----------
        url : str
            URL to check
        timeout : int
            Request timeout in seconds
        
        Returns:
        --------
        tuple : (exists: bool, size_mb: float)
        """
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                size_bytes = int(response.headers.get('content-length', 0))
                size_mb = size_bytes / (1024 * 1024)
                return True, size_mb
            else:
                return False, 0
        except Exception as e:
            return False, 0
    
    def download_file(self, url, output_path, max_retries=3):
        """
        Download a single file with retry logic
        
        Parameters:
        -----------
        url : str
            URL to download
        output_path : Path
            Output file path
        max_retries : int
            Maximum number of retry attempts
        
        Returns:
        --------
        bool : Success status
        """
        temp_path = output_path.with_suffix('.nc.tmp')
        
        # Check if already downloaded and valid
        if output_path.exists():
            try:
                # Quick validation - check file size > 0
                if output_path.stat().st_size > 0:
                    print(f"⊙ Already downloaded: {output_path.name}")
                    return True
            except:
                pass
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                # Check if partial download exists
                resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0
                
                headers = {}
                if resume_byte_pos > 0:
                    headers['Range'] = f'bytes={resume_byte_pos}-'
                    print(f"↻ Resuming from {resume_byte_pos/1e6:.1f} MB: {output_path.name}")
                else:
                    print(f"↓ Downloading: {output_path.name}")
                
                response = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=300  # 5 minutes
                )
                response.raise_for_status()
                
                # Get total size
                total_size = int(response.headers.get('content-length', 0))
                if resume_byte_pos > 0:
                    total_size += resume_byte_pos
                
                # Download with progress
                block_size = 1024 * 1024  # 1 MB chunks
                downloaded = resume_byte_pos
                
                mode = 'ab' if resume_byte_pos > 0 else 'wb'
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                mb_downloaded = downloaded / 1e6
                                mb_total = total_size / 1e6
                                print(f"  Progress: {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
                
                print(f"\n✓ Downloaded: {output_path.name}")
                
                # Move temp to final location
                temp_path.rename(output_path)
                
                return True
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"\n⚠  File not found (404): {url}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return False
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\n⚠  HTTP Error {e.response.status_code}: {e}")
                    print(f"  Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n✗ HTTP error after {max_retries} attempts: {e}")
                    return False
            
            except (requests.exceptions.RequestException, 
                    ConnectionResetError, 
                    ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\n⚠  Download interrupted: {e}")
                    print(f"  Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n✗ Download failed after {max_retries} attempts: {e}")
                    if temp_path.exists():
                        print(f"  Partial download saved: {temp_path}")
                    return False
            
            except Exception as e:
                print(f"\n✗ Unexpected error: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                return False
        
        return False
    
    def generate_date_list(self):
        """
        Generate list of dates to download
        
        Returns:
        --------
        list : List of datetime objects
        """
        dates = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)
        
        return dates
    
    def build_download_tasks(self, product_types=None, time_cycles=None, 
                            forecast_hours=None, station_types=None):
        """
        Build list of all files to download based on configuration
        
        Parameters:
        -----------
        product_types : list, optional
            Product types to download (default: from config)
        time_cycles : list, optional
            Time cycles to download (default: from config)
        forecast_hours : list, optional
            Forecast hours to download (default: from config)
        station_types : list, optional
            Station types to download (default: from config)
        
        Returns:
        --------
        list : List of tuples (date, product_type, time_cycle, forecast_hour, station_type)
        """
        if product_types is None:
            product_types = self.product_types
        if time_cycles is None:
            time_cycles = self.time_cycles
        if forecast_hours is None:
            forecast_hours = self.forecast_hours
        if station_types is None:
            station_types = self.station_types
        
        dates = self.generate_date_list()
        tasks = []
        
        for date in dates:
            for time_cycle in time_cycles:
                for product_type in product_types:
                    
                    if product_type in ['2ds', 'fields']:
                        # Gridded data: need forecast hours
                        for forecast_hour in forecast_hours:
                            tasks.append((date, product_type, time_cycle, forecast_hour, None))
                    
                    elif product_type == 'stations':
                        # Station data: need station types
                        for station_type in station_types:
                            tasks.append((date, product_type, time_cycle, None, station_type))
        
        return tasks
    
    def download_batch(self,
                      product_types=None,
                      time_cycles=None,
                      forecast_hours=None,
                      station_types=None,
                      check_availability=True,
                      parallel=False,
                      max_workers=2):
        """
        Download GoMOFS data for date range
        
        Parameters:
        -----------
        product_types : list, optional
            Product types to download (default: from config)
        time_cycles : list, optional
            Time cycles to download (default: from config)
        forecast_hours : list, optional
            Forecast hours to download (default: from config)
        station_types : list, optional
            Station types to download (default: from config)
        check_availability : bool
            Check file existence before downloading (default: True)
        parallel : bool
            Enable parallel downloads (default: False)
        max_workers : int
            Number of parallel workers (default: 2)
        """
        
        # Build download tasks
        tasks = self.build_download_tasks(product_types, time_cycles, forecast_hours, station_types)
        
        print("\n" + "="*80)
        print(f"DOWNLOADING GoMOFS DATA")
        print("="*80)
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total tasks: {len(tasks)}")
        if parallel:
            print(f"Mode: PARALLEL ({max_workers} workers)")
        else:
            print(f"Mode: SEQUENTIAL")
        print("="*80)
        
        # Check availability if requested
        if check_availability:
            print("\nChecking data availability...")
            available_tasks = []
            
            for date, product_type, time_cycle, forecast_hour, station_type in tasks:
                filename = self.construct_filename(date, product_type, time_cycle, forecast_hour, station_type)
                url = self.construct_url(date, product_type, time_cycle, forecast_hour, station_type)
                exists, size_mb = self.check_file_exists(url)
                
                # Create label
                if product_type == 'stations':
                    label = f"{date.date()} {product_type} {station_type} {time_cycle}"
                else:
                    label = f"{date.date()} {product_type} {forecast_hour} {time_cycle}"
                
                if exists:
                    available_tasks.append((date, product_type, time_cycle, forecast_hour, station_type))
                    print(f"  ✓ {label}: Available ({size_mb:.1f} MB)")
                    print(f"    → File: {filename}")
                    print(f"    → URL:  {url}")
                else:
                    print(f"  ✗ {label}: Not available")
                    print(f"    → File: {filename}")
                    print(f"    → URL:  {url}")
            
            print(f"\nFound {len(available_tasks)}/{len(tasks)} available files")
            
            if not available_tasks:
                print("\n⚠  No files available for download")
                return 0, []
            
            tasks = available_tasks
        
        # Download files
        print("\n" + "="*80)
        print("STARTING DOWNLOADS")
        print("="*80)
        
        success_count = 0
        failed_tasks = []
        
        if parallel:
            # Parallel downloads
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def download_with_index(index_task):
                i, (date, product_type, time_cycle, forecast_hour, station_type) = index_task
                
                filename = self.construct_filename(date, product_type, time_cycle, forecast_hour, station_type)
                url = self.construct_url(date, product_type, time_cycle, forecast_hour, station_type)
                output_path = self.output_dir / filename
                
                print(f"\n[{i}/{len(tasks)}]")
                print(f"File: {filename}")
                print(f"URL:  {url}")
                
                success = self.download_file(url, output_path)
                return (i, date, product_type, time_cycle, forecast_hour, station_type, success)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, task in enumerate(tasks, 1):
                    # Small delay between submissions
                    if i > 1:
                        time.sleep(0.5)
                    future = executor.submit(download_with_index, (i, task))
                    futures.append(future)
                
                for future in as_completed(futures):
                    i, date, product_type, time_cycle, forecast_hour, station_type, success = future.result()
                    if success:
                        success_count += 1
                        if product_type == 'stations':
                            label = f"{date.date()} {product_type} {station_type}"
                        else:
                            label = f"{date.date()} {product_type} {forecast_hour}"
                        print(f"✓ [{i}/{len(tasks)}] Complete: {label}")
                    else:
                        failed_tasks.append((date, product_type, time_cycle, forecast_hour, station_type))
        
        else:
            # Sequential downloads (recommended)
            for i, (date, product_type, time_cycle, forecast_hour, station_type) in enumerate(tasks, 1):
                print(f"\n[{i}/{len(tasks)}]")
                
                if product_type == 'stations':
                    label = f"{date.date()} {product_type} {station_type} {time_cycle}"
                else:
                    label = f"{date.date()} {product_type} {forecast_hour} {time_cycle}"
                
                filename = self.construct_filename(date, product_type, time_cycle, forecast_hour, station_type)
                url = self.construct_url(date, product_type, time_cycle, forecast_hour, station_type)
                output_path = self.output_dir / filename
                
                # Print what we're trying to download
                print(f"File: {filename}")
                print(f"URL:  {url}")
                
                if self.download_file(url, output_path):
                    success_count += 1
                else:
                    failed_tasks.append((date, product_type, time_cycle, forecast_hour, station_type))
                
                # Small delay between downloads
                if i < len(tasks):
                    time.sleep(0.5)
        
        # Summary
        print("\n" + "="*80)
        print(f"✓ DOWNLOAD COMPLETE: {success_count}/{len(tasks)} successful")
        if failed_tasks:
            print(f"\n⚠  {len(failed_tasks)} FAILED DOWNLOADS:")
            for date, product_type, time_cycle, forecast_hour, station_type in failed_tasks[:10]:
                if product_type == 'stations':
                    label = f"{date.date()} {product_type} {station_type} {time_cycle}"
                else:
                    label = f"{date.date()} {product_type} {forecast_hour} {time_cycle}"
                print(f"  - {label}")
            if len(failed_tasks) > 10:
                print(f"  ... and {len(failed_tasks) - 10} more")
            print("\n💡 TIP: Run again to retry failed downloads")
        print("="*80)
        print(f"Data saved to: {self.output_dir}")
        
        return success_count, failed_tasks
    
    def list_available_products(self):
        """
        List available product types and options
        """
        print("\n" + "="*80)
        print("AVAILABLE GoMOFS PRODUCTS")
        print("="*80)
        
        print("\nPRODUCT TYPES:")
        print("  2ds       - 2D surface fields (temperature, salinity at surface)")
        print("  fields    - 3D volume fields (full water column)")
        print("  stations  - Station time series data")
        
        print("\nTIME CYCLES (4 per day):")
        print("  t00z - 00:00 UTC (midnight)")
        print("  t06z - 06:00 UTC (6am)")
        print("  t12z - 12:00 UTC (noon)")
        print("  t18z - 18:00 UTC (6pm)")
        
        print("\nFORECAST HOURS (for 2ds/fields):")
        print("  Nowcast: n001, n002, n003, n004, n005, n006")
        print("  Forecast: f001, f002, f003, ... f072 (1-72 hours)")
        print("  Common: n001 (nowcast hour 1), f024 (24h forecast), f048 (48h forecast)")
        
        print("\nSTATION TYPES (for stations):")
        print("  nowcast  - Analysis/current conditions")
        print("  forecast - 72-hour forecast")
        
        print("\nFILE NAMING EXAMPLES:")
        print("  nos.gomofs.2ds.n001.20180601.t12z.nc")
        print("  nos.gomofs.fields.f024.20180601.t12z.nc")
        print("  nos.gomofs.stations.nowcast.20180601.t12z.nc")
        
        print("="*80)


def main():
    """Main download workflow"""
    
    parser = argparse.ArgumentParser(
        description='Download Gulf of Maine Operational Forecast System (GoMOFS) data from NOAA NCEI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Download specific product type
  python datasets/4-download_roms_data.py --product-types 2ds --start-date 2018-01-01 --end-date 2020-01-10
  
  # Download multiple products
  python gomofs_downloader.py --product-types 2ds fields stations
  
  # Download specific forecast hours
  python gomofs_downloader.py --forecast-hours n000 f024 f048
  
  # Download specific time cycle
  python gomofs_downloader.py --time-cycles t12z
  
  # Override date range
  python gomofs_downloader.py --start-date 2020-06-01 --end-date 2020-06-30
  
  # Download everything
  python gomofs_downloader.py --product-types 2ds fields stations --forecast-hours n000
  
  # Enable parallel downloads
  python gomofs_downloader.py --parallel --workers 3
  
  # List available products
  python gomofs_downloader.py --list-products
  
  # Check availability without downloading
  python gomofs_downloader.py --check-only
  
Note: GoMOFS data availability starts from ~2018
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='datasets/GOM.yaml',
        help='Path to configuration file (default: gomofs_config.yaml)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides config'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), overrides config'
    )
    
    parser.add_argument(
        '--product-types',
        nargs='+',
        choices=['2ds', 'fields', 'stations'],
        help='Product types to download (overrides config)'
    )
    
    parser.add_argument(
        '--time-cycles',
        nargs='+',
        choices=['t00z', 't06z', 't12z', 't18z'],
        help='Time cycles to download (overrides config)'
    )
    
    parser.add_argument(
        '--forecast-hours',
        nargs='+',
        help='Forecast hours for 2ds/fields (e.g., n000 f024 f048, overrides config)'
    )
    
    parser.add_argument(
        '--station-types',
        nargs='+',
        choices=['nowcast', 'forecast'],
        help='Station types to download (overrides config)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel downloads'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 2)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Check file availability without downloading'
    )
    
    parser.add_argument(
        '--list-products',
        action='store_true',
        help='List available product types and options'
    )
    
    parser.add_argument(
        '--no-check',
        action='store_true',
        help='Skip availability check (try to download all)'
    )
    
    args = parser.parse_args()
    
    # List products if requested
    if args.list_products:
        # Create dummy downloader just for listing
        if Path(args.config).exists():
            downloader = GoMOFSDownloader(config_path=args.config)
        else:
            # Create minimal config
            print("No config file found, showing generic product list")
            downloader = GoMOFSDownloader.__new__(GoMOFSDownloader)
        downloader.list_available_products()
        return
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        print("\n  Create gomofs_config.yaml with your settings")
        print("  Or use s3_olci_config.yaml (will use date range from it)")
        return
    
    # Initialize downloader
    downloader = GoMOFSDownloader(config_path=args.config)
    
    # Override dates if provided
    if args.start_date:
        downloader.start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        downloader.end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Override configuration with command-line arguments
    product_types = args.product_types if args.product_types else downloader.product_types
    time_cycles = args.time_cycles if args.time_cycles else downloader.time_cycles
    forecast_hours = args.forecast_hours if args.forecast_hours else downloader.forecast_hours
    station_types = args.station_types if args.station_types else downloader.station_types
    
    # Check only mode
    if args.check_only:
        print("\nCHECK-ONLY MODE: Listing available files")
        
        tasks = downloader.build_download_tasks(
            product_types, time_cycles, forecast_hours, station_types
        )
        
        available_count = 0
        
        for date, product_type, time_cycle, forecast_hour, station_type in tasks:
            url = downloader.construct_url(date, product_type, time_cycle, forecast_hour, station_type)
            exists, size_mb = downloader.check_file_exists(url)
            
            if product_type == 'stations':
                label = f"{date.date()} {product_type} {station_type} {time_cycle}"
            else:
                label = f"{date.date()} {product_type} {forecast_hour} {time_cycle}"
            
            if exists:
                available_count += 1
                print(f"✓ {label}: Available ({size_mb:.1f} MB)")
            else:
                print(f"✗ {label}: Not available")
        
        print(f"\nSummary: {available_count}/{len(tasks)} files available")
        return
    
    # Download data
    downloader.download_batch(
        product_types=product_types,
        time_cycles=time_cycles,
        forecast_hours=forecast_hours,
        station_types=station_types,
        check_availability=not args.no_check,
        parallel=args.parallel,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()