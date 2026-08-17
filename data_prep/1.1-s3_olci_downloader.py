"""
Generic Sentinel-3 OLCI Data Downloader
Downloads L1B data suitable for ACOLITE processing

Reads region configuration from YAML config file.

FIXES:
- Automatic token refresh (tokens expire after 10 minutes)
- Retry on 401 Unauthorized errors
- Proactive re-authentication before token expires
"""

import os
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import zipfile
import yaml
import argparse
import time


class S3OLCIDownloader:
    def __init__(self, config_path="s3_olci_config.yaml"):
        """
        Initialize downloader from config file
        
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
        self.bbox = self.config['region']['bbox']
        
        self.username = self.config['download']['credentials']['username']
        self.password = self.config['download']['credentials']['password']
        self.max_cloud_cover = self.config['download']['max_cloud_cover']
        self.max_products = self.config['download']['max_products']
        
        # Check if delete_after option exists in config
        self.delete_zip_after_extract = self.config['download'].get('delete_zip_after_extract', False)
        
        self.output_dir = Path(self.config['download']['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Copernicus Data Space API endpoints
        self.auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self.download_url = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"
        
        self.access_token = None
        self.token_timestamp = None  # Track when token was obtained
        self.token_lifetime = 480  # Token lifetime in seconds (8 minutes to be safe, actual is 10)
        
        # Print region info
        print("\n" + "="*80)
        print(f"REGION: {self.region_description}")
        print("="*80)
        print(f"Name: {self.region_name}")
        print(f"Bounding Box:")
        print(f"  West:  {self.bbox['west']:.4f}°")
        print(f"  East:  {self.bbox['east']:.4f}°")
        print(f"  South: {self.bbox['south']:.4f}°")
        print(f"  North: {self.bbox['north']:.4f}°")
        print("="*80 + "\n")
    
    def is_token_valid(self):
        """Check if current token is still valid"""
        if self.access_token is None or self.token_timestamp is None:
            return False
        
        elapsed = time.time() - self.token_timestamp
        return elapsed < self.token_lifetime
    
    def authenticate(self, force=False):
        """
        Get access token for Copernicus Data Space
        
        Parameters:
        -----------
        force : bool
            Force re-authentication even if token appears valid
        """
        # Skip if token is still valid and not forcing
        if not force and self.is_token_valid():
            return True
        
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password"
        }
        
        try:
            response = requests.post(self.auth_url, data=data, timeout=30)
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            self.token_timestamp = time.time()
            
            if force:
                print("✓ Re-authenticated (token refreshed)")
            else:
                print("✓ Authentication successful")
            return True
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            self.access_token = None
            self.token_timestamp = None
            return False
    
    def ensure_authenticated(self):
        """Ensure we have a valid token, refresh if needed"""
        if not self.is_token_valid():
            print("⟳ Token expired or missing, re-authenticating...")
            return self.authenticate(force=True)
        return True
    
    def search_products(self, start_date=None, end_date=None):
        """
        Search for Sentinel-3 OLCI L1B products over region of interest
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in format 'YYYY-MM-DD' (uses config if not provided)
        end_date : str, optional
            End date in format 'YYYY-MM-DD' (uses config if not provided)
        """
        
        # Use config dates if not provided
        if start_date is None:
            start_date = self.config['download']['date_range']['start']
        if end_date is None:
            end_date = self.config['download']['date_range']['end']
        
        # Create WKT footprint from bounding box
        footprint = (f"POLYGON(("
                    f"{self.bbox['west']} {self.bbox['south']},"
                    f"{self.bbox['east']} {self.bbox['south']},"
                    f"{self.bbox['east']} {self.bbox['north']},"
                    f"{self.bbox['west']} {self.bbox['north']},"
                    f"{self.bbox['west']} {self.bbox['south']}))")
        
        # Build OData filter query
        filter_query = (
            f"Collection/Name eq 'SENTINEL-3' and "
            f"contains(Name,'OL_1_EFR') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and "
            f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
            f"ContentDate/Start le {end_date}T23:59:59.999Z"
        )
        
        params = {
            "$filter": filter_query,
            "$orderby": "ContentDate/Start asc",
            "$top": 1000
        }
        
        print(f"Searching for products from {start_date} to {end_date}...")
        
        try:
            response = requests.get(self.search_url, params=params, timeout=60)
            response.raise_for_status()
            results = response.json()
            
            products = results.get('value', [])
            print(f"✓ Found {len(products)} products")
            
            return products
        
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []
    
    def download_product(self, product_id, product_name, extract=True, max_retries=5, rate_limit_pause=None):
        """
        Download a single product with retry logic and intelligent rate limit handling
        
        Parameters:
        -----------
        product_id : str
            Product ID from search results
        product_name : str
            Product name
        extract : bool
            Whether to extract the .SEN3 directory from zip
        max_retries : int
            Maximum number of retry attempts (increased to 5 for rate limits)
        rate_limit_pause : callable, optional
            Function to call when rate limited (for coordinating parallel workers)
        """
        # Ensure we have a valid token before downloading
        if not self.ensure_authenticated():
            print("✗ Cannot download - authentication failed")
            return False
        
        output_file = self.output_dir / f"{product_name}.zip"
        temp_file = self.output_dir / f"{product_name}.zip.tmp"
        
        # Check if already downloaded and valid
        if output_file.exists():
            # Verify it's a valid zip
            try:
                with zipfile.ZipFile(output_file, 'r') as zf:
                    zf.testzip()  # Test integrity
                print(f"⊙ Already downloaded: {product_name}")
                if extract:
                    self.extract_product(output_file, delete_after=self.delete_zip_after_extract)
                return True
            except:
                print(f"⚠ Existing file corrupted, re-downloading: {product_name}")
                output_file.unlink()
        
        download_url = f"{self.download_url}({product_id})/$value"
        
        # Retry loop with intelligent backoff
        for attempt in range(max_retries):
            try:
                # Check token validity before each attempt
                if not self.ensure_authenticated():
                    print("✗ Re-authentication failed")
                    return False
                
                # Check if partial download exists
                resume_byte_pos = temp_file.stat().st_size if temp_file.exists() else 0
                
                headers = {"Authorization": f"Bearer {self.access_token}"}
                if resume_byte_pos > 0:
                    headers['Range'] = f'bytes={resume_byte_pos}-'
                    print(f"↻ Resuming download from {resume_byte_pos/1e6:.1f} MB: {product_name}")
                else:
                    print(f"↓ Downloading: {product_name}")
                
                # Increased timeout
                response = requests.get(
                    download_url, 
                    headers=headers, 
                    stream=True, 
                    timeout=600  # 10 minutes
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
                with open(temp_file, mode) as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                mb_downloaded = downloaded / 1e6
                                mb_total = total_size / 1e6
                                print(f"  Progress: {progress:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
                
                print(f"\n✓ Downloaded: {output_file.name}")
                
                # Move temp to final location
                temp_file.rename(output_file)
                
                # Extract the .SEN3 directory
                if extract:
                    self.extract_product(output_file, delete_after=self.delete_zip_after_extract)
                
                return True
                
            except requests.exceptions.HTTPError as e:
                # Handle authentication errors (401) - refresh token and retry
                if e.response.status_code == 401:
                    if attempt < max_retries - 1:
                        print(f"\n⚠ 401 UNAUTHORIZED: Token expired during download")
                        print(f"  Re-authenticating and retrying... (attempt {attempt + 2}/{max_retries})")
                        
                        # Force re-authentication
                        if self.authenticate(force=True):
                            time.sleep(2)  # Brief pause before retry
                            continue
                        else:
                            print(f"\n✗ Re-authentication failed")
                            return False
                    else:
                        print(f"\n✗ Authentication failed after {max_retries} attempts")
                        return False
                
                # Handle rate limiting (429) specially
                elif e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Aggressive backoff for rate limits: 30, 60, 120, 240 seconds
                        wait_time = 30 * (2 ** attempt)
                        print(f"\n⚠ RATE LIMITED (429): Server says too many requests")
                        print(f"  This happens with parallel downloads")
                        print(f"  Waiting {wait_time} seconds before retry... (attempt {attempt + 2}/{max_retries})")
                        
                        # Notify parallel workers to pause if callback provided
                        if rate_limit_pause:
                            rate_limit_pause(wait_time)
                        
                        time.sleep(wait_time)
                    else:
                        print(f"\n✗ Rate limited after {max_retries} attempts")
                        print(f"  💡 TIP: Reduce --workers or switch to sequential mode")
                        if temp_file.exists():
                            print(f"  Partial download saved, will resume next time")
                        return False
                else:
                    # Other HTTP errors
                    if attempt < max_retries - 1:
                        wait_time = 5 * (2 ** attempt)
                        print(f"\n⚠ HTTP Error {e.response.status_code}: {e}")
                        print(f"  Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"\n✗ HTTP error after {max_retries} attempts: {e}")
                        if temp_file.exists():
                            print(f"  Partial download saved")
                        return False
            
            except (requests.exceptions.RequestException, 
                    ConnectionResetError, 
                    ConnectionError) as e:
                # Handle connection errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"\n⚠ Download interrupted: {e}")
                    print(f"  Retrying in {wait_time} seconds... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"\n✗ Download failed after {max_retries} attempts: {e}")
                    if temp_file.exists():
                        print(f"  Partial download saved: {temp_file}")
                        print(f"  Run again to resume from {temp_file.stat().st_size/1e6:.1f} MB")
                    return False
                    
            except Exception as e:
                print(f"\n✗ Unexpected error: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                return False
        
        return False
    
    def extract_product(self, zip_path, delete_after=False):
        """
        Extract .SEN3 directory from zip file
        
        Parameters:
        -----------
        zip_path : Path
            Path to zip file
        delete_after : bool
            Delete zip file after successful extraction (default: False)
        """
        try:
            extract_dir = self.output_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✓ Extracted to: {extract_dir}")
            
            # Delete zip after successful extraction if requested
            if delete_after:
                try:
                    size_mb = zip_path.stat().st_size / 1e6 if zip_path.exists() else 0
                    zip_path.unlink()
                    print(f"✓ Deleted zip file (saved {size_mb:.1f} MB disk space)")
                except Exception as e:
                    print(f"⚠ Could not delete zip file: {e}")
            
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
    
    def display_product_info(self, products):
        """Display information about found products"""
        print("\n" + "="*80)
        print("AVAILABLE PRODUCTS:")
        print("="*80)
        
        for i, prod in enumerate(products, 1):
            name = prod.get('Name', 'N/A')
            date = prod.get('ContentDate', {}).get('Start', 'N/A')
            size_mb = prod.get('ContentLength', 0) / (1024*1024)
            
            print(f"\n{i}. {name}")
            print(f"   Date: {date}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   ID: {prod.get('Id', 'N/A')}")
    
    def download_all(self, products, max_products=None, parallel=False, max_workers=2, extract=True):
        """
        Download multiple products with optional parallel downloading
        
        Parameters:
        -----------
        products : list
            List of product dictionaries from search_products()
        max_products : int, optional
            Maximum number to download (uses config if not provided)
        parallel : bool
            Enable parallel downloads (default: False for better stability)
        max_workers : int
            Number of parallel downloads (recommended: 2-3, default: 2)
            Higher values WILL trigger rate limiting (429 errors)
        extract : bool
            Whether to extract .SEN3 directories from zip files (default: True)
        """
        
        if max_products is None:
            max_products = self.max_products
        
        products_to_download = products[:max_products]
        
        print("\n" + "="*80)
        print(f"DOWNLOADING {len(products_to_download)} PRODUCTS")
        if parallel:
            print(f"Mode: PARALLEL ({max_workers} workers)")
            print("⚠ Parallel downloads may trigger rate limiting")
            print("⚠ If you see 429 errors, reduce workers or use sequential")
        else:
            print(f"Mode: SEQUENTIAL (recommended - most stable)")
        print("="*80)
        
        success_count = 0
        failed_products = []
        
        if parallel:
            # Parallel downloads using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Shared flag for rate limiting coordination
            rate_limited = {'time': 0}
            
            def rate_limit_callback(wait_time):
                """Callback to coordinate all workers when rate limited"""
                rate_limited['time'] = max(rate_limited['time'], wait_time)
            
            def download_with_index(index_product):
                """Wrapper to download with index for progress tracking"""
                i, product = index_product
                product_id = product['Id']
                product_name = product['Name']
                
                # Check if we're currently rate limited
                if rate_limited['time'] > 0:
                    print(f"\n[{i}] Waiting due to rate limit...")
                    time.sleep(rate_limited['time'])
                    rate_limited['time'] = 0
                
                print(f"\n[{i}/{len(products_to_download)}] Starting: {product_name[:50]}...")
                success = self.download_product(product_id, product_name, extract=extract, rate_limit_pause=rate_limit_callback)
                return (i, product_name, success)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all downloads with delays to avoid rate limiting
                futures = []
                for i, product in enumerate(products_to_download, 1):
                    # Space out submissions: 2-5 seconds between each
                    if i > 1:
                        delay = 3  # 3 second delay between starting new downloads
                        print(f"  Spacing requests ({delay}s delay)...", end='\r')
                        time.sleep(delay)
                    
                    future = executor.submit(download_with_index, (i, product))
                    futures.append(future)
                
                # Process completed downloads
                for future in as_completed(futures):
                    i, product_name, success = future.result()
                    if success:
                        success_count += 1
                        print(f"✓ [{i}/{len(products_to_download)}] Complete: {product_name[:50]}")
                    else:
                        failed_products.append(product_name)
                        print(f"✗ [{i}/{len(products_to_download)}] Failed: {product_name[:50]}")
        
        else:
            # Sequential downloads (most stable, recommended)
            for i, product in enumerate(products_to_download, 1):
                print(f"\n[{i}/{len(products_to_download)}]")
                product_id = product['Id']
                product_name = product['Name']
                
                if self.download_product(product_id, product_name, extract=extract):
                    success_count += 1
                else:
                    failed_products.append(product_name)
                
                # Small delay between downloads to be nice to the server
                if i < len(products_to_download):
                    time.sleep(1)
        
        print("\n" + "="*80)
        print(f"✓ DOWNLOAD COMPLETE: {success_count}/{len(products_to_download)} successful")
        if failed_products:
            print(f"\n⚠ {len(failed_products)} FAILED DOWNLOADS:")
            for name in failed_products[:5]:
                print(f"  - {name}")
            if len(failed_products) > 5:
                print(f"  ... and {len(failed_products) - 5} more")
            print("\n💡 TIP: Run the script again to retry failed downloads")
            print("        Partial downloads will resume automatically")
            if parallel:
                print("💡 TIP: If seeing many rate limit errors, try:")
                print("        - Reduce workers: --workers 2 (or 1)")
                print("        - Or use sequential: remove --parallel flag")
        print("="*80)
        print(f"Data saved to: {self.output_dir}")
        if extract:
            print(f"Extracted to: {self.output_dir / 'extracted'}")
        
        return success_count, failed_products


def main():
    """Main download workflow"""
    
    parser = argparse.ArgumentParser(
        description='Download Sentinel-3 OLCI data for atmospheric correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential download (recommended - no 401 errors)
  python datasets/1-s3_olci_downaloder.py --no-extract --parallel --workers 2 --output-dir datasets/GULF_OF_MAINE/raw_data/sentinel_3 --start-date 2018-01-01 --end-date 2025-12-31
  
  # Parallel downloads (faster but may have connection issues)
  python s3_olci_downloader_fixed.py --parallel --workers 2 --output-dir datasets/GULF_OF_MAINE/raw_data/sentinel_3
  
  # Use custom config file
  python s3_olci_downloader_fixed.py --config my_region.yaml
  
  # Override date range from command line
  python s3_olci_downloader_fixed.py --start-date 2017-01-01 --end-date 2020-12-31
  
  # Download without extracting (keep zips only)
  python s3_olci_downloader_fixed.py --no-extract
  
  # Download only 5 products
  python s3_olci_downloader_fixed.py --max-products 5
  
  # Search only (don't download)
  python s3_olci_downloader_fixed.py --search-only
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='datasets/GOM.yaml',
        help='Path to configuration file (default: datasets/GOM.yaml)'
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
        '--max-products',
        type=int,
        help='Maximum number of products to download, overrides config'
    )
    
    parser.add_argument(
        '--search-only',
        action='store_true',
        help='Only search for products, do not download'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel downloads (faster but may trigger rate limiting)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of parallel download workers (default: 2, max recommended: 3)'
    )
    
    parser.add_argument(
        '--delete-after-extract',
        action='store_true',
        help='Delete zip files after successful extraction (saves disk space)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for downloads (overrides config)'
    )
    
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Do not extract .SEN3 directories from zip files (keep zips only)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"✗ Config file not found: {args.config}")
        print("\n  Create a config file or use --config to specify a different file")
        print("  See s3_olci_config.yaml for an example")
        return
    
    # Initialize downloader
    downloader = S3OLCIDownloader(config_path=args.config)
    
    # Override output directory if specified on command line
    if args.output_dir:
        downloader.output_dir = Path(args.output_dir)
        downloader.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"✓ Using custom output directory: {downloader.output_dir}")
    
    # Override delete setting if specified on command line
    if args.delete_after_extract:
        downloader.delete_zip_after_extract = True
        print(f"✓ Zip files will be deleted after successful extraction")
    
    # Notify if extraction is disabled
    if args.no_extract:
        print(f"✓ Extraction disabled - zip files will be kept intact")
    
    # Check credentials
    if downloader.username == "YOUR_USERNAME" or downloader.password == "YOUR_PASSWORD":
        print("\n⚠ Copernicus Data Space credentials not configured!")
        print("="*80)
        print("Please update the config file with your credentials:")
        print("  1. Register at: https://dataspace.copernicus.eu/")
        print(f"  2. Edit {args.config}")
        print("  3. Update 'username' and 'password' under download.credentials")
        print("="*80)
        return
    
    # Authenticate
    if not downloader.authenticate():
        print("\n⚠ Authentication failed")
        print("  Please check your credentials in the config file")
        return
    
    # Search for products
    products = downloader.search_products(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if not products:
        print("\n⚠ No products found")
        print("  Try adjusting the date range in the config file")
        return
    
    # Display available products
    downloader.display_product_info(products)
    
    # Download or exit
    if args.search_only:
        print("\n⚠ Search-only mode: No downloads performed")
        print(f"  Run without --search-only to download products")
        return
    
    # Download products
    downloader.download_all(
        products, 
        max_products=args.max_products,
        parallel=args.parallel,
        max_workers=args.workers,
        extract=not args.no_extract
    )
    
    print("\n" + "="*80)
    print("NEXT STEP: Process with ACOLITE")
    print("="*80)
    print("  python acolite_processor.py")


if __name__ == "__main__":
    main()