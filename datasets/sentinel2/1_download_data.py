"""
Sentinel-2 Download with Auto Token Refresh
Fixed version that handles token expiration (10 min) and refresh (60 min)
"""

import requests
import os
import pandas as pd
from datetime import datetime, timedelta
import zipfile
from pathlib import Path
import time

class TokenManager:
    """Manages access token and automatic refresh"""
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
        self.refresh_expiry = None
        self.get_new_token()
    
    def get_new_token(self):
        """Get a fresh token pair (access + refresh)"""
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password"
        }
        
        try:
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            
            # Access token expires in 600 seconds (10 min), refresh early at 8 min
            self.token_expiry = datetime.now() + timedelta(seconds=480)
            # Refresh token expires in 3600 seconds (60 min), refresh early at 55 min
            self.refresh_expiry = datetime.now() + timedelta(seconds=3300)
            
            print(f"  ✓ New token obtained, expires at {self.token_expiry.strftime('%H:%M:%S')}")
            return True
            
        except Exception as e:
            print(f"  ❌ Token generation failed: {str(e)}")
            raise
    
    def refresh_access_token(self):
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            print("  ⚠️ No refresh token, getting new token...")
            return self.get_new_token()
        
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        try:
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token", self.refresh_token)
            self.token_expiry = datetime.now() + timedelta(seconds=480)  # 8 min buffer
            
            print(f"  ✓ Token refreshed, expires at {self.token_expiry.strftime('%H:%M:%S')}")
            return True
            
        except Exception as e:
            print(f"  ⚠️ Token refresh failed: {str(e)}, getting new token...")
            return self.get_new_token()
    
    def get_valid_token(self):
        """Get a valid token, refreshing if necessary"""
        now = datetime.now()
        
        # Check if refresh token is about to expire
        if self.refresh_expiry and now >= self.refresh_expiry:
            print("  🔄 Refresh token expiring, getting new token pair...")
            self.get_new_token()
        # Check if access token is about to expire
        elif self.token_expiry and now >= self.token_expiry:
            print("  🔄 Access token expiring, refreshing...")
            self.refresh_access_token()
        
        return self.access_token

def update_csv(csv_path, record):
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    if 'product_name' in df.columns and record['product_name'] in df['product_name'].values:
        idx = df[df['product_name'] == record['product_name']].index[0]
        for key, value in record.items():
            df.at[idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(csv_path, index=False)

def search_products_batch(bbox, start_date, end_date, max_cloud, token_manager, skip=0, top=1000, 
                         timeout=120, max_retries=3):
    """Search a single batch of products with retry logic"""
    min_lon, min_lat, max_lon, max_lat = bbox
    wkt = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    
    # Ensure dates are in correct format
    if isinstance(start_date, str) and len(start_date) == 8:
        start_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
    if isinstance(end_date, str) and len(end_date) == 8:
        end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
    
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    filter_query = (f"Collection/Name eq 'SENTINEL-2' and "
                   f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
                   f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
                   f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
                   f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud})")
    
    params = {
        "$filter": filter_query, 
        "$orderby": "ContentDate/Start asc",
        "$top": top,
        "$skip": skip
    }
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Get fresh token for each request
            token = token_manager.get_valid_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('value', []), data
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"      Timeout, retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ❌ Failed after {max_retries} attempts")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      Error: {str(e)}, retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
            else:
                raise

def search_products_chunked(bbox, start_date, end_date, max_cloud, token_manager, chunk_months=3):
    """Search products by breaking time range into smaller chunks"""
    # Convert to datetime
    if isinstance(start_date, str):
        if len(start_date) == 8:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = start_date
    
    if isinstance(end_date, str):
        if len(end_date) == 8:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_dt = end_date
    
    all_products = []
    current_start = start_dt
    
    print(f"  Breaking query into {chunk_months}-month chunks...")
    
    while current_start < end_dt:
        # Calculate chunk end
        chunk_end = min(current_start + timedelta(days=chunk_months*30), end_dt)
        
        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')
        
        print(f"  Chunk: {chunk_start_str} to {chunk_end_str}")
        
        # Search this chunk with pagination
        skip = 0
        top = 1000
        
        while True:
            print(f"    Batch: {skip} to {skip + top}...", end=" ")
            try:
                products, data = search_products_batch(
                    bbox, chunk_start_str, chunk_end_str, max_cloud, token_manager, 
                    skip=skip, top=top, timeout=120, max_retries=3
                )
                
                if not products:
                    print("(no more results)")
                    break
                
                all_products.extend(products)
                print(f"✓ {len(products)} products")
                skip += len(products)
                
                if '@odata.nextLink' not in data and len(products) < top:
                    break
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ Error: {str(e)}")
                break
        
        current_start = chunk_end + timedelta(days=1)
        time.sleep(1)
    
    print(f"  ✓ Total products found: {len(all_products)}")
    return all_products

def download_product(product_id, product_name, output_dir, token_manager, product_num, total_products):
    """Download product with automatic token refresh on 401 errors"""
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    output_path = os.path.join(output_dir, f"{product_name}.zip")
    
    if os.path.exists(output_path):
        print(f"  ✓ [{product_num}/{total_products}] Exists: {product_name}")
        return {'status': 'exists', 'path': output_path, 'name': product_name}
    
    print(f"  📥 [{product_num}/{total_products}] Downloading: {product_name}")
    
    # Retry with token refresh on failure
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Get valid token
            token = token_manager.get_valid_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.get(url, headers=headers, stream=True, timeout=300)
            
            # Handle redirects
            while response.status_code in (301, 302, 303, 307):
                response = requests.get(response.headers['Location'], headers=headers, stream=True, timeout=300)
            
            # If unauthorized, refresh token and retry
            if response.status_code == 401:
                if attempt < max_retries - 1:
                    print(f"    🔄 Token expired, refreshing... (attempt {attempt + 1}/{max_retries})")
                    token_manager.refresh_access_token()
                    time.sleep(2)
                    continue
                else:
                    raise requests.exceptions.HTTPError(f"401 Unauthorized after {max_retries} attempts")
            
            response.raise_for_status()
            
            # Download file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  ✓ [{product_num}/{total_products}] Downloaded: {product_name} ({size_mb:.1f} MB)")
            return {'status': 'success', 'path': output_path, 'name': product_name, 'size_mb': size_mb}
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    ⏳ Timeout, retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ [{product_num}/{total_products}] Failed: {product_name} - Timeout after {max_retries} attempts")
                return {'status': 'failed', 'path': None, 'name': product_name, 'error': 'Timeout'}
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️ Error: {str(e)}, retry {attempt + 1}/{max_retries}...")
                time.sleep(5)
            else:
                print(f"  ❌ [{product_num}/{total_products}] Failed: {product_name} - {str(e)}")
                return {'status': 'failed', 'path': None, 'name': product_name, 'error': str(e)}

def extract_zip(zip_path, extract_dir, product_num, total_products):
    product_name = Path(zip_path).stem
    extracted_path = os.path.join(extract_dir, product_name + '.SAFE')
    
    if os.path.exists(extracted_path):
        print(f"  ✓ [{product_num}/{total_products}] Already extracted: {product_name}")
        return {'status': 'exists', 'path': extracted_path, 'name': product_name}
    
    print(f"  📦 [{product_num}/{total_products}] Extracting: {product_name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"  ✓ [{product_num}/{total_products}] Extracted: {product_name}")
        return {'status': 'success', 'path': extracted_path, 'name': product_name}
    except Exception as e:
        print(f"  ❌ [{product_num}/{total_products}] Extraction failed: {product_name}")
        return {'status': 'failed', 'path': None, 'name': product_name, 'error': str(e)}

def download_sentinel2(bbox, start_date, end_date, output_dir, username, password, 
                      max_cloud=30, max_products=None, extract=True, wait_time=2,
                      chunk_months=3):
    """
    Download and optionally extract Sentinel-2 data with automatic token management
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        start_date: 'YYYYMMDD' or 'YYYY-MM-DD'
        end_date: 'YYYYMMDD' or 'YYYY-MM-DD'
        output_dir: Output directory
        username: CDSE username
        password: CDSE password
        max_cloud: Max cloud coverage (0-100)
        max_products: Max number of products to download (None = all)
        extract: Extract zip files (default: True)
        wait_time: Wait time between downloads in seconds (default: 2)
        chunk_months: Split time range into chunks of this many months (default: 3)
    
    Returns:
        List of paths to extracted/downloaded files
    """
    
    print("="*80)
    print("Sentinel-2 Download with Auto Token Refresh")
    print("="*80)
    
    # Setup directories
    zip_dir = os.path.join(output_dir, "zipped")
    extract_dir = os.path.join(output_dir, "extracted")
    os.makedirs(zip_dir, exist_ok=True)
    if extract:
        os.makedirs(extract_dir, exist_ok=True)
    
    tracking_csv = os.path.join(output_dir, "download_tracking.csv")
    
    # Initialize token manager
    print("\n[1/4] Initializing token manager...")
    token_manager = TokenManager(username, password)
    
    # Search
    print(f"\n[2/4] Searching (bbox={bbox}, dates={start_date} to {end_date}, cloud<={max_cloud}%)...")
    products = search_products_chunked(bbox, start_date, end_date, max_cloud, token_manager, 
                                      chunk_months=chunk_months)
    print(f"✓ Found {len(products)} total products")
    
    if not products:
        print("⚠️ No products found")
        return []
    
    # Prepare products
    df = pd.DataFrame(products)
    if 'Attributes' in df.columns:
        df['CloudCover'] = df['Attributes'].apply(
            lambda x: next((a['Value'] for a in x if a['Name'] == 'cloudCover'), None))
        df['AcqDate'] = df['ContentDate'].apply(
            lambda x: x['Start'][:10] if isinstance(x, dict) else None)
        df = df.sort_values('CloudCover')
    
    df = df.drop_duplicates(subset=['Id'])
    print(f"  → After deduplication: {len(df)} unique products")
    
    # Select products to download
    if max_products is None:
        products_to_dl = df.to_dict('records')
        print(f"  → Processing ALL {len(products_to_dl)} products")
    else:
        products_to_dl = df.head(max_products).to_dict('records')
        print(f"  → Processing top {len(products_to_dl)} products")
    
    # Download (sequential with auto token refresh)
    print(f"\n[3/4] Downloading {len(products_to_dl)} products (auto token refresh enabled)...")
    dl_results = {'success': [], 'failed': [], 'exists': []}
    
    for i, p in enumerate(products_to_dl, 1):
        result = download_product(p['Id'], p['Name'][:-5], zip_dir, token_manager, i, len(products_to_dl))
        
        # Update CSV
        record = {
            'product_name': result['name'],
            'product_id': p['Id'],
            'acquisition_date': p.get('AcqDate', ''),
            'cloud_cover': p.get('CloudCover', ''),
            'download_status': result['status'],
            'download_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'extraction_status': 'pending' if extract else 'skipped',
            'extraction_timestamp': '',
            'zip_path': result.get('path', ''),
            'extracted_path': '',
            'size_mb': result.get('size_mb', ''),
            'error_message': result.get('error', '')
        }
        update_csv(tracking_csv, record)
        
        dl_results[result['status']].append(result)
        
        if i < len(products_to_dl):
            time.sleep(wait_time)
    
    print(f"✓ Downloaded: {len(dl_results['success'])}, Existed: {len(dl_results['exists'])}, Failed: {len(dl_results['failed'])}")
    
    # Extract
    ext_results = {'success': [], 'failed': [], 'exists': []}
    if extract:
        print(f"\n[4/4] Extracting {len(dl_results['success']) + len(dl_results['exists'])} products...")
        zip_files = [r['path'] for r in dl_results['success'] + dl_results['exists']]
        
        for i, zf in enumerate(zip_files, 1):
            result = extract_zip(zf, extract_dir, i, len(zip_files))
            
            record = {
                'product_name': result['name'],
                'extraction_status': result['status'],
                'extraction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'extracted_path': result.get('path', ''),
            }
            if result['status'] == 'failed':
                record['error_message'] = result.get('error', '')
            update_csv(tracking_csv, record)
            
            ext_results[result['status']].append(result)
            
            if i < len(zip_files):
                time.sleep(0.5)
        
        print(f"✓ Extracted: {len(ext_results['success'])}, Existed: {len(ext_results['exists'])}, Failed: {len(ext_results['failed'])}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Tracking: {tracking_csv}")
    print(f"📁 Zips: {zip_dir}")
    if extract:
        print(f"📁 Extracted: {extract_dir}")
    print(f"{'='*80}\n")
    
    if extract:
        return [r['path'] for r in ext_results['success'] + ext_results['exists']]
    else:
        return [r['path'] for r in dl_results['success'] + dl_results['exists']]


if __name__ == "__main__":
    # Example usage
    files = download_sentinel2(
        bbox=(-73.74, -43.92, -73.32, -43.15),
        start_date="20150101",
        end_date="20220131",
        output_dir="./sentinel2_data",
        username='rathorek@oregonstate.edu',
        password='Satellite@321',
        max_cloud=50,
        max_products=None,  # None = download ALL
        extract=True,
        wait_time=2,
        chunk_months=3
    )
    
    print(f"✓ Processed {len(files)} products")
    
    # View tracking
    csv_path = "./sentinel2_data/download_tracking.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n📊 Tracking Summary:")
        print(df[['product_name', 'acquisition_date', 'cloud_cover', 'download_status', 'extraction_status']].head(20))