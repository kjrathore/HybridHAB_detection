"""
Mosaic C2RCC Outputs with Temporal Compositing
Creates mosaics from multiple dates within configurable temporal windows (e.g., 14-day composites)
"""

import os
import glob
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

def find_gpt():
    """Auto-detect SNAP GPT"""
    try:
        result = subprocess.run(['which', 'gpt'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'gpt'
    except:
        pass
    
    paths = ['/usr/local/snap/bin/gpt', '/opt/snap/bin/gpt', 
             os.path.expanduser('~/snap/bin/gpt'), '/Applications/snap/bin/gpt']
    
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("SNAP GPT not found")

def extract_date(filename):
    """Extract date from Sentinel-2 filename"""
    match = re.search(r'_(\d{8})T\d{6}_', filename)
    if match:
        return match.group(1)
    return None

def parse_date(date_str):
    """Convert YYYYMMDD string to datetime object"""
    return datetime.strptime(date_str, '%Y%m%d')

def group_by_temporal_window(products, window_days=14):
    """
    Group products into temporal windows
    
    Args:
        products: List of product file paths
        window_days: Size of temporal window in days (default: 14)
    
    Returns:
        Dictionary with window identifiers as keys and list of products as values
    """
    date_product_map = {}
    for prod in products:
        date_str = extract_date(os.path.basename(prod))
        if date_str:
            date_obj = parse_date(date_str)
            date_product_map[date_obj] = date_product_map.get(date_obj, []) + [prod]
    
    if not date_product_map:
        return {}
    
    sorted_dates = sorted(date_product_map.keys())
    temporal_groups = {}
    
    window_start = sorted_dates[0]
    window_end = window_start + timedelta(days=window_days - 1)
    window_id = f"{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
    current_group = []
    
    for date in sorted_dates:
        if date <= window_end:
            current_group.extend(date_product_map[date])
        else:
            if current_group:
                temporal_groups[window_id] = current_group
            
            window_start = date
            window_end = window_start + timedelta(days=window_days - 1)
            window_id = f"{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
            current_group = date_product_map[date][:]
    
    if current_group:
        temporal_groups[window_id] = current_group
    
    return temporal_groups

def create_mosaic_graph(output_path, num_products, output_format='NetCDF4-CF'):
    """
    Create multi-size mosaic graph with overlay method
    Uses parameters that replicate successful SNAP GUI processing
    """
    
    # Build read nodes
    read_nodes = []
    source_products = []
    for i in range(1, num_products + 1):
        read_nodes.append(f"""  <node id="Read{i}">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${{input_file{i}}}</file>
    </parameters>
  </node>""")
        source_products.append(f"      <sourceProduct.{i} refid=\"Read{i}\"/>")

    # Variables: RRS bands + C2RCC flags
    variables_list = [
        'rrs_B1', 'rrs_B2', 'rrs_B3', 'rrs_B4', 
        'rrs_B5', 'rrs_B6', 'rrs_B7', 'rrs_B8A', 'c2rcc_flags'
    ]
    
    var_xml = []
    for var in variables_list:
        var_xml.append(f"""        <variable>
          <name>{var}</name>
          <expression>{var}</expression>
        </variable>""")

    # CRS definition (WGS84)
    crs_wkt = """GEOGCS[&quot;WGS84(DD)&quot;, 
      DATUM[&quot;WGS84&quot;, 
        SPHEROID[&quot;WGS84&quot;, 6378137.0, 298.257223563]], 
      PRIMEM[&quot;Greenwich&quot;, 0.0], 
      UNIT[&quot;degree&quot;, 0.017453292519943295], 
      AXIS[&quot;Geodetic longitude&quot;, EAST], 
      AXIS[&quot;Geodetic latitude&quot;, NORTH], 
      AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]"""

    # Construct graph with overlay method for temporal compositing
    graph = f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="TemporalMosaic">
  <version>1.0</version>
  
{chr(10).join(read_nodes)}
  
  <node id="Multi-size Mosaic">
    <operator>Multi-size Mosaic</operator>
    <sources>
{chr(10).join(source_products)}
    </sources>
    <parameters>
        <variables>
{chr(10).join(var_xml)}
        </variables>
        <combine>OR</combine>
        <crs>{crs_wkt}</crs>
        <orthorectify>false</orthorectify>
        <elevationModelName>GETASSE30</elevationModelName>
        <westBound>-74.17</westBound>
        <northBound>-42.55</northBound>
        <eastBound>-72.84</eastBound>
        <southBound>-44.25</southBound>
        <pixelSizeX>5.39E-4</pixelSizeX>
        <pixelSizeY>5.39E-4</pixelSizeY>
        <resampling>Nearest</resampling>
        <nativeResolution>true</nativeResolution>
        <overlappingMethod>MOSAIC_TYPE_OVERLAY</overlappingMethod>
    </parameters>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Multi-size Mosaic"/>
    </sources>
    <parameters>
      <file>${{output_file}}</file>
      <formatName>{output_format}</formatName>
    </parameters>
  </node>
</graph>
"""
    with open(output_path, 'w') as f:
        f.write(graph)
    return output_path

def mosaic_products(window_id, products, output_dir, gpt_path, cache, threads, 
                    show_output, output_format):
    """Mosaic products for a temporal window"""
    
    ext = '.nc' if 'NetCDF' in output_format else '.h5' if output_format == 'HDF5' else '.dim'
    output_file = os.path.join(output_dir, f"Mosaic_{window_id}{ext}")
    
    if os.path.exists(output_file):
        print(f"  ✓ Already exists: {window_id}")
        return {'status': 'exists', 'window': window_id, 'output': output_file}
    
    print(f"  🔄 Mosaicing {len(products)} products for window {window_id}")
    sys.stdout.flush()
    
    # Create graph
    graph_xml = os.path.join(output_dir, f"mosaic_graph_{len(products)}.xml")
    create_mosaic_graph(graph_xml, len(products), output_format)
    
    # Build command
    cmd = [gpt_path, graph_xml]
    for i, prod in enumerate(products, 1):
        cmd.append(f"-Pinput_file{i}={prod}")
    cmd.extend([f"-Poutput_file={output_file}", '-c', cache, '-q', str(threads)])
    
    try:
        start = datetime.now()
        
        if show_output:
            print(f"    GPT Output:")
            sys.stdout.flush()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                if line.strip():
                    print(f"    {line.rstrip()}")
                    sys.stdout.flush()
            process.wait()
            returncode = process.returncode
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            returncode = result.returncode
            if returncode != 0:
                print(f"    Error: {result.stderr}")
        
        duration = (datetime.now() - start).total_seconds() / 60
        
        if returncode == 0 and os.path.exists(output_file):
            print(f"  ✓ Success: {window_id} ({duration:.1f} min)")
            return {'status': 'success', 'window': window_id, 'output': output_file, 'duration': duration}
        else:
            print(f"  ❌ Failed: {window_id}")
            return {'status': 'failed', 'window': window_id, 'output': None}
    
    except Exception as e:
        print(f"  ❌ Error: {window_id} - {str(e)}")
        return {'status': 'failed', 'window': window_id, 'output': None, 'error': str(e)}

def create_temporal_mosaics(input_dir, output_dir, temporal_window_days=14, 
                           output_format='NetCDF4-CF', cache='24G', threads=6, 
                           show_output=True, test_mode=False):
    """
    Create temporal composite mosaics
    
    Args:
        input_dir: Directory containing C2RCC products
        output_dir: Directory for mosaic outputs
        temporal_window_days: Size of temporal window in days (default: 14)
        output_format: Output format (NetCDF4-CF, HDF5, or BEAM-DIMAP)
        cache: Memory cache for GPT
        threads: Number of parallel threads
        show_output: Show GPT processing output
        test_mode: Process only first window for testing
    """
    
    print("="*80)
    print("TEMPORAL COMPOSITE MOSAIC CREATION")
    print("="*80)
    print(f"Temporal window: {temporal_window_days} days")
    print(f"Output format: {output_format}")
    print("="*80)
    
    gpt = find_gpt()
    os.makedirs(output_dir, exist_ok=True)
    
    # Find products
    products = glob.glob(os.path.join(input_dir, '*_C2RCC.dim'))
    
    if not products:
        print(f"No C2RCC products found in {input_dir}")
        return []
    
    print(f"Reading format: {Path(products[0]).suffix}")
    
    # Group by temporal windows
    temporal_groups = group_by_temporal_window(products, temporal_window_days)
    
    print(f"Found {len(products)} products")
    print(f"Grouped into {len(temporal_groups)} temporal windows ({temporal_window_days}-day)")
    print(f"Cache: {cache} | Threads: {threads}")
    
    if test_mode:
        print("\n⚠️  TEST MODE: Processing only first window")
    
    print("="*80 + "\n")
    
    results = {'success': [], 'failed': [], 'exists': []}
    csv_path = os.path.join(output_dir, f"mosaic_log_{temporal_window_days}day.csv")
    
    windows_to_process = sorted(temporal_groups.items())
    if test_mode:
        windows_to_process = windows_to_process[:1]
    
    # Initialize or load CSV
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
        columns=['window_id', 'num_products', 'status', 'output', 'duration_min', 
                 'temporal_window_days', 'timestamp'])
    
    for i, (window_id, prods) in enumerate(windows_to_process, 1):
        print(f"[{i}/{len(windows_to_process)}] Window: {window_id} ({len(prods)} products)")
        
        result = mosaic_products(window_id, prods, output_dir, gpt, cache, threads, 
                                show_output, output_format)
        results[result['status']].append(result)
        
        # Update CSV log
        record_data = {
            'window_id': window_id,
            'num_products': len(prods),
            'status': result['status'],
            'output': result.get('output', ''),
            'duration_min': result.get('duration', ''),
            'temporal_window_days': temporal_window_days,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        df = pd.concat([df, pd.DataFrame([record_data])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        if test_mode:
            break
    
    print(f"\n{'='*80}")
    print(f"✓ Success: {len(results['success'])} | Existed: {len(results['exists'])} | Failed: {len(results['failed'])}")
    print(f"📊 Log: {csv_path}")
    print(f"📁 Output: {output_dir}")
    print("="*80)
    
    return [r['output'] for r in results['success'] + results['exists']]


if __name__ == "__main__":
    
    # CONFIGURATION
    INPUT_DIR = 'sentinel2_data/c2rcc'
    OUTPUT_DIR = 'sentinel2_data/mosaics_14day'
    
    TEMPORAL_WINDOW_DAYS = 14  # Size of temporal composite window
    OUTPUT_FORMAT = 'NetCDF4-CF'
    CACHE = '24G'
    THREADS = 6
    SHOW_OUTPUT = True
    TEST_MODE = False
    
    print("\n" + "="*80)
    print("TEMPORAL COMPOSITE MOSAIC SCRIPT")
    print("="*80)
    print(f"\nCreating {TEMPORAL_WINDOW_DAYS}-day composite mosaics")
    print("- Combines multiple dates within temporal windows")
    print("- Uses overlay method for handling overlaps")
    print("- Native resolution preservation")
    print("="*80 + "\n")
    
    # RUN
    mosaics = create_temporal_mosaics(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        temporal_window_days=TEMPORAL_WINDOW_DAYS,
        output_format=OUTPUT_FORMAT,
        cache=CACHE,
        threads=THREADS,
        show_output=SHOW_OUTPUT,
        test_mode=TEST_MODE
    )
    
    if mosaics:
        print(f"\n✓ Created {len(mosaics)} temporal composite mosaics")
        print("\nTo verify mosaics:")
        print(f"  ncdump -h {mosaics[0]}")
        print("  # Or open in SNAP GUI")
    else:
        print("\n⚠️  No mosaics created - check errors above")