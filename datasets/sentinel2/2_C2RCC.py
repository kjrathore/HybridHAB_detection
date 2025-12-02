"""
C2RCC Atmospheric Correction for Sentinel-2
With land masking
Output: Multi-band format for QGIS
"""

import os
import glob
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import re

def create_c2rcc_graph(output_path, resolution, salinity, temperature, output_format):
    """Create C2RCC processing graph with land mask"""
    
    graph = f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="C2RCC_Processing">
  <version>1.0</version>
  
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${{input_file}}</file>
    </parameters>
  </node>
  
  <node id="Resample">
    <operator>S2Resampling</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters>
      <targetResolution>{resolution}</targetResolution>
      <upsampling>Bilinear</upsampling>
      <downsampling>Mean</downsampling>
      <flagDownsampling>FlagMedianAnd</flagDownsampling>
      <resampleOnPyramidLevels>false</resampleOnPyramidLevels>
    </parameters>
  </node>
  
  <node id="C2RCC">
    <operator>C2RCC.MSI</operator>
    <sources>
      <sourceProduct refid="Resample"/>
    </sources>
    <parameters>
      <salinity>{salinity}</salinity>
      <temperature>{temperature}</temperature>
      <ozone>330.0</ozone>
      <press>1000.0</press>
      <outputAsRrs>true</outputAsRrs>
    </parameters>
  </node>
  
  <node id="LandMask">
    <operator>Land-Sea-Mask</operator>
    <sources>
      <sourceProduct refid="C2RCC"/>
    </sources>
    <parameters>
      <useSRTM>true</useSRTM>
      <landMask>true</landMask>
      <shorelineExtension>0</shorelineExtension>
    </parameters>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="LandMask"/>
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

def get_file_extension(output_format):
    """Get file extension based on output format"""
    format_extensions = {
        'GeoTIFF-BigTIFF': '.tif',
        'GeoTIFF': '.tif',
        'NetCDF-CF': '.nc',
        'NetCDF4-CF': '.nc',
        'BEAM-DIMAP': '.dim',
        'HDF5': '.h5'
    }
    return format_extensions.get(output_format, '.tif')

def process_product(input_path, output_dir, graph_xml, cache, threads, output_format):
    """Process single product"""
    product_name = Path(input_path).stem.replace('.SAFE', '')
    file_ext = get_file_extension(output_format)
    output_file = os.path.join(output_dir, f"{product_name}_C2RCC{file_ext}")
    
    if os.path.exists(output_file):
        print(f"  ✓ Already processed: {product_name}")
        return {'status': 'exists', 'product_name': product_name, 'output_path': output_file}
    
    print(f"  🔄 Processing: {product_name}")
    print(f"     Started: {datetime.now().strftime('%H:%M:%S')}")
    sys.stdout.flush()
    
    cmd = ['gpt', graph_xml, f"-Pinput_file={input_path}", 
           f"-Poutput_file={output_file}", '-c', cache, '-q', str(threads)]
    
    try:
        start = datetime.now()
        
        print(f"     GPT Output:")
        sys.stdout.flush()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1)
        for line in process.stdout:
            if line.strip():
                print(f"     {line.rstrip()}")
                sys.stdout.flush()
        process.wait()
        returncode = process.returncode
        
        duration = (datetime.now() - start).total_seconds() / 60
        
        if returncode == 0 and os.path.exists(output_file):
            print(f"  ✓ Success: {product_name} ({duration:.1f} min)")
            return {'status': 'success', 'product_name': product_name, 
                   'output_path': output_file, 'duration': duration}
        else:
            print(f"  ❌ Failed: {product_name}")
            return {'status': 'failed', 'product_name': product_name, 
                   'output_path': None, 'error': 'Processing failed'}
    
    except Exception as e:
        print(f"  ❌ Error: {product_name} - {str(e)}")
        return {'status': 'failed', 'product_name': product_name, 
               'output_path': None, 'error': str(e)}

def process_batch(input_dir, output_dir, resolution=30, salinity=35.0, temperature=18.0,
                 output_format='GeoTIFF-BigTIFF', cache='24G', threads=6):
    """Batch process Sentinel-2 products with land masking"""
    
    print("="*80)
    print("C2RCC BATCH PROCESSING")
    print("="*80)
    print(f"Resolution: {resolution}m | Salinity: {salinity} PSU | Temp: {temperature}°C")
    print(f"Cache: {cache} | Threads: {threads} | Format: {output_format}")
    print(f"Land Mask: Enabled")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    graph_xml = os.path.join(output_dir, "c2rcc_graph.xml")
    create_c2rcc_graph(graph_xml, resolution, salinity, temperature, output_format)
    
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.SAFE')))
    if not input_files:
        print(f"No .SAFE files found in {input_dir}")
        return []
    
    print(f"\nFound {len(input_files)} products\n")
    
    results = {'success': [], 'failed': [], 'exists': []}
    csv_path = os.path.join(output_dir, "processing_log.csv")
    
    for i, input_file in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}]")
        result = process_product(input_file, output_dir, graph_xml, cache, threads, output_format)
        results[result['status']].append(result)
        
        # Update CSV
        record = {
            'product_name': result['product_name'],
            'status': result['status'],
            'output_path': result.get('output_path', ''),
            'duration_min': result.get('duration', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': result.get('error', '')
        }
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Success: {len(results['success'])} | Existed: {len(results['exists'])} | Failed: {len(results['failed'])}")
    print(f"📊 Log: {csv_path}")
    print(f"📁 Output: {output_dir}")
    print("="*80)
    
    processed_files = [r['output_path'] for r in results['success'] + results['exists']]
    
    return processed_files


if __name__ == "__main__":
    
    # CONFIGURATION
    INPUT_DIR = './sentinel2_data/extracted'
    OUTPUT_DIR = './sentinel2_data/c2rcc'
    
    RESOLUTION = 60          # Resolution [10, 20, 60]
    SALINITY = 35.0          # Ocean water (PSU)
    TEMPERATURE = 18.0       # Ocean temperature (°C)
    
    # FORMAT OPTIONS (all work in QGIS with multiple bands):
    # - 'GeoTIFF-BigTIFF': Best for large multi-band files
    # - 'NetCDF4-CF': Excellent for time-series and multi-band
    # - 'BEAM-DIMAP': SNAP native format, preserves all metadata
    OUTPUT_FORMAT = 'BEAM-DIMAP'
    
    CACHE = '40G'            # Memory cache
    THREADS = 8              # Processing threads
    
    # RUN
    processed_files = process_batch(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        resolution=RESOLUTION,
        salinity=SALINITY,
        temperature=TEMPERATURE,
        output_format=OUTPUT_FORMAT,
        cache=CACHE,
        threads=THREADS
    )
    
    print(f"\n✓ Processed {len(processed_files)} products")
    if processed_files:
        print("\nProcessed files:")
        for file in processed_files:
            print(f"  - {os.path.basename(file)}")