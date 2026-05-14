import requests
import pandas as pd

# Configuration
BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
START_YEAR = 2018
END_YEAR = 2025

# WKT Polygons for your regions
regions = {
    "region1": "POLYGON((-66 45, -60 45, -60 40, -66 40, -66 45))",
    "region2": "POLYGON((-76 45, -66 45, -66 36, -76 36, -76 45))"
}

def get_sentinel3_efr_products(poly_wkt, start_date, end_date):
    """Fetches S3 EFR product IDs for a specific region and time window."""
    product_data = {}
    
    # Combined Filter: Collection + ProductType + Area + Time
    # Note: Using ContentDate/Start is recommended for best performance
    filter_query = (
        "Collection/Name eq 'SENTINEL-3' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'OL_1_EFR___') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly_wkt}') and "
        f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date}"
    )
    
    # We select Id and Name to keep the payload small
    params = {
        "$filter": filter_query,
        "$select": "Id,Name",
        "$top": 1000
    }
    
    url = BASE_URL
    while url:
        response = requests.get(url, params=params if url == BASE_URL else None)
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            break
            
        data = response.json()
        for product in data['value']:
            # Use ID as key, Name as value
            product_data[product['Id']] = product['Name']
        
        # Check for pagination
        url = data.get('@OData.nextLink')
        params = None # Parameters are included in the nextLink URL
        
    return product_data

results_reg1 = {}
results_reg2 = {}

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Searching {year}...")
    start_dt = f"{year}-01-01T00:00:00.000Z"
    end_dt = f"{year}-12-31T23:59:59.000Z"
    
    results_reg1.update(get_sentinel3_efr_products(regions["region1"], start_dt, end_dt))
    results_reg2.update(get_sentinel3_efr_products(regions["region2"], start_dt, end_dt))

# Identify Non-Overlapping IDs using Set Symmetric Difference
ids_reg1 = set(results_reg1.keys())
ids_reg2 = set(results_reg2.keys())
non_overlapping_ids = ids_reg1 ^ ids_reg2

# Combine back into a list with Names for the final output
final_list = []
for p_id in non_overlapping_ids:
    name = results_reg1.get(p_id) or results_reg2.get(p_id)
    final_list.append({"Id": p_id, "Name": name})

# Export to CSV
df = pd.DataFrame(final_list)
df.to_csv("s3_efr_non_overlapping.csv", index=False)

print(f"Process Complete.")
print(f"Region 1 Total: {len(ids_reg1)} | Region 2 Total: {len(ids_reg2)}")
print(f"Unique Non-Overlapping Files: {len(final_list)}")