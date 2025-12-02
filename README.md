## To process Sentinel2 data for HAB-Alexandrium Catenella

# Prepare python env using requirements.txt
# activate the env

# Run to download data
> python3 1_download_data.py

# Implement C2RCC
> python3 2_C2RCC.py

# Run to create mosaic, can change the arguments here.
> python3 3.1_create_mosaic.py

