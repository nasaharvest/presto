# Generates a csv file with the class pixel amount for each grid in ESA WorldCover map
# Requires: pip install geopandas rioxarray tqdm -q
# Usage: python3 generate_grid.py
# Output: esa_grid.csv
# Once outputted, the file can be uploaded and made public by running:
# gcloud storage cp esa_grid.csv gs://lem-assets
# gcloud storage acl ch -u AllUsers:R gs://lem-assets/esa_grid.csv

import geopandas as gpd
import numpy as np
import rioxarray
from tqdm import tqdm

legend = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Barren / sparse vegetation",
    70: "Snow and ice",
    80: "Open water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
url = f"{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson"
grid = gpd.read_file(url)
grid["color"] = "lightgray"
grid["class_0"] = 0
for k in legend.keys():
    grid[f"class_{k}"] = 0

for tile_i in tqdm(range(len(grid))):
    if grid.iloc[tile_i]["color"] == "Red":
        continue
    tile_name = grid.iloc[tile_i]["ll_tile"]
    url = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile_name}_Map.tif"
    keys, amounts = np.unique(rioxarray.open_rasterio(url, cache=False), return_counts=True)
    for i in range(len(keys)):
        grid.at[tile_i, f"class_{keys[i]}"] = amounts[i]
    grid.at[tile_i, "color"] = "Red"
    grid.to_csv("esa_grid.csv", index=False)
