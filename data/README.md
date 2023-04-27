`fuel_moisture/single_file_labels.geojson` is an aggregation of all the geojsons in the fuel moisture dataset, stored in a single file for ease of access.

The script used to generate it is:

```python
import geopandas
import pandas as pd
from pathlib import Path

geojsons = list(Path("fuel_moisture").glob("*/*.geojson"))
labels = pd.concat([geopandas.read_file(gjs) for gjs in geojsons])
labels["date"] = pd.to_datetime(labels["date"])
labels[["percent(t)","site","date","geometry"]].to_file("single_file_labels.geojson", driver="GeoJSON")
```
