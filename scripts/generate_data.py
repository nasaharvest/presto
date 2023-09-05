"""
An idempotent data pipeline for creating WebDataset tars from shapely Polygons.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from presto.dataops.dataset import (  # noqa: E402
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)

os.environ["GOOGLE_CLOUD_PROJECT"] = "presto"

# If uploads are failing, disable the default parallel_composite_uploads
# gcloud config set storage/parallel_composite_upload_enabled False

if __name__ == "__main__":
    # Shard 0 Export time Feb 16 11:14:46 - 19:47:16 (8.5 hours)
    # Shard 1-10 Export Feb 16 23:00:12 - Feb 18 07:10:20 (26 hours)

    # shard_ids_ee_gmail = list(range(20, 40))
    shard_ids_ee_umd = list(range(0, 60))
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021().create_webdataset_tars(
        shard_ids=shard_ids_ee_umd
    )
