"""
This file is required to be called main.py for Google Cloud Function.
"""
import tempfile
from pathlib import Path

from pipelines.dynamicworld import DynamicWorld2020_2021, DynamicWorldMonthly2020_2021
from pipelines.ee_pipeline import EE_BUCKET

# https://cloud.google.com/functions/docs/writing/specifying-dependencies-python#packaging_local_dependencies
from pipelines.s1_s2_era5_srtm import S1_S2_ERA5_SRTM_2020_2021
from pipelines.worldcover2020 import WorldCover2020

tempdir = tempfile.gettempdir()
pipelines = [
    S1_S2_ERA5_SRTM_2020_2021(),
    WorldCover2020(),
    DynamicWorld2020_2021(),
    DynamicWorldMonthly2020_2021(),
]
pipelines_dict = {p.name: p for p in pipelines}


def trigger_tif_to_np(event, context):
    """
    This Google Cloud Function is triggered by a new geotiff file in gs://<EE_BUCKET>.
    It downloads the file, converts it to a numpy array, and uploads it to gs://<NPY-BUCKET>.
    """
    bucket = event["bucket"]
    name = event["name"]

    # Replace "retry" accomodates retry_tif_to_np_pipeline()
    # in scripts/retrigger_tif_to_np.py
    pipeline_name = Path(name).parent.stem.replace("-retry", "")
    try:
        pipeline = pipelines_dict[pipeline_name]
    except KeyError:
        raise ValueError(f"Unknown pipeline name: {pipeline_name}")

    if bucket != EE_BUCKET:
        raise ValueError(f"Expected {EE_BUCKET} but got {bucket}")

    pipeline.convert_tif_to_np_on_cloud(name)


if __name__ == "__main__":

    # Test: satellite data tif -> np
    trigger_tif_to_np(
        {
            "bucket": EE_BUCKET,
            "name": "test/S1_S2_ERA5_SRTM_2020_2021/0000000000-0000000000.tif",
        },
        None,
    )
    # Test: WorldCover 2020 tif -> np
    trigger_tif_to_np(
        {
            "bucket": EE_BUCKET,
            "name": "test/WorldCover2020/0000000000-0000000000.tif",
        },
        None,
    )

    # Test: Dynamic World 2020 tif -> np
    trigger_tif_to_np(
        {
            "bucket": EE_BUCKET,
            "name": "test/DynamicWorld2020_2021/0000000000-0000000000.tif",
        },
        None,
    )
