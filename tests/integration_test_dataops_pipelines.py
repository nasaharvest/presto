import os
import tempfile
from unittest import TestCase

import ee
import numpy as np
import rioxarray
import xarray as xr
from openmapflow.ee_boundingbox import EEBoundingBox

from presto.dataops import (
    EE_BUCKET,
    NPY_BUCKET,
    S1_S2_ERA5_SRTM_2020_2021,
    DynamicWorld2020_2021,
    DynamicWorldMonthly2020_2021,
    EEPipeline,
    WorldCover2020,
    gcloud_download,
)

os.environ["GOOGLE_CLOUD_PROJECT"] = "bsos-geog-harvest1"

temp_dir = tempfile.gettempdir()

name, stem = "test", "0000000000-0000000000"


def get_pipelines():
    return [
        p()
        for p in (
            DynamicWorld2020_2021,
            S1_S2_ERA5_SRTM_2020_2021,
            WorldCover2020,
            DynamicWorldMonthly2020_2021,
        )
    ]


def gcloud_download_tif(pipeline: EEPipeline):
    """
    Downloads a tif file from Google Cloud Storage, retrying by updating
    the pipeline folder and file name if not found.
    """
    pipeline_name = pipeline.name
    for _ in range(3):
        stem_name = stem
        for _ in range(4):
            try:
                filename = f"{name}/{pipeline_name}/{stem_name}.tif"
                return gcloud_download(EE_BUCKET, filename)
            except Exception as e:
                print(e)
                stem_name += "-retry"
        pipeline_name += "-retry"

    raise FileNotFoundError(f"Could not find {filename} in Google Cloud Storage.")


def gcloud_download_npy(pipeline: EEPipeline):
    return gcloud_download(NPY_BUCKET, f"{name}/{pipeline.name}/{stem}.npy")


def load_example_tif_file(p: EEPipeline):
    path = gcloud_download_tif(p)
    data = rioxarray.open_rasterio(path)
    if isinstance(data, xr.DataArray):
        print(f"{p.name} shape: {data.shape}")
    path.unlink()
    return data


def load_example_npy_file(p: EEPipeline):
    path = gcloud_download_npy(p)
    data = np.load(path)
    print(f"{p.name} shape: {data.shape}")
    path.unlink()
    return data


class TestRealPipelines(TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        self.san_diego_ee_bbox = EEBoundingBox(32.6, 32.9, -117.2, -116.7).to_ee_polygon()

    def test_create_ee_image(self):
        for p in get_pipelines():
            out = p.create_ee_image(self.san_diego_ee_bbox)
            self.assertIsInstance(out, ee.Image)

    def test_DynamicWorld_export(self):
        DynamicWorld2020_2021().run(self.san_diego_ee_bbox, "test", is_large_polygon=True)

    def test_DynamicWorldMonthly_export(self):
        DynamicWorldMonthly2020_2021().run(self.san_diego_ee_bbox, "test", is_large_polygon=True)

    def test_WorldCover_export(self):
        WorldCover2020().run(self.san_diego_ee_bbox, "test", is_large_polygon=True)

    def test_S1_S2_ERA5_STRM_export(self):
        S1_S2_ERA5_SRTM_2020_2021().run(self.san_diego_ee_bbox, "test", is_large_polygon=True)

    def test_tif_compatibility_across_pipelines(self):
        pipelines = get_pipelines()
        baseline_tif = load_example_tif_file(pipelines[0])

        for p in pipelines[1:]:
            another_tif = load_example_tif_file(p)
            self.assertEqual(
                baseline_tif.shape[-2:],
                another_tif.shape[-2:],
                f"Spatial dimension does not match for {p.name}",
            )
            self.assertTrue(np.allclose(baseline_tif.x.values, another_tif.x.values))
            self.assertTrue(np.allclose(baseline_tif.y.values, another_tif.y.values))

    def test_npy_compatibility_across_pipelines(self):
        """Checks compatibility of already process npy files"""
        pipelines = get_pipelines()
        baseline_npy = load_example_npy_file(pipelines[0])

        for p in pipelines[1:]:
            another_npy = load_example_npy_file(p)
            self.assertEqual(
                baseline_npy.shape[0],
                another_npy.shape[0],
                f"Spatial dimension does not match for {p.name}",
            )

    def test_convert_tif_to_np(self):
        """Checks convert tif to np gives consistent results"""
        pipelines = get_pipelines()
        p = pipelines[0]
        baseline_tif_path = gcloud_download_tif(p)
        baseline_npy, baseline_latlon = p.convert_tif_to_np(baseline_tif_path)

        for p in pipelines[1:]:
            another_tif_path = gcloud_download_tif(p)
            another_npy, another_latlon = p.convert_tif_to_np(another_tif_path)
            self.assertEqual(
                baseline_npy.shape[0],
                another_npy.shape[0],
                f"Spatial dimension does not match for {p.name}",
            )

            self.assertTrue(
                np.allclose(baseline_latlon, another_latlon), f"LatLons don't match for {p.name}"
            )

    def test_dynamic_world_monthly_has_24_timesteps(self):
        pipeline = DynamicWorldMonthly2020_2021()
        tif_path = gcloud_download_tif(pipeline)
        npy, _ = pipeline.convert_tif_to_np(tif_path)
        self.assertEqual(npy.shape[1], 24, "Incorrect number of timesteps")
