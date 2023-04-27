import tempfile
from pathlib import Path
from typing import List, Tuple

import ee
import numpy as np
import xarray as xr
from google.cloud import storage
from openmapflow.ee_exporter import ee_safe_str, get_ee_task_list

# Changing these parameters will require re-running exports for all files
# with a new prefix to differentiate them.
FILE_DIMENSIONS = 256  # Ensures files fit into GCloud Function memory
METRES_PER_PATCH = 50000  # Ensures EarthEngine exports don't time out

# Changing this parameter will require retriggering the tif-to-np function
# for all files in the bucket
SAMPLE_EVERY = 10

EE_BUCKET = "lem-earthengine2"
NPY_BUCKET = "lem-npy2"

tempdir = tempfile.gettempdir()


def gcloud_download(bucket: str, name: str):
    blob = storage.Client().bucket(bucket).blob(name)
    path = Path(f"{tempdir}/{blob.name.replace('/', '_')}")
    print(f"Downloading file: {path}")
    blob.download_to_filename(path)
    return path


def resample_and_flatten(x_np, sample_every: int = SAMPLE_EVERY):
    """Resample and flatten a 3D or 4D array."""
    if len(x_np.shape) == 4:
        if sample_every is not None:
            x_np = x_np[:, :, sample_every::sample_every, sample_every::sample_every]
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1], x_np.shape[2] * x_np.shape[3])
    elif len(x_np.shape) == 3:
        if sample_every is not None:
            x_np = x_np[:, sample_every::sample_every, sample_every::sample_every]
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1] * x_np.shape[2])
    else:
        raise ValueError(f"Unexpected shape: {x_np.shape}")
    x_np = np.moveaxis(x_np, -1, 0)
    return x_np


def resample_and_flatten_tif(tif: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """Resamples and flattens a tif file to a numpy array and lat/lon array"""
    lons, lats = np.meshgrid(tif.coords["x"].values, tif.coords["y"].values, indexing="xy")
    lat_lons = np.moveaxis(np.stack([lats, lons], axis=2), -1, 0)
    return resample_and_flatten(tif.values), resample_and_flatten(lat_lons)


class EEPipeline:
    """
    Used for obtaining, organizing and using data from EarthEngine
    Works by:
    1) Creating an Earth Engine Image Collection (create_ee_image)
    2) Exporting a polygon from above collection to cloud (export_polygon_to_cloud)
    3) Converting exported polygon (tif file) to a numpy array (convert_tif_to_np_on_cloud)
    """

    def __init__(self):
        self.name = self.__class__.__name__
        if self.name == "EEPipeline":
            print("Warning: EEPipeline is an abstract class")
        self._ee_task_list = None
        self._client = None

    def run(
        self, ee_polygon: ee.Geometry.Polygon, prefix: str, is_large_polygon: bool = False
    ) -> List[storage.Blob]:
        """
        params:
            ee_polygon: Earth Engine polygon
            prefix: prefix for the exported and processed files
        returns:
            list of processed blobs in the npy bucket
        """
        if self.name in prefix:
            raise ValueError(f"Prefix should not contain {self.name}")

        assert not prefix.endswith("/")
        # Trailing slash critical for organizing files in GCS
        image_dest = f"{prefix}/{self.name}/"

        ee_description = ee_safe_str(image_dest[:-1])
        if self._ee_task_list is None:
            if not ee.data._credentials:
                ee.Initialize()
            self._ee_task_list = get_ee_task_list()

        if ee_description in self._ee_task_list or len(self._ee_task_list) >= 3000:
            return []

        if self._client is None:
            self._client = storage.Client()

        if is_large_polygon and any(
            True for _ in self._client.list_blobs(NPY_BUCKET, prefix=image_dest)
        ):
            return list(self._client.list_blobs(NPY_BUCKET, prefix=image_dest))

        if is_large_polygon and any(
            True for _ in self._client.list_blobs(EE_BUCKET, prefix=image_dest)
        ):
            return []

        ee.batch.Export.image.toCloudStorage(
            fileNamePrefix=image_dest,
            description=ee_description,
            image=self.create_ee_image(ee_polygon).clip(ee_polygon),
            region=ee_polygon,
            bucket=EE_BUCKET,
            fileDimensions=FILE_DIMENSIONS,
            scale=10,
            maxPixels=1e13,
        ).start()
        self._ee_task_list.append(ee_description)
        return []

    def convert_tif_to_np_on_cloud(self, name: str):
        # Check if npy file already exists
        if self._client is None:
            self._client = storage.Client()
        dest_bucket = self._client.bucket(NPY_BUCKET)
        dest_name = name.replace("-retry", "").replace(".tif", ".npy")
        lat_lon_name = dest_name.replace(self.name, "LatLon")

        dest_blob = dest_bucket.blob(dest_name)
        lat_lon_blob = dest_bucket.blob(lat_lon_name)
        dest_blob_exists = dest_blob.exists()
        lat_lon_blob_exists = lat_lon_blob.exists()

        if dest_blob_exists and lat_lon_blob_exists:
            print(f"{dest_name} and {lat_lon_name} already exist")
            return

        tif_path = gcloud_download(EE_BUCKET, name)

        # Convert the tif file to a numpy array
        x_np, lat_lon_np = self.convert_tif_to_np(tif_path)
        tif_path.unlink()

        np_path = Path(str(tif_path).replace("-retry", "").replace(".tif", ".npy"))
        lat_lon_np_path = Path(str(np_path).replace(".npy", "_lat_lon.npy"))

        # Upload data if not already available
        if not dest_blob_exists:
            np.save(np_path, x_np)
            dest_blob.upload_from_filename(np_path)
            np_path.unlink()
            print(f"Uploaded {dest_name} to {NPY_BUCKET}")

        # Upload lat lons if not already available
        if not lat_lon_blob_exists:
            np.save(lat_lon_np_path, lat_lon_np)
            lat_lon_blob.upload_from_filename(lat_lon_np_path)
            lat_lon_np_path.unlink()
            print(f"Uploaded {lat_lon_name} to {NPY_BUCKET}")

    def create_ee_image(self, ee_polygon: ee.Geometry.Polygon) -> ee.Image:
        raise NotImplementedError

    def convert_tif_to_np(self, tif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def normalize(cls, x):
        raise NotImplementedError
