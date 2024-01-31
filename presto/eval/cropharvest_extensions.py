"""
This file contains extensions of CropHarvest classes
"""
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import ee
import geopandas
import h5py
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from cropharvest import countries
from cropharvest.columns import NullableColumns, RequiredColumns
from cropharvest.config import (
    DAYS_PER_TIMESTEP,
    DEFAULT_NUM_TIMESTEPS,
    EXPORT_END_DAY,
    EXPORT_END_MONTH,
    FEATURES_DIR,
    LABELS_FILENAME,
    TEST_DATASETS,
    TEST_FEATURES_DIR,
    TEST_REGIONS,
)
from cropharvest.datasets import BaseDataset
from cropharvest.datasets import CropHarvestLabels as OrgCropHarvestLabels
from cropharvest.datasets import Task
from cropharvest.engineer import MISSING_DATA, DataInstance
from cropharvest.engineer import Engineer as CropHarvestEngineer
from cropharvest.engineer import TestInstance
from cropharvest.eo import EarthEngineExporter
from cropharvest.utils import (
    NoDataForBoundingBoxError,
    deterministic_shuffle,
    load_normalizing_dict,
    sample_with_memory,
)
from rasterio import mask
from torch.utils.data import Dataset
from tqdm import tqdm

from .. import utils
from ..dataops.pipelines.dynamicworld import DynamicWorld2020_2021, pad_array

logger = logging.getLogger("__main__")


def cropharvest_data_dir() -> Path:
    return utils.data_dir / "cropharvest_data"


class DynamicWorldExporter(EarthEngineExporter):
    output_folder_name = "dynamic_world_data"
    test_output_folder_name = "test_dynamic_world_data"
    data_dir: Path

    @staticmethod
    def load_default_labels(
        dataset: Optional[str], start_from_last, checkpoint: Optional[Path]
    ) -> geopandas.GeoDataFrame:
        labels = geopandas.read_file(cropharvest_data_dir() / LABELS_FILENAME)
        export_end_year = pd.to_datetime(labels[RequiredColumns.EXPORT_END_DATE]).dt.year
        labels["end_date"] = export_end_year.apply(
            lambda x: date(x, EXPORT_END_MONTH, EXPORT_END_DAY)
        )
        labels = labels.assign(
            start_date=lambda x: x["end_date"]
            - timedelta(days=DAYS_PER_TIMESTEP * DEFAULT_NUM_TIMESTEPS)
        )
        labels["export_identifier"] = labels.apply(
            lambda x: f"{x['index']}-{x[RequiredColumns.DATASET]}", axis=1
        )
        if dataset:
            labels = labels[labels.dataset == dataset]

        if start_from_last:
            labels = DynamicWorldExporter._filter_labels(labels, checkpoint)

        return labels

    def _export_for_polygon(
        self,
        polygon: ee.Geometry.Polygon,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        days_per_timestep: int = DAYS_PER_TIMESTEP,
        checkpoint: Optional[Path] = None,
        test: bool = False,
        file_dimensions: Optional[int] = None,
    ) -> bool:
        filename = str(polygon_identifier)
        if (checkpoint is not None) and (checkpoint / f"{filename}.tif").exists():
            logger.warning("File already exists! Skipping")
            return False

        # Description of the export cannot contain certrain characters
        description = filename.replace(".", "-").replace("=", "-").replace("/", "-")[:100]

        assert (
            days_per_timestep == DAYS_PER_TIMESTEP
        ), "Dynamic World exporter does not support non-default days per timestep"

        if self.check_gcp:
            # If test data we check the root in the cloud bucket
            if test and f"{filename}.tif" in self.cloud_tif_list:
                return False
            # If training data we check the tifs folder in thee cloud bucket
            elif not test and (f"tifs/{filename}.tif" in self.cloud_tif_list):
                return False

        # Check if task is already started in EarthEngine
        if self.check_ee and (description in self.ee_task_list):
            return True

        if self.check_ee and len(self.ee_task_list) >= 3000:
            return False

        img = DynamicWorld2020_2021.create_ee_image(polygon, start_date, end_date)

        # and finally, export the image
        kwargs = dict(
            image=img,
            region=polygon,
            filename=filename,
            description=description,
            file_dimensions=file_dimensions,
            test=test,
        )
        if self.dest_bucket:
            kwargs["dest_bucket"] = self.dest_bucket
        elif test:
            kwargs["drive_folder"] = self.test_output_folder_name
        else:
            kwargs["drive_folder"] = self.cur_output_folder

        self._export(**kwargs)
        return True

    @staticmethod
    def tif_to_npy(path_to_file: Path, lat: float, lon: float, num_timesteps: int):
        da = xr.open_rasterio(path_to_file)
        closest_lon = Engineer.find_nearest(da.x, lon)
        closest_lat = Engineer.find_nearest(da.y, lat)

        dw_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        return pad_array(dw_np, num_timesteps), closest_lat, closest_lon


class Engineer(CropHarvestEngineer):
    """
    This class is a bit overloaded - its used for the following:

    1. Turning the tifs (both EO and dynamic world) into npys for the
        algal blooms and fuel moisture datasets. This is handled
        by the `process_{}_files` functions.
    2. Turning the tifs (only dynamic world) into TestInstances for the
        cropharvest test dataset. This is handled by the `process_test_file`
        and `process_test_file_with_region` functions
    """

    @classmethod
    def tif_to_nps(
        cls, satellite_file: Path, dynamic_world_file: Path, row: pd.Series, num_timesteps: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if (not satellite_file.exists()) or (not dynamic_world_file.exists()):
            return None

        da, average_slope = cls.load_tif(
            satellite_file, start_date=row["start_date"], num_timesteps=num_timesteps
        )
        closest_lon = cls.find_nearest(da.x, row[RequiredColumns.LON])
        closest_lat = cls.find_nearest(da.y, row[RequiredColumns.LAT])

        labelled_np = da.sel(x=closest_lon).sel(y=closest_lat).values

        labelled_np = cls.calculate_ndvi(labelled_np)
        labelled_np = cls.remove_bands(labelled_np)

        labelled_array = cls.fillna(labelled_np, average_slope=average_slope)
        if labelled_array is None:
            return None

        months = np.array(
            [
                (row["start_date"] + timedelta(days=x * DAYS_PER_TIMESTEP)).month - 1
                for x in range(num_timesteps)
            ]
        )

        dw_np, _, _ = DynamicWorldExporter.tif_to_npy(
            dynamic_world_file, row[RequiredColumns.LAT], row[RequiredColumns.LON], num_timesteps
        )

        return labelled_array, dw_np, np.array([closest_lat, closest_lon]), months

    @classmethod
    def process_fuel_moisture_files(
        cls, satellite_file: Path, dynamic_world_file: Path, row: pd.Series, num_timesteps: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int, str]]:
        nps = cls.tif_to_nps(satellite_file, dynamic_world_file, row, num_timesteps)
        if nps is None:
            return None
        labelled_array, dw_np, latlon, months = nps

        y = row["percent(t)"]
        site = row["site"]
        is_test = int(row["split"] == "test")

        return labelled_array, dw_np, months, y, latlon, is_test, site

    @classmethod
    def process_algal_bloom_files(
        cls, satellite_file: Path, dynamic_world_file: Path, row: pd.Series, num_timesteps: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, int, str]]:
        nps = cls.tif_to_nps(satellite_file, dynamic_world_file, row, num_timesteps)
        if nps is None:
            return None
        labelled_array, dw_np, latlon, months = nps
        y = row["severity"]
        is_test = int(row["split"] == "test")

        return labelled_array, dw_np, months, y, latlon, is_test, row["uid"]

    @staticmethod
    def process_test_file(
        path_to_file: Path, start_date: datetime
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        da = xr.open_rasterio(path_to_file)

        x_np = da.values
        x_np = x_np.reshape(x_np.shape[0], x_np.shape[1] * x_np.shape[2])
        x_np = np.moveaxis(x_np, -1, 0)
        x_np = pad_array(x_np, DEFAULT_NUM_TIMESTEPS)

        # Get lat lons
        lon, lat = np.meshgrid(da.x.values, da.y.values)
        flat_lat, flat_lon = (
            np.squeeze(lat.reshape(-1, 1), -1),
            np.squeeze(lon.reshape(-1, 1), -1),
        )
        return x_np, flat_lat, flat_lon

    def process_single_file(
        self,
        path_to_file: Path,
        row: pd.Series,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    ) -> Optional[DataInstance]:
        """
        This function is used in CropHarvestEval.create_dynamic_world_test_h5_instances,
        to create the TestInstance for the Togo dataset
        """
        dw_np, closest_lat, closest_lon = DynamicWorldExporter.tif_to_npy(
            path_to_file, row[RequiredColumns.LAT], row[RequiredColumns.LON], num_timesteps
        )

        return DataInstance(
            label_lat=row[RequiredColumns.LAT],
            label_lon=row[RequiredColumns.LON],
            instance_lat=closest_lat,
            instance_lon=closest_lon,
            array=dw_np,
            is_crop=row[RequiredColumns.IS_CROP],
            label=row[NullableColumns.LABEL],
            dataset=row[RequiredColumns.DATASET],
        )

    def process_test_file_with_region(
        self, path_to_file: Path, id_in_region: int
    ) -> Tuple[str, TestInstance]:
        id_components = path_to_file.stem.split("_")
        crop, end_year = id_components[1], id_components[2]
        identifier = "_".join(id_components[:4])
        identifier_plus_idx = f"{identifier}_{id_in_region}"
        start_date = datetime(int(end_year), EXPORT_END_MONTH, EXPORT_END_DAY) - timedelta(
            days=DEFAULT_NUM_TIMESTEPS * DAYS_PER_TIMESTEP
        )

        final_x, flat_lat, flat_lon = self.process_test_file(path_to_file, start_date=start_date)

        # finally, we need to calculate the mask
        region_bbox = TEST_REGIONS[identifier]
        relevant_indices = self.labels.apply(
            lambda x: (
                region_bbox.contains(x[RequiredColumns.LAT], x[RequiredColumns.LON])
                and (x[RequiredColumns.EXPORT_END_DATE].year == int(end_year))
            ),
            axis=1,
        )
        relevant_rows = self.labels[relevant_indices]
        positive_geoms = relevant_rows[relevant_rows[NullableColumns.LABEL] == crop][
            RequiredColumns.GEOMETRY
        ].tolist()
        negative_geoms = relevant_rows[relevant_rows[NullableColumns.LABEL] != crop][
            RequiredColumns.GEOMETRY
        ].tolist()

        with rasterio.open(path_to_file) as src:
            # the mask is True outside shapes, and False inside shapes. We want the opposite
            positive, _, _ = mask.raster_geometry_mask(src, positive_geoms, crop=False)
            negative, _, _ = mask.raster_geometry_mask(src, negative_geoms, crop=False)
        # reverse the booleans so that 1 = in the
        positive = (~positive.reshape(positive.shape[0] * positive.shape[1])).astype(int)
        negative = (~negative.reshape(negative.shape[0] * negative.shape[1])).astype(int) * -1
        y = positive + negative

        # swap missing and negative values, since this will be easier to use in the future
        negative = y == -1
        missing = y == 0
        y[negative] = 0
        y[missing] = MISSING_DATA
        assert len(y) == final_x.shape[0]

        return identifier_plus_idx, TestInstance(x=final_x, y=y, lats=flat_lat, lons=flat_lon)

    def create_h5_test_instances(
        self,
    ) -> None:
        """
        Copied from https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/engineer.py,
        except for a change to the eo_files.glob line
        """
        for region_identifier, _ in TEST_REGIONS.items():
            all_region_files = list(self.test_eo_files.glob(f"{region_identifier}*.tif"))
            if len(all_region_files) == 0:
                logger.info(f"No downloaded files for {region_identifier}")
                continue
            for region_idx, filepath in enumerate(all_region_files):
                instance_name, test_instance = self.process_test_file_with_region(
                    filepath, region_idx
                )
                if test_instance is not None:
                    hf = h5py.File(self.test_savedir / f"{instance_name}.h5", "w")

                    for key, val in test_instance.datasets.items():
                        hf.create_dataset(key, data=val)
                    hf.close()

        for _, dataset in TEST_DATASETS.items():
            x: List[np.ndarray] = []
            y: List[int] = []
            lats: List[float] = []
            lons: List[float] = []
            relevant_labels = self.labels[self.labels[RequiredColumns.DATASET] == dataset]

            for _, row in tqdm(relevant_labels.iterrows()):
                tif_paths = list(
                    self.eo_files.glob(
                        f"{row[RequiredColumns.INDEX]}-{row[RequiredColumns.DATASET]}*.tif"
                    )
                )
                if len(tif_paths) == 0:
                    continue
                else:
                    tif_path = tif_paths[0]
                instance = self.process_single_file(tif_path, row)
                if instance is not None:
                    x.append(instance.array)
                    y.append(instance.is_crop)
                    lats.append(instance.label_lat)
                    lons.append(instance.label_lon)

            # then, combine the instances into a test instance
            test_instance = TestInstance(np.stack(x), np.stack(y), np.stack(lats), np.stack(lons))
            hf = h5py.File(self.test_savedir / f"{dataset}.h5", "w")
            for key, val in test_instance.datasets.items():
                hf.create_dataset(key, data=val)
            hf.close()


class CropHarvestLabels(OrgCropHarvestLabels):
    def construct_fao_classification_labels(
        self, task: Task, filter_test: bool = True
    ) -> List[Tuple[Tuple[Path, Path], int]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]
        if task.bounding_box is not None:
            gpdf = self.filter_geojson(
                gpdf, task.bounding_box, task.include_externally_contributed_labels
            )

        # This should probably be a required column since it has no
        # None values (and shouldn't have any)
        gpdf = gpdf[~gpdf[NullableColumns.CLASSIFICATION_LABEL].isnull()]

        if len(gpdf) == 0:
            raise NoDataForBoundingBoxError

        ys = gpdf[NullableColumns.CLASSIFICATION_LABEL]
        paths = self._dataframe_to_paths(gpdf)

        return [(path, y) for path, y in zip(paths, ys) if (path[0].exists() and path[1].exists())]

    def construct_positive_and_negative_labels(
        self, task: Task, filter_test: bool = True
    ) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]
        if task.bounding_box is not None:
            gpdf = self.filter_geojson(
                gpdf, task.bounding_box, task.include_externally_contributed_labels
            )

        if len(gpdf) == 0:
            raise NoDataForBoundingBoxError

        is_null = gpdf[NullableColumns.LABEL].isnull()
        is_crop = gpdf[RequiredColumns.IS_CROP] == True

        try:
            if task.target_label != "crop":
                positive_labels = gpdf[gpdf[NullableColumns.LABEL] == task.target_label]
                target_label_is_crop = positive_labels.iloc[0][RequiredColumns.IS_CROP]

                is_target = gpdf[NullableColumns.LABEL] == task.target_label

                if not target_label_is_crop:
                    # if the target label is a non crop class (e.g. pasture),
                    # then we can just collect all classes which either
                    # 1) are crop, or 2) are a different non crop class (e.g. forest)
                    negative_labels = gpdf[((is_null & is_crop) | (~is_null & ~is_target))]
                    negative_paths = self._dataframe_to_paths(negative_labels)
                else:
                    # otherwise, the target label is a crop. If balance_negative_crops is
                    # true, then we want an equal number of (other) crops and non crops in
                    # the negative labels
                    negative_non_crop_labels = gpdf[~is_crop]
                    negative_other_crop_labels = gpdf[(is_crop & ~is_null & ~is_target)]
                    negative_non_crop_paths = self._dataframe_to_paths(negative_non_crop_labels)
                    negative_paths = self._dataframe_to_paths(negative_other_crop_labels)

                    if task.balance_negative_crops:
                        multiplier = math.ceil(len(negative_non_crop_paths) / len(negative_paths))
                        negative_paths *= multiplier
                        negative_paths.extend(negative_non_crop_paths)
                    else:
                        negative_paths.extend(negative_non_crop_paths)
            else:
                # otherwise, we will just filter by crop and non crop
                positive_labels = gpdf[is_crop]
                negative_labels = gpdf[~is_crop]
                negative_paths = self._dataframe_to_paths(negative_labels)
        except IndexError:
            raise NoDataForBoundingBoxError

        positive_paths = self._dataframe_to_paths(positive_labels)

        if (len(positive_paths) == 0) or (len(negative_paths) == 0):
            raise NoDataForBoundingBoxError

        return [x for x in positive_paths if (x[0].exists() and x[1].exists())], [
            x for x in negative_paths if (x[0].exists() and x[1].exists())
        ]

    def _paths_from_row(self, row: geopandas.GeoSeries) -> Tuple[Path, Path]:
        suffix = f"{row[RequiredColumns.INDEX]}_{row[RequiredColumns.DATASET]}"
        prefix = self.root / "features"
        return prefix / f"arrays/{suffix}.h5", prefix / f"dynamic_world_arrays/{suffix}.npy"

    def _dataframe_to_paths(self, df: geopandas.GeoDataFrame) -> List[Tuple[Path, Path]]:
        return [self._paths_from_row(row) for _, row in df.iterrows()]


class MultiClassCropHarvest(Dataset):
    def __init__(
        self,
        paths_and_y: List[Tuple[Tuple[Path, Path], str]],
        y_string_to_int: Dict[str, int],
        ignore_dynamic_world: bool = False,
    ):
        self.paths_and_y = paths_and_y
        self.y_string_to_int = y_string_to_int
        self.ignore_dynamic_world = ignore_dynamic_world

    def __len__(self) -> int:
        return len(self.paths_and_y)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        paths, y = self.paths_and_y[index]
        satellite_data = h5py.File(paths[0], "r")
        lat = satellite_data.attrs["instance_lat"]
        lon = satellite_data.attrs["instance_lon"]
        dynamic_world = np.load(paths[1])
        if self.ignore_dynamic_world:
            dynamic_world = np.ones_like(dynamic_world) * DynamicWorld2020_2021.class_amount
        return (
            satellite_data.get("array")[:],
            dynamic_world,
            np.array([lat, lon]),
            self.y_string_to_int[y],
        )

    def as_array(
        self, flatten_x: bool = False, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if num_samples is not None:
            raise NotImplementedError
        indices_to_sample = list(range(len(self)))
        X, dw, latlons, Y = zip(*[self[i] for i in indices_to_sample])
        X_np, DW_np, latlon_np, y_np = np.stack(X), np.stack(dw), np.stack(latlons), np.stack(Y)

        if flatten_x:
            X_np = self._flatten_array(X_np)
            DW_np = self._flatten_array(DW_np)
        return X_np, DW_np, latlon_np, y_np

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0], -1)


class CropHarvest(BaseDataset):
    """Dataset consisting of satellite data and associated labels"""

    def __init__(
        self,
        root,
        task: Optional[Task] = None,
        download=False,
        val_ratio: float = 0.0,
        is_val: bool = False,
        ignore_dynamic_world: bool = False,
        start_month: int = 1,
    ):
        super().__init__(root, download, filenames=(FEATURES_DIR, TEST_FEATURES_DIR))

        labels = CropHarvestLabels(root, download=download)
        if task is None:
            logger.info("Using the default task; crop vs. non crop globally")
            task = Task()
        self.task = task
        self.ignore_dynamic_world = ignore_dynamic_world
        self.start_month = start_month
        self.normalizing_dict = load_normalizing_dict(
            Path(root) / f"{FEATURES_DIR}/normalizing_dict.h5"
        )

        positive_paths, negative_paths = labels.construct_positive_and_negative_labels(
            task, filter_test=True
        )
        if val_ratio > 0.0:
            # the fixed seed is to ensure the validation set is always
            # different from the training set
            positive_paths = deterministic_shuffle(positive_paths, seed=42)
            negative_paths = deterministic_shuffle(negative_paths, seed=42)
            if is_val:
                positive_paths = positive_paths[: int(len(positive_paths) * val_ratio)]
                negative_paths = negative_paths[: int(len(negative_paths) * val_ratio)]
            else:
                positive_paths = positive_paths[int(len(positive_paths) * val_ratio) :]
                negative_paths = negative_paths[int(len(negative_paths) * val_ratio) :]

        self.filepaths: List[Tuple[Path, Path]] = positive_paths + negative_paths
        self.y_vals: List[int] = [1] * len(positive_paths) + [0] * len(negative_paths)
        self.positive_indices = list(range(len(positive_paths)))
        self.negative_indices = list(
            range(len(positive_paths), len(positive_paths) + len(negative_paths))
        )

        # used in the sample() function, to ensure filepaths are sampled without
        # duplication as much as possible
        self.sampled_positive_indices: List[int] = []
        self.sampled_negative_indices: List[int] = []

    def reset_sampled_indices(self) -> None:
        self.sampled_positive_indices = []
        self.sampled_negative_indices = []

    def shuffle(self, seed: int) -> None:
        self.reset_sampled_indices()
        filepaths_and_y_vals = list(zip(self.filepaths, self.y_vals))
        filepaths_and_y_vals = deterministic_shuffle(filepaths_and_y_vals, seed)
        filepaths, y_vals = zip(*filepaths_and_y_vals)
        self.filepaths, self.y_vals = list(filepaths), list(y_vals)

        self.positive_indices, self.negative_indices = self._get_positive_and_negative_indices()

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        hf = h5py.File(self.filepaths[index][0], "r")
        lat = hf.attrs["instance_lat"]
        lon = hf.attrs["instance_lon"]
        dynamic_world = np.load(self.filepaths[index][1])
        if self.ignore_dynamic_world:
            dynamic_world = np.ones_like(dynamic_world) * DynamicWorld2020_2021.class_amount
        return (
            self._normalize(hf.get("array")[:]),
            dynamic_world,
            np.array([lat, lon]),
            self.y_vals[index],
            self.start_month,
        )

    @property
    def k(self) -> int:
        return min(len(self.positive_indices), len(self.negative_indices))

    @property
    def num_bands(self) -> int:
        # array has shape [timesteps, bands]
        return self[0][0].shape[-1]

    def as_array(
        self, flatten_x: bool = False, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Return the training data as a tuple of
        np.ndarrays
        :param flatten_x: If True, the X array will have shape [num_samples, timesteps * bands]
            instead of [num_samples, timesteps, bands]
        :param num_samples: If -1, all data is returned. Otherwise, a balanced dataset of
            num_samples / 2 positive (& negative) samples will be returned
        """

        if num_samples is None:
            indices_to_sample = list(range(len(self)))
        else:
            k = num_samples // 2

            pos_indices, neg_indices = self._get_positive_and_negative_indices()
            if (k > len(pos_indices)) or (k > len(neg_indices)):
                raise ValueError(
                    f"num_samples // 2 ({k}) is greater than the number of "
                    f"positive samples ({len(pos_indices)}) "
                    f"or the number of negative samples ({len(neg_indices)})"
                )
            indices_to_sample = pos_indices[:k] + neg_indices[:k]

        X, dw, latlons, Y, _ = zip(*[self[i] for i in indices_to_sample])
        X_np, dw_np, latlons_np, y_np = np.stack(X), np.stack(dw), np.stack(latlons), np.stack(Y)

        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, dw_np, latlons_np, y_np

    def test_data(
        self, flatten_x: bool = False, max_size: Optional[int] = None
    ) -> Generator[Tuple[str, TestInstance, TestInstance], None, None]:
        r"""
        A generator returning TestInstance objects containing the test
        inputs, ground truths and associated latitudes nad longitudes
        :param flatten_x: If True, the TestInstance.x will have shape
            [num_samples, timesteps * bands] instead of [num_samples, timesteps, bands]
        """
        all_relevant_files = list(
            (self.root / TEST_FEATURES_DIR).glob(f"{self.task.test_identifier}*.h5")
        )
        if len(all_relevant_files) == 0:
            raise RuntimeError(f"Missing test data {self.task.test_identifier}*.h5")
        for filepath in all_relevant_files:
            hf = h5py.File(filepath, "r")
            test_array = TestInstance.load_from_h5(hf)

            dw_hf = h5py.File(self.root / "test_dynamic_world_features" / filepath.name, "r")
            dw_test_array = TestInstance.load_from_h5(dw_hf)

            if (max_size is not None) and (len(test_array) > max_size):
                cur_idx = 0
                while (cur_idx * max_size) < len(test_array):
                    sub_array = test_array[cur_idx * max_size : (cur_idx + 1) * max_size]
                    sub_dw_array = dw_test_array[cur_idx * max_size : (cur_idx + 1) * max_size]
                    sub_array.x = self._normalize(sub_array.x)
                    sub_dw_array.x = pad_array(sub_dw_array.x, num_timesteps=DEFAULT_NUM_TIMESTEPS)
                    if self.ignore_dynamic_world:
                        sub_dw_array.x = (
                            np.ones_like(sub_dw_array.x) * DynamicWorld2020_2021.class_amount
                        )
                    if flatten_x:
                        sub_array.x = self._flatten_array(sub_array.x)
                    test_id = f"{cur_idx}_{filepath.stem}"
                    cur_idx += 1
                    yield test_id, sub_array, sub_dw_array
            else:
                test_array.x = self._normalize(test_array.x)
                dw_test_array.x = pad_array(dw_test_array.x, num_timesteps=DEFAULT_NUM_TIMESTEPS)
                if self.ignore_dynamic_world:
                    dw_test_array.x = (
                        np.ones_like(dw_test_array.x) * DynamicWorld2020_2021.class_amount
                    )
                if flatten_x:
                    test_array.x = self._flatten_array(test_array.x)
                yield filepath.stem, test_array, dw_test_array

    @classmethod
    def create_benchmark_datasets(
        cls,
        root,
        balance_negative_crops: bool = True,
        download: bool = True,
        normalize: bool = True,
        ignore_dynamic_world: bool = False,
        start_month: int = 1,
    ) -> List:
        r"""
        Create the benchmark datasets.
        :param root: The path to the data, where the training data and labels are (or will be)
            saved
        :param balance_negative_crops: Whether to ensure the crops are equally represented in
            a dataset's negative labels. This is only used for datasets where there is a
            target_label, and that target_label is a crop
        :param download: Whether to download the labels and training data if they don't
            already exist
        :param normalize: Whether to normalize the data
        :returns: A list of evaluation CropHarvest datasets according to the TEST_REGIONS and
            TEST_DATASETS in the config
        """

        output_datasets: List = []

        for identifier, bbox in TEST_REGIONS.items():
            country, crop, _, _ = identifier.split("_")

            country_bboxes = countries.get_country_bbox(country)
            for country_bbox in country_bboxes:
                task = Task(
                    country_bbox,
                    crop,
                    balance_negative_crops,
                    f"{country}_{crop}",
                    normalize,
                )
                if task.id not in [x.id for x in output_datasets]:
                    if country_bbox.contains_bbox(bbox):
                        output_datasets.append(
                            cls(
                                root,
                                task,
                                download=download,
                                ignore_dynamic_world=ignore_dynamic_world,
                                start_month=start_month,
                            )
                        )

        for country, test_dataset in TEST_DATASETS.items():
            # TODO; for now, the only country here is Togo, which
            # only has one bounding box. In the future, it might
            # be nice to confirm its the right index (maybe by checking against
            # some points in the test h5py file?)
            country_bbox = countries.get_country_bbox(country)[0]
            output_datasets.append(
                cls(
                    root,
                    Task(country_bbox, None, test_identifier=test_dataset, normalize=normalize),
                    download=download,
                    ignore_dynamic_world=ignore_dynamic_world,
                    start_month=start_month,
                )
            )
        return output_datasets

    def sample(
        self, k: int, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # we will sample to get half positive and half negative
        # examples
        output_x: List[np.ndarray] = []
        output_y: List[np.ndarray] = []

        k = min(k, self.k)

        if deterministic:
            pos_indices = self.positive_indices[:k]
            neg_indices = self.negative_indices[:k]
        else:
            pos_indices, self.sampled_positive_indices = sample_with_memory(
                self.positive_indices, k, self.sampled_positive_indices
            )
            neg_indices, self.sampled_negative_indices = sample_with_memory(
                self.negative_indices, k, self.sampled_negative_indices
            )

        # returns a list of [pos_index, neg_index, pos_index, neg_index, ...]
        indices = [val for pair in zip(pos_indices, neg_indices) for val in pair]
        output_x, dw, latlons, output_y, _ = zip(*[self[i] for i in indices])

        x = np.stack(output_x, axis=0)
        return x, np.stack(dw, axis=0), np.stack(latlons, axis=0), np.array(output_y)

    def __repr__(self) -> str:
        class_name = f"CropHarvest{'Eval' if self.task.test_identifier is not None else ''}"
        return f"{class_name}({self.id}, {self.task.test_identifier})"

    @property
    def id(self) -> str:
        return self.task.id

    def _get_positive_and_negative_indices(self) -> Tuple[List[int], List[int]]:
        positive_indices: List[int] = []
        negative_indices: List[int] = []

        for i, y_val in enumerate(self.y_vals):
            if y_val == 1:
                positive_indices.append(i)
            else:
                negative_indices.append(i)
        return positive_indices, negative_indices

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        if not self.task.normalize:
            return array
        return (array - self.normalizing_dict["mean"]) / self.normalizing_dict["std"]

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0], -1)
