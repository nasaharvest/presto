from datetime import date
from pathlib import Path
from typing import Tuple

import ee
import numpy as np
import xarray as xr

from .ee_pipeline import EEPipeline, resample_and_flatten_tif

NUM_TIMESTEPS = 12


def pad_array(array: np.ndarray, num_timesteps: int):
    if len(array.shape) == 1:
        time_idx = 0
    elif len(array.shape) == 2:
        time_idx = 1

    num_array_timesteps = array.shape[time_idx]
    if num_array_timesteps < num_timesteps:
        zeroth_timestep = array[0] if time_idx == 0 else array[:, 0]
        padding = np.array([zeroth_timestep] * (num_timesteps - num_array_timesteps))
        array = np.concatenate([array, padding.T], axis=time_idx)
    elif num_array_timesteps > num_timesteps:
        array = array[:num_timesteps] if time_idx == 0 else array[:, :num_timesteps]

    assert array.shape[time_idx] == num_timesteps
    return array


class DynamicWorld2020_2021(EEPipeline):
    legend = {
        0: "water",
        1: "trees",
        2: "grass",
        3: "flooded_vegetation",
        4: "crops",
        5: "shrub_and_scrub",
        6: "built",
        7: "bare",
        8: "snow_and_ice",
    }

    legend_idx = {key: i for i, key in enumerate(legend.keys())}
    class_amount = 9
    item_shape = (12,)
    missing_data_class = 9

    def convert_tif_to_np(self, tif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        tif = xr.open_rasterio(tif_path)
        return resample_and_flatten_tif(tif)

    @classmethod
    def normalize(cls, x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def create_ee_image(
        ee_polygon, start_date: date = date(2020, 1, 1), end_date: date = date(2021, 12, 31)
    ) -> ee.Image:
        start_date = date(start_date.year, start_date.month, start_date.day)
        end_date = date(end_date.year, end_date.month, end_date.day)
        date_ranges = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.month < 12:
                next_date = date(current_date.year, current_date.month + 1, 1)
            else:
                next_date = date(current_date.year + 1, 1, 1)
            date_ranges.append((current_date, next_date))
            current_date = next_date

        dw_collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(ee_polygon)
            .filterDate(ee.DateRange(str(start_date), str(end_date)))
            .select("label")
        )
        ee_date_ranges = [ee.DateRange(str(start), str(end)) for start, end in date_ranges]
        dw_month_images = [dw_collection.filterDate(d).mode() for d in ee_date_ranges]
        return ee.Image.cat(dw_month_images)


class DynamicWorldMonthly2020_2021(DynamicWorld2020_2021):
    """
    The only change to the `DynamicWorld2020_2021` class is
    to ensure that at least one value per month is downloaded.
    If no image for that month is available, then the closest value
    will be taken.
    """

    @staticmethod
    def create_ee_image(
        ee_polygon, start_date: date = date(2020, 1, 1), end_date: date = date(2021, 12, 31)
    ) -> ee.Image:
        start_date = date(start_date.year, start_date.month, start_date.day)
        end_date = date(end_date.year, end_date.month, end_date.day)

        # we start by getting all the data for the range
        dw_collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(ee_polygon)
            .filterDate(ee.DateRange(str(start_date), str(end_date)))
            .select("label")
        )

        fifteen_days_in_ms = 1296000000
        output_images = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.month < 12:
                next_date = date(current_date.year, current_date.month + 1, 1)
            else:
                next_date = date(current_date.year + 1, 1, 1)
            mid_date = current_date + (next_date - current_date) / 2
            mid_date_ee = ee.Date(str(date(mid_date.year, mid_date.month, mid_date.day)))

            # first, order by distance from mid_date
            from_mid_date = dw_collection.map(
                lambda image: image.set(
                    "dateDist",
                    ee.Number(image.get("system:time_start")).subtract(mid_date_ee.millis()).abs(),
                )
            )
            from_mid_date = from_mid_date.sort("dateDist", opt_ascending=True)

            # no matter what, we take the first element in the image collection
            # and we add 1 to ensure the less_than condition triggers
            max_diff = ee.Number(from_mid_date.first().get("dateDist")).max(
                ee.Number(fifteen_days_in_ms)
            )

            kept_images = from_mid_date.filterMetadata("dateDist", "not_greater_than", max_diff)
            output_images.append(kept_images.mode())

            current_date = next_date

        return ee.Image.cat(output_images)
