from pathlib import Path
from typing import Tuple

import ee
import numpy as np
import xarray as xr

from .ee_pipeline import EEPipeline, resample_and_flatten_tif


class WorldCover2020(EEPipeline):
    legend = {
        0: "Not classified",
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
    legend_idx = {key: i for i, key in enumerate(legend.keys())}
    class_amount = 12
    item_shape = (class_amount,)

    def create_ee_image(self, ee_polygon) -> ee.Image:
        return ee.ImageCollection("ESA/WorldCover/v100").first()

    def convert_tif_to_np(self, tif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        tif = xr.open_rasterio(tif_path)
        return resample_and_flatten_tif(tif)

    @classmethod
    def normalize(cls, x: np.ndarray) -> np.ndarray:
        return np.vectorize(cls.legend_idx.get)(x)
