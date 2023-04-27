from unittest import TestCase

import numpy as np

from presto.dataops import S1_S2_ERA5_SRTM, DynamicWorld2020_2021, MaskParams
from presto.dataops.dataset import (
    TAR_BUCKET,
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)


class TestRealDataset(TestCase):
    def test_get_S1_S2_ERA5_SRTM_DynamicWorld2020_2021_item_with_mask(self):
        folder = "S1_S2_ERA5_SRTM_2020_2021_DynamicWorld2020_2021_tars"
        url = f"gs://{TAR_BUCKET}/{folder}/dw_144_shard_0.tar"
        mask_params = MaskParams()
        dataset = S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021(mask_params=mask_params)
        iterable_dataset = dataset.as_webdataset(url=url)
        for item in iterable_dataset:
            break

        self.assertEqual(len(item), 9)
        mask, dw_mask, x, y, dw_x, dw_y, _, latlons, _ = item

        # Test shapes
        self.assertEqual(mask.shape, S1_S2_ERA5_SRTM.item_shape)
        self.assertEqual(x.shape, S1_S2_ERA5_SRTM.item_shape)
        self.assertEqual(y.shape, S1_S2_ERA5_SRTM.item_shape)
        self.assertEqual(latlons.shape, (2,))
        self.assertEqual(dw_mask.shape, DynamicWorld2020_2021.item_shape)
        self.assertEqual(dw_x.shape, DynamicWorld2020_2021.item_shape)
        self.assertEqual(dw_y.shape, DynamicWorld2020_2021.item_shape)

        # Check values
        self.assertTrue(list(np.unique(mask)) == [0, 1])

        # Check x has 0s where mask is 1,
        # Unlike cropharvest, some values will be 0 even if not masked out
        self.assertTrue((x[mask] == 0).all())
        self.assertTrue((y[~mask] == 0).all())

        # check that for dw, all the masked values are the missing value tokens
        self.assertTrue((dw_x[dw_mask] == DynamicWorld2020_2021.missing_data_class).all())
        self.assertTrue((dw_y[~dw_mask] == 0).all())
