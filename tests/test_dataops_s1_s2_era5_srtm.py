from unittest import TestCase

import numpy as np

from presto.dataops import TIMESTEPS_IDX
from presto.dataops.masking import make_mask

TEST_MASK_RATIOS = (0.1, 0.5, 0.9)


class TestS1S2ERA5SRTM(TestCase):
    def test_make_mask_rest(self):
        eo_mask, dw_mask = make_mask(strategy="group_bands", mask_ratio=0.75)
        mask = np.concatenate([eo_mask, np.expand_dims(dw_mask, -1)], -1)
        masked_timesteps, _ = np.where(mask)
        timesteps_with_band_masked = np.unique(masked_timesteps) == TIMESTEPS_IDX
        self.assertTrue(timesteps_with_band_masked.all())
