from unittest import TestCase

import numpy as np
import torch

from presto.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS, TIMESTEPS_IDX
from presto.dataops.masking import MASK_STRATEGIES, MaskParams, make_mask
from presto.dataops.pipelines.s1_s2_era5_srtm import S1_S2_ERA5_SRTM
from presto.presto import Presto


class TestMasking(TestCase):
    def test_make_mask_rest(self):
        eo_mask, dw_mask = make_mask(strategy="group_bands", mask_ratio=0.75)
        mask = np.concatenate([eo_mask, np.expand_dims(dw_mask, -1)], -1)
        masked_timesteps, _ = np.where(mask)
        timesteps_with_band_masked = np.unique(masked_timesteps) == TIMESTEPS_IDX
        self.assertTrue(timesteps_with_band_masked.all())

    def test_masks_work_with_different_strategies(self):
        batch_size = 10
        input_eo = S1_S2_ERA5_SRTM.normalize(np.zeros((NUM_TIMESTEPS, NUM_ORG_BANDS)))
        dynamic_world = np.ones((NUM_TIMESTEPS))

        eo_mask_list, dw_list, eo_list = [], [], []
        masker = MaskParams(MASK_STRATEGIES)
        for _ in range(batch_size):
            mask, _, x, _, x_dw, _, _ = masker.mask_data(input_eo.copy(), dynamic_world.copy())
            eo_mask_list.append(torch.from_numpy(mask))
            dw_list.append(torch.from_numpy(x_dw))
            eo_list.append(torch.from_numpy(x))

        latlons = torch.rand((batch_size, 2))
        model = Presto.construct()
        # if the model masking works, then we have correctly masked the right
        # number of tokens for all the different strategies
        _ = model(
            torch.stack(eo_list),
            dynamic_world=torch.stack(dw_list).long(),
            latlons=latlons,
            mask=torch.stack(eo_mask_list),
            month=1,
        )
