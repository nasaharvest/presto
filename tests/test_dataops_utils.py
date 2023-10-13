from unittest import TestCase

import torch

from presto import construct_single_presto_input
from presto.dataops.pipelines.dynamicworld import DynamicWorld2020_2021
from presto.dataops.pipelines.s1_s2_era5_srtm import NORMED_BANDS


class TestDatopsUtils(TestCase):
    def test_construct_single_presto_input(self):
        input_bands = ["B2", "B3", "B4", "B8"]
        x, mask, dw = construct_single_presto_input(
            s2=torch.ones(2, 4), s2_bands=input_bands, normalize=False
        )
        self.assertTrue(torch.equal(dw, torch.ones_like(dw) * DynamicWorld2020_2021.class_amount))
        self.assertEqual(len(dw), x.shape[0])
        self.assertEqual(x.shape, mask.shape)
        self.assertTrue((x[mask == 1] == 0).all())
        self.assertTrue((x[mask == 0] != 0).all())
        for idx, band in enumerate(NORMED_BANDS):
            if band in input_bands:
                self.assertTrue((mask[:, idx] == 0).all())
            else:
                self.assertTrue((mask[:, idx] == 1).all())

    def test_construct_single_presto_input_ndvi(self):
        input_bands = ["B2", "B3", "B4", "B8"]
        x, mask, dw = construct_single_presto_input(
            s2=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]).float(),
            s2_bands=input_bands,
            normalize=True,
        )
        self.assertTrue(torch.equal(dw, torch.ones_like(dw) * DynamicWorld2020_2021.class_amount))
        self.assertEqual(len(dw), x.shape[0])
        self.assertEqual(x.shape, mask.shape)
        # we can't test for equality to 0 since we normalize;
        # that's tested above
        self.assertTrue((x[mask == 0] != 0).all())
        for idx, band in enumerate(NORMED_BANDS):
            if band in input_bands + ["NDVI"]:
                self.assertTrue((mask[:, idx] == 0).all())
