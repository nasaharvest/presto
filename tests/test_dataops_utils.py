from unittest import TestCase

import torch

from presto import construct_batch_presto_input, construct_single_presto_input
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

    def test_construct_batch_presto_input(self):
        input_bands = ["B2", "B3", "B4", "B8"]
        batch_size, num_timesteps = 3, 2
        x, mask, dw = construct_batch_presto_input(
            s2=torch.ones(batch_size, num_timesteps, len(input_bands)),
            s2_bands=input_bands,
            normalize=False,
        )
        self.assertEqual(x.shape[0], batch_size)
        self.assertEqual(mask.shape[0], batch_size)
        self.assertEqual(dw.shape, (batch_size, num_timesteps))
        self.assertTrue(torch.equal(dw, torch.ones_like(dw) * DynamicWorld2020_2021.class_amount))
        self.assertEqual(x.shape, mask.shape)
        self.assertTrue((x[mask == 1] == 0).all())
        self.assertTrue((x[mask == 0] != 0).all())
        for idx, band in enumerate(NORMED_BANDS):
            if band in input_bands:
                self.assertTrue((mask[:, :, idx] == 0).all())
            else:
                self.assertTrue((mask[:, :, idx] == 1).all())

    def test_construct_batch_matches_single(self):
        # a batch built from stacked single inputs should match the per-item outputs
        input_bands = ["B2", "B3", "B4", "B8"]
        item_a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).float()
        item_b = torch.tensor([[2, 4, 6, 8], [1, 3, 5, 7]]).float()

        xa, ma, dwa = construct_single_presto_input(
            s2=item_a, s2_bands=input_bands, normalize=True
        )
        xb, mb, dwb = construct_single_presto_input(
            s2=item_b, s2_bands=input_bands, normalize=True
        )

        x, mask, dw = construct_batch_presto_input(
            s2=torch.stack([item_a, item_b]), s2_bands=input_bands, normalize=True
        )

        self.assertTrue(torch.allclose(x, torch.stack([xa, xb])))
        self.assertTrue(torch.equal(mask, torch.stack([ma, mb])))
        self.assertTrue(torch.equal(dw, torch.stack([dwa, dwb])))
