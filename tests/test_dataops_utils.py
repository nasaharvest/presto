from unittest import TestCase

import torch

from presto import construct_single_presto_input
from presto.dataops.pipelines.dynamicworld import DynamicWorld2020_2021


class TestDatopsUtils(TestCase):
    def test_construct_single_presto_input(self):
        x, mask, dw = construct_single_presto_input(
            s2=torch.ones(2, 3), s2_bands=["B2", "B3", "B4"], normalize=False
        )
        self.assertTrue(torch.equal(dw, torch.ones_like(dw) * DynamicWorld2020_2021.class_amount))
        self.assertEqual(len(dw), x.shape[0])
        self.assertEqual(x.shape, mask.shape)
        self.assertTrue((x[mask == 1] == 0).all())
        self.assertTrue((x[mask == 0] != 0).all())
