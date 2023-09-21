from unittest import TestCase

import numpy as np
import torch
from einops import repeat
from sklearn.model_selection import train_test_split

from presto import Presto
from presto.dataops import NUM_BANDS, DynamicWorld2020_2021
from presto.eval.eurosat_eval import EuroSatDataset, EuroSatEval
from presto.utils import seed_everything


class TestEval(TestCase):
    def test_correct_number_of_tokens_masked_eurosat(self):

        eval_task = EuroSatEval()
        x = torch.ones([1, 1, NUM_BANDS])
        latlons = torch.ones([1, 2])
        dw = torch.ones([1, 1]).int() * DynamicWorld2020_2021.class_amount
        mask = torch.from_numpy(eval_task.update_mask()).bool()
        mask_per_batch = repeat(mask, "t d -> b t d", b=1)

        model = Presto.construct()
        output_tokens, _, _ = model.encoder(
            x, dw, latlons, mask_per_batch, month=0, eval_task=False
        )
        # 4 tokens: latlon, s2_rgb, s2_nir_10, s2_nir_20, s2_red_edge, s2_swir, nvdi
        self.assertEqual(output_tokens.shape[1], 7)

    def test_correct_number_of_tokens_masked_eurosat_rgb(self):

        eval_task = EuroSatEval(rgb=True)
        x = torch.ones([1, 1, NUM_BANDS])
        latlons = torch.ones([1, 2])
        dw = torch.ones([1, 1]).int() * DynamicWorld2020_2021.class_amount
        mask = torch.from_numpy(eval_task.update_mask()).bool()
        mask_per_batch = repeat(mask, "t d -> b t d", b=1)

        model = Presto.construct()
        output_tokens, _, _ = model.encoder(
            x, dw, latlons, mask_per_batch, month=0, eval_task=False
        )
        # 2 tokens: latlon, s2_rgb
        self.assertEqual(output_tokens.shape[1], 2)

    def test_seed_doesnt_affect_splits(self):
        # the informal settlements eval task uses a train_test split with
        # a random state - this test is to make sure that is not affected by
        # different seeds
        x_train, x_test = train_test_split([1, 2, 3, 4, 5], test_size=0.2, random_state=42)
        seed_everything()
        seeded_train, seeded_test = train_test_split(
            [1, 2, 3, 4, 5], test_size=0.2, random_state=42
        )

        self.assertTrue(x_train == seeded_train)
        self.assertTrue(x_test == seeded_test)

    def test_resize_and_average_arrays(self):

        num_dims = 13

        test_array = np.tile(
            np.expand_dims(
                np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]), (0)
            ).astype(float),
            (num_dims, 1, 1),
        )

        eurosat = EuroSatDataset(input_patch_size=2)
        eurosat.num_patches = 4
        eurosat.num_patches_per_dim = 2
        output = eurosat.resize_and_average_arrays(test_array)
        self.assertTrue(np.all(output == np.tile(np.array([[1, 2, 3, 4]]).T, (1, num_dims))))

    def test_splits_correctly_downloaded(self):
        train_split = EuroSatDataset.url_to_list(EuroSatDataset.split_urls["train"])
        assert len(train_split) == 16200
        for i in train_split:
            assert not i.startswith("b'")
