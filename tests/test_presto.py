import json
from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import torch
from einops import repeat
from torch import nn
from torch.optim import AdamW

from presto.dataops import NUM_ORG_BANDS, NUM_TIMESTEPS
from presto.dataops.pipelines.dynamicworld import DynamicWorld2020_2021
from presto.dataops.pipelines.s1_s2_era5_srtm import BANDS_GROUPS_IDX, S1_S2_ERA5_SRTM
from presto.presto import Decoder, Encoder, Presto, month_to_tensor, param_groups_lrd
from presto.utils import config_dir, default_model_path, device
from single_file_presto import Presto as SingleFilePresto


class TestPresto(TestCase):
    def test_encoder_init(self):
        batch_size = 3
        input = S1_S2_ERA5_SRTM.normalize(torch.zeros((batch_size, NUM_TIMESTEPS, NUM_ORG_BANDS)))
        input_mask = torch.zeros_like(input)
        dynamic_world = torch.ones((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))
        model = Encoder()

        x, kept_indices, removed_indices = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups + 1)) + 1 tokens
        self.assertTrue(x.shape[1] == 1 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) + 1)))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) + 1))
        )

        # mask one group
        input_mask[:, :, 0] = 1
        x, kept_indices, removed_indices = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups)) + 1 tokens
        self.assertTrue(x.shape[1] == 1 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX))))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) + 1))
        )

        # mask dynamic world. This is the missing class value
        dynamic_world *= DynamicWorld2020_2021.class_amount
        x, kept_indices, removed_indices = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups)) + 1 tokens
        self.assertTrue(x.shape[1] == 1 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 1)))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) + 1))
        )

    def test_end_to_end(self):
        batch_size = 3
        input = S1_S2_ERA5_SRTM.normalize(torch.zeros((batch_size, NUM_TIMESTEPS, NUM_ORG_BANDS)))
        input_mask = torch.zeros_like(input)
        dynamic_world = torch.ones((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))

        model = Presto.construct()

        # mask one group
        input_mask[:, :, 0] = 1
        output, dw_outut = model(
            input, dynamic_world=dynamic_world, latlons=latlons, mask=input_mask, month=1
        )
        self.assertTrue(output.shape == input.shape)
        self.assertTrue(
            dw_outut.shape == (batch_size, NUM_TIMESTEPS, DynamicWorld2020_2021.class_amount)
        )

    def test_tokens_correctly_masked(self):

        x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        x = repeat(x, "b t -> b t d", d=3)

        mask = torch.zeros_like(x)
        mask[0, 1] = 1
        mask[0, 3] = 1
        mask[1, 2] = 1
        mask[1, 4] = 1

        x, kept_indices, removed_indices = Encoder.mask_tokens(x, mask)

        expected_out = torch.tensor([[1, 3, 5], [1, 2, 4]])
        expected_out = repeat(expected_out, "b t -> b t d", d=3)
        self.assertTrue(torch.equal(x, expected_out))
        self.assertTrue(torch.equal(kept_indices, torch.tensor([[0, 2, 4], [0, 1, 3]])))
        self.assertTrue(torch.equal(removed_indices, torch.tensor([[1, 3], [2, 4]])))

    def test_tokens_correctly_unmasked(self):

        # add a -1, for the latlon embedding. The -1 value
        # is not counted in the indices
        masked_x = torch.tensor([[-1, 1, 3, 5], [-1, 1, 2, 4]])
        masked_x = repeat(masked_x, "b t -> b t d", d=14)

        decoder = Decoder(
            channel_embeddings=nn.Embedding(2, 2),
            encoder_embed_dim=14,
            decoder_embed_dim=14,
            decoder_num_heads=2,
        )
        # the mask token is initialized to 0s
        kept_indices = torch.tensor([[0, 2, 4], [0, 1, 3]])
        removed_indices = torch.tensor([[1, 3], [2, 4]])

        filled_tokens = decoder.add_masked_tokens(masked_x, kept_indices, removed_indices)

        expected_out = torch.tensor([[-1, 1, 0, 3, 0, 5], [-1, 1, 2, 0, 4, 0]])
        expected_out = repeat(expected_out, "b t -> b t d", d=14)
        self.assertTrue(torch.equal(filled_tokens, expected_out))

    def test_encodings_correctly_added_in_decoder(self):

        # 1 batch, 3 timesteps, 14 dimensions (plus the latlon token)
        num_timesteps = 3
        x = torch.zeros((1, (num_timesteps * (len(BANDS_GROUPS_IDX) + 1)) + 1, 14))

        # increasing channel embedding
        embedding = torch.arange(0, (len(BANDS_GROUPS_IDX) + 1)).float()
        channel_embedding = nn.Embedding.from_pretrained(repeat(embedding, "c -> c d", d=2))

        decoder = Decoder(
            channel_embedding,
            encoder_embed_dim=14,
            decoder_embed_dim=14,
            decoder_num_heads=2,
        )

        output = decoder.add_embeddings(x, month=1)

        # check the latlon token has no embeddings
        self.assertTrue(torch.equal(output[:, 0, :], torch.zeros_like(output[:, 0, :])))
        output = output[:, 1:, :]

        expected_positional_encodings = decoder.pos_embed[:, :num_timesteps, :]
        expected_month_encodings = decoder.month_embed(month_to_tensor(1, 1, num_timesteps))
        # then, for each group of channels lets make sure the embeddings are correct
        for idx in range((len(BANDS_GROUPS_IDX) + 1)):
            # each encoding is 6 month dims, 2 channel dims and 6 pos dims
            channel_group = output[:, idx * num_timesteps : (idx + 1) * num_timesteps]
            # make sure all the channel group encodings are correct
            self.assertTrue(
                torch.equal(
                    channel_group[:, :, 6:8], torch.ones_like(channel_group[:, :, 6:8]) * idx
                )
            )
            self.assertTrue(torch.equal(channel_group[:, :, 8:], expected_positional_encodings))
            self.assertTrue(torch.equal(channel_group[:, :, :6], expected_month_encodings))

    def test_finetuning_model_doesnt_affect_grads(self):
        seq2seq_model = Presto.construct()
        seq2seq_model.requires_grad_(False)
        org_model = deepcopy(seq2seq_model)

        finetuning_model = seq2seq_model.construct_finetuning_model(num_outputs=1)

        batch_size = 3
        input = S1_S2_ERA5_SRTM.normalize(torch.zeros((batch_size, NUM_TIMESTEPS, NUM_ORG_BANDS)))
        input_mask = torch.zeros_like(input)
        dynamic_world = torch.ones((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))

        parameters = param_groups_lrd(finetuning_model)
        opt = AdamW(parameters, lr=0.1)

        finetuning_model(
            input, dynamic_world=dynamic_world, latlons=latlons, mask=input_mask, month=1
        ).sum().backward()
        opt.step()

        # also check the finetuning hasn't affected the gradients
        for seq2seq_params, org_params in zip(seq2seq_model.parameters(), org_model.parameters()):
            self.assertTrue(seq2seq_params.grad is None)
            self.assertTrue(org_params.grad is None)
            self.assertTrue(torch.equal(seq2seq_params, org_params))

    def test_default_loading_behaviour(self):
        """
        Checks that the default parameters in construct() match with the
        default state dict saved in the data folder. Also tests that
        all 3 ways of loading the pretrained model are in agreement
        """
        model = Presto.construct()
        model.load_state_dict(torch.load(default_model_path, map_location=device))

        from_function = Presto.load_pretrained()
        for torch_loaded, pretrain_loaded in zip(model.parameters(), from_function.parameters()):
            self.assertTrue(torch.equal(torch_loaded, pretrain_loaded))

        path_to_config = config_dir / "default.json"
        with Path(path_to_config).open("r") as f:
            model_kwargs = json.load(f)
        from_config = Presto.construct(band_groups=BANDS_GROUPS_IDX, **model_kwargs)
        from_config.load_state_dict(torch.load(default_model_path, map_location=device))
        for torch_loaded, config_loaded in zip(model.parameters(), from_config.parameters()):
            self.assertTrue(torch.equal(torch_loaded, config_loaded))

    def test_single_file_presto_matches_presto(self):
        model = Presto.construct()
        model.load_state_dict(torch.load(default_model_path, map_location=device))

        single_file_model = SingleFilePresto.construct()
        single_file_model.load_state_dict(torch.load(default_model_path, map_location=device))

        for model_p, sf_model_p in zip(model.parameters(), single_file_model.parameters()):
            self.assertTrue(torch.equal(model_p, sf_model_p))

        batch_size = 3
        input = S1_S2_ERA5_SRTM.normalize(torch.zeros((batch_size, NUM_TIMESTEPS, NUM_ORG_BANDS)))
        input_mask = torch.zeros_like(input)
        dynamic_world = torch.ones((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))

        output = model(
            input, dynamic_world=dynamic_world, latlons=latlons, mask=input_mask, month=1
        )
        sf_output = single_file_model(
            input, dynamic_world=dynamic_world, latlons=latlons, mask=input_mask, month=1
        )

        for out_tensor, out_sf_tensor in zip(output, sf_output):
            self.assertTrue(torch.equal(out_tensor, out_sf_tensor))
