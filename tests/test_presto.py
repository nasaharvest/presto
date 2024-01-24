import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from unittest import TestCase

import torch
from einops import repeat
from torch import nn
from torch.optim import AdamW

from presto.dataops import (
    BANDS_GROUPS_IDX,
    NUM_BANDS,
    NUM_ORG_BANDS,
    NUM_TIMESTEPS,
    S1_S2_ERA5_SRTM,
    DynamicWorld2020_2021,
)
from presto.dataops.masking import SRTM_INDEX
from presto.presto import Decoder, Encoder, Presto, month_to_tensor
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

        x, orig_indices, upd_mask = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups)) + 2 tokens
        self.assertEqual(x.shape[1], 2 + NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
        self.assertEqual(orig_indices.shape[1], x.shape[1])
        self.assertEqual(upd_mask.shape[1], x.shape[1])

        # mask one entire group
        input_mask[:, :, 0] = 1
        x, orig_indices, upd_mask = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        self.assertEqual(x.shape[1], 2 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 1)))
        self.assertEqual(orig_indices.shape[1], 2 + NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
        self.assertEqual(upd_mask.shape[1], x.shape[1])

        # mask dynamic world. This is the missing class value
        dynamic_world *= DynamicWorld2020_2021.class_amount
        x, orig_indices, upd_mask = model(
            input,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=input_mask,
            month=1,
            eval_task=False,
        )
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups)) + 2 tokens
        self.assertEqual(x.shape[1], 2 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 2)))
        self.assertEqual(orig_indices.shape[1], 2 + NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
        self.assertEqual(upd_mask.shape[1], x.shape[1])

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
        self.assertEqual(output.shape, input.shape)
        self.assertEqual(
            dw_outut.shape, (batch_size, NUM_TIMESTEPS, DynamicWorld2020_2021.class_amount)
        )

    def test_tokens_correctly_masked(self):
        x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        mask = torch.zeros_like(x)
        x = repeat(x, "b t -> b t d", d=3).clone()
        x += torch.arange(3)[None, None]

        mask[0, 1] = 1
        mask[0, 3] = 1
        mask[1, 2] = 1
        mask[1, 4] = 1

        x, orig_indices, upd_mask = Encoder.mask_tokens(x, mask)

        expected_out = torch.tensor([[1, 3, 5], [1, 2, 4]])
        expected_out = repeat(expected_out, "b t -> b t d", d=3).clone()
        expected_out += torch.arange(3)[None, None]

        self.assertTrue(torch.equal(x, expected_out))
        self.assertTrue(
            torch.equal(orig_indices, torch.tensor([[0, 2, 4, 1, 3], [0, 1, 3, 2, 4]]))
        )
        self.assertTrue(torch.equal(upd_mask, torch.tensor([[0, 0, 0], [0, 0, 0]])))

    def test_tokens_correctly_masked_unequal(self):
        x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        mask = torch.zeros_like(x)
        x = repeat(x, "b t -> b t d", d=3)

        mask[0, 1] = 1
        mask[0, 3] = 1
        mask[1, 2] = 1

        x, orig_indices, upd_mask = Encoder.mask_tokens(x, mask)

        expected_out = torch.tensor([[1, 3, 5, 0], [1, 2, 4, 5]])
        expected_out = repeat(expected_out, "b t -> b t d", d=3)
        self.assertTrue(torch.equal(x, expected_out))
        self.assertTrue(
            torch.equal(orig_indices, torch.tensor([[0, 2, 4, 1, 3], [0, 1, 3, 4, 2]]))
        )
        self.assertTrue(torch.equal(upd_mask, torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])))

    def test_tokens_correctly_unmasked(self):
        # add a -1, for the latlon embedding
        masked_x = torch.tensor([[-1, 1, 3, 5], [-1, 1, 2, 4]]).float()
        masked_x = repeat(masked_x, "b t -> b t d", d=14).clone()
        masked_x += torch.arange(14)[None, None]

        decoder = Decoder(
            channel_embeddings=nn.Embedding(2, 2),
            encoder_embed_dim=14,
            decoder_embed_dim=14,
            decoder_num_heads=2,
        )
        # the mask token is initialized to 0s
        orig_indices = torch.tensor([[0, 2, 3, 5, 1, 4], [0, 1, 3, 2, 4, 5]])
        x_mask = torch.zeros((2, 4))

        filled_tokens = decoder.add_masked_tokens(masked_x, orig_indices, x_mask)

        expected_out = torch.tensor([[-1, 0, 1, 3, 0, 5], [-1, 1, 4, 2, 0, 0]]).float()
        expected_out = repeat(expected_out, "b t -> b t d", d=14).clone()
        full_mask = torch.zeros((2, 6))
        full_mask[[0, 0, 1, 1], [1, 4, 4, 5]] = 1
        expected_out[~full_mask.bool()] += torch.arange(14)[None]
        self.assertTrue(torch.equal(filled_tokens, expected_out))

    def test_tokens_correctly_unmasked_unequal(self):
        masked_x = torch.tensor([[-1, 1, 3, 0], [-1, 1, 2, 4]]).float()
        masked_x = repeat(masked_x, "b t -> b t d", d=14)

        decoder = Decoder(
            channel_embeddings=nn.Embedding(2, 2),
            encoder_embed_dim=14,
            decoder_embed_dim=14,
            decoder_num_heads=2,
        )
        # the mask token is initialized to 0s
        orig_indices = torch.tensor([[0, 2, 3, 5, 1, 4], [0, 1, 3, 2, 4, 5]])
        x_mask = torch.zeros((2, 4))
        x_mask[0, -1] = 1

        filled_tokens = decoder.add_masked_tokens(masked_x, orig_indices, x_mask)

        expected_out = torch.tensor([[-1, 0, 1, 3, 0, 0], [-1, 1, 4, 2, 0, 0]]).float()
        expected_out = repeat(expected_out, "b t -> b t d", d=14)
        self.assertTrue(torch.equal(filled_tokens, expected_out))

    def test_encodings_correctly_added_in_decoder(self):

        # 1 batch, 3 timesteps, 13 dimensions (plus the latlon and srtm token)
        num_timesteps = 3
        x = torch.zeros((1, (num_timesteps * len(BANDS_GROUPS_IDX)) + 2, 14))

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
        # also, remove the SRTM token
        srtm_index = decoder.band_group_to_idx["SRTM"] * num_timesteps
        # srtm_token = x[:, srtm_index : srtm_index + 1, :]
        output = output[:, [i for i in range(output.shape[1]) if i != srtm_index], :]
        expected_positional_encodings = decoder.pos_embed[:, :num_timesteps, :]
        expected_month_encodings = decoder.month_embed(month_to_tensor(1, 1, num_timesteps))
        # then, for each group of channels lets make sure the embeddings are correct
        for idx in range(len(BANDS_GROUPS_IDX) + 1):
            # we record the true channel idx since removing SRTM messes things up
            true_channel_idx = idx
            if idx == decoder.band_group_to_idx["SRTM"]:
                continue
            if idx > decoder.band_group_to_idx["SRTM"]:
                idx -= 1
            # each encoding is 6 month dims, 2 channel dims and 6 pos dims
            channel_group = output[:, idx * num_timesteps : (idx + 1) * num_timesteps]
            # make sure all the channel group encodings are correct
            self.assertTrue(
                torch.equal(
                    channel_group[:, :, 6:8],
                    torch.ones_like(channel_group[:, :, 6:8]) * true_channel_idx,
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

        opt = AdamW(finetuning_model.parameters(), lr=0.1)

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
        from_config = Presto.construct(**model_kwargs)
        from_config.load_state_dict(torch.load(default_model_path, map_location=device))
        for torch_loaded, config_loaded in zip(model.parameters(), from_config.parameters()):
            self.assertTrue(torch.equal(torch_loaded, config_loaded))

    def test_reconstruct_inputs(self):

        model = Presto.construct().decoder

        class NoOp(nn.Module):
            def __init__(self, out_dim: int):
                super().__init__()
                self.out_dim = out_dim

            def forward(self, x):
                return x[:, :, : self.out_dim]

        model.eo_decoder_pred = nn.ModuleDict(
            {group_name: NoOp(len(group)) for group_name, group in model.band_groups.items()}
        )

        model.dw_decoder_pred = NoOp(DynamicWorld2020_2021.class_amount)
        batch_size, num_timesteps, num_dimensions = 1, 2, 3

        x = torch.cat(
            [torch.zeros((batch_size, 1, num_dimensions))]  # latlon token
            + [
                torch.ones(((batch_size, num_timesteps if group != "SRTM" else 1, num_dimensions)))
                * idx
                for group, idx in model.band_group_to_idx.items()
            ],
            dim=1,
        )

        eo, dw = model.reconstruct_inputs(x)

        for group, idxs in BANDS_GROUPS_IDX.items():
            relevant_vals = eo[:, :, idxs]
            self.assertTrue(torch.all(relevant_vals == model.band_group_to_idx[group]))
        self.assertTrue(torch.all(dw == model.band_group_to_idx["dynamic_world"]))

    def test_grads(self):
        encoder = Encoder()
        input = torch.ones(3, 12, 18)
        dw_input = torch.ones(3, 12).long()
        latlons = torch.rand((3, 2))
        output = encoder(input, dw_input, latlons).sum()
        output.backward()

        for name, param in encoder.named_parameters():
            if ("pos_embed" not in name) and ("month_embed" not in name):
                # the positional encoder is frozen
                self.assertIsNotNone(param.grad, msg=name)

    def test_finetuning_model_outputs_equivalent(self):

        batch_size = 3
        num_outputs = 2

        seq2seq_model = Presto.construct()
        finetuning_model = seq2seq_model.construct_finetuning_model(num_outputs=num_outputs)

        seq2seq_model.eval()
        finetuning_model.eval()

        for name, param in finetuning_model.encoder.named_parameters():
            self.assertTrue(param.equal(seq2seq_model.encoder.state_dict()[name]))

        with torch.no_grad():
            encoder_input = torch.zeros((batch_size, NUM_TIMESTEPS, NUM_BANDS))
            dw_input = torch.zeros((batch_size, NUM_TIMESTEPS)).long()
            encoder_latlons = torch.rand((batch_size, 2))
            seq2seq_encoder_output = seq2seq_model.encoder(
                encoder_input, dw_input, encoder_latlons
            )
            finetuning_encoder_output = finetuning_model.encoder(
                encoder_input, dw_input, encoder_latlons
            )
        self.assertTrue(finetuning_encoder_output.equal(seq2seq_encoder_output))

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


class TestPrestoEndToEnd(TestCase):
    @classmethod
    def setUpClass(cls):

        embedding_size = 16
        model = Presto.construct(
            encoder_embedding_size=embedding_size, decoder_embedding_size=embedding_size
        )

        class NoOp(nn.Module):
            def __init__(self, out_dim: int):
                super().__init__()
                self.out_dim = out_dim

            def forward(self, x):
                if len(x.shape) == 2:
                    x = x.unsqueeze(-1)
                if x.shape[-1] >= self.out_dim:
                    return x[:, :, : self.out_dim]
                else:
                    return repeat(x[:, :, 0], "b t -> b t d", d=self.out_dim)

        model.encoder.eo_patch_embed = nn.ModuleDict(
            {name: NoOp(embedding_size) for name, _ in model.encoder.band_groups.items()}
        )
        model.encoder.dw_embed = NoOp(embedding_size)
        model.decoder.eo_decoder_pred = nn.ModuleDict(
            {name: NoOp(len(group)) for name, group in model.decoder.band_groups.items()}
        )
        model.decoder.dw_decoder_pred = NoOp(DynamicWorld2020_2021.class_amount)

        def forward_encoder(x, dynamic_world, mask, encoder, eval_task=False):
            # THIS CODE IS FROM WITHIN THE PRESTO FUNCTION, WITH SLIGHT MODIFICATIONS #
            # if the presto code changes this will need to as well #
            all_tokens, all_masks = [], []

            for channel_group, channel_idxs in encoder.band_groups.items():
                tokens = encoder.eo_patch_embed[channel_group](x[:, :, channel_idxs])
                if channel_group == "SRTM":
                    indices = slice(0, 1)
                else:
                    indices = slice(None)

                tokens = tokens[:, indices]
                all_tokens.append(tokens)
                group_mask = torch.max(mask[:, indices, channel_idxs], dim=-1)[0]
                all_masks.append(group_mask)

            # then, dynamic world
            tokens = encoder.dw_embed(dynamic_world)
            all_tokens.append(tokens)

            # now we calculate the mask for these [b, t] tokens
            group_mask = dynamic_world == DynamicWorld2020_2021.class_amount
            all_masks.append(group_mask)

            x = torch.cat(all_tokens, dim=1)  # [batch, timesteps, embedding_dim]
            mask = torch.cat(all_masks, dim=1)  # [batch, timesteps]
            x, orig_indices, upd_mask = encoder.mask_tokens(x, mask)

            # append latlon tokens
            latlon_tokens = torch.ones((x.shape[0], 1, embedding_size)) * -1
            x = torch.cat((latlon_tokens, x), dim=1)
            upd_mask = torch.cat((torch.zeros(x.shape[0])[:, None].to(device), upd_mask), dim=1)
            orig_indices = torch.cat(
                (torch.zeros(upd_mask.shape[0])[:, None].to(device).int(), orig_indices + 1),
                dim=1,
            )
            if eval_task:
                x_for_mean = x * (1 - upd_mask.unsqueeze(-1))
                x_mean = x_for_mean.sum(dim=1) / torch.sum(1 - upd_mask, -1, keepdim=True)
                # skip norm
                return x_mean
            return x, orig_indices, upd_mask

        cls.forward_encoder = partial(forward_encoder, encoder=model.encoder)
        cls.model = model

    def test_masking_and_unmasking_end_to_end(self):
        def forward(x, dynamic_world, mask):
            # THIS CODE IS FROM WITHIN THE PRESTO FUNCTION, WITH SLIGHT MODIFICATIONS #
            # if the presto code changes this will need to as well #
            x, orig_indices, upd_mask = self.forward_encoder(
                x, dynamic_world, mask, eval_task=False
            )
            x = self.model.decoder.add_masked_tokens(x, orig_indices, upd_mask)
            return self.model.decoder.reconstruct_inputs(x)

        batch_size, timesteps = 2, 3
        x = torch.ones((batch_size, timesteps, NUM_BANDS))
        for idx, (_, indices) in enumerate(BANDS_GROUPS_IDX.items()):
            x[:, :, indices] *= idx
        # so masked values are the only values equal to 0
        x += 1

        dw_value = -2
        dynamic_world = torch.ones((batch_size, timesteps)) * dw_value
        mask = torch.zeros_like(x)

        eo, dw = forward(x, dynamic_world, mask)
        for group, idxs in BANDS_GROUPS_IDX.items():
            relevant_vals = eo[:, :, idxs]
            self.assertTrue(
                torch.all(relevant_vals == self.model.decoder.band_group_to_idx[group] + 1)
            )
        self.assertTrue(torch.all(dw == dw_value))

        mask[:, :, BANDS_GROUPS_IDX["SRTM"]] = 1
        mask[1, 1, BANDS_GROUPS_IDX["S2_RGB"]] = 1

        eo, dw = forward(x, dynamic_world, mask)
        for group, idxs in BANDS_GROUPS_IDX.items():
            relevant_vals = eo[:, :, idxs]
            if group == "SRTM":
                # the mask token is initialized to 0
                self.assertTrue(torch.all(relevant_vals == 0))
            elif group == "S2_RGB":
                self.assertTrue(torch.all(relevant_vals[1, 1] == 0))
                self.assertTrue(torch.all(relevant_vals[[0, 0, 0, 1, 1], [0, 1, 2, 0, 2]] != 0))
            else:
                self.assertTrue(
                    torch.all(relevant_vals == self.model.decoder.band_group_to_idx[group] + 1)
                )
        self.assertTrue(torch.all(dw == dw_value))

    def test_mean_tokens_end_to_end(self):
        batch_size, timesteps = 2, 3
        x = torch.ones((batch_size, timesteps, NUM_BANDS))
        sum, count = 0, 0  # to compare mean token values to
        for idx, (_, indices) in enumerate(BANDS_GROUPS_IDX.items()):
            x[:, :, indices] *= idx
            sum, count = sum + timesteps * (idx + 1), count + timesteps
        # so masked values are the only values equal to 0
        x += 1

        sum, count = sum - 1, count + 1  # latlon token is -1 in `forward` above
        # correct for srtm token that appears only once
        sum, count = sum - (timesteps - 1) * (SRTM_INDEX + 1), count - (timesteps - 1)

        dw_value = -2
        dynamic_world = torch.ones((batch_size, timesteps)) * dw_value
        sum, count = sum + timesteps * dw_value, count + timesteps
        mask = torch.zeros_like(x)

        enc = self.forward_encoder(x, dynamic_world, mask, eval_task=True)
        self.assertTrue(torch.all(enc == sum / count))

        mask[:, :, BANDS_GROUPS_IDX["SRTM"]] = 1
        mask[1, 1, BANDS_GROUPS_IDX["S2_RGB"]] = 1

        sum_0, count_0 = sum - (SRTM_INDEX + 1), count - 1  # first sample in batch
        # second sample has S2_RGB masked out in 1 timestep
        sum_1, count_1 = sum_0 - 1 * (list(BANDS_GROUPS_IDX).index("S2_RGB") + 1), count_0 - 1

        enc = self.forward_encoder(x, dynamic_world, mask, eval_task=True)
        self.assertTrue(torch.all(enc[0] == sum_0 / count_0))
        self.assertTrue(torch.all(enc[1] == sum_1 / count_1))
