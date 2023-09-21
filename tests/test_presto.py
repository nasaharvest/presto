import json
from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import torch
from einops import repeat
from torch import nn
from torch.optim import AdamW

from presto.dataops import NUM_BANDS, NUM_ORG_BANDS, NUM_TIMESTEPS
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
        # if nothing is masked, we expect to have (NUM_TIMESTEPS * (band_groups)) + 2 tokens
        self.assertTrue(x.shape[1] == 2 + (NUM_TIMESTEPS * len(BANDS_GROUPS_IDX)))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == 1 + (NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
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
        self.assertTrue(x.shape[1] == 2 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 1)))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == 1 + (NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
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
        self.assertTrue(x.shape[1] == 2 + (NUM_TIMESTEPS * (len(BANDS_GROUPS_IDX) - 2)))
        self.assertTrue(kept_indices.shape[1] + 1 == x.shape[1])  # +1 because of the latlon token
        self.assertTrue(
            kept_indices.shape[1] + removed_indices.shape[1]
            == 1 + (NUM_TIMESTEPS * len(BANDS_GROUPS_IDX))
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
        from_config = Presto.construct(**model_kwargs)
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

    def test_masking_and_unmasking_end_to_end(self):

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

        def forward(x, dynamic_world, mask, model):
            # THIS CODE IS FROM WITHIN THE PRESTO FUNCTION, WITH SLIGHT MODIFICATIONS #
            # if the presto code changes this will need to as well #
            all_tokens, all_masks = [], []

            for channel_group, channel_idxs in model.encoder.band_groups.items():
                tokens = model.encoder.eo_patch_embed[channel_group](x[:, :, channel_idxs])
                if channel_group == "SRTM":
                    indices = slice(0, 1)
                else:
                    indices = slice(None)

                tokens = tokens[:, indices]
                all_tokens.append(tokens)
                group_mask = repeat(
                    torch.max(mask[:, indices, channel_idxs], dim=-1)[0],
                    "b t -> b t d",
                    d=tokens.shape[-1],
                )
                all_masks.append(group_mask)

            # then, dynamic world
            tokens = model.encoder.dw_embed(dynamic_world)
            all_tokens.append(tokens)

            # now we calculate the mask for these [b, t] tokens
            group_mask = repeat(
                dynamic_world == DynamicWorld2020_2021.class_amount,
                "b t -> b t d",
                d=tokens.shape[-1],
            )
            all_masks.append(group_mask)

            x = torch.cat(all_tokens, dim=1)  # [batch, timesteps, embedding_dim]
            mask = torch.cat(all_masks, dim=1)  # [batch, timesteps, embedding_dim]
            x, kept_indices, removed_indices = model.encoder.mask_tokens(x, mask)

            # append latlon tokens
            latlon_tokens = torch.ones((x.shape[0], 1, embedding_size)) * -1
            x = torch.cat((latlon_tokens, x), dim=1)

            x = model.decoder.add_masked_tokens(x, kept_indices, removed_indices)
            return model.decoder.reconstruct_inputs(x)

        batch_size, timesteps = 1, 3
        x = torch.ones((batch_size, timesteps, NUM_BANDS))
        for idx, (_, indices) in enumerate(BANDS_GROUPS_IDX.items()):
            x[:, :, indices] *= idx

        dw_value = -2
        dynamic_world = torch.ones((batch_size, timesteps)) * dw_value
        mask = torch.zeros_like(x)

        eo, dw = forward(x, dynamic_world, mask, model)
        for group, idxs in BANDS_GROUPS_IDX.items():
            relevant_vals = eo[:, :, idxs]
            self.assertTrue(torch.all(relevant_vals == model.decoder.band_group_to_idx[group]))
        self.assertTrue(torch.all(dw == dw_value))

        mask[:, :, BANDS_GROUPS_IDX["SRTM"]] = 1

        eo, dw = forward(x, dynamic_world, mask, model)
        for group, idxs in BANDS_GROUPS_IDX.items():
            relevant_vals = eo[:, :, idxs]
            if group != "SRTM":
                self.assertTrue(torch.all(relevant_vals == model.decoder.band_group_to_idx[group]))
            else:
                # the mask token is initialized to 0
                self.assertTrue(torch.all(relevant_vals == 0))
        self.assertTrue(torch.all(dw == dw_value))
