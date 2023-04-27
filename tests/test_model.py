from unittest import TestCase

import torch

from presto import Presto
from presto.dataops import BANDS_GROUPS_IDX, NUM_BANDS, NUM_TIMESTEPS
from presto.dataops.pipelines.dynamicworld import DynamicWorld2020_2021
from presto.presto import Encoder


class TestModel(TestCase):
    def test_model_init(self):
        batch_size = 3
        input = torch.zeros((batch_size, NUM_TIMESTEPS, NUM_BANDS))
        mask = torch.ones_like(input)
        dynamic_world = torch.ones((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))
        model = Presto.construct(BANDS_GROUPS_IDX)
        output, dynamic_world_output = model(
            input, dynamic_world=dynamic_world, latlons=latlons, mask=mask
        )

        self.assertEqual(output.shape, input.shape)
        self.assertEqual(
            dynamic_world_output.shape,
            (batch_size, NUM_TIMESTEPS, DynamicWorld2020_2021.class_amount),
        )

    def test_grads(self):
        encoder = Encoder(band_groups=BANDS_GROUPS_IDX)
        input = torch.ones(3, 12, 18)
        dw_input = torch.ones(3, 12).long()
        latlons = torch.rand((3, 2))
        output = encoder(input, dw_input, latlons).sum()
        output.backward()

        for name, param in encoder.named_parameters():
            if ("pos_embed" not in name) and ("month_embed" not in name):
                # the positional encoder is frozen
                self.assertIsNotNone(param.grad, msg=name)

    def test_finetuning_model_grads_flow_correctly(self):

        batch_size = 3
        num_outputs = 2

        seq2seq_model = Presto.construct(band_groups=BANDS_GROUPS_IDX)
        finetuning_model = seq2seq_model.construct_finetuning_model(num_outputs=num_outputs)

        input = torch.zeros((batch_size, NUM_TIMESTEPS, NUM_BANDS))
        dw_input = torch.zeros((batch_size, NUM_TIMESTEPS)).long()
        latlons = torch.rand((batch_size, 2))
        output = finetuning_model(input, dw_input, latlons)

        self.assertEqual(output.shape, (batch_size, num_outputs))

        output.sum().backward()

        for param in finetuning_model.head.parameters():
            self.assertIsNotNone(param.grad)

        for name, param in finetuning_model.encoder.named_parameters():
            if ("pos_embed" not in name) and ("month_embed" not in name):
                # the positional encoder is frozen
                self.assertIsNotNone(param.grad, msg=name)

        for param in seq2seq_model.encoder.parameters():
            self.assertIsNone(param.grad, msg=name)

    def test_finetuning_model_outputs_equivalent(self):

        batch_size = 3
        num_outputs = 2

        seq2seq_model = Presto.construct(band_groups=BANDS_GROUPS_IDX)
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
