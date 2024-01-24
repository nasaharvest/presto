from unittest import TestCase

import numpy as np

from presto.dataops.masking import BANDS_GROUPS_IDX, NUM_TIMESTEPS, make_mask

TEST_MASK_RATIOS = [x / 100 for x in range(5, 100, 5)]


class TestS1S2ERA5SRTM(TestCase):
    def test_make_mask_group_bands(self):
        for mask_ratio in TEST_MASK_RATIOS:
            eo_mask, dw_mask = make_mask(strategy="group_bands", mask_ratio=mask_ratio)

            num_masked_channels = 0
            num_masked_tokens = 0
            srtm_masked = False
            for channel, indices in BANDS_GROUPS_IDX.items():
                channel_vals = eo_mask[:, indices]
                if channel == "SRTM":
                    self.assertTrue(channel_vals.max() == channel_vals.min())
                    if channel_vals.max() == 1:
                        num_masked_tokens += 1
                        srtm_masked = True
                else:
                    for timestep in range(channel_vals.shape[0]):
                        # within a token, all timestep should be correctly masked
                        self.assertTrue(
                            channel_vals[timestep].max() == channel_vals[timestep].min()
                        )
                        num_masked_tokens += channel_vals[timestep].max()
                    if channel_vals.max() == channel_vals.min() == 1:
                        num_masked_channels += 1

            for timestep in range(dw_mask.shape[0]):
                num_masked_tokens += dw_mask[timestep].max()
            if dw_mask.max() == dw_mask.min() == 1:
                num_masked_channels += 1
            expected_masked_tokens = int(
                ((NUM_TIMESTEPS * len(BANDS_GROUPS_IDX)) + 1) * mask_ratio
            )
            self.assertTrue(num_masked_tokens == expected_masked_tokens)
            if srtm_masked:
                expected_masked_tokens -= 1
            self.assertTrue(num_masked_channels == int(expected_masked_tokens / NUM_TIMESTEPS))

    def test_make_mask_timesteps(self):

        for mask_ratio in TEST_MASK_RATIOS:
            for masking_strategy in ["random_timesteps", "chunk_timesteps"]:
                eo_mask, dw_mask = make_mask(strategy=masking_strategy, mask_ratio=mask_ratio)

                num_masked_timesteps = 0
                num_masked_tokens = 0
                srtm_masked = False

                channel_vals = eo_mask[:, BANDS_GROUPS_IDX["SRTM"]]
                self.assertTrue(channel_vals.max() == channel_vals.min())
                if channel_vals.max() == 1:
                    num_masked_tokens += 1
                    srtm_masked = True

                eo_mask_no_srtm = eo_mask[
                    :, [x for x in range(eo_mask.shape[1]) if x not in BANDS_GROUPS_IDX["SRTM"]]
                ]
                total_mask = np.concatenate([eo_mask_no_srtm, np.expand_dims(dw_mask, 1)], axis=1)
                for timestep in range(total_mask.shape[0]):

                    timestep_vals = total_mask[timestep, :]
                    if timestep_vals.max() == timestep_vals.min() == 1:
                        num_masked_timesteps += 1

                for channel, indices in BANDS_GROUPS_IDX.items():
                    channel_vals = eo_mask[:, indices]
                    if channel == "SRTM":
                        continue  # already tested
                    else:
                        for timestep in range(channel_vals.shape[0]):
                            # within a token, all timestep should be correctly masked
                            self.assertTrue(
                                channel_vals[timestep].max() == channel_vals[timestep].min()
                            )
                            num_masked_tokens += channel_vals[timestep].max()

                for timestep in range(dw_mask.shape[0]):
                    num_masked_tokens += dw_mask[timestep].max()

            expected_masked_tokens = int(
                ((NUM_TIMESTEPS * len(BANDS_GROUPS_IDX)) + 1) * mask_ratio
            )
            self.assertTrue(num_masked_tokens == expected_masked_tokens)
            if srtm_masked:
                expected_masked_tokens -= 1
            self.assertTrue(
                num_masked_timesteps == int(expected_masked_tokens / len(BANDS_GROUPS_IDX))
            )
