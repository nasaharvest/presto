import torch
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from torch.utils.benchmark import Timer

NUM_ITERATIONS = 100
VARIABLE_LENGTH_RATIO = 0.5
BATCH_SIZE = 4096
REPEATS = 1

setup = f"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import webdataset as wds
from torch import optim

from presto import Presto
from presto.dataops import MASK_STRATEGIES, MaskParams
from presto.dataops.dataset import (
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)
from presto.model import LossWrapper

train_url: str = "data/dw_144_mini_shard_44.tar"
device = torch.device("cuda:0")
path_to_config = Path("config") / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

# ------------ Dataloaders -------------------------------------
# Set mask_ratio to 0.0 here because it will generate masks that result in
#  sequences with equal length
mask_params = MaskParams(MASK_STRATEGIES, 0.0)


def load_dataset(url, shuffle_on_load):
    dataset = S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021(mask_params=mask_params)
    return dataset.as_webdataset(url, shuffle_on_load)


train_dataset = load_dataset("data/dw_144_mini_shard_44.tar", shuffle_on_load=True)
train_dataloader = wds.WebLoader(train_dataset, batch_size={BATCH_SIZE})

mse = LossWrapper(nn.MSELoss())

b = next(iter(train_dataloader))
mask, x, y, start_month = b[0].to(device), b[2].to(device), b[3].to(device), b[6]
dw_mask, x_dw, y_dw = b[1].to(device), b[4].to(device).long(), b[5].to(device).long()
latlons = b[7].to(device)

x = torch.repeat_interleave(x, {REPEATS}, dim=1)
y = torch.repeat_interleave(y, {REPEATS}, dim=1)
x_dw = torch.repeat_interleave(x_dw, {REPEATS}, dim=1)
y_dw = torch.repeat_interleave(y_dw, {REPEATS}, dim=1)
mask = torch.repeat_interleave(mask, {REPEATS}, dim=1)
"""

model_sdpa_setup = """
_model = Presto.construct(**model_kwargs, attn="sdpa", max_sequence_length=480)
model = _model.construct_finetuning_model(num_outputs=1, regression=True)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
model.eval()
"""

model_flash_setup = """
_model = Presto.construct(**model_kwargs, attn="flash", max_sequence_length=480)
model = _model.construct_finetuning_model(num_outputs=1, regression=True)
model.to(device)
model.eval()
"""

bfloat16_setup = """
latlons = latlons.bfloat16()
x = x.bfloat16()
y = y.bfloat16()
model = model.bfloat16()
"""

variable_length_mask_setup = f"""
B, T, C = mask.shape
total_tokens = B * T * C
num_tokens_to_mask = int(total_tokens * {VARIABLE_LENGTH_RATIO})

# Create flat mask and randomly select positions to mask
flat_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
mask_indices = torch.randperm(total_tokens, device=device)[:num_tokens_to_mask]
flat_mask[mask_indices] = True

# Reshape back to (B, T, C)
mask = ~flat_mask.reshape(B, T, C)

# Apply mask: x should be zero where mask is False, y should be zero where mask is True
x = torch.where(mask, torch.zeros_like(x), x)
y = torch.where(mask, y, torch.zeros_like(y))
"""

forward_only = """
with torch.no_grad():
    y_pred = model(x, mask=mask, dynamic_world=x_dw, latlons=latlons, month=start_month)
    loss = mse(y_pred, torch.ones_like(y_pred))
"""

device = torch.device("cuda:0")

# Run these with pytorch 2.0, and without the code changes to presto.py
#   (or just comment the line that imports flash_attn_varlen_qkvpacked_func)

# timer = Timer(
#     stmt=forward_backward,
#     setup=setup + model_sdpa_setup,
#     label="Pytorch 2.0, scaled_dot_product_attention, on A100, set to None",
# )
# print(timer.timeit(NUM_ITERATIONS))
# print(max_memory_allocated(device) / 10**6)
# reset_peak_memory_stats(device)
# timer = Timer(
#     stmt=forward_backward,
#     setup=setup + model_sdpa_setup + variable_length_mask_setup,
#     label="Pytorch 2.0, scaled_dot_product_attention, on A100, varlen mask",
# )
# print(timer.timeit(NUM_ITERATIONS))
# print(max_memory_allocated(device) / 10**6)

# reset_peak_memory_stats(device)
# timer = Timer(
#     stmt=forward_backward,
#     setup=setup + model_sdpa_setup + bfloat16_setup,
#     label="Pytorch 2.0, scaled_dot_product_attention, on A100, bfloat16",
# )
# print(timer.timeit(NUM_ITERATIONS))
# print(max_memory_allocated(device) / 10**6)

# reset_peak_memory_stats(device)
# timer = Timer(
#     stmt=forward_backward,
#     setup=setup + model_sdpa_setup + bfloat16_setup + variable_length_mask_setup,
#     label="Pytorch 2.0, scaled_dot_product_attention, on A100, bfloat16, varlen mask",
# )
# print(timer.timeit(NUM_ITERATIONS))
# print(max_memory_allocated(device) / 10**6)

# exit()
# run
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_sdpa_setup,
    label="Pytorch 2.2, scaled_dot_product_attention, on A100, NOT set to None",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)

reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_sdpa_setup + variable_length_mask_setup,
    label="Pytorch 2.2, scaled_dot_product_attention, on A100, varlen mask",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)

reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_sdpa_setup + bfloat16_setup,
    label="Pytorch 2.2, scaled_dot_product_attention, on A100, bfloat16, set to None",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)

reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_sdpa_setup + bfloat16_setup + variable_length_mask_setup,
    label="Pytorch 2.2, scaled_dot_product_attention, on A100, bfloat16, varlen mask",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)

# # 2.7.4.post1 because latest version gave error `lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found`
# #  may be a LINUX version issue, may be a python version issue? Tested for python 3.9
# # https://github.com/Dao-AILab/flash-attention/issues/1708
# # https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0
# # pip install packaging ninja
# # pip install flash-attn==2.7.4.post1 --no-build-isolation

reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_flash_setup + bfloat16_setup,
    label="Pytorch 2.2, flashattention `flash_attn_varlen_qkvpacked_func`, on A100, bfloat16",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)
reset_peak_memory_stats(device)
timer = Timer(
    stmt=forward_only,
    setup=setup + model_flash_setup + bfloat16_setup + variable_length_mask_setup,
    label="Pytorch 2.2, flashattention `flash_attn_varlen_qkvpacked_func`, on A100, bfloat16, varlen mask",
)
print(timer.timeit(NUM_ITERATIONS))
print(max_memory_allocated(device) / 10**6)
