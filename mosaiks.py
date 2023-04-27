import argparse
import os
from typing import List

import einops
import torch
import webdataset as wds
from tqdm import tqdm

from presto.dataops import NUM_BANDS, MaskParams
from presto.dataops.dataset import (
    TAR_BUCKET,
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)
from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    CropHarvestMultiClassValidation,
    EvalDataset,
    FuelMoistureEval,
)
from presto.model import Mosaiks1d
from presto.utils import device, seed_everything

os.environ["GOOGLE_CLOUD_PROJECT"] = "large-earth-model"

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="")
argparser.add_argument("--k", type=int, default=8192)
argparser.add_argument("--kernel_size", type=int, default=3)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_plots", type=int, default=3)

argparser.add_argument(
    "--train_url",
    type=str,
    default=f"gs://{TAR_BUCKET}/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/"
    + "dw_144_shard_{0..48}.tar",
)

argparser.set_defaults(wandb=False)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
k = args["k"]
kernel_size = args["kernel_size"]
wandb_enabled: bool = args["wandb"]

train_url: str = args["train_url"]

if wandb_enabled:
    import wandb

seed_everything(42)
# ------------ Dataloaders -------------------------------------
print("Setting up dataloaders")
# we only mask dynamic world, so that
# EO is left un-marked
mask_params = MaskParams(("dynamic_world",), 0.9)


def load_dataset(url, shuffle_on_load):
    dataset = S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021(mask_params=mask_params)
    return dataset.as_webdataset(url, shuffle_on_load)


train_dataset = load_dataset(train_url, shuffle_on_load=True)
train_dataloader = wds.WebLoader(train_dataset, batch_size=k)


cropharvest_validation = CropHarvestMultiClassValidation()

# ------------ Model -----------------------------------------
print("Setting up model")
# we already randomly select the start month in the train dataloader,
# so if we just take the first `kernel_size` timesteps here we will
# be getting a variety of seasons
eo_patches = next(iter(train_dataloader))[2][:, :kernel_size, :]
eo_patches = einops.rearrange(eo_patches, "k timestep channel -> k channel timestep")
model = Mosaiks1d(in_channels=NUM_BANDS, k=k, kernel_size=3, patches=eo_patches)
model.to(device)
model.eval()


training_config = {
    "model": model.__class__,
    "k": k,
    "kernel_size": kernel_size,
    "device": device,
    "train_url": train_url,
}

if wandb_enabled:
    run = wandb.init(entity="nasa-harvest", project="lem", config=training_config)

with torch.no_grad():
    results = cropharvest_validation.finetuning_results(
        model, model_modes=["Regression", "Random Forest"]
    )
    wandb.log(results)

print("Loading evaluation tasks")
eval_task_list: List[EvalDataset] = [
    FuelMoistureEval(),
    AlgaeBloomsEval(),
    CropHarvestEval("Kenya"),
    CropHarvestEval("Togo"),
    CropHarvestEval("Brazil"),
]

for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
    print("\n" + eval_task.name, flush=True)
    lr_results = eval_task.finetuning_results(model, model_modes=["Regression", "Random Forest"])
    print(lr_results, flush=True)
    if wandb_enabled:
        wandb.log(lr_results)

    eval_task.clear_data()

if wandb_enabled and run:
    run.finish()
    print(f"Wandb url: {run.url}")
