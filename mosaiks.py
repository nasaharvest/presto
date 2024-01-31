import argparse
import logging
import os
from typing import List

import einops
import webdataset as wds
from tqdm import tqdm

from presto.dataops import NUM_BANDS, MaskParams
from presto.dataops.dataset import (
    TAR_BUCKET,
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)
from presto.eval import AlgaeBloomsEval, CropHarvestEval, EvalTask, FuelMoistureEval
from presto.model import Mosaiks1d
from presto.utils import DEFAULT_SEED, device, initialize_logging, seed_everything

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
    + "dw_144_shard_{0..58}.tar",
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
logger = logging.getLogger("__main__")
initialize_logging(output_dir="", to_file=False)

# ------------ Dataloaders -------------------------------------
logger.info("Setting up dataloaders")
# mosaiks ignores any masking, so there is no need
# to mask anything (0.0 masking ratio)
mask_params = MaskParams(("random_combinations",), 0.0)


def load_dataset(url, shuffle_on_load):
    dataset = S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021(mask_params=mask_params)
    return dataset.as_webdataset(url, shuffle_on_load)


train_dataset = load_dataset(train_url, shuffle_on_load=True)
train_dataloader = wds.WebLoader(train_dataset, batch_size=k)


for seed in [0, DEFAULT_SEED, 84]:
    logger.info("Loading model")
    # ------------ Model -----------------------------------------
    logger.info("Setting up model")
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
        "seed": seed,
    }
    if wandb_enabled:
        run = wandb.init(
            entity="nasa-harvest", project="presto-downstream", config=training_config
        )

    logger.info("Loading evaluation tasks")
    eval_task_list: List[EvalTask] = [
        FuelMoistureEval(seed=seed),
        AlgaeBloomsEval(seed=seed),
        CropHarvestEval("Kenya", seed=seed),
        CropHarvestEval("Togo", seed=seed),
        CropHarvestEval("Brazil", seed=seed),
    ]

    for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
        logger.info(eval_task.name)
        lr_results = eval_task.finetuning_results(
            model, model_modes=["Regression", "Random Forest"]
        )
        logger.info(lr_results)
        if wandb_enabled:
            wandb.log(lr_results)

        eval_task.clear_data()

    if wandb_enabled and run:
        run.finish()
        logger.info(f"Wandb url: {run.url}")
