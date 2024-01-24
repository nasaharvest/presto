import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import List, Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import webdataset as wds
from torch import optim
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from presto import Presto
from presto.dataops import BANDS_GROUPS_IDX, MASK_STRATEGIES, MaskParams, plot_masked
from presto.dataops.dataset import (
    TAR_BUCKET,
    S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021,
)
from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    CropHarvestMultiClassValidation,
    EuroSatEval,
    EvalTask,
    FuelMoistureEval,
    TreeSatEval,
)
from presto.model import LossWrapper, adjust_learning_rate, param_groups_weight_decay
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    device,
    initialize_logging,
    seed_everything,
    timestamp_dirname,
    update_data_dir,
)

logger = logging.getLogger("__main__")
os.environ["GOOGLE_CLOUD_PROJECT"] = "large-earth-model"

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Parent directory to save output to, <output_dir>/wandb/ "
    "and <output_dir>/output/ will be written to. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Data is stored in <data_dir>/data. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument("--n_epochs", type=int, default=20)
argparser.add_argument("--val_per_n_steps", type=int, default=1000)
argparser.add_argument(
    "--cropharvest_per_n_validations",
    type=int,
    default=10,
    help="0 to skip cropharvest validation",
)
argparser.add_argument(
    "--cropharvest_val_n_per_class",
    type=int,
    default=-1,
    help="-1 for no limit",
)
argparser.add_argument("--max_learning_rate", type=float, default=0.001)
argparser.add_argument("--min_learning_rate", type=float, default=0.0)
argparser.add_argument("--warmup_epochs", type=int, default=2)

argparser.add_argument("--weight_decay", type=float, default=0.05)
argparser.add_argument(
    "--dynamic_world_loss_weight",
    type=float,
    default=2,
    help="Each dynamic world instance we be weighted by this amount relative to each eo instance",
)
argparser.add_argument("--batch_size", type=int, default=4096)
argparser.add_argument(
    "--dataloader_length", type=int, default=5950, help="-1 to re-estimate dataloader length"
)
argparser.add_argument(
    "--mask_strategies",
    type=str,
    default=[
        "group_bands",
        "random_timesteps",
        "chunk_timesteps",
        "random_combinations",
    ],
    nargs="+",
    help="`all` will use all available masking strategies (including single bands)",
)
argparser.add_argument("--mask_ratio", type=float, default=0.75)
argparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.add_argument("--wandb_plots", type=int, default=3)
argparser.add_argument("--wandb_org", type=str, default="nasa-harvest")

argparser.add_argument(
    "--train_url",
    type=str,
    default=f"gs://{TAR_BUCKET}/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/"
    + "dw_144_shard_{0..58}.tar",
)
argparser.add_argument(
    "--val_url",
    type=str,
    default=f"gs://{TAR_BUCKET}/S1_S2_ERA5_SRTM_2020_2021_DynamicWorldMonthly2020_2021_tars/"
    + "dw_144_shard_59.tar",
)
argparser.add_argument("--skip_finetuning", dest="skip_finetuning", action="store_true")

argparser.set_defaults(wandb=False)
argparser.set_defaults(skip_finetuning=False)
args = argparser.parse_args().__dict__

model_name = args["model_name"]
seed: int = args["seed"]
path_to_config = args["path_to_config"]
wandb_enabled: bool = args["wandb"]
wandb_plots: int = args["wandb_plots"]
wandb_org: str = args["wandb_org"]

seed_everything(seed)

output_parent_dir = Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
run_id = None
if wandb_enabled:
    import wandb

    run = wandb.init(
        entity=wandb_org,
        project="lem",
        dir=output_parent_dir,
    )
    run_id = cast(Run, run).id

logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(logging_dir)
logger.info("Using output dir: %s" % logging_dir)

data_dir = args["data_dir"]
if data_dir != "":
    update_data_dir(data_dir)

num_epochs = args["n_epochs"]
val_per_n_steps = args["val_per_n_steps"]
cropharvest_per_n_validations = args["cropharvest_per_n_validations"]
cropharvest_val_n_per_class = args["cropharvest_val_n_per_class"]
dynamic_world_loss_weight = args["dynamic_world_loss_weight"]
max_learning_rate = args["max_learning_rate"]
min_learning_rate = args["min_learning_rate"]
warmup_epochs = args["warmup_epochs"]
weight_decay = args["weight_decay"]
batch_size = args["batch_size"]

mask_strategies: Tuple[str, ...] = tuple(args["mask_strategies"])
if (len(mask_strategies) == 1) and (mask_strategies[0] == "all"):
    mask_strategies = MASK_STRATEGIES
mask_ratio: float = args["mask_ratio"]

train_url: str = args["train_url"]
val_url: str = args["val_url"]
dataloader_length: int = args["dataloader_length"]

if (batch_size != argparser.get_default("batch_size")) & (
    dataloader_length == argparser.get_default("dataloader_length")
):
    warnings.warn(
        "Dataloader length calculated for a specific batch size. "
        "Set dataloader_length to -1 to recalculate"
    )

skip_finetuning: bool = args["skip_finetuning"]

if path_to_config == "":
    path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))

# ------------ Dataloaders -------------------------------------
logger.info("Setting up dataloaders")
mask_params = MaskParams(mask_strategies, mask_ratio)


def load_dataset(url, shuffle_on_load):
    dataset = S1_S2_ERA5_SRTM_DynamicWorldMonthly_2020_2021(mask_params=mask_params)
    return dataset.as_webdataset(url, shuffle_on_load)


train_dataset = load_dataset(train_url, shuffle_on_load=True)
val_dataset = load_dataset(val_url, shuffle_on_load=False)
train_dataloader = wds.WebLoader(train_dataset, batch_size=batch_size)
val_dataloader = wds.WebLoader(val_dataset, batch_size=batch_size)

if dataloader_length == -1:
    logger.info("Finding train dataloader length")
    dataloader_length = 0
    for _ in train_dataloader:
        dataloader_length += 1
    logger.info("train_dataloader length: ", dataloader_length)


if cropharvest_per_n_validations != 0:
    cropharvest_validation = CropHarvestMultiClassValidation(
        n_per_class=cropharvest_val_n_per_class if cropharvest_val_n_per_class != -1 else None
    )


# ------------ Model -----------------------------------------
logger.info("Setting up model")
model = Presto.construct(**model_kwargs)
model.to(device)

# ------------ Model hyperparameters -------------------------------------
param_groups = param_groups_weight_decay(model, weight_decay)
optimizer = optim.AdamW(param_groups, lr=max_learning_rate, betas=(0.9, 0.95))
mse = LossWrapper(nn.MSELoss())
ce = LossWrapper(nn.CrossEntropyLoss())

training_config = {
    "model": model.__class__,
    "encoder": model.encoder.__class__,
    "decoder": model.decoder.__class__,
    "optimizer": optimizer.__class__.__name__,
    "eo_loss": mse.loss.__class__.__name__,
    "dynamic_world_loss": ce.loss.__class__.__name__,
    "device": device,
    **args,
    **model_kwargs,
}

if wandb_enabled:
    wandb.config.update(training_config)

    examples = []

    # Dynamic world masking is not visualizable in this setting
    for ex in train_dataset:
        examples.append(ex)
        if len(examples) >= wandb_plots:
            break
    for ex in val_dataset:
        examples.append(ex)
        if len(examples) >= wandb_plots * 2:
            break

    def to_tensor(ex):
        return torch.from_numpy(ex).to(device)

    masks_tensor = torch.stack([to_tensor(ex.mask_eo) for ex in examples])
    xs_tensor = torch.stack([to_tensor(ex.x_eo) for ex in examples])
    x_dws_tensor = torch.stack([to_tensor(ex.x_dw).long() for ex in examples])
    start_months_tensor = torch.stack([torch.tensor(ex.start_month).to(device) for ex in examples])
    latlons_tensor = torch.stack([to_tensor(ex.latlon) for ex in examples])

    def plot_predictions(model):
        with torch.no_grad():
            eo_preds, dw_preds = model(
                xs_tensor,
                mask=masks_tensor,
                dynamic_world=x_dws_tensor,
                latlons=latlons_tensor,
                month=start_months_tensor,
            )
        name_plots_list = []
        for i, example in enumerate(examples):
            if i < wandb_plots:
                title = f"plot_train_{i}_{example.strategy}"
            else:
                title = f"plot_val_{i}_{example.strategy}"
            fig = plot_masked(
                example=example,
                eo_pred=eo_preds[i].cpu().numpy(),
                dw_pred=dw_preds[i].cpu().numpy(),
            )
            name_plots_list.append((title, wandb.Image(fig)))
        return name_plots_list


lowest_validation_loss = None
best_val_epoch = 0
training_step = 0
num_validations = 0

with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:
    for epoch in tqdm_epoch:
        # ------------------------ Training ----------------------------------------
        total_train_loss = 0.0
        total_eo_train_loss = 0.0
        total_dw_train_loss = 0.0
        total_num_eo_values_masked = 0
        total_num_dw_values_masked = 0
        num_updates_being_captured = 0
        train_size = 0
        model.train()
        for epoch_step, b in enumerate(tqdm(train_dataloader, desc="Train", leave=False)):
            mask, x, y, start_month = b[0].to(device), b[2].to(device), b[3].to(device), b[6]
            dw_mask, x_dw, y_dw = b[1].to(device), b[4].to(device).long(), b[5].to(device).long()
            latlons = b[7].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            lr = adjust_learning_rate(
                optimizer,
                epoch_step / dataloader_length + epoch,
                warmup_epochs,
                num_epochs,
                max_learning_rate,
                min_learning_rate,
            )
            # Get model outputs and calculate loss
            y_pred, dw_pred = model(
                x, mask=mask, dynamic_world=x_dw, latlons=latlons, month=start_month
            )
            # set all SRTM timesteps except the first one to unmasked, so that
            # they will get ignored by the loss function even if the SRTM
            # value was masked
            mask[:, 1:, BANDS_GROUPS_IDX["SRTM"]] = False
            loss = mse(y_pred[mask], y[mask])
            dw_loss = ce(dw_pred[dw_mask], y_dw[dw_mask])
            num_eo_masked, num_dw_masked = len(y_pred[mask]), len(dw_pred[dw_mask])
            with torch.no_grad():
                ratio = num_dw_masked / max(num_eo_masked, 1)
                # weight shouldn't be > 1
                weight = min(1, dynamic_world_loss_weight * ratio)

            total_loss = loss + weight * dw_loss
            total_loss.backward()
            optimizer.step()

            current_batch_size = len(x)
            total_train_loss += total_loss.item()
            total_eo_train_loss += loss.item() * num_eo_masked
            total_dw_train_loss += dw_loss.item() * num_dw_masked
            total_num_eo_values_masked += num_eo_masked
            total_num_dw_values_masked += num_dw_masked
            num_updates_being_captured += 1
            train_size += current_batch_size
            training_step += 1

            # ------------------------ Validation --------------------------------------
            if training_step % val_per_n_steps == 0:
                total_val_loss = 0.0
                total_eo_val_loss = 0.0
                total_dw_val_loss = 0.0
                total_val_num_eo_values_masked = 0
                total_val_num_dw_values_masked = 0
                num_val_updates_captured = 0
                val_size = 0
                model.eval()
                with torch.no_grad():
                    for b in tqdm(val_dataloader, desc="Validate"):
                        mask, x, y, start_month = (
                            b[0].to(device),
                            b[2].to(device),
                            b[3].to(device),
                            b[6],
                        )
                        dw_mask, x_dw = b[1].to(device), b[4].to(device).long()
                        y_dw, latlons = b[5].to(device).long(), b[7].to(device)
                        # Get model outputs and calculate loss
                        y_pred, dw_pred = model(
                            x, mask=mask, dynamic_world=x_dw, latlons=latlons, month=start_month
                        )
                        # set all SRTM timesteps except the first one to unmasked, so that
                        # they will get ignored by the loss function even if the SRTM
                        # value was masked
                        mask[:, 1:, BANDS_GROUPS_IDX["SRTM"]] = False
                        loss = mse(y_pred[mask], y[mask])
                        dw_loss = ce(dw_pred[dw_mask], y_dw[dw_mask])
                        num_eo_masked, num_dw_masked = len(y_pred[mask]), len(dw_pred[dw_mask])
                        with torch.no_grad():
                            ratio = num_dw_masked / max(num_eo_masked, 1)
                            # weight shouldn't be > 1
                            weight = min(1, dynamic_world_loss_weight * ratio)
                        total_loss = loss + weight * dw_loss
                        current_batch_size = len(x)
                        val_size += current_batch_size
                        total_val_loss += total_loss.item()
                        total_eo_val_loss += loss.item() * num_eo_masked
                        total_dw_val_loss += dw_loss.item() * num_dw_masked
                        total_val_num_eo_values_masked += num_eo_masked
                        total_val_num_dw_values_masked += num_dw_masked
                        num_val_updates_captured += 1

                # ------------------------ Metrics + Logging -------------------------------
                if (
                    (cropharvest_per_n_validations != 0)
                    and (num_validations % cropharvest_per_n_validations == 0)
                    and wandb_enabled
                ):
                    results = cropharvest_validation.finetuning_results(
                        model, model_modes=["Regression", "Random Forest"]
                    )
                    results["epoch"] = epoch
                    wandb.log(results)

                # train_loss now reflects the value against which we calculate gradients
                train_loss = total_train_loss / num_updates_being_captured
                train_eo_loss = total_eo_train_loss / max(total_num_eo_values_masked, 1)
                train_dw_loss = total_dw_train_loss / max(total_num_dw_values_masked, 1)

                val_loss = total_val_loss / num_val_updates_captured
                val_eo_loss = total_eo_val_loss / max(total_val_num_eo_values_masked, 1)
                val_dw_loss = total_dw_val_loss / max(total_val_num_dw_values_masked, 1)

                if "train_size" not in training_config and "val_size" not in training_config:
                    training_config["train_size"] = train_size
                    training_config["val_size"] = val_size
                    if wandb_enabled:
                        wandb.config.update(training_config)

                to_log = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_eo_loss": train_eo_loss,
                    "val_eo_loss": val_eo_loss,
                    "train_dynamic_world_loss": train_dw_loss,
                    "val_dynamic_world_loss": val_dw_loss,
                    "training_step": training_step,
                    "epoch": epoch,
                    "lr": lr,
                }
                tqdm_epoch.set_postfix(loss=val_loss)

                if lowest_validation_loss is None or val_loss < lowest_validation_loss:
                    lowest_validation_loss = val_loss
                    best_val_epoch = epoch

                    model_path = logging_dir / Path("models")
                    model_path.mkdir(exist_ok=True, parents=True)

                    best_model_path = model_path / f"{model_name}{epoch}.pt"
                    logger.info(f"Saving best model to: {best_model_path}")
                    torch.save(model.state_dict(), best_model_path)

                # reset training logging
                total_train_loss = 0.0
                total_eo_train_loss = 0.0
                total_dw_train_loss = 0.0
                total_num_eo_values_masked = 0
                total_num_dw_values_masked = 0
                num_updates_being_captured = 0
                train_size = 0
                num_validations += 1

                if wandb_enabled:
                    model.eval()
                    for title, plot in plot_predictions(model):
                        to_log[title] = plot
                    wandb.log(to_log)
                    plt.close("all")

                model.train()

logger.info(f"Done training, best model saved to {best_model_path}")

if not skip_finetuning:
    # retreive the best model
    logger.info("Loading best model: %s" % best_model_path)
    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model)

    logger.info("Loading evaluation tasks")
    seeds = [0, DEFAULT_SEED, 84]
    eval_task_list: List[EvalTask] = [
        *[
            CropHarvestEval(country=country, ignore_dynamic_world=idw, seed=seed)
            for country in ["Kenya", "Togo", "Brazil"]
            for idw in [True, False]
            for seed in seeds
        ],
        *[FuelMoistureEval(seed=seed) for seed in seeds],
        *[AlgaeBloomsEval(seed=seed) for seed in seeds],
        *[
            EuroSatEval(rgb=rgb, input_patch_size=ps, seed=seed)
            for rgb in [True, False]
            for ps in [1, 2, 4, 8, 16, 32, 64]
            for seed in seeds
        ],
        *[TreeSatEval(subset=subset, seed=seed) for subset in ["S1", "S2"] for seed in seeds],
        *[
            CropHarvestEval("Togo", ignore_dynamic_world=True, num_timesteps=x, seed=seed)
            for x in range(1, 12)
            for seed in seeds
        ],
        *[
            CropHarvestEval("Kenya", ignore_dynamic_world=True, num_timesteps=x, seed=seed)
            for x in range(1, 12)
            for seed in seeds
        ],
    ]

    result_dict = {}
    for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
        model_modes = ["finetune", "Regression", "Random Forest"]
        if "EuroSat" in eval_task.name:
            model_modes = ["Regression", "Random Forest", "KNNat5", "KNNat20", "KNNat100"]
        if "TreeSat" in eval_task.name:
            model_modes = ["Random Forest"]
        logger.info(eval_task.name)

        results = eval_task.finetuning_results(model, model_modes=model_modes)
        result_dict.update(results)

        if wandb_enabled:
            wandb.log(results)

        logger.info(json.dumps(results, indent=2))
        eval_task.clear_data()

    eval_results_file = logging_dir / "results.json"
    logger.info("Saving eval results to file %s" % eval_results_file)
    with open(eval_results_file, "w") as f:
        json.dump(result_dict, f)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
