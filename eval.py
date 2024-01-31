# sometimes, runs fail
# This (hacky) script lets
# eval results be logged to the
# right wandb run
import argparse
import json
import logging
from pathlib import Path
from typing import List, cast

import torch
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    CroptypeFranceEval,
    EuroSatEval,
    EvalTask,
    FuelMoistureEval,
    TreeSatEval,
)
from presto.presto import Presto
from presto.utils import (
    DEFAULT_SEED,
    config_dir,
    default_model_path,
    device,
    initialize_logging,
    seed_everything,
    timestamp_dirname,
    update_data_dir,
)

seed_everything()
logger = logging.getLogger("__main__")

argparser = argparse.ArgumentParser()
argparser.add_argument("--path_to_state_dict", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Data is stored in <data_dir>/data. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Output is stored in <data_dir>/output. "
    "Leave empty to use the directory you are running this file from.",
)
argparser.add_argument("--fully_supervised", dest="fully_supervised", action="store_true")
argparser.add_argument("--wandb", dest="wandb", action="store_true")
argparser.set_defaults(wandb=False)
argparser.set_defaults(fully_supervised=False)
args = argparser.parse_args().__dict__

path_to_state_dict = args["path_to_state_dict"]
path_to_config = args["path_to_config"]
fully_supervised = args["fully_supervised"]
wandb_enabled = args["wandb"]
data_dir = args["data_dir"]
if data_dir != "":
    update_data_dir(data_dir)

output_parent_dir = Path(args["output_dir"]) if args["output_dir"] else Path(__file__).parent
run_id = None
if wandb_enabled:
    import wandb

    run = wandb.init(
        entity="nasa-harvest",
        project="presto-downstream",
        dir=output_parent_dir,
    )
    run_id = cast(Run, run).id

logging_dir = output_parent_dir / "output" / timestamp_dirname(run_id)
logging_dir.mkdir(exist_ok=True, parents=True)
initialize_logging(logging_dir)
logger.info("Using output dir: %s" % logging_dir)

if path_to_config == "":
    path_to_config = config_dir / "default.json"
logger.info("Loading config from %s" % path_to_config)
model_kwargs = json.load(Path(path_to_config).open("r"))
model = Presto.construct(**model_kwargs)

if not fully_supervised:
    if path_to_state_dict == "":
        path_to_state_dict = default_model_path
    logger.info("Loading params from %s" % path_to_state_dict)
    model.load_state_dict(torch.load(path_to_state_dict, map_location=device))
model.to(device)

logger.info("Loading evaluation tasks")
seeds = [0, DEFAULT_SEED, 84]
eval_task_list: List[EvalTask] = [
    *[
        CropHarvestEval(country="Brazil", ignore_dynamic_world=idw, seed=seed)
        for idw in [True, False]
        for seed in seeds
    ],
    *[
        CropHarvestEval(country="Kenya", ignore_dynamic_world=idw, seed=seed, sample_size=s)
        for idw in [True, False]
        for seed in seeds
        for s in CropHarvestEval.country_to_sizes["Kenya"]
    ],
    *[
        CropHarvestEval(country="Togo", ignore_dynamic_world=idw, seed=seed, sample_size=s)
        for idw in [True, False]
        for seed in seeds
        for s in CropHarvestEval.country_to_sizes["Togo"]
    ],
    *[FuelMoistureEval(seed=seed) for seed in seeds],
    *[AlgaeBloomsEval(seed=seed) for seed in seeds],
    *[
        EuroSatEval(rgb=rgb, input_patch_size=ps, seed=seed, aggregates=["mean"])
        for rgb in [True, False]
        for ps in [1, 2, 4, 8, 16, 32, 64]
        for seed in seeds
    ],
    *[
        TreeSatEval(subset=subset, seed=seed, aggregates=["mean"])
        for subset in ["S1", "S2"]
        for seed in seeds
    ],
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
    *[
        CroptypeFranceEval(input_patch_size=patch_size, aggregates=["mean"], seed=seed)
        for patch_size in [1, 5]
        for seed in seeds
    ],
]

if wandb_enabled:
    eval_config = {
        "model": model.__class__,
        "encoder": model.encoder.__class__,
        "decoder": model.decoder.__class__,
        "device": device,
        "model_parameters": "random" if fully_supervised else path_to_state_dict,
        **args,
        **model_kwargs,
    }
    wandb.config.update(eval_config)

result_dict = {}
for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
    model_modes = ["finetune", "Regression", "Random Forest"]
    if "EuroSat" in eval_task.name:
        model_modes = [
            "Regression",
            "Random Forest",
            "KNNat5",
            "KNNat20",
            "KNNat100",
            "finetune",
        ]
    if "TreeSat" in eval_task.name:
        model_modes = ["finetune", "Random Forest"]
    logger.info(eval_task.name)

    results = eval_task.finetuning_results(model, model_modes=model_modes)
    result_dict.update(results)
    logger.info(json.dumps(results, indent=2))

    if wandb_enabled:
        wandb.log(results)

    eval_task.clear_data()

eval_results_file = logging_dir / "results.json"
logger.info("Saving eval results to file %s" % eval_results_file)
with open(eval_results_file, "w") as f:
    json.dump(result_dict, f)

if wandb_enabled and run:
    run.finish()
    logger.info(f"Wandb url: {run.url}")
