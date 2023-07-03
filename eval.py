# sometimes, runs fail
# This (hacky) script lets
# eval results be logged to the
# right wandb run
import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from presto.dataops import BANDS_GROUPS_IDX
from presto.eval import (
    AlgaeBloomsEval,
    CropHarvestEval,
    EuroSatEval,
    EvalDataset,
    FuelMoistureEval,
    TreeSatEval,
)
from presto.presto import Presto
from presto.utils import config_dir, default_model_path, device, seed_everything

seed_everything()

argparser = argparse.ArgumentParser()
argparser.add_argument("--path_to_state_dict", type=str, default="")
argparser.add_argument("--path_to_config", type=str, default="")
argparser.add_argument("--fully_supervised", dest="fully_supervised", action="store_true")
argparser.set_defaults(fully_supervised=False)
args = argparser.parse_args().__dict__
path_to_state_dict = args["path_to_state_dict"]
path_to_config = args["path_to_config"]
fully_supervised = args["fully_supervised"]

if path_to_config == "":
    path_to_config = config_dir / "default.json"
model_kwargs = json.load(Path(path_to_config).open("r"))
model = Presto.construct(band_groups=BANDS_GROUPS_IDX, **model_kwargs)

if not fully_supervised:
    if path_to_state_dict == "":
        path_to_state_dict = default_model_path
    model.load_state_dict(torch.load(path_to_state_dict, map_location=device))
model.to(device)

print("Loading evaluation tasks")
eval_task_list: List[EvalDataset] = [
    TreeSatEval(),
    TreeSatEval("S1"),
    TreeSatEval("S2"),
    FuelMoistureEval(),
    AlgaeBloomsEval(),
    EuroSatEval(),
    EuroSatEval(rgb=True),
    CropHarvestEval("Kenya", ignore_dynamic_world=True),
    CropHarvestEval("Togo", ignore_dynamic_world=True),
    CropHarvestEval("Brazil", ignore_dynamic_world=True),
    CropHarvestEval("Kenya"),
    CropHarvestEval("Togo"),
    CropHarvestEval("Brazil"),
]
eval_task_list.extend(
    [CropHarvestEval("Togo", ignore_dynamic_world=True, num_timesteps=x) for x in range(1, 12)]
)
eval_task_list.extend(
    [CropHarvestEval("Kenya", ignore_dynamic_world=True, num_timesteps=x) for x in range(1, 12)]
)
# add CropHarvest over time
for eval_task in tqdm(eval_task_list, desc="Full Evaluation"):
    model_modes = ["finetune", "Regression", "Random Forest"]
    if "EuroSat" in eval_task.name:
        model_modes.extend(["KNNat5", "KNNat20", "KNNat100"])
    print("\n" + eval_task.name, flush=True)

    results = eval_task.finetuning_results(model, model_modes=model_modes)
    print(json.dumps(results, indent=2), flush=True)

    eval_task.clear_data()
