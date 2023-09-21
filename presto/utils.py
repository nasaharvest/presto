import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import dateutil.tz
import torch

logger = logging.getLogger("__main__")


def update_data_dir(new_data_dir: str):
    global data_dir
    data_dir = Path(new_data_dir) / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Setting other data dir to be used: %s" % data_dir)


data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True, parents=True)
logger.info("Using data dir: %s" % data_dir)

config_dir = Path(__file__).parent.parent / "config"
default_model_path = data_dir / "default_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED: int = 42


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(output_dir: Union[str, Path], to_file=True, logger_name="__main__"):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    if to_file:
        path = os.path.join(output_dir, "console-output.log")
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Initialized logging to %s" % path)
    return logger


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now(dateutil.tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts
