from pathlib import Path

import torch

data_dir = Path(__file__).parent.parent / "data"
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
