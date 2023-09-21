import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest import TestCase


class TestTrainScript(TestCase):
    def test_train_skip_finetuning(self):

        out = subprocess.check_output(
            [
                "python",
                "train.py",
                "--model_name",
                "test_model",
                "--train_url",
                "data/dw_144_mini_shard_44.tar",
                "--val_url",
                "data/dw_144_mini_shard_44.tar",
                "--val_per_n_steps",
                "1",
                "--cropharvest_per_n_validations",
                "0",
                "--skip_finetuning",
                "--n_epochs",
                "1",
            ]
        )
        print("\u2714 train.py ran successfully")

        match = re.search(
            r"Done training, best model saved to (\S+)$", out.decode("utf-8"), flags=re.MULTILINE
        )
        model_path = Path(match.group(1))

        self.assertTrue(model_path.exists())
        if model_path.exists():
            model_path.unlink()
        print("\u2714 train.py generated a new model")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=2)
    unittest.main(testRunner=runner)
