from typing import Callable, Dict, List, Optional, Sequence, Union, cast

import numpy as np
import torch
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from ..model import FineTuningModel, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .knn import KNNat5, KNNat20, KNNat100


class EvalDataset:

    name: str
    num_outputs: int
    regression: bool

    def _construct_finetuning_model(self, pretrained_model: Seq2Seq) -> FineTuningModel:
        model = cast(Callable, pretrained_model.construct_finetuning_model)(
            num_outputs=self.num_outputs,
            # todo how to pass parameters such as the number of
            # finetuning layers. Through a config or through the
            # function?
            regression=self.regression,
        )
        return model

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        raise NotImplementedError

    def evaluate(
        self,
        finetuned_model: Union[FineTuningModel, BaseEstimator],
        pretrained_model: Optional[Seq2Seq] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        # this function should be wrapped in a no_grad context manager
        raise NotImplementedError

    @staticmethod
    def _mask_to_batch_tensor(
        mask: Optional[np.ndarray], batch_size: int
    ) -> Optional[torch.Tensor]:
        if mask is not None:
            return repeat(torch.from_numpy(mask).to(device), "t c -> b t c", b=batch_size).float()
        return None

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pretrained_model,
        dynamic_world: np.ndarray,
        latlons: np.ndarray,
        month: Union[np.ndarray, int],
        mask: Optional[np.ndarray] = None,
        batch_size: int = 64,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Sequence[BaseEstimator]:

        for model_mode in models:
            if self.regression:
                assert model_mode in ["Regression", "Random Forest"]
            else:
                assert model_mode in [
                    "Regression",
                    "Random Forest",
                    "KNNat5",
                    "KNNat20",
                    "KNNat100",
                ]
        pretrained_model.eval()

        encodings_list = []
        if isinstance(month, int):
            month = np.array([month] * X.shape[0])

        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(X).to(device).float(),
                torch.from_numpy(dynamic_world).to(device).long(),
                torch.from_numpy(latlons).to(device).float(),
                torch.from_numpy(month).to(device).long(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        for (x, dw, latlons, month) in dl:
            batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
            with torch.no_grad():
                encodings = (
                    pretrained_model.encoder(
                        x, dynamic_world=dw, mask=batch_mask, latlons=latlons, month=month
                    )
                    .cpu()
                    .numpy()
                )
                encodings_list.append(encodings)
        encodings_np = np.concatenate(encodings_list)

        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", random_state=DEFAULT_SEED
                ),
                "KNNat5": KNNat5(),
                "KNNat20": KNNat20(),
                "KNNat100": KNNat100(),
            },
            True: {
                "Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=DEFAULT_SEED),
            },
        }
        for model in models:
            fit_models.append(model_dict[self.regression][model].fit(encodings_np, y))
        return fit_models

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        raise NotImplementedError

    def clear_data(self):
        # this doesn't have to be implemented but
        # helps with memory management for large
        # eval tasks
        return None
