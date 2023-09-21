import logging
from abc import ABC
from dataclasses import astuple, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union, cast

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from ..dataops import S1_S2_ERA5_SRTM, DynamicWorld2020_2021
from ..model import FineTuningModel, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .knn import KNNat5, KNNat20, KNNat100

logger = logging.getLogger("__main__")


@dataclass
class Hyperparams:
    lr: float = 3e-4
    max_epochs: int = 20
    batch_size: int = 4096
    patience: int = 3
    num_workers: int = 2


class EvalTask(ABC):
    name: str
    num_outputs: int
    regression: bool
    multilabel: bool

    def __init__(self, seeds: List[int] = [DEFAULT_SEED]):
        assert len(seeds) == 1
        self.seed = seeds[0]
        self.name = f"{self.name}_{self.seed}"

    def _construct_finetuning_model(self, pretrained_model: Seq2Seq) -> FineTuningModel:
        model = cast(Callable, pretrained_model.construct_finetuning_model)(
            num_outputs=self.num_outputs,
            # todo how to pass parameters such as the number of
            # finetuning layers. Through a config or through the
            # function?
            regression=self.regression,
        )
        return model

    @classmethod
    def _construct_model(cls, model) -> BaseEstimator:
        if cls.multilabel:
            model = MultiOutputClassifier(model, n_jobs=cls.num_outputs)
        return model

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        raise NotImplementedError

    def finetune_pytorch_model(
        self,
        model: FineTuningModel,
        hyperparams: Hyperparams,
        optimizer: Optimizer,
        train_dl: DataLoader,
        val_dl: DataLoader,
        train_loss_fn: Callable,
        val_loss_fn: Callable,
        mask: Optional[np.ndarray] = None,
    ) -> FineTuningModel:
        lr, max_epochs, batch_size, patience, _ = astuple(hyperparams)

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        for _ in tqdm(range(max_epochs), desc="Finetuning"):
            model.train()
            epoch_train_loss = 0.0
            for x, dw, latlons, y, month in tqdm(train_dl, desc="Training", leave=False):
                x, dw, latlons, y, month = [t.to(device) for t in (x, dw, latlons, y, month)]
                optimizer.zero_grad()
                b_mask = self._mask_to_batch_tensor(mask, x.shape[0])
                preds = model(
                    x,
                    dynamic_world=dw,
                    mask=b_mask,
                    latlons=latlons,
                    month=month,
                )
                loss = train_loss_fn(preds, y)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_y = [], []
            for x, dw, latlons, y, month in val_dl:
                x, dw, latlons, y, month = [t.to(device) for t in (x, dw, latlons, y, month)]
                batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
                with torch.no_grad():
                    preds = model(
                        x,
                        dynamic_world=dw,
                        mask=batch_mask,
                        latlons=latlons,
                        month=month,
                    )
                    all_preds.append(preds)
                    all_y.append(y)

            val_loss.append(val_loss_fn(torch.cat(all_preds).cpu(), torch.cat(all_y).cpu()))
            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = model.state_dict()
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = model.state_dict()
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        logger.info("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

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
        dl: DataLoader,
        pretrained_model,
        mask: Optional[np.ndarray] = None,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
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

        encoding_list, target_list = [], []
        for x, y, dw, latlons, month in dl:
            x, dw, latlons, y, month = [t.to(device) for t in (x, dw, latlons, y, month)]
            batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
            target_list.append(y.cpu().numpy())
            with torch.no_grad():
                encodings = (
                    pretrained_model.encoder(
                        x, dynamic_world=dw, mask=batch_mask, latlons=latlons, month=month
                    )
                    .cpu()
                    .numpy()
                )
                encoding_list.append(encodings)
        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Regression": self._construct_model(
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=self.seed
                    )
                ),
                "Random Forest": self._construct_model(
                    RandomForestClassifier(class_weight="balanced", random_state=self.seed)
                ),
                "KNNat5": self._construct_model(KNNat5()),
                "KNNat20": self._construct_model(KNNat20()),
                "KNNat100": self._construct_model(KNNat100()),
            },
            True: {
                "Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=self.seed),
            },
        }
        for model in models:
            fit_models.append(clone(model_dict[self.regression][model]).fit(encodings_np, targets))
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


class EvalTaskWithAggregatedOutputs(EvalTask, ABC):

    # used for a histogram aggregation
    lower = None
    upper = None

    def __init__(
        self,
        aggregates: List[str],
        outputs_per_image: int,
        num_histogram_bins: Optional[int] = None,
        histogram_lower: Optional[int] = None,
        histogram_upper: Optional[int] = None,
        seeds: List[int] = [DEFAULT_SEED],
    ):
        for aggregate in aggregates:
            assert aggregate in ("mean", "quantiles", "histogram")
        self.aggregates = aggregates
        self.outputs_per_image = outputs_per_image
        if "histogram" in self.aggregates:
            assert histogram_lower is not None
            self.lower = histogram_lower
            assert histogram_upper is not None
            self.upper = histogram_upper
            assert num_histogram_bins is not None
            self.num_histogram_bins = num_histogram_bins
        self.seeds = seeds

    def _construct_finetuning_model(self, pretrained_model: Seq2Seq) -> FineTuningModel:
        # TODO
        raise NotImplementedError

    def reshape_for_aggregate(
        self,
        encodings: torch.Tensor,
        aggregate: str,
    ) -> torch.Tensor:

        encodings_im = rearrange(
            encodings, "(img p) h_dim -> img p h_dim", p=self.outputs_per_image
        )
        if aggregate == "histogram":

            def histo(i, d):
                return torch.histogram(
                    encodings_im.cpu()[i, :, d],
                    bins=self.num_histogram_bins,
                    range=(self.lower, self.upper),
                ).hist

            return torch.vstack(
                [
                    torch.cat([histo(i, d) for d in range(encodings_im.shape[-1])], dim=-1)
                    for i in range(encodings_im.shape[0])
                ]
            )
        elif aggregate == "quantiles":
            return torch.cat(
                [
                    torch.quantile(encodings_im, 0.25, dim=1),
                    torch.mean(encodings_im, dim=1),
                    torch.quantile(encodings_im, 0.75, dim=1),
                    # the unbiased (default) estimate divides by (n-1) giving NaN
                    #   for self.outputs_per_image == 1
                    torch.std(encodings_im, dim=1, correction=int(encodings_im.shape[1] > 1)),
                    torch.quantile(encodings_im, q=0.5, dim=1),  # median
                ],
                dim=-1,
            )
        else:
            return torch.cat(
                [
                    torch.mean(encodings_im, dim=1),
                    torch.std(encodings_im, dim=1, correction=int(encodings_im.shape[1] > 1)),
                ],
                dim=-1,
            )

    @torch.no_grad()
    def finetune_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model,
        mask: Optional[np.ndarray] = None,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Dict:
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

        aggregate_lists: Dict[str, List] = {aggregate: [] for aggregate in self.aggregates}
        y_list = []
        for x, y, dw, latlons, month in tqdm(dl, desc="Computing embeddings"):
            x, dw, latlons, month = [t.to(device) for t in (x, dw, latlons, month)]
            batch_mask = self._mask_to_batch_tensor(mask, x.shape[0])
            with torch.no_grad():
                encodings = pretrained_model.encoder(
                    x, dynamic_world=dw, mask=batch_mask, latlons=latlons, month=month
                ).cpu()
                for aggregate in self.aggregates:
                    assert not torch.isnan(encodings).any()
                    reshaped_encodings = self.reshape_for_aggregate(encodings, aggregate)
                    assert not torch.isnan(reshaped_encodings).any()
                    aggregate_lists[aggregate].append(reshaped_encodings.cpu().numpy())
                y_list.append(
                    y.cpu()
                    .numpy()
                    .reshape(
                        (
                            encodings.shape[0] // self.outputs_per_image,
                            self.outputs_per_image,
                            *y.shape[1:],
                        )
                    )[:, 0]
                )
        y_per_im = np.concatenate(y_list)
        for aggregate, aggregations in aggregate_lists.items():
            aggregate_lists[aggregate] = np.concatenate(aggregations, axis=0)

        fit_models: Dict[str, List] = {aggregate: [] for aggregate in self.aggregates}
        for seed in self.seeds:
            model_dict = {
                False: {
                    "Regression": self._construct_model(
                        LogisticRegression(max_iter=1000, random_state=seed)
                    ),
                    "Random Forest": self._construct_model(
                        RandomForestClassifier(random_state=seed)
                    ),
                    "KNNat5": KNNat5(),
                    "KNNat20": KNNat20(),
                    "KNNat100": KNNat100(),
                },
                True: {
                    "Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(random_state=seed),
                },
            }

            for aggregate in tqdm(self.aggregates, desc="Fitting sklearn models"):
                encodings_im = aggregate_lists[aggregate]
                for model in tqdm(models, desc="Fitting sklearn models"):
                    fit_models[aggregate].append(
                        clone(model_dict[self.regression][model]).fit(encodings_im, y_per_im)
                    )
        return fit_models


class EvalDatasetWithPatches(Dataset, ABC):
    start_month: int

    def __init__(
        self,
        input_patch_size: int = 1,
        num_patches_per_dim: int = 1,
        split: str = "train",
        merge_train_val: bool = True,
    ):
        self.input_patch_size = input_patch_size
        self.num_patches_per_dim = num_patches_per_dim
        self.num_patches = self.num_patches_per_dim**2
        self.images = self.split_images(merge_train_val)[split]

    @staticmethod
    def split_images(merge_train_val: bool = True):
        raise NotImplementedError

    def image_to_eo_array(self, tif_file: str):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        image = self.images[idx]
        x, lonlats, label = self.image_to_eo_array(image.strip())

        # works both for scalar and array-shaped (multilabel) labels
        labels = np.tile(label, reps=(self.num_patches, 1) if len(label) > 1 else self.num_patches)
        x = np.expand_dims(self.resize_and_average_arrays(x), 1)
        latlons = self.resize_and_average_arrays(lonlats)[:, [1, 0]]
        # all dynamic world values are considered masked
        dw = np.ones((len(labels), 1)) * DynamicWorld2020_2021.class_amount
        month = np.ones((len(labels))) * self.start_month

        assert len(x) == len(dw) == len(latlons) == len(labels)

        return (
            torch.from_numpy(S1_S2_ERA5_SRTM.normalize(x)).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(dw).long(),
            torch.from_numpy(latlons).float(),
            torch.from_numpy(month).long(),
        )

    def resize_and_average_arrays(self, arrays: np.ndarray) -> np.ndarray:
        # flatten to array of pixels, normalize, unflatten to patches
        arrays = rearrange(
            arrays,
            "c (n1 p1) (n2 p2) -> (n1 n2) c p1 p2",
            n1=self.num_patches_per_dim,
            n2=self.num_patches_per_dim,
            p1=self.input_patch_size,
            p2=self.input_patch_size,
        )
        return reduce(arrays, "b c h w -> b c", "mean")

    @staticmethod
    def collate_fn(data):
        x, labels, dw, latlons, month = default_collate(data)
        return (
            rearrange(x, "b bp t d -> (b bp) t d"),
            # ... is an optional dimension: for TreeSat labels which are arrays
            rearrange(labels, "b bp ... -> (b bp) ..."),
            rearrange(dw, "b bp t -> (b bp) t"),
            rearrange(latlons, "b bp d -> (b bp) d"),
            rearrange(month, "b bp -> (b bp)"),
        )

    def __len__(self):
        return len(self.images)
