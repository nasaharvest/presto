import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import geopandas
import numpy as np
import torch
from cropharvest.config import LABELS_FILENAME
from cropharvest.datasets import Task
from cropharvest.engineer import TestInstance
from cropharvest.utils import extract_archive
from google.cloud import storage
from openmapflow.utils import memoized
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, default_collate
from tqdm import tqdm

from ..dataops import S1_S2_ERA5_SRTM, TAR_BUCKET
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .cropharvest_extensions import (
    DEFAULT_NUM_TIMESTEPS,
    CropHarvest,
    CropHarvestLabels,
    DynamicWorldExporter,
    Engineer,
    MultiClassCropHarvest,
    cropharvest_data_dir,
)
from .eval import EvalTask, Hyperparams


@memoized
def get_eval_datasets(
    normalize: bool = False, ignore_dynamic_world: bool = False, start_month: int = 1
):
    return CropHarvest.create_benchmark_datasets(
        cropharvest_data_dir(),
        normalize=normalize,
        ignore_dynamic_world=ignore_dynamic_world,
        start_month=start_month,
    )


def download_cropharvest_data(root_name: str = ""):
    root = Path(root_name) if root_name != "" else cropharvest_data_dir()
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)
    for gcloud_path in ["features/dynamic_world_arrays", "test_dynamic_world_features"]:
        if not (root / gcloud_path).exists():
            blob = (
                storage.Client().bucket(TAR_BUCKET).blob(f"eval/cropharvest/{gcloud_path}.tar.gz")
            )
            blob.download_to_filename(root / f"{gcloud_path}.tar.gz")
            extract_archive(root / f"{gcloud_path}.tar.gz", remove_tar=True)


class CropHarvestEval(EvalTask):
    regression = False
    multilabel = False
    num_outputs = 1
    start_month = 1
    num_timesteps = None

    country_to_sizes: Dict[str, List] = {
        "Kenya": [20, 32, 64, 96, 128, 160, 192, 224, 256, None],
        "Togo": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
    }

    def __init__(
        self,
        country: str,
        ignore_dynamic_world: bool = False,
        num_timesteps: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = DEFAULT_SEED,
    ):
        download_cropharvest_data()

        evaluation_datasets = get_eval_datasets(False, ignore_dynamic_world, self.start_month)
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset = evaluation_datasets[0]
        assert self.dataset.task.normalize is False
        self.ignore_dynamic_world = ignore_dynamic_world
        self.num_timesteps = num_timesteps
        self.sample_size = sample_size

        suffix = "_no_dynamic_world" if ignore_dynamic_world else ""
        suffix = f"_{sample_size}" if sample_size else ""
        suffix = f"{suffix}_{num_timesteps}" if num_timesteps is not None else suffix

        self.name = f"CropHarvest_{country}{suffix}"
        super().__init__(seed)

    @staticmethod
    def export_dynamic_world(test: bool = False):
        exporter = DynamicWorldExporter()
        if not test:
            exporter.export_for_labels()
        else:
            exporter.export_for_test()

    @staticmethod
    def dynamic_world_tifs_to_npy():
        def process_filename(filestem: str) -> Tuple[int, str]:
            r"""
            Given an exported sentinel file, process it to get the dataset
            it came from, and the index of that dataset
            """
            parts = filestem.split("_")[0].split("-")
            index = parts[0]
            dataset = "-".join(parts[1:])
            return int(index), dataset

        input_folder = cropharvest_data_dir() / DynamicWorldExporter.output_folder_name
        output_folder = cropharvest_data_dir() / "features/dynamic_world_arrays"
        labels = geopandas.read_file(cropharvest_data_dir() / LABELS_FILENAME)

        for filepath in tqdm(list(input_folder.glob("*.tif"))):
            index, dataset = process_filename(filepath.stem)
            output_filename = f"{index}_{dataset}.npy"
            if not (output_folder / output_filename).exists():
                rows = labels[((labels["dataset"] == dataset) & (labels["index"] == index))]
                row = rows.iloc[0]
                array, _, _ = DynamicWorldExporter.tif_to_npy(
                    filepath, row["lat"], row["lon"], DEFAULT_NUM_TIMESTEPS
                )
                assert len(array) == DEFAULT_NUM_TIMESTEPS
                np.save(output_folder / output_filename, array)

    @staticmethod
    def create_dynamic_world_test_h5_instances():
        engineer = Engineer(cropharvest_data_dir())
        engineer.test_eo_files = cropharvest_data_dir() / "test_dynamic_world_data"
        engineer.test_savedir = cropharvest_data_dir() / "test_dynamic_world_features"
        engineer.eo_files = cropharvest_data_dir() / "dynamic_world_data"
        engineer.create_h5_test_instances()

    def truncate_timesteps(self, x):
        if (self.num_timesteps is None) or (x is None):
            return x
        else:
            return x[:, : self.num_timesteps]

    @staticmethod
    def collate_fn(data):
        x, dw, latlons, labels, month = default_collate(data)
        return (x.float(), dw.long(), latlons.float(), torch.unsqueeze(labels.float(), 1), month)

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[FineTuningModel, BaseEstimator],
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, (Mosaiks1d, Seq2Seq))

        with tempfile.TemporaryDirectory() as results_dir:
            for test_id, test_instance, test_dw_instance in self.dataset.test_data(max_size=10000):
                savepath = Path(results_dir) / f"{test_id}.nc"

                test_x = self.truncate_timesteps(
                    torch.from_numpy(S1_S2_ERA5_SRTM.normalize(test_instance.x)).to(device).float()
                )
                # mypy fails with these lines uncommented, but this is how we will
                # pass the other values to the model
                test_latlons_np = np.stack([test_instance.lats, test_instance.lons], axis=-1)
                test_latlon = torch.from_numpy(test_latlons_np).to(device).float()
                test_dw = self.truncate_timesteps(
                    torch.from_numpy(test_dw_instance.x).to(device).long()
                )
                batch_mask = self.truncate_timesteps(
                    self._mask_to_batch_tensor(mask, test_x.shape[0])
                )

                if isinstance(finetuned_model, FineTuningModel):
                    finetuned_model.eval()
                    preds = (
                        finetuned_model(
                            test_x,
                            dynamic_world=test_dw,
                            mask=batch_mask,
                            latlons=test_latlon,
                            month=self.start_month,
                        )
                        .squeeze(dim=1)
                        .cpu()
                        .numpy()
                    )
                else:
                    cast(Seq2Seq, pretrained_model).eval()
                    encodings = (
                        cast(Seq2Seq, pretrained_model)
                        .encoder(
                            test_x,
                            dynamic_world=test_dw,
                            mask=batch_mask,
                            latlons=test_latlon,
                            month=self.start_month,
                        )
                        .cpu()
                        .numpy()
                    )
                    preds = finetuned_model.predict_proba(encodings)[:, 1]
                ds = test_instance.to_xarray(preds)
                ds.to_netcdf(savepath)

            all_nc_files = list(Path(results_dir).glob("*.nc"))
            combined_instance, combined_preds = TestInstance.load_from_nc(all_nc_files)
            combined_results = combined_instance.evaluate_predictions(combined_preds)

        prefix = finetuned_model.__class__.__name__
        return {f"{self.name}: {prefix}_{key}": val for key, val in combined_results.items()}

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        for x in model_modes:
            assert x in ["finetune", "Regression", "Random Forest"]
        results_dict = {}

        sklearn_modes = [x for x in model_modes if x != "finetune"]
        if len(sklearn_modes) > 0:
            array, dw, latlons, y = self.dataset.as_array(num_samples=self.sample_size)
            month = np.array([self.start_month] * array.shape[0])
            dl = DataLoader(
                TensorDataset(
                    torch.from_numpy(
                        self.truncate_timesteps(S1_S2_ERA5_SRTM.normalize(array))
                    ).float(),
                    torch.from_numpy(y).long(),
                    torch.from_numpy(self.truncate_timesteps(dw)).long(),
                    torch.from_numpy(latlons).float(),
                    torch.from_numpy(month).long(),
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )
            sklearn_models = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                mask=self.truncate_timesteps(mask),
                models=sklearn_modes,
            )
            for sklearn_model in sklearn_models:
                results_dict.update(self.evaluate(sklearn_model, pretrained_model, mask))
        return results_dict


class CropHarvestMultiClassValidation(CropHarvestEval):
    regression = False
    num_outputs = 10

    def __init__(
        self,
        val_ratio: float = 0.2,
        n_per_class: Optional[int] = 100,
        ignore_dynamic_world: bool = False,
        seed: int = DEFAULT_SEED,
    ):
        download_cropharvest_data()
        task = Task(normalize=False)
        labels = CropHarvestLabels(cropharvest_data_dir())
        paths_and_y = labels.construct_fao_classification_labels(task, filter_test=True)

        y = [x[1] for x in paths_and_y]
        unique_ys = np.unique(y)
        y_string_to_int = {val: idx for idx, val in enumerate(np.unique(y))}

        train_paths_and_y, val_paths_and_y = train_test_split(
            paths_and_y, test_size=val_ratio, stratify=y, random_state=42
        )

        if n_per_class is not None:
            indices_to_keep = []
            y_train = np.array([x[1] for x in train_paths_and_y])
            for y_val in unique_ys:
                y_val_indices = np.where(y_train == y_val)[0]
                indices_to_keep.append(y_val_indices[:n_per_class])
            train_paths_and_y = [train_paths_and_y[i] for i in np.concatenate(indices_to_keep)]
            assert len(train_paths_and_y) <= n_per_class * len(unique_ys)
        self.dataset = MultiClassCropHarvest(
            train_paths_and_y, y_string_to_int, ignore_dynamic_world=ignore_dynamic_world
        )
        self.eval_dataset = MultiClassCropHarvest(
            val_paths_and_y, y_string_to_int, ignore_dynamic_world=ignore_dynamic_world
        )

        name_suffix = f"_{n_per_class}" if n_per_class is not None else ""
        dw_suffix = "_no_dynamic_world" if ignore_dynamic_world else ""
        self.name = f"CropHarvest_multiclass_global{name_suffix}{dw_suffix}_{seed}"
        self.seed = seed
        self.sample_size = None

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[FineTuningModel, BaseEstimator],
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:

        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, (Seq2Seq, Mosaiks1d))

        dl = DataLoader(
            self.eval_dataset,
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        test_preds, test_true = [], []
        for x, dw, latlons, y in dl:
            x = S1_S2_ERA5_SRTM.normalize(x).to(device).float()
            b_mask = self._mask_to_batch_tensor(mask, x.shape[0])

            if isinstance(finetuned_model, FineTuningModel):
                finetuned_model.eval()
                preds = (
                    finetuned_model(
                        x,
                        dynamic_world=dw.to(device).long(),
                        mask=b_mask,
                        latlons=latlons.to(device).float(),
                        month=self.start_month,
                    )
                    .cpu()
                    .numpy()
                )
                preds = np.argmax(preds, axis=-1)
            elif isinstance(finetuned_model, BaseEstimator):
                cast(Seq2Seq, pretrained_model).eval()
                encodings = (
                    cast(Seq2Seq, pretrained_model)
                    .encoder(
                        x,
                        dynamic_world=dw.to(device).long(),
                        mask=b_mask,
                        latlons=latlons.to(device).float(),
                        month=self.start_month,
                    )
                    .cpu()
                    .numpy()
                )
                preds = finetuned_model.predict(encodings)
            test_preds.append(preds)
            test_true.append(y)

        test_preds_np = np.concatenate(test_preds)
        test_true_np = np.concatenate(test_true)

        prefix = finetuned_model.__class__.__name__
        return {
            f"{self.name}: {prefix}_num_samples": len(test_true_np),
            f"{self.name}: {prefix}_f1_score": f1_score(
                test_true_np, test_preds_np, average="weighted"
            ),
        }

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        # TODO - where are these controlled?
        hyperparams = Hyperparams(max_epochs=3, batch_size=64)
        model = self._construct_finetuning_model(pretrained_model)

        # TODO - should this be more intelligent? e.g. first learn the
        # (randomly initialized) head before modifying parameters for
        # the whole model?
        opt = Adam(model.parameters(), lr=hyperparams.lr)
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        train_dl = DataLoader(
            self.dataset,
            batch_size=hyperparams.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
        )

        train_loss = []
        for _ in range(hyperparams.max_epochs):
            model.train()
            epoch_train_loss = 0.0
            for x, dw, latlons, y in train_dl:
                x = S1_S2_ERA5_SRTM.normalize(x).to(device).float()
                opt.zero_grad()
                b_mask = self._mask_to_batch_tensor(mask, x.shape[0])
                preds = model(
                    x,
                    dynamic_world=dw.to(device).long(),
                    mask=b_mask,
                    latlons=latlons.to(device).float(),
                    month=self.start_month,
                )
                loss = loss_fn(preds, y.to(device))
                epoch_train_loss += loss.item()
                loss.backward()
                opt.step()
            train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        return model
