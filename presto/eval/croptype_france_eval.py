import json
import logging
import os
import re
from math import cos, radians
from typing import Dict, List, Optional, cast

import numpy as np
import torch
from einops import repeat
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import utils
from ..dataops import NUM_BANDS, NUM_ORG_BANDS, DynamicWorld2020_2021
from ..dataops.pipelines.s1_s2_era5_srtm import (
    BANDS,
    BANDS_GROUPS_IDX,
    REMOVED_BANDS,
    S1_S2_ERA5_SRTM,
    S2_BANDS,
)
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .eval import (
    EvalDatasetWithPatches,
    EvalTaskWithAggregatedOutputs,
    Hyperparams,
    PrestoFinetuningWithAggregates,
)

logger = logging.getLogger("__main__")
data_subdir = "croptype-france/"
npy_subdir = "croptype-france/s2-2017-IGARSS-NNI-NPY/"
label_file = "croptype-france/train_val_test_labels.json"
split_file = "croptype-france/train_val_test_split.json"

IMAGE_SIZE = 5
ORIG_IMAGE_SIZE = 32
# We leave out the atmospheric bands (bands 1, 9, and 10), keeping C = 10 spectral bands
TASK_REMOVED_BANDS = ["B1", "B9", "B10"]


class CroptypeFranceDataset(EvalDatasetWithPatches):
    """
    Data published at https://zenodo.org/records/5815523
    As used by https://github.com/linlei1214/SITS-Former
    Preprocessing from https://github.com/linlei1214/SITS-Former/blob/master/PrepareData.ipynb
    """

    # exclude classes with less than 200 samples
    labels = [1, 4, 5, 6, 9, 12, 14, 16, 18, 19, 23, 28, 31, 34, 36]
    label_dict = {c: i for i, c in enumerate(labels)}

    # https://eatlas.org.au/data/uuid/f7468d15-12be-4e3f-a246-b2882a324f59
    # data is from the 31TFM Sentinel-2 tile, this is the tile's centroid
    lat, lon = 46.441, 5.017

    day_of_year = [
        3,
        13,
        43,
        73,
        93,
        113,
        133,
        153,
        163,
        173,
        183,
        188,
        193,
        198,
        203,
        218,
        233,
        238,
        248,
        253,
        263,
        278,
        283,
        288,
    ]
    months = np.digitize(day_of_year, np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))

    def __init__(
        self, input_patch_size: int = 1, split: str = "train", merge_train_val: bool = True
    ):
        assert input_patch_size in [1, IMAGE_SIZE]
        super().__init__(input_patch_size, int(IMAGE_SIZE / input_patch_size), split, False)
        self.targets = json.load((utils.data_dir / label_file).open("r"))[split]
        self.split = split

    @staticmethod
    def prepare():
        # Based on https://github.com/linlei1214/SITS-Former/blob/master/PrepareData.ipynb
        raw_data_dir = utils.data_dir / npy_subdir / "DATA"
        arrays = [f for f in os.listdir(raw_data_dir) if f.endswith(".npy")]
        patch_ids = [int(f.split(".")[0]) for f in arrays]
        patch_ids = [str(pid) for pid in np.sort(patch_ids)]

        label_key = "label_44class"
        all_targets = {}

        with open(utils.data_dir / npy_subdir / "META" / "labels.json", "r") as label_file:
            label_dict = json.load(label_file)

        for pid in patch_ids:
            label = label_dict[label_key][pid]
            if label in CroptypeFranceDataset.labels:
                all_targets[pid] = CroptypeFranceDataset.label_dict[label]

        # Don't shuffle, instead load ids from PrepareData.ipynb so they are identical (?)
        #  to those used by SITS-Former
        with open(utils.data_dir / data_subdir / "PrepareData.ipynb") as f:
            lines = f.readlines()

        shuffled_indices = []
        for line in lines:
            match = re.search(r"idx = ([0-9]+),", line)
            if match is not None:
                shuffled_indices.append(int(match.group(1)))

        train_num = [0 for _ in range(len(CroptypeFranceDataset.labels))]
        val_num = [0 for _ in range(len(CroptypeFranceDataset.labels))]
        test_num = [0 for _ in range(len(CroptypeFranceDataset.labels))]

        train, val, test = [], [], []
        train_targets, val_targets, test_targets = {}, {}, {}

        for idx in tqdm(shuffled_indices, desc="Processing arrays..."):
            pid = patch_ids[idx]
            path_to_npy = raw_data_dir / "{}.npy".format(pid)
            pixels = np.load(path_to_npy)

            if pid in all_targets:  # means the patch is part of the chosen subclasses
                class_label = all_targets[pid]

                row, col = CroptypeFranceDataset.select_row_col(pixels)
                if row > 0 and col > 0:

                    if train_num[class_label] < 100:
                        train_num[class_label] += 1
                        train.append(pid)
                        train_targets[pid] = class_label
                    elif val_num[class_label] < 100:
                        val_num[class_label] += 1
                        val.append(pid)
                        val_targets[pid] = class_label
                    else:
                        test_num[class_label] += 1
                        test.append(pid)
                        test_targets[pid] = class_label

        targets = {"train": train_targets, "val": val_targets, "test": test_targets}
        return {"train": train, "val": val, "test": test}, targets

    @staticmethod
    def degree_per_metre(lat: float):
        # https://gis.stackexchange.com/questions/75528/understanding-terms-in
        # -length-of-degree-formula
        # see the link above to explain the magic numbers
        m_per_degree_lat = (
            111132.954
            + (-559.822 * cos(radians(2.0 * lat)))
            + (1.175 * cos(radians(4.0 * lat)))
            + (-0.0023 * cos(radians(6 * lat)))
        )
        m_per_degree_lon = (
            (111412.84 * cos(radians(lat)))
            + (-93.5 * cos(radians(3 * lat)))
            + (0.118 * cos(radians(5 * lat)))
        )

        return 1 / m_per_degree_lat, 1 / m_per_degree_lon

    @staticmethod
    def select_row_col(pixels):
        mask = pixels[0, 0, :, :]
        mask[np.nonzero(mask)] = 1
        row, col = -1, -1
        flag = False
        for i in range(ORIG_IMAGE_SIZE - IMAGE_SIZE - 1):
            for j in range(ORIG_IMAGE_SIZE - IMAGE_SIZE - 1):
                if np.all(mask[i : i + IMAGE_SIZE, j : j + IMAGE_SIZE] == 1):
                    row, col = i, j
                    flag = True
                    break
            if flag:
                break
        return row, col

    @staticmethod
    def split_images(merge_train_val: bool = False) -> Dict[str, List[str]]:
        split_path = utils.data_dir / split_file
        label_path = utils.data_dir / label_file

        if split_path.exists() and label_path.exists():
            splits = json.load(split_path.open("r"))
        else:
            # this code was only run once (the dictionary is then saved)
            # but is saved here for clarity
            splits, targets = CroptypeFranceDataset.prepare()
            json.dump(splits, split_path.open("w"))
            json.dump(targets, label_path.open("w"))
        return splits

    def image_to_eo_array(self, patch_id: str):
        path_to_npy = utils.data_dir / npy_subdir / "DATA" / f"{patch_id}.npy"
        pixels = np.load(path_to_npy)
        class_label = self.targets[patch_id]
        row, col = self.select_row_col(pixels)
        if not row >= 0 or not col >= 0:
            raise ValueError
        out_patch = pixels[:, :, row : row + IMAGE_SIZE, col : col + IMAGE_SIZE]

        eo_style_array = np.zeros([len(self.months), NUM_ORG_BANDS, IMAGE_SIZE, IMAGE_SIZE])

        assert set(REMOVED_BANDS) <= set(TASK_REMOVED_BANDS)
        kept_overall_bands = [
            idx
            for idx, x in enumerate(BANDS)
            if ((x in S2_BANDS) and (x not in TASK_REMOVED_BANDS))
        ]

        eo_style_array[:, kept_overall_bands] = out_patch
        m_lat, m_lon = self.degree_per_metre(self.lat)
        lonlats = np.tile(
            np.array([self.lon, self.lat])[:, None, None], (1, IMAGE_SIZE, IMAGE_SIZE)
        ) + np.stack(
            [
                np.repeat(np.array([[i * m_lon for i in range(5)]]), 5, 0),
                np.repeat(np.array([[i * m_lat for i in range(5)]]), 5, 0).T,
            ]
        )
        return (
            eo_style_array,
            lonlats,
            np.array([class_label], dtype=int),
            np.array(self.months, dtype=int),
        )

    def __getitem__(self, idx: int):
        image = self.images[idx]
        x, lonlats, label, month = self.image_to_eo_array(image.strip())

        # works both for scalar and array-shaped (multilabel) labels
        labels = np.tile(label, reps=self.num_patches)
        months = np.tile(month, reps=(self.num_patches, 1))
        x = self.resize_and_average_arrays(x)
        latlons = self.resize_and_average_arrays(lonlats)[:, [1, 0]]
        # all dynamic world values are considered masked
        dw = np.ones((len(labels), len(month))) * DynamicWorld2020_2021.class_amount

        assert len(x) == len(dw) == len(latlons) == len(labels) == len(months)

        return (
            torch.from_numpy(S1_S2_ERA5_SRTM.normalize(x)).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(dw).long(),
            torch.from_numpy(latlons).float(),
            torch.from_numpy(months).long(),
        )


class CroptypeFranceEval(EvalTaskWithAggregatedOutputs):
    regression = False
    multilabel = False
    num_outputs = len(CroptypeFranceDataset.labels)

    def __init__(
        self, input_patch_size: int = 1, aggregates: List[str] = ["mean"], seed: int = DEFAULT_SEED
    ) -> None:
        # for each image, we will take an `input_patch_size x input_patch_size`
        # patch, and take a spatial mean of it
        self.input_patch_size = input_patch_size
        assert input_patch_size in [1, IMAGE_SIZE]
        self.num_patches_per_dim = int(IMAGE_SIZE / input_patch_size)
        self.num_patches = self.num_patches_per_dim**2
        self.batch_size = 128
        self.name = f"CroptypeFrance_{input_patch_size}"
        super().__init__(aggregates, self.num_patches, seed)

    def update_mask(self, mask: Optional[np.ndarray] = None):
        channels_lists = [x for key, x in BANDS_GROUPS_IDX.items() if "S2" in key]
        channels_lists.append(BANDS_GROUPS_IDX["NDVI"])
        # flatten the list of lists
        default_channels = [item for sublist in channels_lists for item in sublist]

        # everything is masked by default
        default_mask = np.ones([NUM_BANDS])
        # unmask the s2 bands
        default_mask[default_channels] = 0
        default_mask = repeat(default_mask, "d -> t d", t=len(CroptypeFranceDataset.months))

        if mask is not None:
            return np.clip(mask + default_mask, a_min=0, a_max=1)
        else:
            return np.clip(default_mask, a_min=0, a_max=1)

    @torch.no_grad()
    def evaluate_for_finetuned_model(
        self,
        finetuned_model: FineTuningModel,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        hparams = Hyperparams(batch_size=self.batch_size)
        updated_mask = self.update_mask(mask)
        test_dataset = CroptypeFranceDataset(self.input_patch_size, split="test")
        test_dl = DataLoader(
            test_dataset,
            batch_size=hparams.batch_size,
            shuffle=False,
            num_workers=0,  # issues with too many open files
            collate_fn=test_dataset.finetuning_collate_fn,
        )
        pred_list, labels = [], []
        for x, dw, latlons, label, month in tqdm(test_dl, desc="Computing test predictions"):
            x, dw, latlons, month = [t.to(device) for t in (x, dw, latlons, month)]
            batch_mask = self._mask_to_batch_tensor(updated_mask, x.shape[0])
            batch_preds = finetuned_model(
                x,
                dynamic_world=dw,
                mask=batch_mask,
                latlons=latlons,
                month=month,
            )
            labels.append(label.cpu().numpy())
            pred_list.append(batch_preds.cpu().numpy())

        target = np.concatenate(labels)
        test_preds_np = np.argmax(np.concatenate(pred_list, axis=0), axis=1)
        prefix = f"finetuning_{finetuned_model.aggregate}"

        # https://github.com/linlei1214/SITS-Former/blob/master/code/trainer/finetune.py#L115
        results_dict = self.metrics(prefix, target, test_preds_np)
        return results_dict

    @torch.no_grad()
    def evaluate_for_sklearn(
        self,
        finetuned_model: Dict,
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        hparams = Hyperparams(batch_size=self.batch_size)
        updated_mask = self.update_mask(mask)

        assert isinstance(pretrained_model, (Seq2Seq, Mosaiks1d))
        pred_dict: Dict[str, Dict[str, List]] = {}
        for aggregate, model_list in finetuned_model.items():
            pred_dict[aggregate] = {model.__class__.__name__: [] for model in model_list}

        test_dataset = CroptypeFranceDataset(self.input_patch_size, split="test")
        test_dl = DataLoader(
            test_dataset,
            batch_size=hparams.batch_size // self.num_patches,
            shuffle=False,
            collate_fn=CroptypeFranceDataset.collate_fn,
            num_workers=Hyperparams.num_workers,
        )

        labels = []
        for x, label, dw, latlons, month in tqdm(test_dl, desc="Computing test predictions"):
            x, dw, latlons, month = [t.to(device) for t in (x, dw, latlons, month)]
            batch_mask = self._mask_to_batch_tensor(updated_mask, x.shape[0])
            cast(Seq2Seq, pretrained_model).eval()
            encodings = cast(Seq2Seq, pretrained_model).encoder(
                x,
                dynamic_world=dw,
                mask=batch_mask,
                latlons=latlons,
                month=month,
            )
            labels.append(
                label.cpu()
                .numpy()
                .reshape(
                    (
                        encodings.shape[0] // self.outputs_per_image,
                        self.outputs_per_image,
                        *label.shape[1:],
                    )
                )[:, 0]
            )
            for aggregate, model_list in finetuned_model.items():
                assert not torch.isnan(encodings).any()
                reshaped_encodings = PrestoFinetuningWithAggregates.reshape_for_aggregate(
                    encodings, aggregate, self.outputs_per_image
                ).cpu()
                assert not torch.isnan(reshaped_encodings).any()
                for model in model_list:
                    preds = model.predict(reshaped_encodings.numpy())
                    pred_dict[aggregate][model.__class__.__name__].append(preds)

        target = np.concatenate(labels)
        results_dict = {}
        for aggregate, model_pred_dict in pred_dict.items():
            for model_name, pred_list in model_pred_dict.items():
                test_preds_np = np.concatenate(pred_list, axis=0)
                prefix = f"{model_name}_{aggregate}"
                results_dict.update(self.metrics(prefix, target, test_preds_np))
        return results_dict

    def metrics(self, prefix: str, target: np.ndarray, test_preds_np: np.ndarray) -> Dict:
        results = {
            f"{self.name}: {prefix}_num_samples": len(target),
            f"{self.name}: {prefix}_f1_score": f1_score(
                target,
                test_preds_np,
                average="macro",
                labels=list(range(self.num_outputs)),
            ),
            f"{self.name}: {prefix}_kappa_score": cohen_kappa_score(
                target, test_preds_np, labels=list(range(self.num_outputs))
            ),
            f"{self.name}: {prefix}_accuracy_score": accuracy_score(target, test_preds_np),
        }
        class_matrix = confusion_matrix(test_preds_np, target)
        accuracies = class_matrix.diagonal() / class_matrix.sum(axis=1)
        for f1, acc, label in zip(
            f1_score(target, test_preds_np, average=None),
            accuracies,
            CroptypeFranceDataset.labels,
        ):
            results[f"{self.name}: {prefix}_f1_score_{label}"] = f1
            results[f"{self.name}: {prefix}_accuracy_score_{label}"] = acc
        return results

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        mask = self.update_mask(mask)
        for model_mode in model_modes:
            assert model_mode in [
                "Regression",
                "Random Forest",
                "KNNat5",
                "KNNat20",
                "KNNat100",
                "finetune",
            ]
        results_dict = {}
        if "finetune" in model_modes:
            for aggregate in self.aggregates:
                model = self.finetune_with_aggregate(pretrained_model, mask, aggregate)
                results_dict.update(self.evaluate_for_finetuned_model(model, mask))
        sklearn_modes = [x for x in model_modes if x != "finetune"]
        if (len(sklearn_modes) > 0) and (len(self.sklearn_aggregates) > 0):
            train_dl = DataLoader(
                CroptypeFranceDataset(self.input_patch_size, split="train"),
                shuffle=False,
                batch_size=Hyperparams.batch_size // self.num_patches,
                collate_fn=CroptypeFranceDataset.collate_fn,
                num_workers=Hyperparams.num_workers,
            )
            sklearn_model_dict = self.finetune_sklearn_model(
                train_dl,
                pretrained_model,
                mask=mask,
                models=sklearn_modes,
            )
            results_dict.update(
                self.evaluate_for_sklearn(sklearn_model_dict, pretrained_model, mask)
            )
        return results_dict

    def finetune_with_aggregate(
        self, pretrained_model, mask: Optional[np.ndarray], aggregate: str
    ) -> FineTuningModel:
        hparams = hparams = Hyperparams(max_epochs=50, patience=50, batch_size=self.batch_size)
        model = self._construct_finetuning_model_with_aggregates(pretrained_model, aggregate)
        optimizer = AdamW(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        def val_loss_fn(preds, target):
            metrics = self.metrics("val", target.cpu(), preds.argmax(-1).cpu())
            # multiply by -1 so smaller is better
            return -1 * metrics[f"{self.name}: val_f1_score"]

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_dl = DataLoader(
            CroptypeFranceDataset(self.input_patch_size, split="train"),
            shuffle=True,
            batch_size=hparams.batch_size,
            collate_fn=CroptypeFranceDataset.finetuning_collate_fn,
            num_workers=hparams.num_workers,
            generator=generator,
        )
        val_dl = DataLoader(
            CroptypeFranceDataset(self.input_patch_size, split="val"),
            shuffle=False,
            batch_size=hparams.batch_size,
            collate_fn=CroptypeFranceDataset.finetuning_collate_fn,
            num_workers=hparams.num_workers,
        )
        return self.finetune_pytorch_model(
            model, hparams, optimizer, train_dl, val_dl, loss_fn, val_loss_fn, mask
        )
