import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
import xarray
from einops import repeat
from pyproj import Transformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import utils
from ..dataops import NUM_BANDS, NUM_ORG_BANDS
from ..dataops.pipelines.s1_s2_era5_srtm import (
    BANDS,
    BANDS_GROUPS_IDX,
    REMOVED_BANDS,
    S2_BANDS,
)
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import device
from .eval import EvalDatasetWithPatches, EvalTaskWithAggregatedOutputs, Hyperparams

tif_files_dir = "eurosat/EuroSAT_MS"

IMAGE_SIZE = 64


class EuroSatDataset(EvalDatasetWithPatches):

    labels_to_int = {
        "AnnualCrop": 0,
        "Forest": 1,
        "HerbaceousVegetation": 2,
        "Highway": 3,
        "Industrial": 4,
        "Pasture": 5,
        "PermanentCrop": 6,
        "Residential": 7,
        "River": 8,
        "SeaLake": 9,
    }

    # this is not the true start month!
    start_month = 1

    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt",
    }

    def __init__(
        self, input_patch_size: int = 1, split: str = "train", merge_train_val: bool = True
    ):
        assert IMAGE_SIZE % input_patch_size == 0
        super().__init__(
            input_patch_size, int(IMAGE_SIZE / input_patch_size), split, merge_train_val
        )

    def image_to_eo_array(self, tif_filename: str):
        tif_file = self.image_name_to_path(tif_filename)
        image = xarray.open_rasterio(tif_file)
        # from (e.g.) +init=epsg:32630 to epsg:32630
        crs = image.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        indices_to_remove = []
        for band in REMOVED_BANDS:
            indices_to_remove.append(S2_BANDS.index(band))
        kept_s2_bands = [i for i in range(len(S2_BANDS)) if i not in indices_to_remove]
        kept_overall_bands = [
            idx for idx, x in enumerate(BANDS) if ((x in S2_BANDS) and (x not in REMOVED_BANDS))
        ]

        eo_style_array = np.zeros([NUM_ORG_BANDS, IMAGE_SIZE, IMAGE_SIZE])
        eo_style_array[kept_overall_bands] = image.values[kept_s2_bands]
        lon, lat = transformer.transform(image.x, image.y)
        lonlats = np.meshgrid(lon, lat, indexing="xy")

        return (
            eo_style_array,
            lonlats,
            np.array([self.labels_to_int[tif_file.parents[0].name]]),
        )

    @staticmethod
    def url_to_list(url: str) -> List[str]:
        data = urllib.request.urlopen(url).read()
        return data.decode("utf-8").split("\n")

    @staticmethod
    def split_images(merge_train_val: bool = True) -> Dict[str, List[str]]:
        # updated to use the splits stored in
        # https://storage.googleapis.com/remote_sensing_representations
        # as per torchgeo
        filename = (
            "eurosat/train_test_split.json"
            if merge_train_val
            else "eurosat/train_val_test_split.json"
        )
        split_path = utils.data_dir / filename
        if split_path.exists():
            train_test_split = json.load(split_path.open("r"))
        else:
            # this code was only run once (the dictionary is then saved)
            # but is saved here for clarity
            train_images = EuroSatDataset.url_to_list(EuroSatDataset.split_urls["train"])
            test_images = EuroSatDataset.url_to_list(EuroSatDataset.split_urls["test"])
            train_test_split = {"train": train_images, "test": test_images}
            if merge_train_val:
                train_test_split["train"] += EuroSatDataset.url_to_list(
                    EuroSatDataset.split_urls["val"]
                )
            else:
                train_test_split["val"] = EuroSatDataset.url_to_list(
                    EuroSatDataset.split_urls["val"]
                )
            json.dump(train_test_split, split_path.open("w"))
        return train_test_split

    @staticmethod
    def image_name_to_path(name: str) -> Path:
        class_name = name.split("_")[0]
        if name.endswith("jpg"):
            name = f"{name.split('.')[0]}.tif"
        return utils.data_dir / tif_files_dir / class_name / name


class EuroSatEval(EvalTaskWithAggregatedOutputs):
    regression = False
    multilabel = False
    num_outputs = 10

    def __init__(
        self,
        rgb: bool = False,
        input_patch_size: int = 1,
        aggregates: List[str] = ["mean", "quantiles", "histogram"],
        num_histogram_bins: Optional[int] = 10,
        histogram_lower: Optional[int] = -1,
        histogram_upper: Optional[int] = 1,
    ) -> None:
        self.rgb = rgb

        # for each image, we will take an `input_patch_size x input_patch_size`
        # patch, and take a spatial mean of it
        self.input_patch_size = input_patch_size
        assert IMAGE_SIZE % input_patch_size == 0
        self.num_patches_per_dim = int(IMAGE_SIZE / input_patch_size)
        self.num_patches = self.num_patches_per_dim**2

        self.name = f"EuroSat_{input_patch_size}" if not rgb else f"EuroSat_RGB_{input_patch_size}"

        super().__init__(
            aggregates, self.num_patches, num_histogram_bins, histogram_lower, histogram_upper
        )

    def update_mask(self, mask: Optional[np.ndarray] = None):
        if not self.rgb:
            channels_lists = [x for key, x in BANDS_GROUPS_IDX.items() if "S2" in key]
            channels_lists.append(BANDS_GROUPS_IDX["NDVI"])
            # flatten the list of lists
            default_channels = [item for sublist in channels_lists for item in sublist]
        else:
            # a bit hacky but it will have to do
            default_channels = BANDS_GROUPS_IDX["S2_RGB"]

        # everything is masked by default
        default_mask = np.ones([NUM_BANDS])
        # unmask the s2 bands
        default_mask[default_channels] = 0
        default_mask = repeat(default_mask, "d -> t d", t=1)

        if mask is not None:
            return np.clip(mask + default_mask, a_min=0, a_max=1)
        else:
            return np.clip(default_mask, a_min=0, a_max=1)

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model: Union[FineTuningModel, Dict],
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        updated_mask = self.update_mask(mask)

        if isinstance(finetuned_model, Dict):
            assert isinstance(pretrained_model, (Seq2Seq, Mosaiks1d))
            pred_dict: Dict[str, Dict[str, List]] = {}
            for aggregate, model_list in finetuned_model.items():
                pred_dict[aggregate] = {model.__class__.__name__: [] for model in model_list}
        else:
            raise NotImplementedError

        test_dataset = EuroSatDataset(self.input_patch_size, split="test")
        test_dl = DataLoader(
            test_dataset,
            batch_size=Hyperparams.batch_size // self.num_patches,
            shuffle=False,
            collate_fn=EuroSatDataset.collate_fn,
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
                reshaped_encodings = self.reshape_for_aggregate(encodings, aggregate).cpu()
                assert not torch.isnan(reshaped_encodings).any()
                for model in model_list:
                    preds = model.predict(reshaped_encodings.numpy())
                    pred_dict[aggregate][model.__class__.__name__].append(preds)

        target = np.concatenate(labels)
        results_dict = {}
        int_to_labels, _ = zip(*sorted(test_dataset.labels_to_int.items(), key=lambda l_i: l_i[1]))
        for aggregate, model_pred_dict in pred_dict.items():
            for model_name, pred_list in model_pred_dict.items():
                test_preds_np = np.concatenate(pred_list, axis=0)
                prefix = f"{model_name}_{aggregate}"
                results_dict.update(
                    {
                        f"{self.name}: {prefix}_num_samples": len(target),
                        f"{self.name}: {prefix}_f1_score": f1_score(
                            target, test_preds_np, average="weighted"
                        ),
                        f"{self.name}: {prefix}_accuracy_score": accuracy_score(
                            target, test_preds_np
                        ),
                    }
                )
                class_matrix = confusion_matrix(test_preds_np, target)
                accuracies = class_matrix.diagonal() / class_matrix.sum(axis=1)
                for f1, acc, label in zip(
                    f1_score(target, test_preds_np, average=None), accuracies, int_to_labels
                ):
                    results_dict[f"{self.name}: {prefix}_f1_score_{label}"] = f1
                    results_dict[f"{self.name}: {prefix}_accuracy_score_{label}"] = acc
        return results_dict

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        mask = self.update_mask(mask)
        for model_mode in model_modes:
            assert model_mode in [
                # "finetune",  # not yet implemented
                "Regression",
                "Random Forest",
                "KNNat5",
                "KNNat20",
                "KNNat100",
            ]
        # if "finetune" in model_modes:
        #     model = self.finetune(pretrained_model, mask)
        #     results_dict.update(self.evaluate(model, None, mask))
        # sklearn_modes = [x for x in model_modes if x != "finetune"]
        sklearn_modes = model_modes
        if len(sklearn_modes) > 0:
            train_dl = DataLoader(
                EuroSatDataset(self.input_patch_size, split="train", merge_train_val=True),
                shuffle=False,
                batch_size=Hyperparams.batch_size // self.num_patches,
                collate_fn=EuroSatDataset.collate_fn,
                num_workers=Hyperparams.num_workers,
            )
            sklearn_model_dict = self.finetune_sklearn_model(
                train_dl,
                pretrained_model,
                mask=mask,
                models=sklearn_modes,
            )
            return self.evaluate(sklearn_model_dict, pretrained_model, mask)
        return {}

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        # TODO - where are these controlled?
        raise NotImplementedError
        # hyperparams = Hyperparams()
        # prop_val_images = 0.1
        # model = self._construct_finetuning_model(pretrained_model)

        # # TODO - should this be more intelligent? e.g. first learn the
        # # (randomly initialized) head before modifying parameters for
        # # the whole model?
        # opt = Adam(model.parameters(), lr=hyperparams.lr)
        # loss_fn = nn.CrossEntropyLoss(reduction="mean")

        # x, dw, latlons, target, image_names = self.load_npys(test=False)

        # unique_image_names = np.unique(image_names)
        # num = round(prop_val_images * len(unique_image_names))
        # val_images = np.random.choice(unique_image_names, size=num, replace=False)
        # val_filter = np.isin(image_names, val_images)

        # train_dl = DataLoader(
        #     TensorDataset(
        #         torch.from_numpy(x[~val_filter]).to(device).float(),
        #         torch.from_numpy(dw[~val_filter]).to(device).long(),
        #         torch.from_numpy(latlons[~val_filter]).to(device).float(),
        #         torch.from_numpy(target[~val_filter]).to(device).long(),
        #         torch.full(target[~val_filter].shape[:1], self.start_month, device=device).long(),
        #     ),
        #     batch_size=hyperparams.batch_size,
        #     shuffle=True,
        # )

        # val_dl = DataLoader(
        #     TensorDataset(
        #         torch.from_numpy(x[val_filter]).to(device).float(),
        #         torch.from_numpy(dw[val_filter]).to(device).long(),
        #         torch.from_numpy(latlons[val_filter]).to(device).float(),
        #         torch.from_numpy(target[val_filter]).to(device).long(),
        #         torch.full(target[val_filter].shape[:1], self.start_month, device=device).long(),
        #     ),
        #     batch_size=hyperparams.batch_size,
        #     shuffle=False,
        # )

        # return self.finetune_pytorch_model(
        #     model, hyperparams, opt, train_dl, val_dl, loss_fn, loss_fn, mask
        # )
