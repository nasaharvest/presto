import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import xarray
from einops import repeat
from pyproj import Transformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .. import utils
from ..dataops import NUM_BANDS, NUM_ORG_BANDS
from ..dataops.pipelines.s1_s2_era5_srtm import (
    BANDS,
    BANDS_GROUPS_IDX,
    REMOVED_BANDS,
    S1_BANDS,
)
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import DEFAULT_SEED, device
from .eval import EvalDatasetWithPatches, EvalTaskWithAggregatedOutputs, Hyperparams

treesat_dir = "treesat"
s1_files_dir = "s1/60m"
s2_files_dir = "s2/60m"
labels_file = "TreeSatBA_v9_60m_multi_labels.json"

# https://zenodo.org/record/6780578
# Band order is B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, and B09.
# Spatial resolution is 10 m.
S2_BAND_ORDERING = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]
# Band order is VV, VH, and VV/VH ratio. Spatial resolution is 10 m.
S1_BAND_ORDERING = ["VV", "VH", "VV/VH"]


IMAGE_SIZE = 6


class TreeSatDataset(EvalDatasetWithPatches):
    labels_to_int = {
        "Abies": 0,
        "Acer": 1,
        "Alnus": 2,
        "Betula": 3,
        "Cleared": 4,
        "Fagus": 5,
        "Fraxinus": 6,
        "Larix": 7,
        "Picea": 8,
        "Pinus": 9,
        "Populus": 10,
        "Prunus": 11,
        "Pseudotsuga": 12,
        "Quercus": 13,
        "Tilia": 14,
    }

    # this is not the true start month!
    # the data is a mosaic of summer months
    start_month = 6

    def __init__(self, input_patch_size: int = 1, split: str = "train"):
        assert IMAGE_SIZE % input_patch_size == 0
        super().__init__(input_patch_size, int(IMAGE_SIZE / input_patch_size), split)

        with (utils.data_dir / treesat_dir / labels_file).open("r") as f:
            self.labels_dict = json.load(f)

    @staticmethod
    def image_name_to_paths(tif_file: str) -> Tuple[Path, Path]:
        s1_path = utils.data_dir / treesat_dir / s1_files_dir / Path(tif_file).name
        s2_path = utils.data_dir / treesat_dir / s2_files_dir / Path(tif_file).name
        return s1_path, s2_path

    @staticmethod
    def split_images(merge_train_val: bool = True):
        with (utils.data_dir / treesat_dir / "train_filenames.lst").open("r") as f:
            train_files = [line for line in f]
        with (utils.data_dir / treesat_dir / "test_filenames.lst").open("r") as f:
            test_files = [line for line in f]
        return {"train": train_files, "test": test_files}

    def image_to_eo_array(self, tif_file: str):
        s1_image, s2_image = self.image_name_to_paths(tif_file)
        s2 = xarray.open_rasterio(s2_image)
        s1 = xarray.open_rasterio(s1_image)
        # from (e.g.) +init=epsg:32630 to epsg:32630
        crs = s2.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        kept_treesat_s2_band_idx = [
            i for i, val in enumerate(S2_BAND_ORDERING) if val not in REMOVED_BANDS
        ]
        kept_kept_treesat_s2_band_names = [
            val for val in S2_BAND_ORDERING if val not in REMOVED_BANDS
        ]
        treesat_to_cropharvest_s2_map = [
            BANDS.index(val) for val in kept_kept_treesat_s2_band_names
        ]

        kept_treesat_s1_band_idx = [i for i, val in enumerate(S1_BAND_ORDERING) if val in S1_BANDS]
        kept_kept_treesat_s1_band_names = [val for val in S1_BAND_ORDERING if val in S1_BANDS]
        treesat_to_cropharvest_s1_map = [
            BANDS.index(val) for val in kept_kept_treesat_s1_band_names
        ]

        labels_np = np.zeros(len(self.labels_to_int))
        positive_classes = self.labels_dict[tif_file]
        for name, percentage in positive_classes:
            labels_np[self.labels_to_int[name]] = percentage

        eo_style_array = np.zeros([NUM_ORG_BANDS, IMAGE_SIZE, IMAGE_SIZE])
        eo_style_array[treesat_to_cropharvest_s2_map] = s2.values[kept_treesat_s2_band_idx]
        eo_style_array[treesat_to_cropharvest_s1_map] = s1.values[kept_treesat_s1_band_idx]
        lon, lat = transformer.transform(s2.x, s2.y)
        lonlats = np.meshgrid(lon, lat, indexing="xy")

        return eo_style_array, lonlats, self.min_threshold(labels_np)

    @staticmethod
    def min_threshold(labels: np.ndarray, binarize: bool = True):
        # this is what is also done in
        # https://git.tu-berlin.de/rsim/treesat_benchmark/-/blob/master/TreeSat_Benchmark/trainers/utils.py#L27
        lower_bound = 0.07  # anything below this is ignored
        bounded = np.where(
            labels > lower_bound,
            np.ones_like(lower_bound) if binarize else labels,
            np.zeros_like(lower_bound),
        )
        return bounded


class TreeSatEval(EvalTaskWithAggregatedOutputs):

    regression = False
    multilabel = True
    # different than the paper but this is
    # from all the unique classes in the labels json
    # (above)
    num_outputs = 15

    def __init__(
        self,
        subset: Optional[str] = None,
        input_patch_size: int = 1,
        aggregates: List[str] = ["mean"],
        num_histogram_bins: Optional[int] = 10,
        histogram_lower: Optional[int] = -1,
        histogram_upper: Optional[int] = 1,
        seeds: List[int] = [DEFAULT_SEED, 0, 84],
    ) -> None:
        if subset is not None:
            assert subset in ["S1", "S2"]
            self.name = f"TreeSatAI_{subset}_{input_patch_size}"
        else:
            self.name = f"TreeSatAI_{input_patch_size}"
        self.subset = subset

        # for each image, we will take an `input_patch_size x input_patch_size`
        # patch, and take a spatial mean of it
        self.input_patch_size = input_patch_size
        assert IMAGE_SIZE % input_patch_size == 0
        self.num_patches_per_dim = int(IMAGE_SIZE / input_patch_size)
        self.num_patches = self.num_patches_per_dim**2

        super().__init__(
            aggregates,
            self.num_patches,
            num_histogram_bins,
            histogram_lower,
            histogram_upper,
            seeds,
        )

    def update_mask(self, mask: Optional[np.ndarray] = None):
        if self.subset is None:
            channels_list = [
                x for k, x in BANDS_GROUPS_IDX.items() if (("S1" in k) or ("S2" in k))
            ]
            default_channels = [item for sublist in channels_list for item in sublist]
        else:
            channels_list = [x for k, x in BANDS_GROUPS_IDX.items() if (self.subset in k)]
            if self.subset == "S2":
                channels_list.append(BANDS_GROUPS_IDX["NDVI"])
            default_channels = [item for sublist in channels_list for item in sublist]

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

        def model_name(model) -> str:
            if hasattr(model, "random_state"):
                model_name = f"{model.__class__.__name__}_{model.random_state}"
            elif hasattr(model, "estimator"):
                # this is true for the multi output classifier
                model_name = f"{model.__class__.__name__}_{model.estimator.random_state}"
            else:
                model_name = model.__class__.__name__
            return model_name

        if isinstance(finetuned_model, Dict):
            assert isinstance(pretrained_model, (Seq2Seq, Mosaiks1d))
            pred_dict: Dict[str, Dict[str, List]] = {}
            for aggregate, model_list in finetuned_model.items():
                pred_dict[aggregate] = {model_name(model): [] for model in model_list}
        else:
            raise NotImplementedError

        test_dl = DataLoader(
            TreeSatDataset(self.input_patch_size, split="test"),
            batch_size=Hyperparams.batch_size // self.num_patches,
            shuffle=False,
            collate_fn=TreeSatDataset.collate_fn,
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
                    preds_list = model.predict_proba(reshaped_encodings.numpy())

                    # this is a list of probabilities; we want to take the sum of
                    # positive predictions
                    preds = np.zeros((preds_list[0].shape[0], self.num_outputs))
                    for idx, pred in enumerate(preds_list):
                        if pred.shape[1] == 2:
                            # if not, there are no positive samples
                            preds[:, idx] = pred[:, 1]
                    pred_dict[aggregate][model_name(model)].append(preds)

        target = np.concatenate(labels)
        results_dict = {}
        for aggregate, model_pred_dict in pred_dict.items():
            for model_name_str, pred_list in model_pred_dict.items():
                test_preds_np = np.concatenate(pred_list, axis=0)
                test_preds_binary = test_preds_np > 0.5

                prefix = f"{model_name_str}_{aggregate}"
                results_dict.update(
                    {
                        f"{self.name}: {prefix}_num_samples": len(target),
                        f"{self.name}: {prefix}_mAP_score_weighted": average_precision_score(
                            target, test_preds_np, average="weighted"
                        ),
                        f"{self.name}: {prefix}_mAP_score_micro": average_precision_score(
                            target, test_preds_np, average="micro"
                        ),
                        f"{self.name}: {prefix}_f1_score_weighted": f1_score(
                            target, test_preds_binary, average="weighted"
                        ),
                        f"{self.name}: {prefix}_f1_score_micro": f1_score(
                            target, test_preds_binary, average="micro"
                        ),
                        f"{self.name}: {prefix}_precision_micro": precision_score(
                            target, test_preds_binary, average="micro"
                        ),
                        f"{self.name}: {prefix}_precision_weighted": precision_score(
                            target, test_preds_binary, average="weighted"
                        ),
                        f"{self.name}: {prefix}_recall_micro": recall_score(
                            target, test_preds_binary, average="micro"
                        ),
                        f"{self.name}: {prefix}_recall_weighted": recall_score(
                            target, test_preds_binary, average="weighted"
                        ),
                        f"{self.name}: {prefix}_accuracy_score": accuracy_score(
                            target, test_preds_binary
                        ),
                    }
                )
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
                "Regression",
                "Random Forest",
                "KNNat5",
                "KNNat20",
                "KNNat100",
            ]
        if len(model_modes) > 0:
            dl = DataLoader(
                TreeSatDataset(self.input_patch_size),
                shuffle=False,
                batch_size=Hyperparams.batch_size // self.num_patches,
                collate_fn=TreeSatDataset.collate_fn,
                num_workers=Hyperparams.num_workers,
            )
            sklearn_model_dict = self.finetune_sklearn_model(
                dl,
                pretrained_model,
                mask=mask,
                models=model_modes,
            )
            return self.evaluate(sklearn_model_dict, pretrained_model, mask)
        return {}

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        raise NotImplementedError
        # hyperparams = Hyperparams()
        # prop_val_images = 0.1
        # model = self._construct_finetuning_model(pretrained_model)
        #
        # opt = Adam(model.parameters(), lr=hyperparams.lr)
        # loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        #
        # x, dw, latlons, target, image_names = self.load_npys(test=False)
        #
        # unique_image_names = np.unique(image_names)
        # num = round(prop_val_images * len(unique_image_names))
        # val_images = np.random.choice(unique_image_names, size=num, replace=False)
        # val_filter = np.isin(image_names, val_images)
        #
        # train_dl = DataLoader(
        #     TensorDataset(
        #         torch.from_numpy(x[~val_filter]).to(device).float(),
        #         torch.from_numpy(dw[~val_filter]).to(device).long(),
        #         torch.from_numpy(latlons[~val_filter]).to(device).float(),
        #         torch.from_numpy(target[~val_filter]).to(device).float(),
        #         torch.full(target[~val_filter].shape[:1], self.start_month, device=device).long(),
        #     ),
        #     batch_size=hyperparams.batch_size,
        #     shuffle=True,
        # )
        #
        # val_dl = DataLoader(
        #     TensorDataset(
        #         torch.from_numpy(x[val_filter]).to(device).float(),
        #         torch.from_numpy(dw[val_filter]).to(device).long(),
        #         torch.from_numpy(latlons[val_filter]).to(device).float(),
        #         torch.from_numpy(target[val_filter]).to(device).float(),
        #         torch.full(target[val_filter].shape[:1], self.start_month, device=device).long(),
        #     ),
        #     batch_size=hyperparams.batch_size,
        #     shuffle=False,
        # )
        #
        # return self.finetune_pytorch_model(
        #     model, hyperparams, opt, train_dl, val_dl, loss_fn, loss_fn, mask
        # )
