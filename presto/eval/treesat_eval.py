from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import xarray
from einops import repeat
from google.cloud import storage
from pyproj import Transformer
from scipy import stats
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..dataops import NUM_BANDS, NUM_ORG_BANDS, TAR_BUCKET, DynamicWorld2020_2021
from ..dataops.pipelines.s1_s2_era5_srtm import (
    BANDS,
    BANDS_GROUPS_IDX,
    REMOVED_BANDS,
    S1_BANDS,
    S1_S2_ERA5_SRTM,
)
from ..model import FineTuningModel, Mosaiks1d, Seq2Seq
from ..utils import data_dir, device
from .eval import EvalDataset

treesat_folder = data_dir / "treesat"
s1_files = treesat_folder / "s1/60m"
s2_files = treesat_folder / "s2/60m"

# https://zenodo.org/record/6780578
# Band order is B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, and B09.
# Spatial resolution is 10 m.
S2_BAND_ORDERING = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]
# Band order is VV, VH, and VV/VH ratio. Spatial resolution is 10 m.
S1_BAND_ORDERING = ["VV", "VH", "VV/VH"]


# takes a (6, 6) treesat tif file, and returns a
# (9,1,18) cropharvest eo-style file (with all bands "masked"
# except for S1 and S2)
INDICES_IN_TIF_FILE = list(range(0, 6, 2))


class TreeSatEval(EvalDataset):

    regression = False
    num_outputs = 20

    # this is not the true start month!
    # the data is a mosaic of summer months
    start_month = 6

    def __init__(self, subset: Optional[str] = None) -> None:

        if subset is not None:
            assert subset in ["S1", "S2"]
            self.name = f"TreeSatAI_{subset}"
        else:
            self.name = "TreeSatAI"
        self.subset = subset

    @staticmethod
    def s2_image_path_to_s1_path_and_class(path: Path) -> Tuple[Path, str]:
        class_name = path.name.split("_")[0]
        s1_path = s1_files / path.name
        return s1_path, class_name

    @staticmethod
    def split_images():
        with (treesat_folder / "train_filenames.lst").open("r") as f:
            train_files = [line for line in f]
        with (treesat_folder / "test_filenames.lst").open("r") as f:
            test_files = [line for line in f]
        return {"train": train_files, "test": test_files}

    @classmethod
    def image_to_eo_array(cls, tif_file: Path):

        s1_image, class_name = cls.s2_image_path_to_s1_path_and_class(tif_file)
        s2 = xarray.open_rasterio(tif_file)
        s1 = xarray.open_rasterio(s1_image)
        # from (e.g.) +init=epsg:32630 to epsg:32630
        crs = s2.crs.split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326")

        arrays, latlons, labels, image_names = [], [], [], []

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

        for x_idx in INDICES_IN_TIF_FILE:
            for y_idx in INDICES_IN_TIF_FILE:
                s2_vals = s2.values[:, x_idx, y_idx]
                s1_vals = s1.values[:, x_idx, y_idx]
                x, y = s2.x[x_idx], s2.y[y_idx]
                lon, lat = transformer.transform(x, y)
                latlons.append(np.array([lat, lon]))
                eo_style_array = np.zeros([NUM_ORG_BANDS])
                eo_style_array[treesat_to_cropharvest_s2_map] = s2_vals[kept_treesat_s2_band_idx]
                eo_style_array[treesat_to_cropharvest_s1_map] = s1_vals[kept_treesat_s1_band_idx]
                arrays.append(np.expand_dims(eo_style_array, 0))
                labels.append(class_name)
                image_names.append(tif_file.name)
        return (
            np.stack(arrays, axis=0),
            np.stack(latlons, axis=0),
            np.array(labels),
            np.array(image_names),
        )

    @classmethod
    def tifs_to_arrays(cls, mode: str = "train"):
        features_folder = data_dir / f"treesat/{mode}_features"
        features_folder.mkdir(exist_ok=True)

        images_to_process = cls.split_images()[mode]

        arrays, latlons, labels, image_names = [], [], [], []

        for image in tqdm(images_to_process):
            output = cls.image_to_eo_array(s2_files / image.strip())
            arrays.append(output[0])
            latlons.append(output[1])
            labels.append(output[2])
            image_names.append(output[3])

        np.save(features_folder / "arrays.npy", np.concatenate(arrays, axis=0))
        np.save(features_folder / "latlons.npy", np.concatenate(latlons, axis=0))
        np.save(features_folder / "labels.npy", np.concatenate(labels, axis=0))
        np.save(features_folder / "image_names.npy", np.concatenate(image_names, axis=0))

    @staticmethod
    def load_npy_gcloud(path: Path) -> np.ndarray:
        if not path.exists():
            blob = (
                storage.Client()
                .bucket(TAR_BUCKET)
                .blob(f"eval/treesat/{path.parent.name}/{path.name}")
            )
            blob.download_to_filename(path)
        return np.load(path)

    @classmethod
    def load_npys(
        cls, test: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mode = "test" if test else "train"
        npy_folder = data_dir / f"treesat/{mode}_features"
        npy_folder.mkdir(exist_ok=True)

        labels = cls.load_npy_gcloud(npy_folder / "labels.npy")
        # this returns ints instead of strings, so that we can seamlessly
        # pass it to the sklearn models
        _, labels_to_int = np.unique(labels, return_inverse=True)

        # all dynamic world values are considered masked
        dw = np.ones((len(labels), 1)) * DynamicWorld2020_2021.class_amount

        return (
            S1_S2_ERA5_SRTM.normalize(cls.load_npy_gcloud(npy_folder / "arrays.npy")),
            dw,
            cls.load_npy_gcloud(npy_folder / "latlons.npy"),
            labels_to_int,
            cls.load_npy_gcloud(npy_folder / "image_names.npy"),
        )

    def update_mask(self, mask: Optional[np.ndarray] = None):
        # if not self.rgb:
        #     channels_lists = [x for key, x in BANDS_GROUPS_IDX.items() if "S2" in key]
        #     # flatten the list of lists
        #     default_channels = [item for sublist in channels_lists for item in sublist]
        # else:
        # a bit hacky but it will have to do
        if self.subset is None:
            channels_list = [
                x for k, x in BANDS_GROUPS_IDX.items() if (("S1" in k) or ("S2" in k))
            ]
            default_channels = [item for sublist in channels_list for item in sublist]
        else:
            channels_list = [x for k, x in BANDS_GROUPS_IDX.items() if (self.subset in k)]
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
        finetuned_model: Union[FineTuningModel, BaseEstimator],
        pretrained_model=None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        updated_mask = self.update_mask(mask)

        batch_size = 64

        if isinstance(finetuned_model, BaseEstimator):
            assert isinstance(pretrained_model, (Seq2Seq, Mosaiks1d))

        x, dw, latlon, target, image_names = self.load_npys(test=True)
        pix_per_image = len(INDICES_IN_TIF_FILE) ** 2
        assert len(target) % pix_per_image == 0
        assert len(np.unique(image_names)) == len(target) / pix_per_image
        target = np.reshape(target, (int(len(target) / pix_per_image), pix_per_image))
        image_names = np.reshape(
            image_names, (int(len(image_names) / pix_per_image), pix_per_image)
        )
        assert (image_names.T == image_names.T[0, :]).all()
        assert (target.T == target.T[0, :]).all()
        image_names = image_names[:, 0]
        target = target[:, 0]

        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(x).to(device).float(),
                torch.from_numpy(dw).to(device).long(),
                torch.from_numpy(latlon).to(device).float(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        test_preds = []
        for (x, dw, latlons) in dl:
            batch_mask = self._mask_to_batch_tensor(updated_mask, x.shape[0])
            if isinstance(finetuned_model, FineTuningModel):
                finetuned_model.eval()
                preds = (
                    finetuned_model(
                        x,
                        dynamic_world=dw,
                        mask=batch_mask,
                        latlons=latlons,
                        month=self.start_month,
                    )
                    .cpu()
                    .numpy()
                )
            elif isinstance(finetuned_model, BaseEstimator):
                cast(Seq2Seq, pretrained_model).eval()
                encodings = (
                    cast(Seq2Seq, pretrained_model)
                    .encoder(
                        x,
                        dynamic_world=dw,
                        mask=batch_mask,
                        latlons=latlons,
                        month=self.start_month,
                    )
                    .cpu()
                    .numpy()
                )
                preds = finetuned_model.predict_proba(encodings)

            test_preds.append(preds)
        test_preds_np = np.concatenate(test_preds, axis=0)
        test_preds_np = np.reshape(
            test_preds_np,
            (int(len(test_preds_np) / pix_per_image), pix_per_image, test_preds_np.shape[-1]),
        )
        # then, take the mode of the model predictions
        test_preds_np_argmax = stats.mode(
            np.argmax(test_preds_np, axis=-1), axis=1, keepdims=False
        )[0]

        test_preds_np_mean = np.mean(softmax(test_preds_np, axis=-1), axis=1)
        classes = list(range(test_preds_np_mean.shape[-1]))

        prefix = finetuned_model.__class__.__name__
        return {
            f"{self.name}: {prefix}_num_samples": len(target),
            f"{self.name}: {prefix}_mAP_score_weighted": average_precision_score(
                label_binarize(target, classes=classes), test_preds_np_mean, average="weighted"
            ),
            f"{self.name}: {prefix}_mAP_score_micro": average_precision_score(
                label_binarize(target, classes=classes), test_preds_np_mean, average="micro"
            ),
            f"{self.name}: {prefix}_f1_score_weighted": f1_score(
                target, test_preds_np_argmax, average="weighted"
            ),
            f"{self.name}: {prefix}_f1_score_micro": f1_score(
                target, test_preds_np_argmax, average="micro"
            ),
            f"{self.name}: {prefix}_precision_micro": precision_score(
                target, test_preds_np_argmax, average="micro"
            ),
            f"{self.name}: {prefix}_precision_weighted": precision_score(
                target, test_preds_np_argmax, average="weighted"
            ),
            f"{self.name}: {prefix}_recall_micro": recall_score(
                target, test_preds_np_argmax, average="micro"
            ),
            f"{self.name}: {prefix}_recall_weighted": recall_score(
                target, test_preds_np_argmax, average="weighted"
            ),
            f"{self.name}: {prefix}_accuracy_score": accuracy_score(target, test_preds_np_argmax),
        }

    def finetuning_results(
        self,
        pretrained_model,
        model_modes: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> Dict:
        mask = self.update_mask(mask)
        for model_mode in model_modes:
            assert model_mode in [
                "finetune",
                "Regression",
                "Random Forest",
                "KNNat5",
                "KNNat20",
                "KNNat100",
            ]
        results_dict = {}
        if "finetune" in model_modes:
            model = self.finetune(pretrained_model, mask)
            results_dict.update(self.evaluate(model, None, mask))
        sklearn_modes = [x for x in model_modes if x != "finetune"]
        if len(sklearn_modes) > 0:
            x, dw, latlons, target, _ = self.load_npys(test=False)
            sklearn_models = self.finetune_sklearn_model(
                x,
                target,
                pretrained_model,
                dynamic_world=dw,
                latlons=latlons,
                mask=mask,
                month=self.start_month,
                models=sklearn_modes,
            )
            for sklearn_model in sklearn_models:
                results_dict.update(self.evaluate(sklearn_model, pretrained_model, mask))
        return results_dict

    def finetune(self, pretrained_model, mask: Optional[np.ndarray] = None) -> FineTuningModel:
        # TODO - where are these controlled?
        lr, max_epochs, batch_size = 3e-4, 3, 64
        model = self._construct_finetuning_model(pretrained_model)

        # TODO - should this be more intelligent? e.g. first learn the
        # (randomly initialized) head before modifying parameters for
        # the whole model?
        opt = Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(reduction="mean")

        x, dw, latlons, target, _ = self.load_npys(test=False)

        dl = DataLoader(
            TensorDataset(
                torch.from_numpy(x).to(device).float(),
                torch.from_numpy(dw).to(device).long(),
                torch.from_numpy(latlons).to(device).float(),
                torch.from_numpy(target).to(device).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        train_loss = []
        for _ in range(max_epochs):
            model.train()
            epoch_train_loss = 0.0
            for x, dw, latlons, y in dl:
                opt.zero_grad()
                b_mask = self._mask_to_batch_tensor(mask, x.shape[0])
                preds = model(
                    x,
                    dynamic_world=dw,
                    mask=b_mask,
                    latlons=latlons,
                    month=self.start_month,
                )
                loss = loss_fn(preds, y)
                epoch_train_loss += loss.item()
                loss.backward()
                opt.step()
            train_loss.append(epoch_train_loss / len(dl))

        model.eval()
        return model
