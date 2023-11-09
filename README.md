# The Pretrained Remote Sensing Transformer (Presto)

This code accompanies our paper, [Lightweight, Pre-trained Transformers for Remote Sensing Timeseries](https://arxiv.org/abs/2304.14065).

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```
[`wandb`](https://pypi.org/project/wandb/) can additionally be installed for full functionality of the `train.py` script.

## Entrypoints

Three entrypoints to the code are available: [`train.py`](train.py), [`eval.py`](eval.py) and [`mosaiks.py`](mosaiks.py).

In addition, a [jupyter notebook](downstream_task_demo.ipynb) is available demonstrating how Presto can be finetuned on different downstream tasks.

Finally, Presto can also be loaded directly from the python package.
We also have included Presto contained in a single file (i.e. with no imports from elsewhere in the package) at [`single_file_presto.py`](single_file_presto.py), if you want to easily integrate it into a different application.
We [test](tests/test_presto.py) that these models are equivalent:
```python
# either import works. The single_file_presto has no load_pretrained function, since this
# requires knowing where the pretrained file is. The state dict can be loaded directly
# from data/default_models.pt
from single_file_presto import Presto
from presto import Presto

# to make a randomly initialized encoder-decoder model
encoder_decoder = Presto.construct()
# alternatively, the pre-trained model can also be loaded
encoder_decoder = Presto.load_pretrained()

# to isolate the encoder
encoder_only = encoder_decoder.encoder
# to add a linear transformation to the encoder's output for finetuning
finetuning_model = encoder_decoder.construct_finetuning_model(num_outputs=1, regression=True)
```
The default arguments to `construct` are the same as the default parameters described in [`default.json`](config/default.json).

Presto expects the following values as input, and returns the following outputs:
```python
reconstructed_x, reconstructed_dynamic_world = encoder_decoder(x, dynamic_world, latlons, mask, month)

globally_pooled_tokens = encoder(x, dynamic_world, latlons, mask, month, eval_task=True)

predictions = finetuning_model(x, dynamic_world, latlons, mask, month)
```
- `x`: *torch.Tensor* of shape `[batch_size, num_timesteps, bands]` where `bands` is described by [`NORMED_BANDS`](presto/dataops/pipelines/s1_s2_era5_srtm.py).
- `dynamic_world`: *torch.Tensor* of shape `[batch_size, num_timesteps]`. If no Dynamic World classes are available, this tensor should be filled with the value [`DynamicWorld2020_2021.class_amount`](presto/dataops/pipelines/dynamicworld.py) (i.e. `9`), in which case it is ignored.
- `latlons`: *torch.Tensor* of shape `[batch_size, 2]` describing the latitude and longitude of each input instance.
- `mask`: An optional *torch.Tensor* of shape `[batch_size, num_timesteps, bands]`. `mask[i, j, k] == 1` means `x[i, j, k]` is considered masked. If the mask is `None`, no values in `x` are ignored.
- `month`: An *int* or *torch.Tensor* describing the first month of the instances being passed. If an *int*, all instances in the batch are assumed to have the same starting month.

The number of timesteps passed is optional, and can be any value between 1 and 24 (2 years of data).

3 of the input tensors (`x`, `dynamic_world`, `mask`) can be generated using `presto.construct_single_presto_input`.
An example of this is in the [downstream task jupyter notebook](downstream_task_demo.ipynb).
For example, if I have access to some RGB imagery, it can be turned into Presto-compatible inputs:

```python
import presto
x, mask, dynamic_world = presto.construct_single_presto_input(
    s2=rgb_imagery,  # of shape [num_timesteps, 3]
    s2_bands=["B2", "B3", "B4"]
)
```
Here, `x` will contain only the (normalized) RGB values in the correct indices, and `mask` will communicate to Presto to ignore every other input.
Similarly, `dynamic_world` will contain only `DynamicWorld2020_2021.class_amount`, so Presto will ignore it.

### Training

The [`train.py`](train.py) script contains code for self-supervised training. This can be run locally on a small subset of the data with:

```bash
# Barebones local run
python train.py \
    --train_url "data/dw_144_mini_shard_44.tar" \
    --val_url "data/dw_144_mini_shard_44.tar" \
    --val_per_n_steps 1 \
    --cropharvest_per_n_validations 0 \
    --skip_finetuning
```

### Evaluation

A trained model (or a randomly initialized model) can be run against the evaluation tasks using [`eval.py`](eval.py). If an `--id` and `--epoch` is passed to the script, a model will be loaded from `models/{id}/{epoch}.pt` - otherwise, a randomly initialized model will be evaluated.

### Mosaiks

The MOSAIKS1D benchmark can be run against evaluation tasks using the [`mosaiks.py`](mosaiks.py) script.

## Generating new data

Diagram: [url](https://docs.google.com/presentation/d/1rxUXdSmKtfHSusBFzG3Apx-pLR0UE9oVIOaEB997bAM/edit?usp=sharing)

**Prerequisites:**
- Account with Google Cloud access and Earth Engine access
    ```bash
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-387.0.0-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-387.0.0-linux-x86_64.tar.gz
    exec bash
    ./google-cloud-sdk/install.sh
    gcloud init
    earthengine authenticate
    ```
- Create buckets for processing
    ```bash
    gcloud storage mb -l us-central1 $(python -c "from dataops import EE_BUCKET; print(EE_BUCKET)")
    gcloud storage mb -l us-central1 $(python -c "from dataops import NPY_BUCKET; print(NPY_BUCKET)")
    gcloud storage mb -l us-central1 $(python -c "from dataops import TAR_BUCKET; print(TAR_BUCKET)")
    ```
- Deploy tif-to-np Cloud Function
    ```bash
    sh scripts/deploy_tif_to_np.sh
    ```
Once prerequisites are satisfied, data can be generated by running:
```bash
python scripts/generate_data.py
```
⚠️ This script assumes you have a Google Cloud project named ``presto`` - you need to change this in the script if the name of the project is different. ⚠️

The script will generate:
- `data/tile_processing.txt` A summary of tiles being processed
- `data/tile_stats.yaml` Stats for all tiles available for training

Behind the scenes for each tile the script will:
1. Begin Earth Engine exports to get data for tile from specific data pipeline. Examples:
    - `gs://<EE_BUCKET>/<SHARD_1>/<PIPELINE_1>/*.tif`
    - `gs://<EE_BUCKET>/<SHARD_1>/<PIPELINE_2>/*.tif`
    - `gs://<EE_BUCKET>/<SHARD_1>/<PIPELINE_3>/*.tif`
2. Process each tif file to npy. Examples:
    - `gs://<NPY_BUCKET>/<SHARD_1>/<PIPELINE_1>/*.npy`
    - `gs://<NPY_BUCKET>/<SHARD_1>/<PIPELINE_2>/*.npy`
    - `gs://<NPY_BUCKET>/<SHARD_1>/<PIPELINE_3>/*.npy`
3. Combine all npy files into a tar file accessible through webdataset. Example:
    - `gs://<TAR_BUCKET>/<DATASET_NAME>/<SHARD_1>.tar`

## Accessing new data
```python
In [0]:
import webdataset as wds
import pandas as pd
uri = "gs://lem-assets2/S1_S2_ERA5_SRTM_2020_2021_DynamicWorld2020_2021_tars/dw_144_shard_0.tar"
dataset = wds.WebDataset(f"pipe:gcloud storage cat {uri}").decode()
for sample in dataset:
    break

In [1]: list(sample.keys())
Out[1]: ['__key__', '__url__', 'dynamicworld2020_2021.npy', 's1_s2_era5_srtm_2020_2021.npy', 'worldcover2020.npy']

In [2]: sample["s1_s2_era5_srtm_2020_2021.npy"].shape
Out[2]: (625, 24, 18)

In [3]: sample["latlon.npy"].shape
Out[3]: (625, 2)

In [4]: sample["worldcover2020.npy"].shape
Out[4]: (625, 1)

In [5]: sample["dynamicworld2020_2021.npy"].shape
Out[5]: (625, 24)

In [6]: pd.Series(sample["dynamicworld2020_2021.npy"].flatten()).value_counts()
Out[6]:
0    14978
7       22
dtype: int64

```

## Reference
If you find this code useful, please cite the following paper:
```
@misc{tseng2023lightweight,
      title={Lightweight, Pre-trained Transformers for Remote Sensing Timeseries},
      author={Gabriel Tseng and Ruben Cartuyvels and Ivan Zvonkov and Mirali Purohit and David Rolnick and Hannah Kerner},
      year={2023},
      eprint={2304.14065},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
