"""
Renames files in a bucket from .tif to -retry.tif
OR renames entire pipeline folder from /path/to/pipeline/* to /path/to/pipeline-retry/*
This is handy to retrigger the tif_to_np Cloud Function if it has previously failed and
a fixed version of the function has been redeployed OR an update has been made.
"""
from typing import Optional

from google.cloud import storage
from tqdm import tqdm

client = storage.Client()


def retry_tif_to_np_files(bucket_name: str, prefix: str):
    """Retrigger by renaming one file at a time using Python API"""
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    for blob in tqdm(blobs, desc=f"Renaming blobs: {prefix}"):
        bucket.rename_blob(blob, blob.name.replace(".tif", "-retry.tif"))


def retry_tif_to_np_pipeline(bucket_name: str, prefix: str, amount: Optional[int] = None):
    """
    Retrigger by renaming entire pipeline folder using
    gcloud storage mv , parallelized and much faster
    """
    assert prefix[-1] != "/"
    src = f"gs://{bucket_name}/{prefix}"
    dest = f"gs://{bucket_name}/{prefix}-retry"
    print("Run the following command:")
    if "*" not in prefix:
        print(f"gcloud storage mv \\\n\t{src} \\\n\t{dest}")
        return

    assert amount is not None

    def sub_idx(s: str, i: int):
        return s.replace("*", str(i))

    for i in range(amount):
        print(f"gcloud storage mv \\\n\t{sub_idx(src, i)} \\\n\t{sub_idx(dest, i)}")


if __name__ == "__main__":
    retry_tif_to_np_pipeline(
        "lem-earthengine2",
        "dw_144_shard_0",
    )
