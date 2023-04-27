from dataclasses import dataclass
from typing import Tuple
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
from openmapflow.ee_exporter import ee_safe_str

from presto.dataops import EE_BUCKET, NPY_BUCKET, EEPipeline, resample_and_flatten


class MockBlob:
    def __init__(self, name):
        self.name = name


@dataclass
class MockCloudStorage:

    ee_bucket_files: Tuple[str, ...] = ()
    npy_bucket_files: Tuple[str, ...] = ()

    def list_blobs(self, bucket_name: str, prefix: str):
        if bucket_name == EE_BUCKET:
            files = self.ee_bucket_files
        elif bucket_name == NPY_BUCKET:
            files = self.npy_bucket_files
        else:
            raise ValueError(f"Unexpected bucket name: {bucket_name}")
        return iter([MockBlob(f) for f in files if f.startswith(prefix)])


class MockPipeline(EEPipeline):
    def create_ee_image(self, ee_polygon):
        return MagicMock()

    def convert_tif_to_np(self, tif_path) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones((1, 1)), np.ones((1, 1))


class TestEEPipeline(TestCase):
    def test_resample_and_flatten_4_dim(self):
        input_array = np.zeros((24, 18, 100, 100))
        actual = resample_and_flatten(input_array)
        expected = np.zeros((81, 24, 18))  # Because 9 samples from 100 pixels
        self.assertEqual(expected.shape, actual.shape)

    def test_resample_and_flatten_3_dim(self):
        input_array = np.zeros((250, 100, 100))
        actual = resample_and_flatten(input_array)
        expected = np.zeros((81, 250))  # Because 9 samples from 100 pixels
        self.assertEqual(expected.shape, actual.shape)

    def test_resample_and_flatten_2_dim(self):
        input_array = np.zeros((100, 100))
        self.assertRaises(ValueError, resample_and_flatten, input_array)

    def test_ee_pipeline_bad_prefix1(self):
        self.assertRaises(ValueError, MockPipeline().run, None, prefix="MockPipeline")

    def test_ee_pipeline_bad_prefix2(self):
        self.assertRaises(AssertionError, MockPipeline().run, None, prefix="tile_1/")

    @patch("presto.dataops.pipelines.ee_pipeline.ee")
    @patch("presto.dataops.pipelines.ee_pipeline.get_ee_task_list")
    @patch("presto.dataops.pipelines.ee_pipeline.storage")
    def test_ee_pipeline_npy_files_available(self, mock_storage, mock_get_ee_task_list, mock_ee):
        # ----------- Setup ---------------
        mock_get_ee_task_list.return_value = []
        mock_cloud_storage = MockCloudStorage(
            ee_bucket_files=("tile_1/MockPipeline/1.tif", "tile_1/MockPipeline/2.tif"),
            npy_bucket_files=("tile_1/MockPipeline/1.npy", "tile_1/MockPipeline/2.npy"),
        )
        mock_list_blobs = mock_storage.Client().list_blobs
        mock_list_blobs.side_effect = mock_cloud_storage.list_blobs
        mock_polygon = mock_ee.geometry.Polygon([])

        # ----------- Run pipeline ---------------
        npy_blobs = MockPipeline().run(mock_polygon, prefix="tile_1", is_large_polygon=True)

        # ----------- Check calls -----------
        self.assertEqual(2, mock_list_blobs.call_count)
        mock_list_blobs.assert_has_calls(
            [
                call(NPY_BUCKET, prefix="tile_1/MockPipeline/"),
                call(NPY_BUCKET, prefix="tile_1/MockPipeline/"),
            ]
        )
        mock_ee.batch.Export.image.toCloudStorage.assert_not_called()

        # ----------- Check output -----------
        self.assertEqual(2, len(npy_blobs))
        self.assertEqual("tile_1/MockPipeline/1.npy", npy_blobs[0].name)
        self.assertEqual("tile_1/MockPipeline/2.npy", npy_blobs[1].name)

    @patch("presto.dataops.pipelines.ee_pipeline.ee")
    @patch("presto.dataops.pipelines.ee_pipeline.get_ee_task_list")
    @patch("presto.dataops.pipelines.ee_pipeline.storage")
    def test_ee_pipeline_ee_files_available(self, mock_storage, mock_get_ee_task_list, mock_ee):
        # ----------- Setup ---------------
        mock_get_ee_task_list.return_value = []
        mock_cloud_storage = MockCloudStorage(
            ee_bucket_files=("tile_1/MockPipeline/1.tif", "tile_1/MockPipeline/2.tif"),
            npy_bucket_files=(),
        )
        mock_list_blobs = mock_storage.Client().list_blobs
        mock_list_blobs.side_effect = mock_cloud_storage.list_blobs
        mock_polygon = mock_ee.geometry.Polygon([])

        # ----------- Run pipeline ---------------
        npy_blobs = MockPipeline().run(mock_polygon, prefix="tile_1", is_large_polygon=True)

        # ----------- Check calls -----------
        self.assertEqual(2, mock_list_blobs.call_count)
        mock_list_blobs.assert_has_calls(
            [
                call(NPY_BUCKET, prefix="tile_1/MockPipeline/"),
                call(EE_BUCKET, prefix="tile_1/MockPipeline/"),
            ]
        )
        mock_ee.batch.Export.image.toCloudStorage.assert_not_called()

        # ----------- Check output -----------
        self.assertEqual(0, len(npy_blobs))

    @patch("presto.dataops.pipelines.ee_pipeline.ee")
    @patch("presto.dataops.pipelines.ee_pipeline.get_ee_task_list")
    @patch("presto.dataops.pipelines.ee_pipeline.storage")
    def test_ee_pipeline_ee_files_exporting(self, mock_storage, mock_get_ee_task_list, mock_ee):

        # ----------- Setup ---------------
        mock_get_ee_task_list.return_value = [ee_safe_str("tile_1/MockPipeline")]
        mock_cloud_storage = MockCloudStorage(ee_bucket_files=(), npy_bucket_files=())
        mock_list_blobs = mock_storage.Client().list_blobs
        mock_list_blobs.side_effect = mock_cloud_storage.list_blobs
        mock_polygon = mock_ee.geometry.Polygon([])

        # ----------- Run pipeline ---------------
        npy_blobs = MockPipeline().run(mock_polygon, prefix="tile_1", is_large_polygon=True)

        # ----------- Check calls -----------
        mock_ee.batch.Export.image.toCloudStorage.assert_not_called()

        # ----------- Check output -----------
        self.assertEqual(0, len(npy_blobs))

    @patch("presto.dataops.pipelines.ee_pipeline.ee")
    @patch("presto.dataops.pipelines.ee_pipeline.get_ee_task_list")
    @patch("presto.dataops.pipelines.ee_pipeline.storage")
    def test_ee_pipeline_ee_queue_full(self, mock_storage, mock_get_ee_task_list, mock_ee):

        # ----------- Setup ---------------
        mock_get_ee_task_list.return_value = [f"other_ee_task_{i}" for i in range(3001)]
        mock_cloud_storage = MockCloudStorage(ee_bucket_files=(), npy_bucket_files=())
        mock_list_blobs = mock_storage.Client().list_blobs
        mock_list_blobs.side_effect = mock_cloud_storage.list_blobs
        mock_polygon = mock_ee.geometry.Polygon([])

        # ----------- Run pipeline ---------------
        npy_blobs = MockPipeline().run(mock_polygon, prefix="tile_1", is_large_polygon=True)

        # ----------- Check calls -----------
        mock_ee.batch.Export.image.toCloudStorage.assert_not_called()

        # ----------- Check output -----------
        self.assertEqual(0, len(npy_blobs))

    @patch("presto.dataops.pipelines.ee_pipeline.ee")
    @patch("presto.dataops.pipelines.ee_pipeline.get_ee_task_list")
    @patch("presto.dataops.pipelines.ee_pipeline.storage")
    def test_ee_pipeline_run_once(self, mock_storage, mock_get_ee_task_list, mock_ee):

        # ----------- Setup ---------------
        mock_get_ee_task_list.return_value = []
        mock_cloud_storage = MockCloudStorage(ee_bucket_files=(), npy_bucket_files=())
        mock_list_blobs = mock_storage.Client().list_blobs
        mock_list_blobs.side_effect = mock_cloud_storage.list_blobs
        mock_polygon = mock_ee.geometry.Polygon([])

        # ----------- Run pipeline ---------------
        mock_pipeline = MockPipeline()
        self.assertEqual(mock_pipeline._ee_task_list, None)

        npy_blobs = mock_pipeline.run(mock_polygon, prefix="tile_1", is_large_polygon=True)
        self.assertEqual(mock_pipeline._ee_task_list, ["tile_1-MockPipeline"])

        # ----------- Check calls -----------
        self.assertEqual(2, mock_list_blobs.call_count)
        mock_list_blobs.assert_has_calls(
            [
                call(NPY_BUCKET, prefix="tile_1/MockPipeline/"),
                call(EE_BUCKET, prefix="tile_1/MockPipeline/"),
            ]
        )
        mock_ee.batch.Export.image.toCloudStorage.assert_called_once()

        # ----------- Check output -----------
        self.assertEqual(0, len(npy_blobs))
