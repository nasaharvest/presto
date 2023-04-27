from unittest import TestCase
from unittest.mock import MagicMock, patch

import ee
import shapely

from presto.dataops import EE_BUCKET, NPY_BUCKET
from presto.dataops.dataset import TAR_BUCKET, Dataset


class MockBlob:
    def __init__(self, name):
        self.name = name


class TestDataset(TestCase):
    def check_ee_prereq(self):
        try:
            ee.Initialize()
        except ee.EEException:
            self.skipTest("Earth Engine could not be initialized")

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_big_polygon_already_available(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"

        mock_pipeline_1.run.return_value = []
        mock_pipeline_2.run.return_value = []
        polygon = shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])
        mock_storage.Client().list_blobs.return_value = iter(
            [MockBlob(f"{dataset.name}_tars/mock_tile_{i}.tar") for i in range(4)]
        )
        status = dataset._create_webdataset_tar_from_big_polygon(polygon, "mock_tile")
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 1)
        self.assertEqual(mock_pipeline_1.run.call_count, 0)
        self.assertEqual(mock_pipeline_2.run.call_count, 0)
        self.assertTrue("Earth Engine export: 0" in status)
        self.assertTrue("Numpy processing: 0" in status)
        self.assertTrue("Complete tars: 4" in status)

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_no_npy_files(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"
        mock_pipeline_1.run.return_value = []
        mock_pipeline_2.run.return_value = []
        polygon = shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])
        status = dataset._create_webdataset_tar_from_big_polygon(polygon, "mock_tile")

        # Polygon is divided into 4 tiles, each tile lists blobs for data and latlons
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 5)
        self.assertEqual(mock_pipeline_1.run.call_count, 4)
        self.assertEqual(mock_pipeline_2.run.call_count, 4)
        self.assertTrue("Earth Engine export: 4" in status)
        self.assertTrue("Numpy processing: 0" in status)
        self.assertTrue("Complete tars: 0" in status)

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_big_polygon_some_npy_files(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"
        mock_pipeline_1.run.return_value = ["mock_file_1.npy", "mock_file_2.npy"]
        mock_pipeline_2.run.return_value = []
        polygon = shapely.geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])
        status = dataset._create_webdataset_tar_from_big_polygon(polygon, "mock_tile")

        # Polygon is divided into 4 tiles, each tile lists blobs for data and latlons
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 5)
        self.assertEqual(mock_pipeline_1.run.call_count, 4)
        self.assertEqual(mock_pipeline_2.run.call_count, 4)
        self.assertTrue("Earth Engine export: 0" in status)
        self.assertTrue("Numpy processing: 4" in status)
        self.assertTrue("Complete tars: 0" in status)

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_small_polygons_already_available(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"

        n_polygons = 100
        mock_pipeline_1.run.return_value = []
        mock_pipeline_2.run.return_value = []

        polygons = [
            shapely.geometry.Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])
        ] * n_polygons
        polygon_ids = [f"mock_polygon_{i}" for i in range(n_polygons)]
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])

        def side_effect(bucket, prefix=None):
            return {
                EE_BUCKET: iter(
                    [MockBlob(f"folder/mock_polygon_{i}/{i}.tif") for i in range(n_polygons)]
                ),
                NPY_BUCKET: iter(
                    [MockBlob(f"folder/mock_polygon_{i}/{i}.npy") for i in range(n_polygons)]
                ),
                TAR_BUCKET: iter([MockBlob(f"{dataset.name}_tars/mock_polygon.tar")]),
            }[bucket]

        mock_storage.Client().list_blobs.side_effect = side_effect

        status = dataset._create_webdataset_tar_from_small_polygons(
            polygons, polygon_ids, "mock_polygon"
        )

        # GCloud call count 1 tar check, 1 ee check, 1 npy check
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 1)
        self.assertEqual(mock_pipeline_1.run.call_count, 0)
        self.assertEqual(mock_pipeline_2.run.call_count, 0)
        self.assertTrue(f"Earth Engine files: {n_polygons}/{n_polygons}" in status)
        self.assertTrue(f"Numpy processing: {n_polygons}/{n_polygons}" in status)
        self.assertTrue("Complete tar: 1/1" in status)

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_small_polygons_no_npy_files(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"
        mock_pipeline_1.run.return_value = []
        mock_pipeline_2.run.return_value = []
        n_polygons = 100

        def side_effect(bucket, prefix=None):
            return {
                EE_BUCKET: iter([MockBlob(f"prefix/mock_{i}.tif") for i in range(n_polygons)]),
                NPY_BUCKET: iter([]),
                TAR_BUCKET: iter([]),
            }[bucket]

        mock_storage.Client().list_blobs.side_effect = side_effect

        polygons = [
            shapely.geometry.Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])
        ] * n_polygons
        polygon_ids = [f"mock_polygon_{i}" for i in range(n_polygons)]
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])
        status = dataset._create_webdataset_tar_from_small_polygons(
            polygons, polygon_ids, "mock_polygon"
        )

        # Gcloud call count 1 tar check, 1 ee check, 1 npy check
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 3)
        self.assertEqual(mock_pipeline_1.run.call_count, n_polygons)
        self.assertEqual(mock_pipeline_2.run.call_count, n_polygons)
        self.assertTrue(f"Earth Engine files: {n_polygons}/{n_polygons}" in status)
        self.assertTrue(f"Numpy processing: 0/{n_polygons}" in status)
        self.assertTrue("Complete tar: 0/1" in status)

    @patch("presto.dataops.dataset.storage")
    def test_create_webdataset_tar_small_polygons_some_npy_files(self, mock_storage):
        self.check_ee_prereq()
        mock_pipeline_1, mock_pipeline_2 = MagicMock(), MagicMock()
        mock_pipeline_1.name = "mock_pipeline_1"
        mock_pipeline_2.name = "mock_pipeline_2"
        mock_pipeline_1.run.return_value = ["mock_file_1.npy", "mock_file_2.npy"]
        mock_pipeline_2.run.return_value = []

        n_polygons = 100

        def side_effect(bucket, prefix=None):
            return {
                EE_BUCKET: iter([MockBlob(f"prefix/mock_{i}.tif") for i in range(n_polygons)]),
                NPY_BUCKET: iter(
                    [MockBlob(f"prefix/mock_{i}.tif") for i in range(n_polygons // 2)]
                ),
                TAR_BUCKET: iter([]),
            }[bucket]

        mock_storage.Client().list_blobs.side_effect = side_effect

        polygons = [
            shapely.geometry.Polygon([(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)])
        ] * n_polygons
        polygon_ids = [f"mock_polygon_{i}" for i in range(n_polygons)]
        dataset = Dataset([mock_pipeline_1, mock_pipeline_2])
        status = dataset._create_webdataset_tar_from_small_polygons(
            polygons, polygon_ids, "mock_polygon"
        )

        # Gcloud call count 1 tar check, 1 ee check, 1 npy check
        self.assertEqual(mock_storage.Client().list_blobs.call_count, 3)
        self.assertEqual(mock_pipeline_1.run.call_count, n_polygons)
        self.assertEqual(mock_pipeline_2.run.call_count, n_polygons)
        self.assertTrue(f"Earth Engine files: {n_polygons}/{n_polygons}" in status)
        self.assertTrue(f"Numpy processing: {n_polygons // 2}/{n_polygons}" in status)
        self.assertTrue("Complete tar: 0/1" in status)
