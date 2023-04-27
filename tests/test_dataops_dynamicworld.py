from unittest import TestCase

import numpy as np

from presto.dataops.pipelines.dynamicworld import pad_array


class TestRealDataset(TestCase):
    def test_pad_array_1d(self):

        array = np.array([0, 1, 2, 3, 4, 5])
        output = pad_array(array, 8)
        self.assertTrue(np.equal(output, np.array([0, 1, 2, 3, 4, 5, 0, 0])).all())

        truncated_output = pad_array(array, 3)
        self.assertTrue(np.equal(truncated_output, np.array([0, 1, 2])).all())

    def test_pad_array_2d(self):

        array = np.array([[0, 1, 2, 3, 4, 5], [-1, 1, 2, 3, 4, 5]])
        output = pad_array(array, 8)
        self.assertTrue(
            np.equal(
                output, np.array([[0, 1, 2, 3, 4, 5, 0, 0], [-1, 1, 2, 3, 4, 5, -1, -1]])
            ).all()
        )

        truncated_output = pad_array(array, 3)
        self.assertTrue(np.equal(truncated_output, np.array([[0, 1, 2], [-1, 1, 2]])).all())
