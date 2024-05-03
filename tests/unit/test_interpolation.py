import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.interpolation import Interpolation


class TestInterpolation(TestCase):
    def test_local_interpolation(self):
        """
        Test if local interpolation works as expected.
        """
        coords = [[-2, 0, 0], [-1, 0, 0], [2, 0, 0]]
        inter_point = [1, 0, 0]
        vector_data = [[-2, -2, -2], [-1, -1, -1], [2, 2, 2]]
        scalar_data = [[-2], [-1], [2]]
        expected_vector_interpolation_output = [55 / 49, 55 / 49, 55 / 49]
        expected_scalar_interpolation_output = 55 / 49

        interpolation = Interpolation(MagicMock())
        interpolated_vector_data = interpolation.interpolate(
            coords, inter_point, vector_data
        )
        interpolated_scalar_data = interpolation.interpolate(
            coords, inter_point, scalar_data
        )
        self.assertTrue(
            np.allclose(interpolated_vector_data, expected_vector_interpolation_output)
        )
        self.assertAlmostEqual(
            interpolated_scalar_data, expected_scalar_interpolation_output
        )

    def test_nearest_neighbor(self):
        """
        Test if finding nearest neighbor works as expected if interpolation point
        itself is not part of neighbor coordinates.
        """
        neighbors = [[0, 2, 0], [0, 3, 0], [0, 0, 4], [-5, 0, 0], [0, 0, 0]]
        inter_coord = [0, 0, 0]
        expected_nearest_neighbor_index = [4, 0, 1]
        k = 3

        interpolation = Interpolation(MagicMock())
        nearest_neighbor_index = interpolation.get_nearest_neighbor_indices(
            neighbors, inter_coord, k
        )
        self.assertListEqual(
            nearest_neighbor_index.tolist(), expected_nearest_neighbor_index
        )
