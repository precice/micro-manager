import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.interpolation import interpolate


class TestInterpolation(TestCase):

    def test_local_interpolation(self):
        """
        Test if local interpolation works as expected
        """
        micro_output = [{"vector data": [1,1,1], "scalar data": [1]}] * 4
        micro_output.append({"vector data": [3,3,3], "scalar data": [0]})
        micro_output.append({"vector data": [0,0,0], "scalar data": [3]})
        coords_known = [[2,0,0],[-2,0,0], [0,2,0], [0,-2,0], [0,0,2], [0,0,-1]]
        unknown_coord = [0,0,0]
        expected_interpolation_result = {"vector data": np.array([1,1,1]), "scalar data": np.array([2])}
        interpolation_result = interpolate(MagicMock(), coords_known, unknown_coord , micro_output)
        for key in interpolation_result.keys():
            self.assertTrue(np.allclose(interpolation_result[key], expected_interpolation_result[key]))
        
    def test_local_extrapolation(self):
        micro_output = [{"vector data": [3,2,1], "scalar data": [0]},
                        {"vector data": [3,2,1], "scalar data": [0]},
                        {"vector data": [3,2,1], "scalar data": [0]},
                        {"vector data": [4,3,2], "scalar data": [4]},
                        {"vector data": [3,2,1], "scalar data": [2]}]
        coords_known = [[0,-2,0],[0,-4,0],[0,0,2], [2,0,0],[1,0,0]]
        unknown_coord = [0,0,0]
        expected_interpolation_result = {"vector data": [3,2,1], "scalar data": [2]}
        interpolation_result = interpolate(MagicMock(), coords_known, unknown_coord , micro_output)
        self.assertDictEqual(interpolation_result, expected_interpolation_result)