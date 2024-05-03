from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import os
import h5py

from micro_manager.read_write_data import HDFParameterfile


class TestHDFFunctionalities(TestCase):
    def test_create_file(self):
        """
        Test if file creation works as expected
        """
        dir_name = os.path.dirname(os.path.realpath(__file__))
        file_name = dir_name + "/database_0.hdf5"
        if os.path.isfile(file_name):
            os.remove(file_name)
        data_manager = HDFParameterfile(MagicMock(), 0, 1)
        data_manager.create_file(dir_name)
        self.assertTrue(os.path.isfile(file_name))
        os.remove(file_name)

    def test_collect_parameter_files(self):
        pass

    def test_dict_to_hdf(self):
        """
        Test if the write_dict_to_hdf method correctly writes a dictionary or list of dictionaries to an hdf5 file.
        """
        file_name = (
            os.path.dirname(os.path.realpath(__file__)) + "/test_dict_to_hdf.hdf5"
        )

        data_manager = HDFParameterfile(MagicMock(), 0, 1)
        input_data = [
            {
                "macro_vector_data": np.array([0, 1, 2]),
                "macro_scalar_data": 1,
                "micro_vector_data": np.array([3, 1, 2]),
                "micro_scalar_data": 2,
            }
        ] * 2
        expected_macro_vector_data = [np.array([0, 1, 2]), np.array([0, 1, 2])]
        expected_macro_scalar_data = [1, 1]
        expected_micro_vector_data = [np.array([3, 1, 2]), np.array([3, 1, 2])]
        expected_micro_scalar_data = [2, 2]

        data_manager.write_dict_to_hdf(file_name, input_data, "w")

        test_file = h5py.File(file_name, "r")

        for i in range(2):
            self.assertEqual(
                (test_file["macro_scalar_data"][i]), (expected_macro_scalar_data[i])
            )
            self.assertListEqual(
                (test_file["macro_vector_data"][i]).tolist(),
                (expected_macro_vector_data[i]).tolist(),
            )
            self.assertEqual(
                (test_file["micro_scalar_data"][i]), (expected_micro_scalar_data[i])
            )
            self.assertListEqual(
                (test_file["micro_vector_data"][i]).tolist(),
                (expected_micro_vector_data[i]).tolist(),
            )

    def test_sim_output_to_hdf(self):
        pass

    def test_hdf_to_parameter_dict(self):
        """
        Test if read_parameter_hdf_to_dict method correctly reads parameter data from an hdf5 file.
        """
        expected_macro_scalar = 1
        expected_macro_vector = np.array([0, 1, 2])
        file_name = os.path.dirname(os.path.realpath(__file__)) + "/read_test.hdf5"
        read_data_names = {"macro_vector_data": True, "macro_scalar_data": False}
        data_manager = HDFParameterfile(MagicMock(), 0, 1)
        [
            {"macro_vector_data": np.array([0, 1, 2]), "macro_scalar_data": 1},
            {"macro_vector_data": np.array([0, 1, 2]), "macro_scalar_data": 1},
        ]
        read = data_manager.read_parameter_hdf_to_dict(file_name, read_data_names)
        for i in range(2):
            self.assertEqual(read[i]["macro_scalar_data"], expected_macro_scalar)
            self.assertListEqual(
                read[i]["macro_vector_data"].tolist(), expected_macro_vector.tolist()
            )

    def test_micro_hdf_to_dict(self):
        pass

    def test_hdf_to_sim_output(self):
        pass

    def test_all_hdf_to_dict(self):
        pass
