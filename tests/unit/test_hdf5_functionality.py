from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import os
import h5py

from snapshot.read_write_hdf import ReadWriteHDF


class TestHDFFunctionalities(TestCase):
    def test_create_file(self):
        """
        Test if file creation works as expected
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hdf_files")
        file_name = "create_file.hdf5"
        entire_path = os.path.join(path, file_name)
        if os.path.isfile(entire_path):
            os.remove(entire_path)
        data_manager = ReadWriteHDF(MagicMock())
        complete_path = data_manager.create_file(path, file_name)
        self.assertTrue(os.path.isfile(complete_path))
        os.remove(complete_path)

    def test_collect_output_files(self):
        """
        Test if collection of output files form different ranks works as expected
        """
        dir_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "hdf_files"
        )
        files = ["output_1.hdf5", "output_2.hdf5"]
        if os.path.isfile(os.path.join(dir_name, "output.hdf5")):
            os.remove(os.path.join(dir_name, "output.hdf5"))
        output1 = h5py.File(os.path.join(dir_name, "output_1.hdf5"), "r")
        output2 = h5py.File(os.path.join(dir_name, "output_2.hdf5"), "r")
        data_manager = ReadWriteHDF(MagicMock())
        data_manager.collect_output_files(dir_name, files)
        output = h5py.File(os.path.join(dir_name, "output.hdf5"), "r")
        self.assertEqual(
            output["macro_scalar_data"][0], output1["macro_scalar_data"][0]
        )
        self.assertEqual(
            output["macro_scalar_data"][1], output2["macro_scalar_data"][0]
        )
        self.assertListEqual(
            output["macro_vector_data"][0].tolist(),
            output1["macro_vector_data"][0].tolist(),
        )
        self.assertListEqual(
            output["macro_vector_data"][1].tolist(),
            output2["macro_vector_data"][0].tolist(),
        )
        self.assertEqual(
            output["micro_scalar_data"][0], output1["micro_scalar_data"][0]
        )
        self.assertEqual(
            output["micro_scalar_data"][1], output2["micro_scalar_data"][0]
        )
        self.assertListEqual(
            output["micro_vector_data"][0].tolist(),
            output1["micro_vector_data"][0].tolist(),
        )
        self.assertListEqual(
            output["micro_vector_data"][1].tolist(),
            output2["micro_vector_data"][0].tolist(),
        )
        output1.close()
        output2.close()
        output.close()
        os.remove(os.path.join(dir_name, "output.hdf5"))

    def test_simulation_output_to_hdf(self):
        """
        Test if the write_sim_dict_to_hdf method correctly writes a dictionary or list of dictionaries to an hdf5 file.
        """
        file_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "hdf_files",
            "write_output.hdf5",
        )
        if os.path.isfile(file_name):
            os.remove(file_name)

        data_manager = ReadWriteHDF(MagicMock())
        macro_data = {
            "macro_vector_data": np.array([3, 1, 2]),
            "macro_scalar_data": 2,
        }
        micro_data = {
            "micro_vector_data": np.array([3, 2, 1]),
            "micro_scalar_data": 1,
        }

        expected_micro_vector_data = np.array([3, 2, 1])
        expected_micro_scalar_data = 1

        expected_macro_vector_data = np.array([3, 1, 2])
        expected_macro_scalar_data = 2

        data_manager.write_sim_output_to_hdf(file_name, macro_data, micro_data)

        test_file = h5py.File(file_name, "r")

        self.assertEqual(
            (test_file["micro_scalar_data"][0]), expected_micro_scalar_data
        )
        self.assertListEqual(
            (test_file["micro_vector_data"][0]).tolist(),
            (expected_micro_vector_data).tolist(),
        )
        self.assertEqual(
            (test_file["macro_scalar_data"][0]), expected_macro_scalar_data
        )
        self.assertListEqual(
            (test_file["macro_vector_data"][0]).tolist(),
            (expected_macro_vector_data).tolist(),
        )
        os.remove(file_name)

    def test_hdf_to_dict(self):
        """
        Test if read_parameter_hdf_to_dict method correctly reads parameter data from an hdf5 file.
        """
        expected_macro_scalar = 1
        expected_macro_vector = np.array([1, 2, 3])
        file_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "hdf_files", "output_1.hdf5"
        )
        read_data_names = {"macro_vector_data": True, "macro_scalar_data": False}
        data_manager = ReadWriteHDF(MagicMock())
        read = data_manager.read_hdf_to_dict(file_name, read_data_names)
        for i in range(len(read)):
            self.assertEqual(read[i]["macro_scalar_data"], expected_macro_scalar)
            self.assertListEqual(
                read[i]["macro_vector_data"].tolist(), expected_macro_vector.tolist()
            )
