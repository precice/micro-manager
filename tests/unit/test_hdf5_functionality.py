from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import os
import h5py

from micro_manager.snapshot.dataset import ReadWriteHDF


class TestHDFFunctionalities(TestCase):
    def test_create_file(self):
        """
        Test if file creation works as expected.
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hdf_files")
        file_name = "create_file.hdf5"
        entire_path = os.path.join(path, file_name)
        # Ensure output file does not exist
        if os.path.isfile(entire_path):
            os.remove(entire_path)
        data_manager = ReadWriteHDF(MagicMock())
        data_manager.create_file(entire_path)
        self.assertTrue(os.path.isfile(entire_path))
        with h5py.File(entire_path, "r") as f:
            self.assertEqual(f.attrs["status"], "writing")
        os.remove(entire_path)

    def test_collect_output_files(self):
        """
        Test if collection of output files works as expected.
        """
        dir_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "hdf_files"
        )
        files = ["output_1.hdf5", "output_2.hdf5"]
        # Create ouput files to merge
        input_data = [
            {
                "Macro-Vector-Data": np.array([1, 2, 3]),
                "Macro-Scalar-Data": 1,
                "Micro-Vector-Data": np.array([-1, -2, -3]),
                "Micro-Scalar-Data": -1,
            },
            {
                "Macro-Vector-Data": np.array([4, 5, 6]),
                "Macro-Scalar-Data": 2,
                "Micro-Vector-Data": np.array([-4, -5, -6]),
                "Micro-Scalar-Data": -2,
            },
        ]
        for data, file in zip(input_data, files):
            with h5py.File(os.path.join(dir_name, file), "w") as f:
                for key in data.keys():
                    current_data = np.asarray(data[key])
                    f.create_dataset(
                        key,
                        data=current_data,
                        shape=(1, *current_data.shape),
                        chunks=(1, *current_data.shape),
                    )
        # Ensure output file does not exist
        if os.path.isfile(os.path.join(dir_name, "snapshot_data.hdf5")):
            os.remove(os.path.join(dir_name, "snapshot_data.hdf5"))
        length = 2
        data_manager = ReadWriteHDF(MagicMock())
        data_manager.collect_output_files(dir_name, files, length)
        output = h5py.File(os.path.join(dir_name, "snapshot_data.hdf5"), "r")

        for i in range(length):
            self.assertEqual(
                output["Macro-Scalar-Data"][i], input_data[i]["Macro-Scalar-Data"]
            )

            self.assertListEqual(
                output["Macro-Vector-Data"][i].tolist(),
                input_data[i]["Macro-Vector-Data"].tolist(),
            )
            self.assertEqual(
                output["Micro-Scalar-Data"][i], input_data[i]["Micro-Scalar-Data"]
            )

            self.assertListEqual(
                output["Micro-Vector-Data"][i].tolist(),
                input_data[i]["Micro-Vector-Data"].tolist(),
            )

        output.close()
        os.remove(os.path.join(dir_name, "snapshot_data.hdf5"))

    def test_simulation_output_to_hdf(self):
        """
        Test if the write_output_to_hdf method correctly writes a dictionary to an HDF5 file.
        """
        file_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "hdf_files",
            "write_output.hdf5",
        )
        if os.path.isfile(file_name):
            os.remove(file_name)

        # Create artificial output data
        macro_data = {
            "Macro-Vector-Data": np.array([3, 1, 2]),
            "Macro-Scalar-Data": 2,
        }
        micro_data = {
            "Micro-Vector-Data": np.array([3, 2, 1]),
            "Micro-Scalar-Data": 1,
        }

        expected_micro_vector_data = np.array([3, 2, 1])
        expected_micro_scalar_data = 1

        expected_macro_vector_data = np.array([3, 1, 2])
        expected_macro_scalar_data = 2

        data_manager = ReadWriteHDF(MagicMock())
        for i in range(2):
            data_manager.write_output_to_hdf(file_name, macro_data, micro_data, i, 2)

        test_file = h5py.File(file_name, "r")

        self.assertEqual(
            (test_file["Micro-Scalar-Data"][0]), expected_micro_scalar_data
        )
        self.assertListEqual(
            (test_file["Micro-Vector-Data"][0]).tolist(),
            (expected_micro_vector_data).tolist(),
        )
        self.assertEqual(
            (test_file["Macro-Scalar-Data"][0]), expected_macro_scalar_data
        )
        self.assertListEqual(
            (test_file["Macro-Vector-Data"][0]).tolist(),
            (expected_macro_vector_data).tolist(),
        )
        os.remove(file_name)

    def test_hdf_to_dict(self):
        """
        Test if read__hdf method correctly reads parameter data from an HDF5 file.
        """
        expected_macro_scalar = 1
        expected_macro_vector = np.array([0, 1, 2])
        file_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "hdf_files",
            "test_parameter.hdf5",
        )
        read_data_names = {"Macro-Vector-Data": True, "Macro-Scalar-Data": False}
        data_manager = ReadWriteHDF(MagicMock())
        read = data_manager.read_hdf(file_name, read_data_names, 0, -1)
        for i in range(len(read)):
            self.assertEqual(read[i]["Macro-Scalar-Data"], expected_macro_scalar)
            self.assertListEqual(
                read[i]["Macro-Vector-Data"].tolist(), expected_macro_vector.tolist()
            )

    def test_get_parameter_space_length(self):
        """
        Test if reading the length of the parameter space works as expected.
        """
        file_name = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "hdf_files",
            "test_parameter.hdf5",
        )
        data_manager = ReadWriteHDF(MagicMock())
        self.assertEqual(data_manager.get_parameter_space_size(file_name), 1)
