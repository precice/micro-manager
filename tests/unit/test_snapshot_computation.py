import numpy as np
import os
from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.snapshot.snapshot import MicroManagerSnapshot
from micro_manager.micro_simulation import create_simulation_class
from micro_manager.config import Config


class MicroSimulation:
    def __init__(self, sim_id):
        self.very_important_value = 0

    def solve(self, macro_data, dt):
        assert macro_data["Macro-Scalar-Data"] == 1
        assert macro_data["Macro-Vector-Data"].tolist() == [0, 1, 2]
        return {
            "Micro-Scalar-Data": macro_data["Macro-Scalar-Data"] + 1,
            "Micro-Vector-Data": macro_data["Macro-Vector-Data"] + 1,
        }

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def get_global_id(self):
        pass


class TestFunctionCalls(TestCase):
    def setUp(self):
        self.fake_read_data_names = ["Macro-Scalar-Data", "Macro-Vector-Data"]
        self.fake_read_data = {
            "Macro-Scalar-Data": 1,
            "Macro-Vector-Data": np.array([0, 1, 2]),
        }

        self.fake_write_data_names = ["Micro-Scalar-Data", "Micro-Vector-Data"]
        self.fake_write_data = [
            {
                "Micro-Scalar-Data": 1,
                "Micro-Vector-Data": np.array([0, 1, 2]),
            }
        ] * 4

    def test_snapshot_constructor(self):
        """
        Test if the constructor of the MicroManagerSnapshot class passes correct values to member variables.
        """
        snapshot_object = MicroManagerSnapshot("snapshot-config.json")

        self.assertListEqual(
            snapshot_object._read_data_names, self.fake_read_data_names
        )
        self.assertListEqual(
            snapshot_object._write_data_names, self.fake_write_data_names
        )
        self.assertEqual(
            snapshot_object._post_processing_file_name, "snapshot_post_processing"
        )

    def test_initialize(self):
        """
        Test if the initialize function of the MicroManagerSnapshot class works as expected.
        """
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
        file_name = "snapshot_data.hdf5"
        complete_path = os.path.join(path, file_name)
        if os.path.isfile(complete_path):
            os.remove(complete_path)

        snapshot_object = MicroManagerSnapshot("snapshot-config.json")

        snapshot_object.initialize()
        self.assertEqual(snapshot_object._global_number_of_sims, 1)
        self.assertListEqual(
            snapshot_object._read_data_names, self.fake_read_data_names
        )
        self.assertListEqual(
            snapshot_object._write_data_names, self.fake_write_data_names
        )
        self.assertTrue(os.path.isfile(complete_path))
        os.remove(complete_path)
        os.rmdir(path)

    def test_solve_micro_sims(self):
        """
        Test if the internal function _solve_micro_simulations works as expected.
        """
        snapshot_object = MicroManagerSnapshot("snapshot-config.json")

        snapshot_object._micro_problem = MicroSimulation

        snapshot_object._micro_sims = create_simulation_class(
            MagicMock(),
            snapshot_object._micro_problem,
            None,
            1,
            None,
        )(0)

        micro_sim_output = snapshot_object._solve_micro_simulation(self.fake_read_data)
        self.assertEqual(micro_sim_output["Micro-Scalar-Data"], 2)
        self.assertListEqual(
            micro_sim_output["Micro-Vector-Data"].tolist(),
            (self.fake_read_data["Macro-Vector-Data"] + 1).tolist(),
        )

    def test_solve(self):
        """
        Test if the solve function of the MicroManagerSnapshot class works as expected.
        """

        snapshot_object = MicroManagerSnapshot("snapshot-config.json")
        snapshot_object._data_storage = MagicMock()

        # Replace initialize call
        snapshot_object._file_name = "output.hdf5"
        snapshot_object._output_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "output",
            snapshot_object._file_name,
        )

        snapshot_object._output_subdirectory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "output"
        )
        snapshot_object._macro_parameters = [self.fake_read_data]
        snapshot_object._local_number_of_sims = 1
        snapshot_object._global_number_of_sims = 1
        snapshot_object._global_ids_of_local_sims = [0]
        snapshot_object._micro_problem = MicroSimulation

        snapshot_object.solve()

        self.assertEqual(
            snapshot_object._micro_sims.very_important_value, 0
        )  # test inheritance

    def test_config(self):
        """
        Test if the functions in the SnapshotConfig class work.
        """
        config = Config("snapshot-config.json")
        config.set_logger(MagicMock())
        config.read_json_snapshot()

        self.assertEqual(
            config._parameter_file_name.split("/")[-1], "test_parameter.hdf5"
        )
        self.assertEqual(config._micro_file_name, "test_snapshot_computation")
        self.assertListEqual(config._read_data_names, self.fake_read_data_names)
        self.assertListEqual(config._write_data_names, self.fake_write_data_names)
        self.assertEqual(config._postprocessing_file_name, "snapshot_post_processing")
        self.assertTrue(config._initialize_once)


if __name__ == "__main__":
    import unittest

    unittest.main()
