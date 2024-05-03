import numpy as np
from unittest import TestCase
from unittest.mock import MagicMock
import micro_manager


class MicroSimulation:
    def __init__(self, sim_id):
        self.very_important_value = 0

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        assert macro_data["macro-scalar-data"] == 1
        assert macro_data["macro-vector-data"].tolist() == [0, 1, 2]
        return {
            "micro-scalar-data": macro_data["macro-scalar-data"] + 1,
            "micro-vector-data": macro_data["macro-vector-data"] + 1,
        }


class TestFunctioncalls(TestCase):
    def setUp(self):
        self.fake_read_data_names = {
            "macro-scalar-data": False,
            "macro-vector-data": True,
        }
        self.fake_read_data = [
            {"macro-scalar-data": 1, "macro-vector-data": np.array([0, 1, 2])}
        ] * 4
        self.fake_write_data_names = {
            "micro-scalar-data": False,
            "micro-vector-data": True,
            "micro_sim_time": False,
        }
        self.fake_write_data = [
            {
                "micro-scalar-data": 1,
                "micro-vector-data": np.array([0, 1, 2]),
                "micro_sim_time": 0,
            }
        ] * 4

    def test_config(self):
        """
        Test if the functions in the SnapshotConfig class work.
        """
        config = micro_manager.SnapshotConfig(MagicMock(), "snapshot-config.json")

        self.assertEqual(config._parameter_file_name.split("/")[-1], "database.hdf5")
        self.assertEqual(config._micro_file_name, "test_snapshot_computation")
        self.assertDictEqual(config._read_data_names, self.fake_read_data_names)
        self.assertDictEqual(self.fake_write_data_names, config._write_data_names)


if __name__ == "__main__":
    import unittest

    unittest.main()
