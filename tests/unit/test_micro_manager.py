from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

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
        self.fake_read_data_names = ["macro-scalar-data", "macro-vector-data"]
        self.fake_read_data = [
            {"macro-scalar-data": 1, "macro-vector-data": np.array([0, 1, 2])}
        ] * 4
        self.fake_write_data_names = [
            "micro-scalar-data",
            "micro-vector-data",
            "solve_cpu_time",
            "active_state",
            "active_steps",
        ]
        self.fake_write_data = [
            {
                "micro-scalar-data": 1,
                "micro-vector-data": np.array([0, 1, 2]),
                "solve_cpu_time": 0,
                "active_state": 0,
                "active_steps": 0,
            }
        ] * 4
        self.macro_bounds = [0.0, 25.0, 0.0, 25.0, 0.0, 25.0]

    def test_micromanager_constructor(self):
        """
        Test if the constructor of the MicroManager class passes correct values to member variables.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")

        self.assertListEqual(manager._macro_bounds, self.macro_bounds)
        self.assertListEqual(manager._read_data_names, self.fake_read_data_names)
        self.assertListEqual(self.fake_write_data_names, manager._write_data_names)
        self.assertEqual(manager._micro_n_out, 10)

    def test_initialization(self):
        """
        Test if the initialize function of the MicroManager class initializes member variables to correct values
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")
        manager.initialize()

        self.assertEqual(manager._micro_dt, 0.1)  # from Interface.initialize
        self.assertEqual(manager._global_number_of_sims, 4)
        self.assertListEqual(manager._macro_bounds, self.macro_bounds)
        self.assertListEqual(manager._mesh_vertex_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(len(manager._micro_sims), 4)
        self.assertEqual(
            manager._micro_sims[0].very_important_value, 0
        )  # test inheritance
        self.assertListEqual(manager._read_data_names, self.fake_read_data_names)
        self.assertListEqual(self.fake_write_data_names, manager._write_data_names)

    def test_read_write_data_from_precice(self):
        """
        Test if the internal functions _read_data_from_precice and _write_data_to_precice work as expected.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")

        manager._global_ids_of_local_sims = 4
        manager._mesh_vertex_ids = np.array([0, 1, 2, 3])

        manager._write_data_to_precice(self.fake_write_data)
        read_data = manager._read_data_from_precice(1.0)

        for data, fake_data in zip(read_data, self.fake_read_data):
            self.assertEqual(data["macro-scalar-data"], 1)
            self.assertListEqual(
                data["macro-vector-data"].tolist(),
                fake_data["macro-vector-data"].tolist(),
            )

    def test_solve_micro_sims(self):
        """
        Test if the internal function _solve_micro_simulations works as expected.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")
        manager.initialize()

        manager._local_number_of_sims = 4
        manager._micro_sims = [MicroSimulation(i) for i in range(4)]
        manager._micro_sims_active_steps = np.zeros(4, dtype=np.int32)

        micro_sims_output = manager._solve_micro_simulations(self.fake_read_data, 1.0)

        for data, fake_data in zip(micro_sims_output, self.fake_write_data):
            self.assertEqual(data["micro-scalar-data"], 2)

            self.assertListEqual(
                data["micro-vector-data"].tolist(),
                (fake_data["micro-vector-data"] + 1).tolist(),
            )

    def test_config(self):
        """
        Test if the functions in the Config class work.
        """
        config = micro_manager.Config("micro-manager-config.json")
        config.set_logger(MagicMock())
        config.read_json_micro_manager()
        self.assertEqual(
            config._precice_config_file_name.split("/")[-1], "dummy-config.xml"
        )
        self.assertEqual(config._micro_file_name, "test_micro_manager")
        self.assertEqual(config._macro_mesh_name, "dummy-macro-mesh")
        self.assertEqual(config._micro_output_n, 10)
        self.assertListEqual(config._read_data_names, self.fake_read_data_names)
        self.assertListEqual(self.fake_write_data_names, config._write_data_names)
        self.assertEqual(config._adaptivity, True)
        self.assertListEqual(config._data_for_adaptivity, self.fake_read_data_names)
        self.assertEqual(config._adaptivity_type, "local")
        self.assertEqual(config._adaptivity_history_param, 0.5)
        self.assertEqual(config._adaptivity_coarsening_constant, 0.3)
        self.assertEqual(config._adaptivity_refining_constant, 0.4)
        self.assertEqual(config._adaptivity_every_implicit_iteration, False)


if __name__ == "__main__":
    import unittest

    unittest.main()
