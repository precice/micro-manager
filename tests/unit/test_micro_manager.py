from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

import micro_manager
from micro_manager.simulation_container import SimulationContainer


class MicroSimulation:
    def __init__(self, sim_id):
        self.very_important_value = 0

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        assert macro_data["Macro-Scalar-Data"] == 1
        assert macro_data["Macro-Vector-Data"].tolist() == [0, 1, 2]
        return {
            "Micro-Scalar-Data": macro_data["Macro-Scalar-Data"] + 1,
            "Micro-Vector-Data": macro_data["Macro-Vector-Data"] + 1,
        }

    def get_global_id(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass


class TestFunctionCalls(TestCase):
    def setUp(self):
        self.fake_read_data_names = ["Macro-Scalar-Data", "Macro-Vector-Data"]
        self.fake_read_data = [
            {"Macro-Scalar-Data": 1, "Macro-Vector-Data": np.array([0, 1, 2])}
        ] * 4
        self.fake_write_data_names = [
            "Micro-Scalar-Data",
            "Micro-Vector-Data",
            "Active-State",
            "Active-Steps",
        ]
        self.fake_write_data = [
            {
                "Micro-Scalar-Data": 1,
                "Micro-Vector-Data": np.array([0, 1, 2]),
                "Active-State": 0,
                "Active-Steps": 0,
            }
        ] * 4
        self.macro_bounds = [0.0, 25.0, 0.0, 25.0, 0.0, 25.0]

    def test_micromanager_constructor(self):
        """
        Test if the constructor of the MicroManager class passes correct values to member variables.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")

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
        self.assertEqual(manager._sim_container.global_num_sims, 4)
        self.assertListEqual(manager._mesh_vertex_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(len(manager._sim_container), 4)
        self.assertEqual(
            manager._sim_container[0].very_important_value, 0
        )  # test inheritance
        self.assertListEqual(manager._read_data_names, self.fake_read_data_names)
        self.assertListEqual(self.fake_write_data_names, manager._write_data_names)

    def test_read_write_data_from_precice(self):
        """
        Test if the internal functions _read_data_from_precice and _write_data_to_precice work as expected.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")
        container = SimulationContainer()
        container.initialize(4, 4, [0, 1, 2, 3], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        manager._sim_container = container
        manager._mesh_vertex_ids = container.local_coords

        manager._write_data_to_precice(self.fake_write_data)
        read_data = manager._read_data_from_precice(1.0)

        for data, fake_data in zip(read_data, self.fake_read_data):
            self.assertEqual(data["Macro-Scalar-Data"], 1)
            self.assertListEqual(
                data["Macro-Vector-Data"].tolist(),
                fake_data["Macro-Vector-Data"].tolist(),
            )

    def test_solve_micro_sims(self):
        """
        Test if the internal function _solve_micro_simulations works as expected.
        """
        manager = micro_manager.MicroManagerCoupling("micro-manager-config.json")
        manager.initialize()

        # manually initialize container
        container = SimulationContainer()
        container.initialize(4, 4, [0, 1, 2, 3], np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        manager._sim_container = container
        for lid in container.range_lid:
            container[lid] = MicroSimulation(lid)
        manager._micro_sims_active_steps = np.zeros(4, dtype=np.int32)

        micro_sims_output = manager._solve_micro_simulations(self.fake_read_data, 1.0)

        for data, fake_data in zip(micro_sims_output, self.fake_write_data):
            self.assertEqual(data["Micro-Scalar-Data"], 2)

            self.assertListEqual(
                data["Micro-Vector-Data"].tolist(),
                (fake_data["Micro-Vector-Data"] + 1).tolist(),
            )

    def test_config(self):
        """
        Test if the functions in the Config class work.
        """
        config = micro_manager.Config("micro-manager-config.json")
        config.set_logger(MagicMock())
        config.read_json_micro_manager()
        self.assertEqual(
            config.precice_config_file_name().split("/")[-1], "dummy-config.xml"
        )
        self.assertEqual(config.micro_file_name(), "test_micro_manager")
        self.assertEqual(config.macro_mesh_name(), "Macro-Mesh")
        self.assertEqual(config.micro_output_n(), 10)
        self.assertListEqual(config.read_data_names(), self.fake_read_data_names)
        self.assertListEqual(self.fake_write_data_names, config.write_data_names())
        self.assertEqual(config.enable_adaptivity(), True)
        self.assertListEqual(config.data_for_adaptivity(), self.fake_read_data_names)
        self.assertEqual(config.adaptivity_type(), "local")
        self.assertEqual(config.adaptivity_history_param(), 0.5)
        self.assertEqual(config.adaptivity_coarsening_constant(), 0.3)
        self.assertEqual(config.adaptivity_refining_constant(), 0.4)
        self.assertEqual(config.enable_adaptivity_each_implicit_iteration(), False)


if __name__ == "__main__":
    import unittest

    unittest.main()
