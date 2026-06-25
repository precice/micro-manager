from unittest import TestCase

import numpy as np

import micro_manager
from micro_manager.adaptivity.adaptivity import NoOpAdaptivity
from micro_manager.simulation_container import SimulationContainer
from micro_manager.interpolation import Interpolator


class MicroSimulation:
    def __init__(self, sim_id):
        self.sim_id = sim_id

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        if self.sim_id == 2:
            raise Exception("Simulation experienced a crash")

        return {
            "Micro-Vector-Data": macro_data["Macro-Vector-Data"],
            "Micro-Scalar-Data": macro_data["Macro-Scalar-Data"],
        }

    def get_state(self):
        return None

    def set_state(self, state):
        pass

    def get_global_id(self):
        return self.sim_id


class TestSimulationCrashHandling(TestCase):
    def test_crash_handling(self):
        """
        Test if the Micro Manager catches a simulation crash and handles it adequately.
        A crash if caught by interpolation within _solve_micro_simulations.
        Note: running this test requires the sci-kit learn package to be installed.
        """

        macro_data = []
        for i in [-2, -1, 1, 2]:
            macro_data.append(
                {"Macro-Vector-Data": np.array([i, i, i]), "Macro-Scalar-Data": [i]}
            )
        expected_crash_vector_data = np.array([55 / 49, 55 / 49, 55 / 49])
        expected_crash_scalar_data = 55 / 49

        manager = micro_manager.MicroManagerCoupling("micro-manager-config_crash.json")
        manager.initialize()

        manager._number_of_nearest_neighbors = 3  # reduce number of neighbors to 3
        container = SimulationContainer(manager._mpi)
        container.initialize(
            4, 4, [0, 1, 2, 3], np.array([[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]])
        )
        manager._coupling._mesh_vertex_coords = container.local_coords
        # make sure adaptivity is off overriding config
        manager._adaptivity_controller = NoOpAdaptivity(container)
        for lid in container.range_lid:
            container[lid] = MicroSimulation(lid)
        manager._sim_container = container

        micro_sims_output = manager._solve_micro_simulations(macro_data, 1.0)

        # Crashed simulation has interpolated value
        data_crashed = micro_sims_output[2]
        self.assertEqual(data_crashed["Micro-Scalar-Data"], expected_crash_scalar_data)
        self.assertListEqual(
            data_crashed["Micro-Vector-Data"].tolist(),
            expected_crash_vector_data.tolist(),
        )
        # Non-crashed simulations should remain constant
        data_normal = micro_sims_output[1]
        self.assertEqual(
            data_normal["Micro-Scalar-Data"], macro_data[1]["Macro-Scalar-Data"]
        )
        self.assertListEqual(
            data_normal["Micro-Vector-Data"].tolist(),
            macro_data[1]["Macro-Vector-Data"].tolist(),
        )

    def test_crash_handling_with_adaptivity(self):
        """
        Test if the micro manager catches a simulation crash and handles it adequately with adaptivity.
        A crash if caught by interpolation within _solve_micro_simulations_with_adaptivity.
        Note: running this test requires the sci-kit learn package to be installed.
        """

        macro_data = []
        for i in [-2, -1, 1, 2, 10]:
            macro_data.append(
                {"Macro-Vector-Data": np.array([i, i, i]), "Macro-Scalar-Data": [i]}
            )
        expected_crash_vector_data = np.array([55 / 49, 55 / 49, 55 / 49])
        expected_crash_scalar_data = 55 / 49

        manager = micro_manager.MicroManagerCoupling("micro-manager-config_crash.json")
        manager.initialize()

        container = SimulationContainer(manager._mpi)
        container.initialize(
            5,
            5,
            [0, 1, 2, 3, 4],
            np.array([[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]]),
        )
        Interpolator.get_config("crash")["k"] = 3  # reduce number of neighbors to 3
        manager._crash_handler.reset()
        manager._coupling._mesh_vertex_coords = container.local_coords
        for lid in container.range_lid:
            container[lid] = MicroSimulation(lid)
        manager._sim_container = container

        manager._adaptivity_controller._sim_container = container
        manager._adaptivity_controller._similarity_dists = np.zeros(shape=(5, 5))
        manager._adaptivity_controller._is_sim_active = np.array(
            [True, True, True, True, False]
        )
        manager._adaptivity_controller._sim_is_associated_to = {4: 2}
        manager._adaptivity_controller._sim_active_steps = {
            gid: 0 for gid in container.local_gids
        }
        manager._adaptivity_controller.update_buffers(alloc=True)

        micro_sims_output = manager._solve_micro_simulations(macro_data, 1.0)

        # Crashed simulation has interpolated value
        data_crashed = micro_sims_output[2]
        self.assertEqual(data_crashed["Micro-Scalar-Data"], expected_crash_scalar_data)
        self.assertListEqual(
            data_crashed["Micro-Vector-Data"].tolist(),
            expected_crash_vector_data.tolist(),
        )

        # Inactive simulation that is associated with crashed simulation has same value
        data_associated = micro_sims_output[4]
        self.assertEqual(
            data_associated["Micro-Scalar-Data"], expected_crash_scalar_data
        )
        self.assertListEqual(
            data_associated["Micro-Vector-Data"].tolist(),
            expected_crash_vector_data.tolist(),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
