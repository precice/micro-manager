from unittest import TestCase

import numpy as np

import micro_manager


class MicroSimulation:
    def __init__(self, sim_id):
        self.sim_id = sim_id

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        if self.sim_id == 2:
            raise Exception("Simulation experienced a crash")

        return {
            "micro-vector-data": macro_data["macro-vector-data"],
            "micro-scalar-data": macro_data["macro-scalar-data"],
        }


class TestSimulationCrashHandling(TestCase):
    def test_crash_handling(self):
        """
        Test if the Micro Manager catches a simulation crash and handles it adequately.
        A crash if caught by interpolation within _solve_micro_simulations.
        """

        macro_data = []
        for i in [-2, -1, 1, 2]:
            macro_data.append(
                {"macro-vector-data": np.array([i, i, i]), "macro-scalar-data": [i]}
            )
        expected_crash_vector_data = np.array([55 / 49, 55 / 49, 55 / 49])
        expected_crash_scalar_data = 55 / 49

        manager = micro_manager.MicroManagerCoupling("micro-manager-config_crash.json")
        manager.initialize()

        manager._number_of_nearest_neighbors = 3  # reduce number of neighbors to 3
        manager._local_number_of_sims = 4
        manager._has_sim_crashed = [False] * 4
        manager._mesh_vertex_coords = np.array(
            [[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0]]
        )
        manager._is_adaptivity_on = (
            False  # make sure adaptivity is off overriding config
        )
        manager._micro_sims = [MicroSimulation(i) for i in range(4)]

        micro_sims_output = manager._solve_micro_simulations(macro_data)

        # Crashed simulation has interpolated value
        data_crashed = micro_sims_output[2]
        self.assertEqual(data_crashed["micro-scalar-data"], expected_crash_scalar_data)
        self.assertListEqual(
            data_crashed["micro-vector-data"].tolist(),
            expected_crash_vector_data.tolist(),
        )
        # Non-crashed simulations should remain constant
        data_normal = micro_sims_output[1]
        self.assertEqual(
            data_normal["micro-scalar-data"], macro_data[1]["macro-scalar-data"]
        )
        self.assertListEqual(
            data_normal["micro-vector-data"].tolist(),
            macro_data[1]["macro-vector-data"].tolist(),
        )

    def test_crash_handling_with_adaptivity(self):
        """
        Test if the micro manager catches a simulation crash and handles it adequately with adaptivity.
        A crash if caught by interpolation within _solve_micro_simulations_with_adaptivity.
        """

        macro_data = []
        for i in [-2, -1, 1, 2, 10]:
            macro_data.append(
                {"macro-vector-data": np.array([i, i, i]), "macro-scalar-data": [i]}
            )
        expected_crash_vector_data = np.array([55 / 49, 55 / 49, 55 / 49])
        expected_crash_scalar_data = 55 / 49

        manager = micro_manager.MicroManagerCoupling("micro-manager-config_crash.json")
        manager.initialize()

        manager._number_of_nearest_neighbors = 3  # reduce number of neighbors to 3
        manager._local_number_of_sims = 5
        manager._micro_sims_active_steps = np.zeros(5, dtype=np.int32)
        manager._has_sim_crashed = [False] * 5
        manager._mesh_vertex_coords = np.array(
            [[-2, 0, 0], [-1, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]]
        )
        manager._micro_sims = [MicroSimulation(i) for i in range(5)]

        is_sim_active = np.array([True, True, True, True, False])
        sim_is_associated_to = np.array([-2, -2, -2, -2, 2])
        micro_sims_output = manager._solve_micro_simulations_with_adaptivity(
            macro_data, is_sim_active, sim_is_associated_to
        )

        # Crashed simulation has interpolated value
        data_crashed = micro_sims_output[2]
        self.assertEqual(data_crashed["micro-scalar-data"], expected_crash_scalar_data)
        self.assertListEqual(
            data_crashed["micro-vector-data"].tolist(),
            expected_crash_vector_data.tolist(),
        )

        # Inactive simulation that is associated with crashed simulation has same value
        data_associated = micro_sims_output[4]
        self.assertEqual(
            data_associated["micro-scalar-data"], expected_crash_scalar_data
        )
        self.assertListEqual(
            data_associated["micro-vector-data"].tolist(),
            expected_crash_vector_data.tolist(),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
