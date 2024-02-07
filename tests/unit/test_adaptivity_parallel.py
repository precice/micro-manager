from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.adaptivity.global_adaptivity import GlobalAdaptivityCalculator
import numpy as np
from mpi4py import MPI


class TestGlobalAdaptivity(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    def test_update_inactive_sims_global_adaptivity(self):
        """
        Test functionality to update inactive simulations in a particular setting, for a global adaptivity setting.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._rank == 0:
            global_ids = [0, 1, 2]
        elif self._rank == 1:
            global_ids = [3, 4]

        is_sim_active = np.array([False, False, True, True, False])
        sim_is_associated_to = [3, 3, -2, -2, 2]
        expected_is_sim_active = np.array([True, False, True, True, True])
        expected_sim_is_associated_to = [-2, 3, -2, -2, -2]

        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        adaptivity_controller = GlobalAdaptivityCalculator(
            configurator,
            MagicMock(),
            5,
            global_ids,
            rank=self._rank,
            comm=self._comm)

        # Force the activation of sim #0 and #4
        def check_for_activation(i, sim_dists, active):
            if i == 0 or i == 4:
                return True
            else:
                return False

        adaptivity_controller._check_for_activation = check_for_activation

        class MicroSimulation():
            def __init__(self, global_id) -> None:
                self._global_id = global_id
                self._state = [global_id] * 3

            def get_global_id(self):
                return self._global_id

            def set_state(self, state):
                self._state = state

            def get_state(self):
                return self._state.copy()

        dummy_micro_sims = []
        for i in global_ids:
            dummy_micro_sims.append(MicroSimulation(i))

        is_sim_active, sim_is_associated_to = adaptivity_controller._update_inactive_sims(
            np.array([0]), is_sim_active, sim_is_associated_to, dummy_micro_sims)

        self.assertTrue(np.array_equal(expected_is_sim_active, is_sim_active))
        self.assertTrue(np.array_equal(expected_sim_is_associated_to, sim_is_associated_to))

        if self._rank == 0:
            self.assertTrue(np.array_equal([3, 3, 3], dummy_micro_sims[0].get_state()))
        elif self._rank == 1:
            self.assertTrue(np.array_equal([2, 2, 2], dummy_micro_sims[1].get_state()))

    def test_update_all_active_sims_global_adaptivity(self):
        """
        Test functionality to calculate adaptivity when all simulations are active, for a global adaptivity setting.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._rank == 0:
            global_ids = [0, 1, 2]
            data_for_adaptivity = {"data1": [1.0, 1.0, 1.0], "data2": [13.0, 13.0, 13.0]}
        elif self._rank == 1:
            global_ids = [3, 4]
            data_for_adaptivity = {"data1": [1.0, 1.0], "data2": [13.0, 13.0]}

        similarity_dists = np.zeros((5, 5))
        is_sim_active = np.array([True, True, True, True, True])
        sim_is_associated_to = [-2, -2, -2, -2, -2]
        expected_is_sim_active = np.array([False, False, False, False, True])
        expected_sim_is_associated_to = [4, 4, 4, 4, -2]

        configurator = MagicMock()
        configurator.get_adaptivity_hist_param = MagicMock(return_value=0.1)
        configurator.get_adaptivity_refining_const = MagicMock(return_value=0.05)
        configurator.get_adaptivity_coarsening_const = MagicMock(return_value=0.2)
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L2rel")
        adaptivity_controller = GlobalAdaptivityCalculator(
            configurator,
            MagicMock(),
            5,
            global_ids,
            rank=self._rank,
            comm=self._comm)

        adaptivity_controller._adaptivity_data_names = {"data1": "scalar", "data2": "scalar"}

        class MicroSimulation():
            def __init__(self, global_id) -> None:
                self._global_id = global_id
                self._state = [global_id] * 3

            def get_global_id(self):
                return self._global_id

            def set_state(self, state):
                self._state = state

            def get_state(self):
                return self._state.copy()

        dummy_micro_sims = []
        for i in global_ids:
            dummy_micro_sims.append(MicroSimulation(i))

        _, is_sim_active, sim_is_associated_to = adaptivity_controller.compute_adaptivity(0.1, dummy_micro_sims, similarity_dists, is_sim_active, sim_is_associated_to, data_for_adaptivity)

        self.assertTrue(np.array_equal(expected_is_sim_active, is_sim_active))
        self.assertTrue(np.array_equal(expected_sim_is_associated_to, sim_is_associated_to))

    def test_communicate_micro_output(self):
        """
        Test functionality to communicate micro output from active sims to their associated inactive sims, for a global adaptivity setting.
        Run this test in parallel using MPI with 2 ranks.
        """
        output_0 = {"data0.1": 1.0, "data0.2": [1.0, 2.0]}
        output_1 = {"data1.1": 10.0, "data1.2": [10.0, 20.0]}

        if self._rank == 0:
            global_ids = [0, 1, 2]
            sim_output = [None, None, output_0]
            expected_sim_output = [output_1, output_1, output_0]
        elif self._rank == 1:
            global_ids = [3, 4]
            sim_output = [output_1, None]
            expected_sim_output = [output_1, output_0]

        is_sim_active = np.array([False, False, True, True, False])
        sim_is_associated_to = [3, 3, -2, -2, 2]

        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        adaptivity_controller = GlobalAdaptivityCalculator(
            configurator,
            MagicMock(),
            5,
            global_ids,
            rank=self._rank,
            comm=self._comm)

        adaptivity_controller.communicate_micro_output(is_sim_active, sim_is_associated_to, sim_output)

        self.assertTrue(np.array_equal(expected_sim_output, sim_output))
