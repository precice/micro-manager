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
            is_sim_on_this_rank = [True, True, True, False, False]
            global_ids = [0, 1, 2]
        elif self._rank == 1:
            is_sim_on_this_rank = [False, False, False, True, True]
            global_ids = [3, 4]

        is_sim_active = np.array([False, False, True, True, False])
        rank_of_sim = [0, 0, 0, 1, 1]
        sim_is_associated_to = [3, 3, -2, -2, 2]
        expected_is_sim_active = np.array([True, False, True, True, True])
        expected_sim_is_associated_to = [-2, 3, -2, -2, -2]

        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        adaptivity_controller = GlobalAdaptivityCalculator(
            configurator,
            MagicMock(),
            is_sim_on_this_rank,
            rank_of_sim,
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

    def test_communicate_micro_output(self):
        """
        Test functionality to communicate micro output from active sims to their associated inactive sims, for a global adaptivity setting.
        Run this test in parallel using MPI with 2 ranks.
        """
        output_0 = {"data0.1": 1.0, "data0.2": [1.0, 2.0]}
        output_1 = {"data1.1": 10.0, "data1.2": [10.0, 20.0]}

        if self._rank == 0:
            is_sim_on_this_rank = [True, True, True, False, False]
            global_ids = [0, 1, 2]
            sim_output = [None, None, output_0]
            expected_sim_output = [output_1, output_1, output_0]
        elif self._rank == 1:
            is_sim_on_this_rank = [False, False, False, True, True]
            global_ids = [3, 4]
            sim_output = [output_1, None]
            expected_sim_output = [output_1, output_0]

        is_sim_active = np.array([False, False, True, True, False])
        rank_of_sim = [0, 0, 0, 1, 1]
        sim_is_associated_to = [3, 3, -2, -2, 2]

        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        adaptivity_controller = GlobalAdaptivityCalculator(
            configurator,
            MagicMock(),
            is_sim_on_this_rank,
            rank_of_sim,
            global_ids,
            rank=self._rank,
            comm=self._comm)

        adaptivity_controller.communicate_micro_output(is_sim_active, sim_is_associated_to, sim_output)

        self.assertTrue(np.array_equal(expected_sim_output, sim_output))
