import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mpi4py import MPI

from micro_manager.adaptivity.global_adaptivity_lb import GlobalAdaptivityLBCalculator


class MicroSimulation:
    def __init__(self, global_id) -> None:
        self._global_id = global_id
        self._state = [global_id] * 3

    def get_global_id(self):
        return self._global_id

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state.copy()


class TestGlobalAdaptivityLB(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._configurator = MagicMock()
        self._configurator.get_micro_file_name = MagicMock(
            return_value="test_global_adaptivity_lb"
        )
        self._configurator.get_adaptivity_similarity_measure = MagicMock(
            return_value="L1"
        )
        self._configurator.get_output_dir = MagicMock(return_value="output_dir")

        self._global_number_of_sims = 5

    def test_get_ranks_of_sims(self):
        """ """
        if self._rank == 0:
            global_ids = [0, 1, 2]
            expected_ranks_of_sims = [0, 0, 0, 1, 1]
        elif self._rank == 1:
            global_ids = [3, 4]
            expected_ranks_of_sims = [0, 0, 0, 1, 1]

        adaptivity_controller = GlobalAdaptivityLBCalculator(
            self._configurator,
            self._global_number_of_sims,
            global_ids,
            rank=self._rank,
            comm=self._comm,
        )

        actual_ranks_of_sims = adaptivity_controller._get_ranks_of_sims()

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))

    def test_redistribute_active_sims_two_ranks(self):
        """
        Test load balancing functionality to redistribute active simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._rank == 0:
            global_ids = [0, 1, 2]
            expected_global_ids = [1, 2]
        elif self._rank == 1:
            global_ids = [3, 4]
            expected_global_ids = [0, 3, 4]

        expected_ranks_of_sims = [1, 0, 0, 1, 1]

        adaptivity_controller = GlobalAdaptivityLBCalculator(
            self._configurator,
            self._global_number_of_sims,
            global_ids,
            rank=self._rank,
            comm=self._comm,
        )

        adaptivity_controller._is_sim_active = np.array(
            [True, True, False, False, False]
        )
        adaptivity_controller._sim_is_associated_to = [-2, -2, 0, 1, 0]

        micro_sims = []
        for i in range(self._global_number_of_sims):
            if i in global_ids:
                micro_sims.append(MicroSimulation(i))
            else:
                micro_sims.append(None)

        adaptivity_controller._redistribute_active_sims(micro_sims)

        actual_global_ids = []
        for i in range(self._global_number_of_sims):
            if micro_sims[i] is not None:
                actual_global_ids.append(micro_sims[i].get_global_id())

        self.assertEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = adaptivity_controller._get_ranks_of_sims()

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))

    @unittest.skip("This test is not fully implemented yet.")
    def test_redistribute_active_sims_four_ranks(self):
        """
        Test load balancing functionality to redistribute active simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._rank == 0:
            global_ids = [0, 1, 2, 3]
            expected_global_ids = [1, 2]
        elif self._rank == 1:
            global_ids = [4, 5, 6, 7, 8, 9]
            expected_global_ids = [0, 3, 4]
        elif self._rank == 2:
            global_ids = [10, 11, 12, 13, 14, 15]
        elif self._rank == 3:
            global_ids = [16, 17, 18, 19]

        expected_ranks_of_sims = [1, 0, 0, 1, 1]

        adaptivity_controller = GlobalAdaptivityLBCalculator(
            self._configurator,
            self._global_number_of_sims,
            global_ids,
            rank=self._rank,
            comm=self._comm,
        )

        adaptivity_controller._is_sim_active = np.array(
            [True, True, False, False, False]
        )
        adaptivity_controller._sim_is_associated_to = [-2, -2, 0, 1, 0]

        micro_sims = []
        for i in range(self._global_number_of_sims):
            if i in global_ids:
                micro_sims.append(MicroSimulation(i))
            else:
                micro_sims.append(None)

        adaptivity_controller._redistribute_active_sims(micro_sims)

        actual_global_ids = []
        for i in range(self._global_number_of_sims):
            if micro_sims[i] is not None:
                actual_global_ids.append(micro_sims[i].get_global_id())

        self.assertEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = adaptivity_controller._get_ranks_of_sims()

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))

    def test_redistribute_inactive_sims(self):
        """
        Test load balancing functionality to redistribute active simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._rank == 0:
            global_ids = [0, 2]
            expected_global_ids = [0, 2, 4]
        elif self._rank == 1:
            global_ids = [1, 3, 4]
            expected_global_ids = [1, 3]

        expected_ranks_of_sims = [0, 1, 0, 1, 0]

        adaptivity_controller = GlobalAdaptivityLBCalculator(
            self._configurator,
            self._global_number_of_sims,
            global_ids,
            rank=self._rank,
            comm=self._comm,
        )

        adaptivity_controller._is_sim_active = np.array(
            [True, True, False, False, False]
        )
        adaptivity_controller._sim_is_associated_to = [-2, -2, 0, 1, 0]

        micro_sims = []
        for i in range(self._global_number_of_sims):
            if i in global_ids:
                micro_sims.append(MicroSimulation(i))
            else:
                micro_sims.append(None)

        adaptivity_controller._redistribute_inactive_sims(micro_sims)

        actual_global_ids = []
        for i in range(self._global_number_of_sims):
            if micro_sims[i] is not None:
                actual_global_ids.append(micro_sims[i].get_global_id())

        self.assertEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = adaptivity_controller._get_ranks_of_sims()

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))
