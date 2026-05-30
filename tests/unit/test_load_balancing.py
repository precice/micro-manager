import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mpi4py import MPI

from micro_manager.load_balancing import LoadBalancer, ActiveBalancer
from micro_manager.micro_simulation import create_simulation_class
from micro_manager.tools.p2p import get_ranks_of_sims


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

    def solve(self, micro_input, dt):
        pass


class ModelManager:
    def __init__(self, cls):
        self._cls = cls

    def get_instance(self, gid, micro_problem_cls):
        assert micro_problem_cls == self._cls
        return micro_problem_cls(gid)

    def is_stateless(self, _):
        return True

    def get_instance_by_name(self, gid: int, name: str, *, late_init: bool = False):
        return self._cls(gid, late_init=late_init)


class TestLBTime(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_redistribute_no_time_two_ranks(self):
        """
        Test load balancing functionality to redistribute active simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        global_number_of_sims = 8

        # assume lpt partitioning
        if self._rank == 0:
            global_ids = [2, 3]
            expected_global_ids = [1, 3, 5, 7]
        else:
            global_ids = [0, 1, 4, 5, 6, 7]
            expected_global_ids = [0, 2, 4, 6]
        expected_ranks_of_sims = [1, 0, 1, 0, 1, 0, 1, 0]

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            micro_sims.append(sim_cls(i))

        config = MagicMock()
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        load_balancer = LoadBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            None,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )

        load_balancer.balance()

        actual_global_ids = []
        for sim in micro_sims:
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(sorted(actual_global_ids), expected_global_ids)

        actual_ranks_of_sims = get_ranks_of_sims(
            global_ids, self._rank, self._comm, global_number_of_sims
        )
        self.assertListEqual(list(expected_ranks_of_sims), list(actual_ranks_of_sims))

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_redistribute_no_time_with_inactive_sims_two_ranks(self):
        """
        Test load balancing functionality to redistribute inactive simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        global_number_of_sims = 8

        class GlobalAdaptivityCalculator:
            def __init__(self, active_ids):
                self._active_ids = active_ids

            def get_active_sim_local_ids(self):
                return self._active_ids

            def set_is_on_rank(self, *args, **kwargs):
                pass

        # assume lpt partitioning
        if self._rank == 0:
            global_ids = [2, 3]
            expected_global_ids = [1, 3, 5, 7]
            adaptivity = GlobalAdaptivityCalculator([1])
        else:
            global_ids = [0, 1, 4, 5, 6, 7]
            expected_global_ids = [0, 2, 4, 6]
            adaptivity = GlobalAdaptivityCalculator([1, 2, 3, 4, 5])
        expected_ranks_of_sims = [1, 0, 1, 0, 1, 0, 1, 0]

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            if i in [0, 2]:
                micro_sims.append(None)
            else:
                micro_sims.append(sim_cls(i))

        config = MagicMock()
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        load_balancer = LoadBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )

        load_balancer.balance()

        self.assertListEqual(sorted(global_ids), expected_global_ids)

        actual_ranks_of_sims = get_ranks_of_sims(
            global_ids, self._rank, self._comm, global_number_of_sims
        )
        self.assertListEqual(list(expected_ranks_of_sims), list(actual_ranks_of_sims))


class GlobalAdaptivityCalculator:
    def __init__(self, is_active, active_lids, active_gids, associated, on_rank):
        self._is_sim_active = is_active
        self._active_lids = active_lids
        self._active_gids = active_gids
        self._sim_is_associated_to = associated
        self._on_rank = on_rank

    def get_active_sim_local_ids(self):
        return self._active_lids

    def get_active_sim_global_ids(self):
        return self._active_gids

    def set_is_on_rank(self, *args, **kwargs):
        pass


class TestLBActive(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_redistribute_active_sims_two_ranks(self):
        """
        Test load balancing functionality to redistribute active simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        config = MagicMock()
        config.load_balancing_threshold = MagicMock(return_value=0)
        config.enable_load_balancing_inactive = MagicMock(return_value=False)
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")

        global_number_of_sims = 8

        if self._rank == 0:
            global_ids = [0, 1, 2, 3]
            expected_global_ids = [2, 3]
            active_lids = [0, 1, 2, 3]
            active_gids = [0, 1, 2, 3]
        elif self._rank == 1:
            global_ids = [4, 5, 6, 7]
            expected_global_ids = [4, 5, 6, 7, 0, 1]
            active_lids = []
            active_gids = []
        expected_ranks_of_sims = [1, 1, 0, 0, 1, 1, 1, 1]

        adaptivity_controller = GlobalAdaptivityCalculator(
            np.array([True, True, True, True, False, False, False, False]),
            active_lids,
            active_gids,
            [-2] * global_number_of_sims,
            [gid in global_ids for gid in range(global_number_of_sims)],
        )

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            micro_sims.append(sim_cls(i))

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )

        load_balancer.balance()

        actual_global_ids = []
        for sim in micro_sims:
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = get_ranks_of_sims(
            actual_global_ids, self._rank, self._comm, global_number_of_sims
        )
        self.assertListEqual(expected_ranks_of_sims, list(actual_ranks_of_sims))

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_redistribute_inactive_sims_two_ranks(self):
        """
        Test load balancing functionality to redistribute inactive simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        config = MagicMock()
        config.load_balancing_threshold = MagicMock(return_value=0)
        config.enable_load_balancing_inactive = MagicMock(return_value=True)
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        global_number_of_sims = 5

        if self._rank == 0:
            global_ids = [0, 2]
            expected_global_ids = [0, 2, 4]
            active_lids = [0]
            active_gids = [0]
        elif self._rank == 1:
            global_ids = [1, 3, 4]
            expected_global_ids = [1, 3]
            active_lids = [0]
            active_gids = [1]
        expected_ranks_of_sims = [0, 1, 0, 1, 0]

        adaptivity_controller = GlobalAdaptivityCalculator(
            np.array([True, True, False, False, False]),
            active_lids,
            active_gids,
            [-2, -2, 0, 1, 0],
            [gid in global_ids for gid in range(global_number_of_sims)],
        )

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            micro_sims.append(sim_cls(i))

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )
        load_balancer._bypass_skip = True
        load_balancer._bypass_active = True

        load_balancer.balance()
        self.assertEqual(global_ids, expected_global_ids)

        self.assertListEqual(global_ids, expected_global_ids)
        actual_ranks_of_sims = get_ranks_of_sims(
            global_ids, self._rank, self._comm, global_number_of_sims
        )

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 4, "This test only works with 4 ranks."
    )
    def test_redistribute_active_sims_four_ranks(self):
        """
        Test load balancing functionality to redistribute active simulations. The load balancing is one in two steps.
        Run this test in parallel using MPI with 4 ranks.
        """
        config = MagicMock()
        config.load_balancing_threshold = MagicMock(return_value=0)
        config.enable_load_balancing_inactive = MagicMock(return_value=False)
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")

        global_number_of_sims = 15

        # active_gids_global = [0, 1, 2, 8, 12, 13, 14]
        if self._rank == 0:
            global_ids = [0, 1, 2, 3]
            expected_global_ids = [1, 2, 3]
            active_lids = [0, 1, 2]
            active_gids = [0, 1, 2]
        elif self._rank == 1:
            global_ids = [4, 5, 6, 7]
            expected_global_ids = [4, 5, 6, 7, 0]
            active_lids = []
            active_gids = []
        elif self._rank == 2:
            global_ids = [8, 9, 10, 11]
            expected_global_ids = [8, 9, 10, 11, 12]
            active_lids = [0]
            active_gids = [8]
        elif self._rank == 3:
            global_ids = [12, 13, 14]
            expected_global_ids = [13, 14]
            active_lids = [0, 1, 2]
            active_gids = [12, 13, 14]
        expected_ranks_of_sims = [1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]

        adaptivity_controller = GlobalAdaptivityCalculator(
            np.array(
                [
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ),
            active_lids,
            active_gids,
            [-2] * global_number_of_sims,
            [gid in global_ids for gid in range(global_number_of_sims)],
        )

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            micro_sims.append(sim_cls(i))

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )

        load_balancer.balance()

        actual_global_ids = []
        for sim in micro_sims:
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = get_ranks_of_sims(
            actual_global_ids, self._rank, self._comm, global_number_of_sims
        )
        self.assertListEqual(expected_ranks_of_sims, list(actual_ranks_of_sims))

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 4, "This test only works with 4 ranks."
    )
    def test_redistribute_inactive_sims_four_ranks(self):
        """
        Test load balancing functionality to redistribute inactive simulations.
        Run this test in parallel using MPI with 4 ranks.
        """
        config = MagicMock()
        config.load_balancing_threshold = MagicMock(return_value=0)
        config.enable_load_balancing_inactive = MagicMock(return_value=True)
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        global_number_of_sims = 15

        # global_active_gids = [0, 1, 2, 8, 12, 13, 14]
        if self._rank == 0:
            global_ids = [1, 2, 3]
            expected_global_ids = [1, 2, 4, 9, 10]
            active_lids = [0, 1]
            active_gids = [1, 2]
        elif self._rank == 1:
            global_ids = [4, 5, 6, 7, 12]
            expected_global_ids = [6, 7, 12]
            active_lids = [4]
            active_gids = [12]
        elif self._rank == 2:
            global_ids = [0, 8, 9, 10, 11]
            expected_global_ids = [0, 8, 11, 3]
            active_lids = [0, 1]
            active_gids = [0, 8]
        elif self._rank == 3:
            global_ids = [13, 14]
            expected_global_ids = [13, 14, 5]
            active_lids = [0, 1]
            active_gids = [13, 14]
        expected_ranks_of_sims = [2, 0, 0, 2, 0, 3, 1, 1, 2, 0, 0, 2, 1, 3, 3]

        adaptivity_controller = GlobalAdaptivityCalculator(
            np.array(
                [
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ),
            active_lids,
            active_gids,
            [-2, -2, -2, 0, 1, 13, 12, 12, -2, 1, 2, 8, -2, -2, -2],
            [gid in global_ids for gid in range(global_number_of_sims)],
        )

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        micro_sims = []
        for i in global_ids:
            micro_sims.append(sim_cls(i))

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            lambda sim: sim.get_state(),
            lambda sim, state: sim.set_state(state),
            MagicMock(),
            config,
            micro_sims,
            global_ids,
            global_number_of_sims,
            self._comm,
            self._rank,
        )
        load_balancer._bypass_skip = True
        load_balancer._bypass_active = True

        load_balancer.balance()

        self.assertEqual(global_ids, expected_global_ids)

        self.assertListEqual(global_ids, expected_global_ids)
        actual_ranks_of_sims = get_ranks_of_sims(
            global_ids, self._rank, self._comm, global_number_of_sims
        )

        self.assertTrue(np.array_equal(expected_ranks_of_sims, actual_ranks_of_sims))
