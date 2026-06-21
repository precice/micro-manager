import unittest
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Any, Optional
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from micro_manager.adaptivity.adaptivity_interface import AdaptivityInterface
from micro_manager.adaptivity.adaptivity import NoOpAdaptivity
from micro_manager.load_balancing import LoadBalancer, ActiveBalancer
from micro_manager.micro_simulation import create_simulation_class
from micro_manager.simulation_container import SimulationContainer
from micro_manager.tools.mpi_handler import MPIHandler, MPI


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


class GlobalAdaptivityDummy(AdaptivityInterface):
    def __init__(
        self,
        active_gids: List[int],
        container: SimulationContainer,
        assoc: Dict[int, int],
        mpi: MPIHandler,
    ):
        self._mpi = mpi
        self._container = container
        self._active_lids = []
        self._active_gids = active_gids
        self._inactive_lids = []
        self._inactive_gids = []
        self._active_steps = {gid: 0 for gid in container.local_gids}
        self._assoc = assoc
        active_gid_set = set(active_gids)
        for lid, gid in enumerate(container.local_gids):
            if gid in active_gid_set:
                self._active_lids.append(lid)
            else:
                self._inactive_lids.append(lid)
                self._inactive_gids.append(gid)
        assert len(self._inactive_gids) == len(self._inactive_lids)
        assert all([gid in set(assoc.keys()) for gid in self._inactive_gids])

    def get_active_steps(self) -> Dict[int, int]:
        return self._active_steps

    def get_active_lids(self) -> List[int]:
        return self._active_lids

    def get_inactive_lids(self) -> List[int]:
        return self._inactive_lids

    def get_active_gids(self) -> List[int]:
        return self._active_gids

    def get_inactive_gids(self) -> List[int]:
        return self._inactive_gids

    def get_full_field_micro_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_output: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        bcast_out = self._mpi.comm.allgather(
            list(zip(self._container.local_gids, micro_output))
        )
        glob_out = [None] * self._container.global_num_sims
        for loc_list in bcast_out:
            for gid, m_out in loc_list:
                glob_out[gid] = m_out
        for i_lid, i_gid in zip(self._inactive_lids, self._inactive_gids):
            micro_output[i_lid] = deepcopy(glob_out[self._assoc[i_gid]])
        return micro_output

    def get_adaptivity_buffer(self) -> Dict[str, List[Any]]:
        return {}

    def get_associated_map(self) -> Dict[int, int]:
        return self._assoc


class TestLBTime(TestCase):
    def setUp(self):
        self._mpi = MPIHandler(MPI.COMM_WORLD)

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
        if self._mpi.rank == 0:
            global_ids = [2, 3]
            expected_global_ids = [1, 3, 5, 7]
        else:
            global_ids = [0, 1, 4, 5, 6, 7]
            expected_global_ids = [0, 2, 4, 6]
        expected_ranks_of_sims = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0}

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        config = MagicMock()
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        load_balancer = LoadBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            NoOpAdaptivity(container),
            container,
            MagicMock(),
            config,
            self._mpi,
        )

        load_balancer.balance()

        actual_global_ids = []
        for lid in container.range_lid:
            sim = container[lid]
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(sorted(actual_global_ids), expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_redistribute_no_time_with_inactive_sims_two_ranks(self):
        """
        Test load balancing functionality to redistribute inactive simulations.
        Run this test in parallel using MPI with 2 ranks.
        """
        global_number_of_sims = 8

        # assume lpt partitioning
        if self._mpi.rank == 0:
            global_ids = [2, 3]
            expected_global_ids = [1, 3, 5, 7]
            active_gids = [3]
        else:
            global_ids = [0, 1, 4, 5, 6, 7]
            expected_global_ids = [0, 2, 4, 6]
            active_gids = [1, 4, 5, 6, 7]
        assoc_map = {0: 1, 2: 3}
        expected_ranks_of_sims = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0}

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            sim = None
            if gid not in [0, 2]:
                sim = sim_cls(gid)
            container[lid] = sim

        adaptivity = GlobalAdaptivityDummy(
            active_gids,
            container,
            assoc_map,
            self._mpi,
        )

        config = MagicMock()
        config.enable_load_balancing = MagicMock(return_value=True)
        config.load_balancing_partitioning = MagicMock(return_value="lpt")
        load_balancer = LoadBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity,
            container,
            MagicMock(),
            config,
            self._mpi,
        )

        load_balancer.balance()

        self.assertListEqual(sorted(container.local_gids), expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)


class TestLBActive(TestCase):
    def setUp(self):
        self._mpi = MPIHandler(MPI.COMM_WORLD)

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

        if self._mpi.rank == 0:
            global_ids = [0, 1, 2, 3]
            expected_global_ids = [2, 3]
            active_gids = [0, 1, 2, 3]
        elif self._mpi.rank == 1:
            global_ids = [4, 5, 6, 7]
            expected_global_ids = [4, 5, 6, 7, 0, 1]
            active_gids = []
        expected_ranks_of_sims = {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller = GlobalAdaptivityDummy(
            active_gids,
            container,
            {4: 0, 5: 1, 6: 2, 7: 3},
            self._mpi,
        )

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            container,
            MagicMock(),
            config,
            self._mpi,
        )

        load_balancer.balance()

        actual_global_ids = []  #
        for lid, gid in enumerate(container.local_gids):
            sim = container[lid]
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)

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

        if self._mpi.rank == 0:
            global_ids = [0, 2]
            expected_global_ids = [0, 2, 4]
            active_gids = [0]
        elif self._mpi.rank == 1:
            global_ids = [1, 3, 4]
            expected_global_ids = [1, 3]
            active_gids = [1]
        assoc_map = {2: 0, 3: 0, 4: 1}
        expected_ranks_of_sims = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller = GlobalAdaptivityDummy(
            active_gids,
            container,
            assoc_map,
            self._mpi,
        )

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            container,
            MagicMock(),
            config,
            self._mpi,
        )
        load_balancer._bypass_skip = True
        load_balancer._bypass_active = True

        load_balancer.balance()

        self.assertListEqual(container.local_gids, expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)

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
        if self._mpi.rank == 0:
            global_ids = [0, 1, 2, 3]
            expected_global_ids = [1, 2, 3]
            active_gids = [0, 1, 2]
        elif self._mpi.rank == 1:
            global_ids = [4, 5, 6, 7]
            expected_global_ids = [4, 5, 6, 7, 0]
            active_gids = []
        elif self._mpi.rank == 2:
            global_ids = [8, 9, 10, 11]
            expected_global_ids = [8, 9, 10, 11, 12]
            active_gids = [8]
        elif self._mpi.rank == 3:
            global_ids = [12, 13, 14]
            expected_global_ids = [13, 14]
            active_gids = [12, 13, 14]
        assoc_map = defaultdict(lambda: 0)
        expected_ranks_of_sims = [1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        expected_ranks_of_sims = {
            gid: rnk for gid, rnk in enumerate(expected_ranks_of_sims)
        }

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller = GlobalAdaptivityDummy(
            active_gids,
            container,
            assoc_map,
            self._mpi,
        )

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            container,
            MagicMock(),
            config,
            self._mpi,
        )

        load_balancer.balance()

        actual_global_ids = []
        for lid, gid in enumerate(container.local_gids):
            sim = container[lid]
            actual_global_ids.append(sim.get_global_id())
        self.assertListEqual(actual_global_ids, expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)

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
        if self._mpi.rank == 0:
            global_ids = [1, 2, 3]
            expected_global_ids = [1, 2, 4, 9, 10]
            active_gids = [1, 2]
        elif self._mpi.rank == 1:
            global_ids = [4, 5, 6, 7, 12]
            expected_global_ids = [6, 7, 12]
            active_gids = [12]
        elif self._mpi.rank == 2:
            global_ids = [0, 8, 9, 10, 11]
            expected_global_ids = [0, 8, 11, 3]
            active_gids = [0, 8]
        elif self._mpi.rank == 3:
            global_ids = [13, 14]
            expected_global_ids = [13, 14, 5]
            active_gids = [13, 14]
        assoc_map = defaultdict(lambda: 0)
        expected_ranks_of_sims = [2, 0, 0, 2, 0, 3, 1, 1, 2, 0, 0, 2, 1, 3, 3]
        expected_ranks_of_sims = {
            gid: rnk for gid, rnk in enumerate(expected_ranks_of_sims)
        }

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            global_number_of_sims,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )
        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller = GlobalAdaptivityDummy(
            active_gids,
            container,
            assoc_map,
            self._mpi,
        )

        load_balancer = ActiveBalancer(
            MagicMock(),
            ModelManager(sim_cls),
            adaptivity_controller,
            container,
            MagicMock(),
            config,
            self._mpi,
        )
        load_balancer._bypass_skip = True
        load_balancer._bypass_active = True

        load_balancer.balance()

        self.assertListEqual(container.local_gids, expected_global_ids)

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)
