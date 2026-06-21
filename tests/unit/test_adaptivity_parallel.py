from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from micro_manager.simulation_container import SimulationContainer
from micro_manager.tools.mpi_handler import MPIHandler, MPI
from micro_manager.micro_simulation import create_simulation_class
from micro_manager.adaptivity.global_adaptivity import GlobalAdaptivityCalculator


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
    def get_instance(self, gid, micro_problem_cls):
        return micro_problem_cls(gid)


class TestGlobalAdaptivity(TestCase):
    def setUp(self):
        self._mpi = MPIHandler(MPI.COMM_WORLD)

        self._configurator = MagicMock()
        self._configurator.output_dir = MagicMock(return_value="output_dir")
        self._configurator.micro_file_name = MagicMock(
            return_value="test_adaptivity_parallel"
        )

    def test_update_inactive_sims_global_adaptivity(self):
        """
        Test functionality to update inactive simulations in a particular setting.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._mpi.rank == 0:
            global_ids = [0, 1, 2]
        elif self._mpi.rank == 1:
            global_ids = [3, 4]

        expected_is_sim_active = np.array([True, False, True, True, True])
        expected_sim_is_associated_to = {1: 3}

        self._configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")

        sim_cls = create_simulation_class(
            MagicMock(), MicroSimulation, __file__, 1, None, "test_micro_manager"
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            5,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in range(len(global_ids))],
        )

        adaptivity_controller = GlobalAdaptivityCalculator(
            self._configurator,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=self._mpi,
            micro_problem_cls=sim_cls,
            model_manager=ModelManager(),
        )

        adaptivity_controller._is_sim_active = np.array(
            [False, False, True, True, False]
        )
        adaptivity_controller._sim_is_associated_to = {0: 3, 1: 3, 4: 2}

        # Force the activation of sim #0 and #4
        def check_for_activation(i, active):
            if i == 0 or i == 4:
                return True
            else:
                return False

        adaptivity_controller._check_for_activation = check_for_activation

        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller._update_inactive_sims()

        self.assertTrue(
            np.array_equal(expected_is_sim_active, adaptivity_controller._is_sim_active)
        )
        self.assertDictEqual(
            expected_sim_is_associated_to, adaptivity_controller.get_associated_map()
        )

        if self._mpi.rank == 0:
            self.assertTrue(np.array_equal([3, 3, 3], container[0].get_state()))
        elif self._mpi.rank == 1:
            self.assertTrue(np.array_equal([2, 2, 2], container[1].get_state()))

    def test_update_all_active_sims_global_adaptivity(self):
        """
        Test functionality to calculate adaptivity when all simulations are active.
        Run this test in parallel using MPI with 2 ranks.
        """
        if self._mpi.rank == 0:
            global_ids = [0, 1, 2]
            data_for_adaptivity = {
                "data1": [43.9, 1.0, 1.0],
                "data2": [1355.57, 13.0, 13.0],
            }
        elif self._mpi.rank == 1:
            global_ids = [3, 4]
            data_for_adaptivity = {
                "data1": [1.0, 43.9],
                "data2": [13.0, 1355.57],
            }

        expected_is_sim_active = np.array([False, False, True, False, True])
        expected_sim_is_associated_to = {0: 4, 1: 2, 3: 2}

        self._configurator.adaptivity_history_param = MagicMock(return_value=0.1)
        self._configurator.adaptivity_refining_constant = MagicMock(return_value=0.5)
        self._configurator.adaptivity_coarsening_constant = MagicMock(return_value=0.3)
        self._configurator.adaptivity_similarity_measure = MagicMock(
            return_value="L2rel"
        )

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            5,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in range(len(global_ids))],
        )

        adaptivity_controller = GlobalAdaptivityCalculator(
            self._configurator,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=self._mpi,
            micro_problem_cls=sim_cls,
            model_manager=ModelManager(),
        )

        adaptivity_controller._data_names = ["data1", "data2"]
        adaptivity_controller._micro_data_names = adaptivity_controller._data_names

        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller.update_buffers(data_for_adaptivity, None, invert=True, alloc=True)
        adaptivity_controller.compute(0.1)

        self.assertTrue(
            np.array_equal(expected_is_sim_active, adaptivity_controller._is_sim_active)
        )
        self.assertDictEqual(
            expected_sim_is_associated_to, adaptivity_controller.get_associated_map()
        )

    def test_communicate_micro_output(self):
        """
        Test functionality to communicate micro output from active sims to their associated inactive sims.
        Run this test in parallel using MPI with 2 ranks.
        """
        output_0 = {"data0.1": 1.0, "data0.2": [1.0, 2.0]}
        output_1 = {"data1.1": 10.0, "data1.2": [10.0, 20.0]}

        if self._mpi.rank == 0:
            global_ids = [0, 1, 2]
            sim_output = [None, None, output_0]
            expected_sim_output = [output_1, output_1, output_0]
        elif self._mpi.rank == 1:
            global_ids = [3, 4]
            sim_output = [output_1, None]
            expected_sim_output = [output_1, output_0]

        self._configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")

        sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            __file__,
            1,
            None,
        )

        container = SimulationContainer(self._mpi)
        container.initialize(
            5,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in range(len(global_ids))],
        )

        adaptivity_controller = GlobalAdaptivityCalculator(
            self._configurator,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=self._mpi,
            micro_problem_cls=sim_cls,
            model_manager=ModelManager(),
        )

        for lid, gid in enumerate(container.local_gids):
            container[lid] = sim_cls(gid)

        adaptivity_controller._is_sim_active = np.array(
            [False, False, True, True, False]
        )
        adaptivity_controller._sim_is_associated_to = {0: 3, 1: 3, 4: 2}

        adaptivity_controller._communicate_micro_output(sim_output)

        self.assertTrue(np.array_equal(expected_sim_output, sim_output))

    def test_get_ranks_of_sims(self):
        """
        Test functionality to get ranks on which particular simulations are living.
        Run this test in parallel using MPI with 2 ranks.
        5 simulations are distributed across the two ranks.
        The first three simulations are on rank 0, and the last two on rank 1.
        The expected ranks of simulations are [0, 0, 0, 1, 1].
        """
        self._configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")

        expected_ranks_of_sims = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        if self._mpi.rank == 0:
            global_ids = [0, 1, 2]
        elif self._mpi.rank == 1:
            global_ids = [3, 4]

        container = SimulationContainer(self._mpi)
        container.initialize(
            5,
            len(global_ids),
            global_ids,
            [np.zeros(3) for _ in global_ids],
        )

        actual_ranks_of_sims = container.get_ranks_of_sims()
        self.assertDictEqual(expected_ranks_of_sims, actual_ranks_of_sims)
