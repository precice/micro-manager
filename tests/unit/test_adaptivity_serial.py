from math import exp
from typing import List, Dict, Any, Optional
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from micro_manager.adaptivity.adaptivity import AdaptivityCalculator
from micro_manager.adaptivity.local_adaptivity import LocalAdaptivityCalculator
from micro_manager.simulation_container import SimulationContainer
from micro_manager.tools.mpi_handler import MPIHandler, MPI


class MicroSimulation:
    def __init__(self, global_id):
        self._global_id = global_id
        self.name = MicroSimulation.__name__

    def get_global_id(self):
        return self._global_id

    def set_global_id(self, global_id):
        pass

    def set_state(self, state):
        pass

    def get_state(self):
        pass

    def solve(self, micro_input, dt):
        pass

    def destroy(self):
        pass


class ModelManager:
    def get_instance(self, gid, micro_problem_cls):
        return micro_problem_cls(gid)


class AdaptivityCalculatorInstantiable(AdaptivityCalculator):
    """
    Workaround to bypass ABC instantiation limitations. Only used for testing internals.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_active_lids(self) -> List[int]:
        return []

    def get_inactive_lids(self) -> List[int]:
        return []

    def get_active_gids(self) -> List[int]:
        return []

    def get_inactive_gids(self) -> List[int]:
        return []

    def get_full_field_micro_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_output: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        return []


class TestLocalAdaptivity(TestCase):
    def setUp(self):
        self._number_of_sims = 5
        self._dt = 0.1
        self._dim = 3

        self._micro_scalar_data = np.zeros(5)
        np.put(self._micro_scalar_data, [0, 1, 2], [3.0, 3.0, 3.0])
        np.put(self._micro_scalar_data, [3, 4], [5.0, 5.0])

        self._micro_vector_data = np.zeros((5, 3))
        # First three simulations have similar micro_vector_data
        for i in range(3):
            self._micro_vector_data[i, :] = 5.0

        # Last two simulations have similar micro_vector_data
        for i in range(3, 5):
            self._micro_vector_data[i, :] = 10.0

        self._macro_scalar_data = np.zeros(5)
        np.put(self._micro_scalar_data, [0, 1, 2], [130.0, 130.0, 130.0])
        np.put(self._micro_scalar_data, [3, 4], [250.0, 250.0])

        self._macro_vector_data = np.zeros((5, 3))
        # First three simulations have similar micro_vector_data
        for i in range(3):
            self._macro_vector_data[i, :] = 100.0

        # Last two simulations have similar micro_vector_data
        for i in range(3, 5):
            self._macro_vector_data[i, :] = 300.0

        # Adaptivity constants
        self._refine_const = 0.5
        self._coarse_const = 0.5
        self._coarse_tol = 0.2

        self._data_diff = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    dist += abs(
                        self._micro_vector_data[i, d] - self._micro_vector_data[j, d]
                    )
                    dist += abs(
                        self._macro_vector_data[i, d] - self._macro_vector_data[j, d]
                    )
                self._data_diff[i, j] = dist

        self._similarity_dists = self._dt * self._data_diff

    def test_update_similarity_dists(self):
        """
        Test functionality of calculating the similarity distance matrix in class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.output_dir = MagicMock(return_value="output_dir")
        configurator.micro_file_name = MagicMock(return_value="test_adaptivity_serial")

        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(
            self._number_of_sims,
            self._number_of_sims,
            list(range(self._number_of_sims)),
            [np.zeros(3) for _ in range(self._number_of_sims)],
        )

        adaptivity_controller = AdaptivityCalculatorInstantiable(
            configurator,
            nsims=self._number_of_sims,
            sim_container=container,
            profiler=MagicMock(),
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
            base_logger=MagicMock(),
            mpi=mpi,
        )
        adaptivity_controller._hist_param = 0.5
        adaptivity_controller._adaptivity_data_names = [
            "Micro-Scalar-Data",
            "Micro-Vector-Data",
            "Macro-Scalar-Data",
            "Macro-Vector-Data",
        ]

        adaptivity_data = dict()
        adaptivity_data["Micro-Scalar-Data"] = self._micro_scalar_data
        adaptivity_data["Micro-Vector-Data"] = self._micro_vector_data
        adaptivity_data["Macro-Scalar-Data"] = self._macro_scalar_data
        adaptivity_data["Macro-Vector-Data"] = self._macro_vector_data

        adaptivity_controller._similarity_dists = self._similarity_dists

        old_similarity_dists = adaptivity_controller._similarity_dists.copy()

        adaptivity_controller._update_similarity_dists(self._dt, adaptivity_data)

        expected_similarity_dists = (
            exp(-adaptivity_controller._hist_param * self._dt) * old_similarity_dists
            + self._dt * self._data_diff
        )

        self.assertTrue(
            np.array_equal(
                expected_similarity_dists, adaptivity_controller._similarity_dists
            )
        )

    def test_update_active_sims(self):
        """
        Test functionality of updating active simulations in class LocalAdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.output_dir = MagicMock(return_value="output_dir")
        configurator.micro_file_name = MagicMock(return_value="test_adaptivity_serial")

        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(
            self._number_of_sims,
            self._number_of_sims,
            list(range(self._number_of_sims)),
            [np.zeros(3) for _ in range(self._number_of_sims)],
        )

        adaptivity_controller = LocalAdaptivityCalculator(
            configurator,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=mpi,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "Macro-Scalar-Data",
            "Macro-Vector-Data",
        ]

        adaptivity_controller._similarity_dists = self._similarity_dists

        # Third and fifth micro sim are active, rest are inactive
        expected_is_sim_active = np.array([False, False, True, False, True])

        adaptivity_controller._update_active_sims()

        self.assertTrue(
            np.array_equal(expected_is_sim_active, adaptivity_controller._is_sim_active)
        )

    def test_adaptivity_norms(self):
        """
        Test functionality for calculating similarity criteria between pairs of simulations using different norms in class AdaptivityCalculator.
        """
        fake_data = np.array([[1], [2], [3]])
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l1(fake_data),
                np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            )
        )
        # norm taken over last axis -> same as before
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l2(fake_data),
                np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l1rel(fake_data),
                np.array([[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l2rel(fake_data),
                np.array([[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]]),
            )
        )

        fake_2d_data = np.array([[1, 2], [3, 4]])
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l1(fake_2d_data), np.array([[0, 4], [4, 0]])
            )
        )
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l2(fake_2d_data),
                np.array(
                    [
                        [0, np.sqrt((1 - 3) ** 2 + (2 - 4) ** 2)],
                        [np.sqrt((1 - 3) ** 2 + (2 - 4) ** 2), 0],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l1rel(fake_2d_data),
                np.array(
                    [
                        [0, abs((1 - 3) / max(1, 3) + (2 - 4) / max(2, 4))],
                        [abs((1 - 3) / max(1, 3) + (2 - 4) / max(2, 4)), 0],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                AdaptivityCalculator._l2rel(fake_2d_data),
                np.array(
                    [
                        [
                            0,
                            np.sqrt(
                                (1 - 3) ** 2 / max(1, 3) ** 2
                                + (2 - 4) ** 2 / max(2, 4) ** 2
                            ),
                        ],
                        [
                            np.sqrt(
                                (1 - 3) ** 2 / max(1, 3) ** 2
                                + (2 - 4) ** 2 / max(2, 4) ** 2
                            ),
                            0,
                        ],
                    ]
                ),
            )
        )

    def test_adaptivity_norms_with_zeros_no_warning(self):
        """
        Test that L1rel/L2rel must not raise division-by-zero
        warning when data contains zeros.
        """
        import warnings

        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(
            self._number_of_sims,
            self._number_of_sims,
            list(range(self._number_of_sims)),
            [np.zeros(3) for _ in range(self._number_of_sims)],
        )

        configurator = MagicMock()
        configurator.adaptivity_similarity_measure = MagicMock(return_value="L2rel")
        configurator.output_dir = MagicMock(return_value="output_dir")
        configurator.micro_file_name = MagicMock(return_value="test_adaptivity_serial")
        adaptivity_l2rel = AdaptivityCalculatorInstantiable(
            configurator,
            nsims=3,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=mpi,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        configurator_l1 = MagicMock()
        configurator_l1.adaptivity_similarity_measure = MagicMock(return_value="L1rel")
        configurator_l1.output_dir = MagicMock(return_value="output_dir")
        configurator_l1.micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )
        adaptivity_l1rel = AdaptivityCalculatorInstantiable(
            configurator_l1,
            nsims=3,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=mpi,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        # Data with zeros - previously triggered RuntimeWarning: invalid value in true_divide
        data_with_zeros = np.array([[0.0], [0.0], [1.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error", RuntimeWarning)
            result_l2rel = adaptivity_l2rel._l2rel(data_with_zeros)
        self.assertEqual(
            len(w), 0, "L2rel must not raise RuntimeWarning with zero data"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error", RuntimeWarning)
            result_l1rel = adaptivity_l1rel._l1rel(data_with_zeros)
        self.assertEqual(
            len(w), 0, "L1rel must not raise RuntimeWarning with zero data"
        )

        # When both are 0, relative diff should be 0 (since numerator is 0)
        self.assertEqual(result_l2rel[0, 1], 0.0)
        self.assertEqual(result_l1rel[0, 1], 0.0)

    def test_associate_active_to_inactive(self):
        """
        Test functionality to associate inactive sims to active ones, in the class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.output_dir = MagicMock(return_value="output_dir")
        configurator.micro_file_name = MagicMock(return_value="test_adaptivity_serial")

        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(
            self._number_of_sims,
            self._number_of_sims,
            list(range(self._number_of_sims)),
            [np.zeros(3) for _ in range(self._number_of_sims)],
        )

        adaptivity_controller = AdaptivityCalculatorInstantiable(
            configurator,
            nsims=self._number_of_sims,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=mpi,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "Macro-Scalar-Data",
            "Macro-Vector-Data",
        ]

        adaptivity_controller._similarity_dists = self._similarity_dists
        adaptivity_controller._max_similarity_dist = np.amax(self._similarity_dists)

        adaptivity_controller._is_sim_active = np.array(
            [True, False, False, True, False]
        )
        expected_sim_is_associated_to = {1: 0, 2: 0, 4: 3}

        adaptivity_controller._associate_inactive_to_active()
        self.assertDictEqual(
            expected_sim_is_associated_to, adaptivity_controller.get_associated_map()
        )

    def test_update_inactive_sims_local_adaptivity(self):
        """
        Test functionality to update inactive simulations in a particular setting, for a local adaptivity setting.
        """
        configurator = MagicMock()
        configurator.adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.output_dir = MagicMock(return_value="output_dir")
        configurator.micro_file_name = MagicMock(return_value="test_adaptivity_serial")

        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(
            self._number_of_sims,
            self._number_of_sims,
            list(range(self._number_of_sims)),
            [np.zeros(3) for _ in range(self._number_of_sims)],
        )

        adaptivity_controller = LocalAdaptivityCalculator(
            configurator,
            sim_container=container,
            profiler=MagicMock(),
            base_logger=MagicMock(),
            mpi=mpi,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "Macro-Scalar-Data",
            "Macro-Vector-Data",
        ]

        # Third and fifth micro sim are active, rest are deactivate
        expected_is_sim_active = np.array([True, False, False, True, False])
        expected_sim_is_associated_to = {1: 0, 2: 0, 4: 3}

        adaptivity_controller._similarity_dists = self._similarity_dists
        adaptivity_controller._is_sim_active = np.array(
            [True, False, False, False, False]
        )
        adaptivity_controller._sim_is_associated_to = {1: 0, 2: 0, 3: 0, 4: 3}

        for i in range(self._number_of_sims):
            container[i] = MicroSimulation(i)

        adaptivity_controller._update_inactive_sims()

        self.assertTrue(
            np.array_equal(expected_is_sim_active, adaptivity_controller._is_sim_active)
        )
        self.assertDictEqual(
            expected_sim_is_associated_to, adaptivity_controller.get_associated_map()
        )
