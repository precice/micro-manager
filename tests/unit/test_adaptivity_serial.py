from math import exp
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from micro_manager.adaptivity.adaptivity import AdaptivityCalculator
from micro_manager.adaptivity.local_adaptivity import LocalAdaptivityCalculator
from micro_manager.config import Config


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

    def test_get_similarity_dists(self):
        """
        Test functionality of calculating the similarity distance matrix in class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")

        adaptivity_controller = AdaptivityCalculator(configurator, 0)
        adaptivity_controller._hist_param = 0.5
        adaptivity_controller._adaptivity_data_names = [
            "micro-scalar-data",
            "micro-vector-data",
            "macro-scalar-data",
            "macro-vector-data",
        ]

        similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))

        adaptivity_data = dict()
        adaptivity_data["micro-scalar-data"] = self._micro_scalar_data
        adaptivity_data["micro-vector-data"] = self._micro_vector_data
        adaptivity_data["macro-scalar-data"] = self._macro_scalar_data
        adaptivity_data["macro-vector-data"] = self._macro_vector_data

        similarity_dists = adaptivity_controller._get_similarity_dists(
            self._dt, self._similarity_dists, adaptivity_data
        )

        expected_similarity_dists = (
            exp(-adaptivity_controller._hist_param * self._dt) * self._similarity_dists
            + self._dt * self._data_diff
        )

        self.assertTrue(np.array_equal(expected_similarity_dists, similarity_dists))

    def test_update_active_sims(self):
        """
        Test functionality of updating active simulations in class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")

        adaptivity_controller = AdaptivityCalculator(configurator, 0)
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "macro-scalar-data",
            "macro-vector-data",
        ]

        # Third and fifth micro sim are active, rest are inactive
        expected_is_sim_active = np.array([False, False, True, False, True])

        is_sim_active = np.array(
            [True, True, True, True, True]
        )  # Activate all micro sims before calling functionality

        is_sim_active = adaptivity_controller._update_active_sims(
            self._similarity_dists, is_sim_active
        )

        self.assertTrue(np.array_equal(expected_is_sim_active, is_sim_active))

    def test_adaptivity_norms(self):
        """
        Test functionality for calculating similarity criteria between pairs of simulations using different norms in class AdaptivityCalculator.
        """
        calc = AdaptivityCalculator(Config("micro-manager-config.json"), 0)

        fake_data = np.array([[1], [2], [3]])
        self.assertTrue(
            np.allclose(
                calc._l1(fake_data), np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
            )
        )
        # norm taken over last axis -> same as before
        self.assertTrue(
            np.allclose(
                calc._l2(fake_data), np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
            )
        )
        self.assertTrue(
            np.allclose(
                calc._l1rel(fake_data),
                np.array([[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                calc._l2rel(fake_data),
                np.array([[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]]),
            )
        )

        fake_2d_data = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.allclose(calc._l1(fake_2d_data), np.array([[0, 4], [4, 0]])))
        self.assertTrue(
            np.allclose(
                calc._l2(fake_2d_data),
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
                calc._l1rel(fake_2d_data),
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
                calc._l2rel(fake_2d_data),
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

    def test_associate_active_to_inactive(self):
        """
        Test functionality to associate inactive sims to active ones, in the class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")

        adaptivity_controller = AdaptivityCalculator(configurator, 0)
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "macro-scalar-data",
            "macro-vector-data",
        ]

        is_sim_active = np.array([True, False, False, True, False])
        expected_sim_is_associated_to = np.array([-2, 0, 0, -2, 3])

        sim_is_associated_to = np.array([-2, -2, -2, -2, -2])

        sim_is_associated_to = adaptivity_controller._associate_inactive_to_active(
            self._similarity_dists, is_sim_active, sim_is_associated_to
        )

        self.assertTrue(
            np.array_equal(expected_sim_is_associated_to, sim_is_associated_to)
        )

    def test_update_inactive_sims_local_adaptivity(self):
        """
        Test functionality to update inactive simulations in a particular setting, for a local adaptivity setting.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")

        adaptivity_controller = LocalAdaptivityCalculator(
            configurator, participant=MagicMock(), rank=0, comm=MagicMock(), num_sims=5
        )
        adaptivity_controller._refine_const = self._refine_const
        adaptivity_controller._coarse_const = self._coarse_const
        adaptivity_controller._adaptivity_data_names = [
            "macro-scalar-data",
            "macro-vector-data",
        ]

        # Third and fifth micro sim are active, rest are deactivate
        expected_is_sim_active = np.array([True, False, False, True, False])
        expected_sim_is_associated_to = np.array([-2, 0, 0, -2, 3])

        similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(
                    self._micro_scalar_data[i] - self._micro_scalar_data[j]
                )
                similarity_dist += abs(
                    self._macro_scalar_data[i] - self._macro_scalar_data[j]
                )
                for d in range(self._dim):
                    similarity_dist += abs(
                        self._micro_vector_data[i, d] - self._micro_vector_data[j, d]
                    )
                    similarity_dist += abs(
                        self._macro_vector_data[i, d] - self._macro_vector_data[j, d]
                    )
                similarity_dists[i, j] = self._dt * similarity_dist

        is_sim_active = np.array([True, False, False, False, False])
        sim_is_associated_to = np.array([-2, 0, 0, 0, 3])

        class MicroSimulation:
            def get_global_id(self):
                return 1

            def set_global_id(self, global_id):
                pass

            def set_state(self, state):
                pass

            def get_state(self):
                pass

        dummy_micro_sims = []
        for i in range(self._number_of_sims):
            dummy_micro_sims.append(MicroSimulation())

        (
            is_sim_active,
            sim_is_associated_to,
        ) = adaptivity_controller._update_inactive_sims(
            similarity_dists, is_sim_active, sim_is_associated_to, dummy_micro_sims
        )

        self.assertTrue(np.array_equal(expected_is_sim_active, is_sim_active))
        self.assertTrue(
            np.array_equal(expected_sim_is_associated_to, sim_is_associated_to)
        )
