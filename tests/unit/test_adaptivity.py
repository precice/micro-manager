from unittest import TestCase
from micro_manager.adaptivity import AdaptiveController
from micro_manager.config import Config
import numpy as np


class TestAdaptivity(TestCase):

    def setUp(self):
        self._adaptivity_controller = AdaptiveController(Config("./tests/unit/test_adaptivity_config.json"))
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

    def test_set_number_of_sims(self):
        self._adaptivity_controller.set_number_of_sims(self._number_of_sims)
        self.assertEqual(self._number_of_sims, self._adaptivity_controller._number_of_sims)

    def test_get_similarity_dists(self):
        self._adaptivity_controller._number_of_sims = self._number_of_sims
        expected_similarity_dist = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                similarity_dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    similarity_dist += abs(self._micro_vector_data[i, d] - self._micro_vector_data[j, d])
                    similarity_dist += abs(self._macro_vector_data[i, d] - self._macro_vector_data[j, d])
                expected_similarity_dist[i, j] = self._dt * similarity_dist

        actual_similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        actual_similarity_dists = self._adaptivity_controller.get_similarity_dists(
            self._dt, actual_similarity_dists, self._micro_scalar_data)
        actual_similarity_dists = self._adaptivity_controller.get_similarity_dists(
            self._dt, actual_similarity_dists, self._micro_vector_data)
        actual_similarity_dists = self._adaptivity_controller.get_similarity_dists(
            self._dt, actual_similarity_dists, self._macro_scalar_data)
        actual_similarity_dists = self._adaptivity_controller.get_similarity_dists(
            self._dt, actual_similarity_dists, self._macro_vector_data)

        self.assertTrue(np.array_equal(expected_similarity_dist, actual_similarity_dists))
