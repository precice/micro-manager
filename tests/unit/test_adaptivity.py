from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.adaptivity.local_adaptivity import LocalAdaptivityCalculator
from micro_manager.adaptivity.adaptivity import AdaptivityCalculator
from micro_manager.config import Config
import numpy as np


class TestLocalAdaptivity(TestCase):

    def setUp(self):
        self._adaptivity_controller = LocalAdaptivityCalculator(configurator=MagicMock(), logger=MagicMock())
        self._adaptivity_controller._refine_const = 0.4
        self._adaptivity_controller._coarse_const = 0.3
        self._adaptivity_controller._hist_param = 0.5
        self._adaptivity_controller._adaptivity_data_names = ["macro-scalar-data", "macro-vector-data"]
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

    def test_get_similarity_dists(self):
        expected_similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                similarity_dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    similarity_dist += abs(self._micro_vector_data[i, d] - self._micro_vector_data[j, d])
                    similarity_dist += abs(self._macro_vector_data[i, d] - self._macro_vector_data[j, d])
                expected_similarity_dists[i, j] = self._dt * similarity_dist

        actual_similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        actual_similarity_dists = self._adaptivity_controller._get_similarity_dists(
            self._dt, actual_similarity_dists, self._micro_scalar_data)
        actual_similarity_dists = self._adaptivity_controller._get_similarity_dists(
            self._dt, actual_similarity_dists, self._micro_vector_data)
        actual_similarity_dists = self._adaptivity_controller._get_similarity_dists(
            self._dt, actual_similarity_dists, self._macro_scalar_data)
        actual_similarity_dists = self._adaptivity_controller._get_similarity_dists(
            self._dt, actual_similarity_dists, self._macro_vector_data)

        self.assertTrue(np.array_equal(expected_similarity_dists, actual_similarity_dists))

    def test_update_active_sims(self):
        # Third and fifth micro sim are active, rest are inactive
        expected_is_sim_active = np.array([False, False, True, False, True])

        similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                similarity_dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    similarity_dist += abs(self._micro_vector_data[i, d] - self._micro_vector_data[j, d])
                    similarity_dist += abs(self._macro_vector_data[i, d] - self._macro_vector_data[j, d])
                similarity_dists[i, j] = self._dt * similarity_dist

        actual_is_sim_active = np.array([1, 1, 1, 1, 1])  # Activate all micro sims before calling functionality

        class MicroSimulation():
            pass

        dummy_micro_sims = []
        for i in range(self._number_of_sims):
            dummy_micro_sims.append(MicroSimulation())

        actual_is_sim_active = self._adaptivity_controller._update_active_sims(
            similarity_dists, actual_is_sim_active)

        self.assertTrue(np.array_equal(expected_is_sim_active, actual_is_sim_active))

    def test_update_inactive_sims(self):
        # Third and fifth micro sim are active, rest are deactivate
        expected_is_sim_active = np.array([False, False, True, False, True])
        expected_sim_is_associated_to = np.array([2, 2, -2, 4, -2])

        similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                similarity_dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    similarity_dist += abs(self._micro_vector_data[i, d] - self._micro_vector_data[j, d])
                    similarity_dist += abs(self._macro_vector_data[i, d] - self._macro_vector_data[j, d])
                similarity_dists[i, j] = self._dt * similarity_dist

        is_sim_active = np.array([True, False, False, False, False])

        class MicroSimulation():
            def get_global_id(self):
                return 1

            def set_global_id(self, global_id):
                pass

        dummy_micro_sims = []
        for i in range(self._number_of_sims):
            dummy_micro_sims.append(MicroSimulation())

        is_sim_active, sim_is_associated_to = self._adaptivity_controller._update_inactive_sims(
            similarity_dists, is_sim_active, sim_is_associated_to)

        self.assertTrue(np.array_equal(expected_is_sim_active, is_sim_active))
        self.assertTrue(np.array_equal(expected_sim_is_associated_to, sim_is_associated_to))

    def test_associate_active_to_inactive(self):
        is_sim_active = np.array([False, False, True, False, True])

        similarity_dists = np.zeros((self._number_of_sims, self._number_of_sims))
        for i in range(self._number_of_sims):
            for j in range(self._number_of_sims):
                similarity_dist = abs(self._micro_scalar_data[i] - self._micro_scalar_data[j])
                similarity_dist += abs(self._macro_scalar_data[i] - self._macro_scalar_data[j])
                for d in range(self._dim):
                    similarity_dist += abs(self._micro_vector_data[i, d] - self._micro_vector_data[j, d])
                    similarity_dist += abs(self._macro_vector_data[i, d] - self._macro_vector_data[j, d])
                similarity_dists[i, j] = self._dt * similarity_dist

        class MicroSimulation():
            def __init__(self, global_id):
                self._global_id = global_id

            def get_global_id(self):
                return self._global_id

        dummy_micro_sims = []
        for i in range(self._number_of_sims):
            dummy_micro_sims.append(MicroSimulation(i))

        self._adaptivity_controller._associate_inactive_to_active(similarity_dists, is_sim_active, dummy_micro_sims)

        self.assertEqual(dummy_micro_sims[0].get_associated_active_id(), 2)
        self.assertEqual(dummy_micro_sims[1].get_associated_active_id(), 2)
        self.assertEqual(dummy_micro_sims[3].get_associated_active_id(), 4)

    def test_adaptivity_norms(self):
        calc = AdaptivityCalculator(Config('micro-manager-unit-test-adaptivity-config.json'), 0)

        fake_data = np.array([[1], [2], [3]])
        self.assertTrue(np.allclose(calc._l1(fake_data), np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])))
        # norm taken over last axis -> same as before
        self.assertTrue(np.allclose(calc._l2(fake_data), np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])))
        self.assertTrue(np.allclose(calc._l1rel(fake_data), np.array(
            [[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]])))
        self.assertTrue(np.allclose(calc._l2rel(fake_data), np.array(
            [[0, 0.5, 2 / 3], [0.5, 0, 1 / 3], [2 / 3, 1 / 3, 0]])))

        fake_2d_data = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.allclose(calc._l1(fake_2d_data), np.array([[0, 4], [4, 0]])))
        self.assertTrue(np.allclose(calc._l2(fake_2d_data), np.array([[0, np.sqrt((1 - 3)**2 + (2 - 4)**2)],
                                                                      [np.sqrt((1 - 3)**2 + (2 - 4)**2), 0]])))
        self.assertTrue(np.allclose(calc._l1rel(fake_2d_data), np.array(
            [[0, abs((1 - 3) / max(1, 3) + (2 - 4) / max(2, 4))], [abs((1 - 3) / max(1, 3) + (2 - 4) / max(2, 4)), 0]])))
        self.assertTrue(np.allclose(calc._l2rel(fake_2d_data), np.array([[0, np.sqrt(
            (1 - 3)**2 / max(1, 3)**2 + (2 - 4)**2 / max(2, 4)**2)], [np.sqrt((1 - 3)**2 / max(1, 3)**2 + (2 - 4)**2 / max(2, 4)**2), 0]])))
