from math import exp
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mpi4py import MPI

from micro_manager.adaptivity.adaptivity import AdaptivityCalculator
from micro_manager.adaptivity.local_adaptivity import LocalAdaptivityCalculator


class MicroSimulation:
    def __init__(self, global_id):
        self._global_id = global_id

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


class ModelManager:
    def get_instance(self, gid, micro_problem_cls):
        return micro_problem_cls(gid)


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

        # Convert 2D matrix to 1D strictly lower triangular vector
        self._similarity_dists = self._convert_2d_to_1d(self._dt * self._data_diff)

    def test_update_similarity_dists(self):
        """
        Test functionality of calculating the similarity distance matrix in class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )

        adaptivity_controller = AdaptivityCalculator(
            configurator,
            nsims=self._number_of_sims,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
            base_logger=MagicMock(),
            rank=0,
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

        # Convert expected 2D result back to 1D for comparison
        expected_2d = (
            exp(-adaptivity_controller._hist_param * self._dt)
            * self._convert_1d_to_2d(old_similarity_dists, self._number_of_sims)
            + self._dt * self._data_diff
        )
        expected_similarity_dists = self._convert_2d_to_1d(expected_2d)

        self.assertTrue(
            np.allclose(
                expected_similarity_dists, adaptivity_controller._similarity_dists
            )
        )

    def test_update_active_sims(self):
        """
        Test functionality of updating active simulations in class LocalAdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )

        adaptivity_controller = LocalAdaptivityCalculator(
            configurator,
            self._number_of_sims,
            base_logger=MagicMock(),
            rank=0,
            comm=MagicMock(),
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
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )

        # Test with 3-element data
        nsims_3 = 3
        adaptivity_l1 = AdaptivityCalculator(
            configurator,
            nsims=nsims_3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        fake_data = np.array([[1], [2], [3]])
        adaptivity_l1._similarity_dists = np.zeros(nsims_3 * (nsims_3 - 1) // 2)
        adaptivity_l1._l1(fake_data, dt=1.0)
        # Expected strictly lower triangular: (1,0)=1, (2,0)=2, (2,1)=1
        expected_l1_3 = np.array([1.0, 2.0, 1.0])
        self.assertTrue(np.allclose(adaptivity_l1._similarity_dists, expected_l1_3))

        adaptivity_l2 = AdaptivityCalculator(
            configurator,
            nsims=nsims_3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        configurator.get_adaptivity_similarity_measure.return_value = "L2"
        adaptivity_l2._similarity_dists = np.zeros(nsims_3 * (nsims_3 - 1) // 2)
        adaptivity_l2._l2(fake_data, dt=1.0)
        # L2 norm of scalars is same as L1 for 1D data
        expected_l2_3 = np.array([1.0, 2.0, 1.0])
        self.assertTrue(np.allclose(adaptivity_l2._similarity_dists, expected_l2_3))

        adaptivity_l1rel = AdaptivityCalculator(
            configurator,
            nsims=nsims_3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        configurator.get_adaptivity_similarity_measure.return_value = "L1rel"
        adaptivity_l1rel._similarity_dists = np.zeros(nsims_3 * (nsims_3 - 1) // 2)
        adaptivity_l1rel._l1rel(fake_data, dt=1.0)
        # Expected: (1,0)=|2-1|/max(2,1)=1/2, (2,0)=|3-1|/max(3,1)=2/3, (2,1)=|3-2|/max(3,2)=1/3
        expected_l1rel_3 = np.array([0.5, 2.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(
            np.allclose(adaptivity_l1rel._similarity_dists, expected_l1rel_3)
        )

        adaptivity_l2rel = AdaptivityCalculator(
            configurator,
            nsims=nsims_3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        configurator.get_adaptivity_similarity_measure.return_value = "L2rel"
        adaptivity_l2rel._similarity_dists = np.zeros(nsims_3 * (nsims_3 - 1) // 2)
        adaptivity_l2rel._l2rel(fake_data, dt=1.0)
        # L2rel of scalars is same as L1rel for 1D data
        expected_l2rel_3 = np.array([0.5, 2.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(
            np.allclose(adaptivity_l2rel._similarity_dists, expected_l2rel_3)
        )

        # Test with 2-element 2D data
        nsims_2 = 2
        configurator.get_adaptivity_similarity_measure.return_value = "L1"
        adaptivity_l1_2d = AdaptivityCalculator(
            configurator,
            nsims=nsims_2,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        fake_2d_data = np.array([[1, 2], [3, 4]])
        adaptivity_l1_2d._similarity_dists = np.zeros(nsims_2 * (nsims_2 - 1) // 2)
        adaptivity_l1_2d._l1(fake_2d_data, dt=1.0)
        # Expected: (1,0)=|3-1|+|4-2|=2+2=4
        expected_l1_2d = np.array([4.0])
        self.assertTrue(np.allclose(adaptivity_l1_2d._similarity_dists, expected_l1_2d))

        configurator.get_adaptivity_similarity_measure.return_value = "L2"
        adaptivity_l2_2d = AdaptivityCalculator(
            configurator,
            nsims=nsims_2,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_l2_2d._similarity_dists = np.zeros(nsims_2 * (nsims_2 - 1) // 2)
        adaptivity_l2_2d._l2(fake_2d_data, dt=1.0)
        # Expected: (1,0)=sqrt((3-1)^2+(4-2)^2)=sqrt(8)=2*sqrt(2)
        expected_l2_2d = np.array([np.sqrt(8.0)])
        self.assertTrue(np.allclose(adaptivity_l2_2d._similarity_dists, expected_l2_2d))

        configurator.get_adaptivity_similarity_measure.return_value = "L1rel"
        adaptivity_l1rel_2d = AdaptivityCalculator(
            configurator,
            nsims=nsims_2,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_l1rel_2d._similarity_dists = np.zeros(nsims_2 * (nsims_2 - 1) // 2)
        adaptivity_l1rel_2d._l1rel(fake_2d_data, dt=1.0)
        # Expected: (1,0)=|3-1|/max(1,3)+|4-2|/max(2,4)=2/3+2/4=2/3+1/2
        expected_l1rel_2d = np.array([2.0 / 3.0 + 1.0 / 2.0])
        self.assertTrue(
            np.allclose(adaptivity_l1rel_2d._similarity_dists, expected_l1rel_2d)
        )

        configurator.get_adaptivity_similarity_measure.return_value = "L2rel"
        adaptivity_l2rel_2d = AdaptivityCalculator(
            configurator,
            nsims=nsims_2,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )
        adaptivity_l2rel_2d._similarity_dists = np.zeros(nsims_2 * (nsims_2 - 1) // 2)
        adaptivity_l2rel_2d._l2rel(fake_2d_data, dt=1.0)
        # Expected: (1,0)=sqrt((3-1)^2/max(1,3)^2+(4-2)^2/max(2,4)^2)=sqrt(4/9+4/16)
        expected_l2rel_2d = np.array([np.sqrt((2.0 / 3.0) ** 2 + (2.0 / 4.0) ** 2)])
        self.assertTrue(
            np.allclose(adaptivity_l2rel_2d._similarity_dists, expected_l2rel_2d)
        )

    def test_adaptivity_norms_with_zeros_no_warning(self):
        """
        Test that L1rel/L2rel must not raise division-by-zero
        warning when data contains zeros.
        """
        import warnings

        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L2rel")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )
        adaptivity_l2rel = AdaptivityCalculator(
            configurator,
            nsims=3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        configurator_l1 = MagicMock()
        configurator_l1.get_adaptivity_similarity_measure = MagicMock(
            return_value="L1rel"
        )
        configurator_l1.get_output_dir = MagicMock(return_value="output_dir")
        configurator_l1.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )
        adaptivity_l1rel = AdaptivityCalculator(
            configurator_l1,
            nsims=3,
            base_logger=MagicMock(),
            rank=0,
            micro_problem_cls=MicroSimulation,
            model_manager=ModelManager(),
        )

        # Data with zeros - previously triggered RuntimeWarning: invalid value in true_divide
        data_with_zeros = np.array([[0.0], [0.0], [1.0]])
        # Initialize vectors for testing
        nsims_zero_test = 3
        zero_vec_size = nsims_zero_test * (nsims_zero_test - 1) // 2

        # Test L2rel with zero data
        adaptivity_l2rel._similarity_dists = np.zeros(zero_vec_size)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error", RuntimeWarning)
            adaptivity_l2rel._l2rel(data_with_zeros, dt=1.0)
        self.assertEqual(
            len(w), 0, "L2rel must not raise RuntimeWarning with zero data"
        )
        # When both are 0, relative diff should be 0 (since numerator is 0)
        # (0,0) pair is at index 0
        self.assertEqual(adaptivity_l2rel._similarity_dists[0], 0.0)

        # Test L1rel with zero data
        adaptivity_l1rel._similarity_dists = np.zeros(zero_vec_size)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error", RuntimeWarning)
            adaptivity_l1rel._l1rel(data_with_zeros, dt=1.0)
        self.assertEqual(
            len(w), 0, "L1rel must not raise RuntimeWarning with zero data"
        )
        # When both are 0, relative diff should be 0 (since numerator is 0)
        # (0,0) pair is at index 0
        self.assertEqual(adaptivity_l1rel._similarity_dists[0], 0.0)

    def test_associate_active_to_inactive(self):
        """
        Test functionality to associate inactive sims to active ones, in the class AdaptivityCalculator.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )

        adaptivity_controller = AdaptivityCalculator(
            configurator,
            nsims=self._number_of_sims,
            base_logger=MagicMock(),
            rank=0,
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
        adaptivity_controller._max_similarity_dist = (
            np.amax(self._similarity_dists) if len(self._similarity_dists) > 0 else 0.0
        )

        adaptivity_controller._is_sim_active = np.array(
            [True, False, False, True, False]
        )
        expected_sim_is_associated_to = np.array([-2, 0, 0, -2, 3])

        adaptivity_controller._associate_inactive_to_active()

        self.assertTrue(
            np.array_equal(
                expected_sim_is_associated_to,
                adaptivity_controller._sim_is_associated_to,
            )
        )

    def test_update_inactive_sims_local_adaptivity(self):
        """
        Test functionality to update inactive simulations in a particular setting, for a local adaptivity setting.
        """
        configurator = MagicMock()
        configurator.get_adaptivity_similarity_measure = MagicMock(return_value="L1")
        configurator.get_output_dir = MagicMock(return_value="output_dir")
        configurator.get_micro_file_name = MagicMock(
            return_value="test_adaptivity_serial"
        )

        adaptivity_controller = LocalAdaptivityCalculator(
            configurator,
            self._number_of_sims,
            base_logger=MagicMock(),
            rank=0,
            comm=MPI.COMM_WORLD,
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
        expected_sim_is_associated_to = np.array([-2, 0, 0, -2, 3])

        adaptivity_controller._similarity_dists = self._similarity_dists
        adaptivity_controller._max_similarity_dist = (
            np.amax(self._similarity_dists) if len(self._similarity_dists) > 0 else 0.0
        )
        adaptivity_controller._is_sim_active = np.array(
            [True, False, False, False, False]
        )
        adaptivity_controller._sim_is_associated_to = np.array([-2, 0, 0, 0, 3])

        dummy_micro_sims = []
        for i in range(self._number_of_sims):
            dummy_micro_sims.append(MicroSimulation(i))

        adaptivity_controller._update_inactive_sims(dummy_micro_sims)

        self.assertTrue(
            np.array_equal(expected_is_sim_active, adaptivity_controller._is_sim_active)
        )
        self.assertTrue(
            np.array_equal(
                expected_sim_is_associated_to,
                adaptivity_controller._sim_is_associated_to,
            )
        )

    def _convert_2d_to_1d(self, matrix_2d: np.ndarray) -> np.ndarray:
        """
        Convert a 2D symmetric matrix to 1D strictly lower triangular vector.
        """
        n = matrix_2d.shape[0]
        vec_size = n * (n - 1) // 2
        vector_1d = np.zeros(vec_size)

        idx = 0
        for i in range(1, n):
            for j in range(i):
                vector_1d[idx] = matrix_2d[i, j]
                idx += 1

        return vector_1d

    def _convert_1d_to_2d(self, vector_1d: np.ndarray, n: int) -> np.ndarray:
        """
        Convert a 1D strictly lower triangular vector back to 2D symmetric matrix.
        """
        matrix_2d = np.zeros((n, n))

        idx = 0
        for i in range(1, n):
            for j in range(i):
                matrix_2d[i, j] = vector_1d[idx]
                matrix_2d[j, i] = vector_1d[idx]
                idx += 1

        return matrix_2d
