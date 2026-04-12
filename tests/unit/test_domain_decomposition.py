from unittest import TestCase
from unittest.mock import MagicMock
import numpy as np
from micro_manager.domain_decomposition import DomainDecomposer


class TestDomainDecomposition(TestCase):
    def setUp(self) -> None:
        self._macro_bounds_3d = [
            -1,
            1,
            -2,
            2,
            -2,
            8,
        ]  # Cuboid which is not symmetric around origin

        self._macro_bounds_2d = [
            0,
            1,
            0,
            2,
        ]

        self._configuration_mock = MagicMock()

    def test_rank2_out_of_4_2d(self):
        """
        Check bounds for rank 2 in a setting of axis-wise ranks: [2, 2]
        """
        self._configuration_mock.get_decomposition_type.return_value = "uniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_2d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [2, 2]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=2, size=4)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, 1, 2]))

    def test_rank1_out_of_4_3d(self):
        """
        Check bounds for rank 1 in a setting of axis-wise ranks: [2, 2, 1]
        """
        self._configuration_mock.get_decomposition_type.return_value = "uniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [2, 2, 1]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=1, size=4)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0.0, 1, -2, 0.0, -2, 8]))

    def test_rank5_outof_10_3d(self):
        """
        Test domain decomposition for rank 5 in a setting of axis-wise ranks: [1, 2, 5]
        """
        self._configuration_mock.get_decomposition_type.return_value = "uniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [1, 2, 5]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=5, size=10)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [-1, 1, 0, 2, 2, 4]))

    def test_rank10_out_of_32_3d(self):
        """
        Test domain decomposition for rank 10 in a setting of axis-wise ranks: [4, 1, 8]
        """
        self._configuration_mock.get_decomposition_type.return_value = "uniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [4, 1, 8]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=10, size=32)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, -2, 2, 0.5, 1.75]))

    def test_rank7_out_of_16_3d(self):
        """
        Test domain decomposition for rank 7 in a setting of axis-wise ranks: [8, 2, 1]
        """
        self._configuration_mock.get_decomposition_type.return_value = "uniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [8, 2, 1]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=7, size=16)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0.75, 1, -2, 0, -2, 8]))


class TestNonUniformDomainDecomposition(TestCase):
    def setUp(self) -> None:
        self._macro_bounds_3d = [
            -1,
            1,
            -2,
            2,
            -2,
            8,
        ]
        self._macro_bounds_2d = [
            0,
            1,
            0,
            2,
        ]

        self._configuration_mock = MagicMock()

    def test_nonuniform_rank2_out_of_4_2d(self):
        """
        Check non-uniform bounds for rank 2 in a setting of axis-wise ranks: [2, 2].
        Along each axis, the local width doubles from one rank to the next.
        """
        self._configuration_mock.get_decomposition_type.return_value = "nonuniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_2d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [2, 2]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=2, size=4)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0]))

    def test_nonuniform_rank1_out_of_4_3d(self):
        """
        Check non-uniform bounds for rank 1 in a setting of axis-wise ranks: [2, 2, 1].
        """
        self._configuration_mock.get_decomposition_type.return_value = "nonuniform"
        self._configuration_mock.get_macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.get_ranks_per_axis.return_value = [2, 2, 1]

        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=1, size=4)
        mesh_bounds = domain_decomposer.get_local_mesh_bounds()

        self.assertTrue(
            np.allclose(mesh_bounds, [-1.0 / 3.0, 1.0, -2.0, -2.0 / 3.0, -2.0, 8.0])
        )

    def test_nonuniform_invalid_processor_count_raises(self):
        """
        A mismatch between `ranks_per_axis` and communicator size should raise a ValueError.
        """
        domain_decomposer = DomainDecomposer(self._configuration_mock, rank=0, size=4)

        with self.assertRaises(ValueError):
            domain_decomposer.get_nonuniform_local_mesh_bounds(
                self._macro_bounds_2d, [3, 2]
            )


class TestDuplicateCoordFiltering(TestCase):
    """
    Test that duplicate vertex coordinates returned by preCICE on rank boundaries
    are correctly filtered, with lower-ranked ranks taking ownership.
    """

    def setUp(self) -> None:
        self._configuration_mock = MagicMock()

    def test_no_duplicates(self):
        """
        If there are no shared coords across ranks, nothing should be filtered.
        """
        all_coords = [
            np.array([[0.0, 0.0], [0.5, 0.0]]),
            np.array([[1.0, 0.0], [1.5, 0.0]]),
        ]
        all_ids = [[0, 1], [2, 3]]

        coords, ids = DomainDecomposer(
            self._configuration_mock, rank=0, size=2
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords), 2)

        coords, ids = DomainDecomposer(1, 2).filter_duplicate_coords(
            all_coords, all_ids
        )
        self.assertEqual(len(coords), 2)

    def test_duplicate_on_boundary_rank0_keeps(self):
        """
        A coord shared between rank 0 and rank 1 should be kept by rank 0
        and dropped by rank 1.
        """
        shared = [0.5, 0.0]
        all_coords = [
            np.array([[0.0, 0.0], shared]),
            np.array([shared, [1.0, 0.0]]),
        ]
        all_ids = [[0, 1], [1, 2]]

        # Rank 0 should keep both its coords
        coords0, ids0 = DomainDecomposer(
            self._configuration_mock, rank=0, size=2
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords0), 2)
        self.assertTrue(np.allclose(coords0[1], shared))

        # Rank 1 should drop the shared coord
        coords1, ids1 = DomainDecomposer(
            self._configuration_mock, rank=1, size=2
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords1), 1)
        self.assertTrue(np.allclose(coords1[0], [1.0, 0.0]))

    def test_duplicate_on_boundary_three_ranks(self):
        """
        A coord shared across three ranks should only be kept by rank 0.
        """
        shared = [0.5, 0.5]
        all_coords = [
            np.array([shared, [0.0, 0.0]]),
            np.array([shared, [1.0, 0.0]]),
            np.array([shared, [2.0, 0.0]]),
        ]
        all_ids = [[0, 1], [0, 2], [0, 3]]

        coords0, _ = DomainDecomposer(
            self._configuration_mock, rank=0, size=3
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords0), 2)

        coords1, _ = DomainDecomposer(
            self._configuration_mock, rank=1, size=3
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords1), 1)
        self.assertTrue(np.allclose(coords1[0], [1.0, 0.0]))

        coords2, _ = DomainDecomposer(
            self._configuration_mock, rank=2, size=3
        ).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords2), 1)
        self.assertTrue(np.allclose(coords2[0], [2.0, 0.0]))
