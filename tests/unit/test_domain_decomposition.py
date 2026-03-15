from unittest import TestCase
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

    def test_rank2_out_of_4_2d(self):
        """
        Check bounds for rank 2 in a setting of axis-wise ranks: [2, 2]
        """
        rank = 2
        size = 4
        ranks_per_axis = [2, 2]
        domain_decomposer = DomainDecomposer(rank, size)
        domain_decomposer._dims = 2
        mesh_bounds = domain_decomposer.get_local_mesh_bounds(
            self._macro_bounds_2d, ranks_per_axis
        )

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, 1, 2]))

    def test_rank1_out_of_4_3d(self):
        """
        Check bounds for rank 1 in a setting of axis-wise ranks: [2, 2, 1]
        """
        rank = 1
        size = 4
        ranks_per_axis = [2, 2, 1]
        domain_decomposer = DomainDecomposer(rank, size)
        domain_decomposer._dims = 3
        mesh_bounds = domain_decomposer.get_local_mesh_bounds(
            self._macro_bounds_3d, ranks_per_axis
        )

        self.assertTrue(np.allclose(mesh_bounds, [0.0, 1, -2, 0.0, -2, 8]))

    def test_rank5_outof_10_3d(self):
        """
        Test domain decomposition for rank 5 in a setting of axis-wise ranks: [1, 2, 5]
        """
        rank = 5
        size = 10
        ranks_per_axis = [1, 2, 5]
        domain_decomposer = DomainDecomposer(rank, size)
        domain_decomposer._dims = 3
        mesh_bounds = domain_decomposer.get_local_mesh_bounds(
            self._macro_bounds_3d, ranks_per_axis
        )

        self.assertTrue(np.allclose(mesh_bounds, [-1, 1, 0, 2, 2, 4]))

    def test_rank10_out_of_32_3d(self):
        """
        Test domain decomposition for rank 10 in a setting of axis-wise ranks: [4, 1, 8]
        """
        rank = 10
        size = 32
        ranks_per_axis = [4, 1, 8]
        domain_decomposer = DomainDecomposer(rank, size)
        domain_decomposer._dims = 3
        mesh_bounds = domain_decomposer.get_local_mesh_bounds(
            self._macro_bounds_3d, ranks_per_axis
        )

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, -2, 2, 0.5, 1.75]))

    def test_rank7_out_of_16_3d(self):
        """
        Test domain decomposition for rank 7 in a setting of axis-wise ranks: [8, 2, 1]
        """
        rank = 7
        size = 16
        ranks_per_axis = [8, 2, 1]
        domain_decomposer = DomainDecomposer(rank, size)
        domain_decomposer._dims = 3
        mesh_bounds = domain_decomposer.get_local_mesh_bounds(
            self._macro_bounds_3d, ranks_per_axis
        )

        self.assertTrue(np.allclose(mesh_bounds, [0.75, 1, -2, 0, -2, 8]))


class TestDuplicateCoordFiltering(TestCase):
    """
    Test that duplicate vertex coordinates returned by preCICE on rank boundaries
    are correctly filtered, with lower-ranked ranks taking ownership.
    """

    def test_no_duplicates(self):
        """
        If there are no shared coords across ranks, nothing should be filtered.
        """
        all_coords = [
            np.array([[0.0, 0.0], [0.5, 0.0]]),
            np.array([[1.0, 0.0], [1.5, 0.0]]),
        ]
        all_ids = [[0, 1], [2, 3]]

        coords, ids = DomainDecomposer(0, 2).filter_duplicate_coords(
            all_coords, all_ids
        )
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
        coords0, ids0 = DomainDecomposer(0, 2).filter_duplicate_coords(
            all_coords, all_ids
        )
        self.assertEqual(len(coords0), 2)
        self.assertTrue(np.allclose(coords0[1], shared))

        # Rank 1 should drop the shared coord
        coords1, ids1 = DomainDecomposer(1, 2).filter_duplicate_coords(
            all_coords, all_ids
        )
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

        coords0, _ = DomainDecomposer(0, 3).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords0), 2)

        coords1, _ = DomainDecomposer(1, 3).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords1), 1)
        self.assertTrue(np.allclose(coords1[0], [1.0, 0.0]))

        coords2, _ = DomainDecomposer(2, 3).filter_duplicate_coords(all_coords, all_ids)
        self.assertEqual(len(coords2), 1)
        self.assertTrue(np.allclose(coords2[0], [2.0, 0.0]))
