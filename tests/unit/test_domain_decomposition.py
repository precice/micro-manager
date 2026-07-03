from unittest import TestCase, skip
from unittest.mock import MagicMock
import numpy as np
from micro_manager.domain_decomposition import (
    DomainDecomposer,
    NonUniformGridDecomp,
    UniformGridDecomp,
)


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
        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_2d
        )
        self._configuration_mock.ranks_per_axis.return_value = [2, 2]

        mpi = MagicMock()
        mpi.rank = 2
        mpi.size = 4
        domain_decomposer = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, 1, 2]))

    def test_rank1_out_of_4_3d(self):
        """
        Check bounds for rank 1 in a setting of axis-wise ranks: [2, 2, 1]
        """
        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.ranks_per_axis.return_value = [2, 2, 1]

        mpi = MagicMock()
        mpi.rank = 1
        mpi.size = 4
        domain_decomposer = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0.0, 1, -2, 0.0, -2, 8]))

    def test_rank5_outof_10_3d(self):
        """
        Test domain decomposition for rank 5 in a setting of axis-wise ranks: [1, 2, 5]
        """
        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.ranks_per_axis.return_value = [1, 2, 5]

        mpi = MagicMock()
        mpi.rank = 5
        mpi.size = 10
        domain_decomposer = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [-1, 1, 0, 2, 2, 4]))

    def test_rank10_out_of_32_3d(self):
        """
        Test domain decomposition for rank 10 in a setting of axis-wise ranks: [4, 1, 8]
        """
        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.ranks_per_axis.return_value = [4, 1, 8]

        mpi = MagicMock()
        mpi.rank = 10
        mpi.size = 32
        domain_decomposer = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0, 0.5, -2, 2, 0.5, 1.75]))

    def test_rank7_out_of_16_3d(self):
        """
        Test domain decomposition for rank 7 in a setting of axis-wise ranks: [8, 2, 1]
        """
        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.ranks_per_axis.return_value = [8, 2, 1]

        mpi = MagicMock()
        mpi.rank = 7
        mpi.size = 16
        domain_decomposer = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

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
        self._configuration_mock.decomposition_type.return_value = "nonuniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_2d
        )
        self._configuration_mock.ranks_per_axis.return_value = [2, 2]
        self._configuration_mock.minimum_access_region_size.return_value = []

        mpi = MagicMock()
        mpi.rank = 2
        mpi.size = 4
        domain_decomposer = NonUniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(np.allclose(mesh_bounds, [0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0]))

    def test_nonuniform_rank15_out_of_128_2d(self):
        """
        Check non-uniform bounds for rank 15 in a setting of axis-wise ranks: [16, 8].
        Along each axis, the local width doubles from one rank to the next.
        """
        self._configuration_mock.decomposition_type.return_value = "nonuniform"
        self._configuration_mock.macro_domain_bounds.return_value = [0, 1, 0, 0.5]
        self._configuration_mock.ranks_per_axis.return_value = [16, 8]
        self._configuration_mock.minimum_access_region_size.return_value = [
            1.0 / 256.0,
            1.0 / 128.0,
        ]

        # In a 16 x 8 grid, rank 15 is in the lower right corner.
        mpi = MagicMock()
        mpi.rank = 15
        mpi.size = 128
        domain_decomposer = NonUniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(
            np.allclose(mesh_bounds, [0.756153, 1.0, 0.0, 0.019664], atol=1e-5)
        )

        # In a 16 x 8 grid, rank 112 is in the lower right corner.
        mpi = MagicMock()
        mpi.rank = 112
        mpi.size = 128
        domain_decomposer = NonUniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(
            np.allclose(mesh_bounds, [0.0, 1.0 / 256.0, 0.364631, 0.5], atol=1e-5)
        )

    def test_nonuniform_rank1_out_of_4_3d(self):
        """
        Check non-uniform bounds for rank 1 in a setting of axis-wise ranks: [2, 2, 1].
        """
        self._configuration_mock.decomposition_type.return_value = "nonuniform"
        self._configuration_mock.macro_domain_bounds.return_value = (
            self._macro_bounds_3d
        )
        self._configuration_mock.ranks_per_axis.return_value = [2, 2, 1]
        self._configuration_mock.minimum_access_region_size.return_value = []

        mpi = MagicMock()
        mpi.rank = 1
        mpi.size = 4
        domain_decomposer = NonUniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        )
        mesh_bounds = domain_decomposer.get_mesh_bounds()

        self.assertTrue(
            np.allclose(mesh_bounds, [-1.0 / 3.0, 1.0, -2.0, -2.0 / 3.0, -2.0, 8.0])
        )

    def test_nonuniform_invalid_processor_count_raises(self):
        """
        A mismatch between `ranks_per_axis` and communicator size should raise a ValueError.
        """
        self._configuration_mock.decomposition_type.return_value = "nonuniform"

        mpi = MagicMock()
        mpi.rank = 0
        mpi.size = 4

        with self.assertRaises(ValueError):
            domain_decomposer = NonUniformGridDecomp(
                self._configuration_mock, mpi, MagicMock()
            )
            domain_decomposer.get_mesh_bounds()


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

        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.ranks_per_axis.return_value = [2, 1]
        self._configuration_mock.macro_domain_bounds.return_value = [-2, -2, 2, 2]

        mpi = MagicMock()
        mpi.rank = 0
        mpi.size = 2
        coords, ids = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
        self.assertEqual(len(coords), 2)

        mpi = MagicMock()
        mpi.rank = 1
        mpi.size = 2
        coords, ids = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
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

        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.ranks_per_axis.return_value = [2, 1]
        self._configuration_mock.macro_domain_bounds.return_value = [-2, -2, 2, 2]

        # Rank 0 should keep both its coords
        mpi = MagicMock()
        mpi.rank = 0
        mpi.size = 2
        coords0, ids0 = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
        self.assertEqual(len(coords0), 2)
        self.assertTrue(np.allclose(coords0[1], shared))

        # Rank 1 should drop the shared coord
        mpi = MagicMock()
        mpi.rank = 1
        mpi.size = 2
        coords1, ids1 = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
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

        self._configuration_mock.decomposition_type.return_value = "uniform"
        self._configuration_mock.ranks_per_axis.return_value = [3, 1]
        self._configuration_mock.macro_domain_bounds.return_value = [-3, -3, 3, 3]

        mpi = MagicMock()
        mpi.rank = 0
        mpi.size = 3
        coords0, _ = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
        self.assertEqual(len(coords0), 2)

        mpi = MagicMock()
        mpi.rank = 1
        mpi.size = 3
        coords1, _ = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
        self.assertEqual(len(coords1), 1)
        self.assertTrue(np.allclose(coords1[0], [1.0, 0.0]))

        mpi = MagicMock()
        mpi.rank = 2
        mpi.size = 3
        coords2, _ = UniformGridDecomp(
            self._configuration_mock, mpi, MagicMock()
        ).filter_duplicates(all_coords, all_ids)
        self.assertEqual(len(coords2), 1)
        self.assertTrue(np.allclose(coords2[0], [2.0, 0.0]))
