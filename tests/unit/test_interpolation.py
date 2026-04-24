import numpy as np
import unittest
from unittest import TestCase
from unittest.mock import MagicMock
from micro_manager.interpolation import (
    Interpolation,
    NDtree,
    HilbertDirect,
    Projector,
    STDProjector,
    IdentityProjector,
    InterleavedDomain,
    RBF_PU
)
from mpi4py import MPI


class TestInterpolation(TestCase):
    def test_local_interpolation(self):
        """
        Test if local interpolation works as expected.
        """
        coords = [[-2, 0, 0], [-1, 0, 0], [2, 0, 0]]
        inter_point = [1, 0, 0]
        vector_data = [[-2, -2, -2], [-1, -1, -1], [2, 2, 2]]
        scalar_data = [[-2], [-1], [2]]
        expected_vector_interpolation_output = [55 / 49, 55 / 49, 55 / 49]
        expected_scalar_interpolation_output = 55 / 49

        interpolation = Interpolation(MagicMock())
        interpolated_vector_data = interpolation.interpolate(
            coords, inter_point, vector_data
        )
        interpolated_scalar_data = interpolation.interpolate(
            coords, inter_point, scalar_data
        )
        self.assertTrue(
            np.allclose(interpolated_vector_data, expected_vector_interpolation_output)
        )
        self.assertAlmostEqual(
            interpolated_scalar_data, expected_scalar_interpolation_output
        )

    def test_nearest_neighbor(self):
        """
        Test if finding nearest neighbor works as expected if interpolation point
        itself is not part of neighbor coordinates.
        Note: running this test requires the sci-kit learn package to be installed.
        """
        neighbors = [[0, 2, 0], [0, 3, 0], [0, 0, 4], [-5, 0, 0], [0, 0, 0]]
        inter_coord = [0, 0, 0]
        expected_nearest_neighbor_index = [4, 0, 1]
        k = 3

        interpolation = Interpolation(MagicMock())
        nearest_neighbor_index = interpolation.get_nearest_neighbor_indices(
            neighbors, inter_coord, k
        )
        self.assertListEqual(
            nearest_neighbor_index.tolist(), expected_nearest_neighbor_index
        )

    def test_interpolation_exact_point(self):
        """
        Test that if interpolation point exactly matches a neighbor, that value is returned.
        """
        coords = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        point = [1, 0, 0]
        values = [10, 20, 30]

        interpolation = Interpolation(MagicMock())
        result = interpolation.interpolate(coords, point, values)
        self.assertEqual(result, 20)

    def test_nearest_neighbor_k_larger_than_coords(self):
        """
        Test that k is reset when larger than number of available neighbors.
        """
        coords = [[0, 0, 0], [1, 0, 0]]
        inter_point = [0.5, 0, 0]
        k = 5  # larger than len(coords)

        mock_logger = MagicMock()
        interpolation = Interpolation(mock_logger)
        indices = interpolation.get_nearest_neighbor_indices(coords, inter_point, k)
        self.assertEqual(len(indices), 2)
        mock_logger.log_info.assert_called_once()


class TestNDtree(TestCase):
    def test_node_properties(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 2, 2, np.ones(2))
            self.assertEqual(node.dim, 2)
            self.assertEqual(node.num_max_split, 4)
            self.assertEqual(node.filling, 0)
            node.data.append(0)
            self.assertEqual(node.filling, 1)

    def test_node_clear(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 2, 2, np.ones(2))
            node.data.append(0)
            node.data_reserve_count = 1
            child = NDtree.Node(mode, -np.ones(2), np.ones(2), 1, 2, np.ones(2))
            child.data.append(0)
            child.data_reserve_count = 1
            node.children = [child]

            node.clear()
            self.assertEqual(len(node.data), 0)
            self.assertEqual(node.data_reserve_count, 0)
            self.assertEqual(len(node.data), 0)
            self.assertEqual(child.data_reserve_count, 0)

    def test_node_propagate_reserve_count(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 2, 2, np.ones(2))
            node.data_reserve_count = 1
            self.assertEqual(node.propagate_up_reserve_counts(), 1)
            child = NDtree.Node(mode, -np.ones(2), np.ones(2), 1, 2, np.ones(2))
            child.data_reserve_count = 1
            node.children = [child, child]
            self.assertEqual(node.propagate_up_reserve_counts(), 3)
            self.assertEqual(node.data_reserve_count, 3)

    def test_node_find_min_depth(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
            node.children = [
                NDtree.Node(mode, np.array([-1, -1]), np.array([0, 0]), 1, 4, np.array([0, 0])),
                NDtree.Node(mode, np.array([ 0, -1]), np.array([1, 0]), 1, 4, np.array([1, 0])),
                NDtree.Node(mode, np.array([-1,  0]), np.array([0, 1]), 1, 4, np.array([0, 1])),
                NDtree.Node(mode, np.array([ 0,  0]), np.array([1, 1]), 1, 4, np.array([1, 1])),
            ]
            node.children[0].children = [
                NDtree.Node(mode, np.array([  -1,   -1]), np.array([-0.5, -0.5]), 0, 4, np.zeros(2)),
                NDtree.Node(mode, np.array([-0.5,   -1]), np.array([   0, -0.5]), 0, 4, np.zeros(2)),
                NDtree.Node(mode, np.array([  -1, -0.5]), np.array([-0.5,    0]), 0, 4, np.zeros(2)),
                NDtree.Node(mode, np.array([-0.5, -0.5]), np.array([   0,    0]), 0, 4, np.zeros(2)),
            ]
            node.children[1].children = [
                NDtree.Node(mode, np.array([   0,   -1]), np.array([ 0.5, -0.5]), 0, 4, np.array([0, 0])),
                NDtree.Node(mode, np.array([ 0.5,   -1]), np.array([   1, -0.5]), 0, 4, np.array([1, 0])),
                NDtree.Node(mode, np.array([   0, -0.5]), np.array([ 0.5,    0]), 0, 4, np.array([0, 0])),
                NDtree.Node(mode, np.array([ 0.5, -0.5]), np.array([   1,    0]), 0, 4, np.array([1, 0])),
            ]
            node.children[0].children[0].data_reserve_count = 1
            node.children[0].children[1].data_reserve_count = 2
            node.children[0].children[2].data_reserve_count = 2
            node.children[0].children[3].data_reserve_count = 1
            node.children[1].children[0].data_reserve_count = 3
            node.children[1].children[1].data_reserve_count = 3
            node.children[1].children[2].data_reserve_count = 4
            node.children[1].children[3].data_reserve_count = 4
            node.propagate_up_reserve_counts()

            self.assertEqual(node.find_min_depth_for_n_neighbors(3, 0, np.array([-1, -1])), 1)
            self.assertEqual(node.find_min_depth_for_n_neighbors(3, 0, np.array([1, -1])), 2)

    def test_node_filled_coords(self):
        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.insert(np.array([-0.5, -0.5]))
        node.insert(np.array([0.5, 0.5]))
        node.insert(np.array([0.5, 0.5]))
        for c in node.children:
            c.data_reserve_count = len(c.data)

        coords = node.get_filled_coords(np.array([0, 0]), np.array([2, 2]))
        true_targets = np.array([[0, 0], [1, 1], [1, 1]])
        for i in range(3):
            self.assertTrue(np.all(true_targets[i] == coords[i]))

    def test_node_split(self):
        # DISC
        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.split()
        self.assertTrue(node.children is not None)
        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.insert(np.array([-0.5, -0.5]))
        c_list = node.children
        node.split()
        self.assertTrue(c_list == node.children)

        # IND
        node = NDtree.Node(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.insert(np.array([-0.5, -0.5]))
        node.insert(np.array([-0.5, -0.5]))
        node.insert(np.array([ 0.5,  0.5]))
        node.split()
        self.assertEqual(len(node.children[0].data), 2)
        self.assertEqual(len(node.children[3].data), 1)

    def test_node_insert(self):
        # DISC
        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        node.insert(np.array([-1, -1]))
        node.insert(np.array([1, 1]))
        self.assertEqual(len(node.children[0].children[0].data), 1)
        self.assertEqual(len(node.children[3].children[3].data), 1)

        # IND
        node = NDtree.Node(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.insert(np.array([-0.5, -0.5]))
        node.insert(np.array([-0.5, -0.5]))
        node.insert(np.array([0.5, 0.5]))
        node.insert(np.array([0.5, 0.5]))
        self.assertEqual(len(node.data), 4)
        node.insert(np.array([0.5, 0.5]))
        self.assertTrue(node.children is not None)
        self.assertEqual(len(node.children[0].data), 2)
        self.assertEqual(len(node.children[3].data), 3)

    def test_node_get_coord(self):
        node = NDtree.Node(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        self.assertRaises(AssertionError, lambda: node.get_coord_of(0, 0, 0))

        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.split()
        self.assertTrue(np.all(node.get_coord_of(np.array([-0.5, -0.5]), np.array([0, 0]), np.array([2, 2])) == np.array([0, 0])))
        self.assertTrue(np.all(node.get_coord_of(np.array([ 0.5, -0.5]), np.array([0, 0]), np.array([2, 2])) == np.array([1, 0])))
        self.assertTrue(np.all(node.get_coord_of(np.array([-0.5,  0.5]), np.array([0, 0]), np.array([2, 2])) == np.array([0, 1])))
        self.assertTrue(np.all(node.get_coord_of(np.array([ 0.5,  0.5]), np.array([0, 0]), np.array([2, 2])) == np.array([1, 1])))

    def test_node_within(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 0, 4, np.ones(2))
            self.assertTrue(node.is_within(np.array([-1, -1])))
            self.assertTrue(node.is_within(np.array([ 1, -1])))
            self.assertTrue(node.is_within(np.array([-1,  1])))
            self.assertTrue(node.is_within(np.array([ 1,  1])))

            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 0, 4, np.array([1, 0]))
            self.assertTrue(node.is_within(np.array([-1, -1])))
            self.assertTrue(node.is_within(np.array([ 1, -1])))
            self.assertFalse(node.is_within(np.array([-1, 1])))
            self.assertFalse(node.is_within(np.array([ 1, 1])))

            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 0, 4, np.array([0, 1]))
            self.assertTrue(node.is_within(np.array([-1, -1])))
            self.assertFalse(node.is_within(np.array([ 1, -1])))
            self.assertTrue(node.is_within(np.array([-1, 1])))
            self.assertFalse(node.is_within(np.array([1,  1])))

            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 0, 4, np.array([0, 0]))
            self.assertTrue(node.is_within(np.array([-1, -1])))
            self.assertFalse(node.is_within(np.array([1, -1])))
            self.assertFalse(node.is_within(np.array([-1, 1])))
            self.assertFalse(node.is_within(np.array([1, 1])))
            self.assertTrue(node.is_within(np.array([0, 0])))

    def test_node_height(self):
        # DISC
        node = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        self.assertEqual(node.get_height(), 0)
        node.insert(np.array([-0.5, -0.5]))
        self.assertEqual(node.get_height(), 2)

        # IND
        node = NDtree.Node(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 2, 2, np.ones(2))
        self.assertEqual(node.get_height(), 0)
        node.insert(np.array([-1, -1]))
        self.assertEqual(node.get_height(), 0)
        node.insert(np.array([-1, -1]))
        node.insert(np.array([1, 1]))
        self.assertEqual(node.get_height(), 1)
        node.insert(np.array([-1, -1]))
        self.assertEqual(node.get_height(), 2)

    def test_node_serialize(self):
        for mode in [NDtree.Mode.DISCRETIZE, NDtree.Mode.INDEX]:
            node = NDtree.Node(mode, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
            self.assertListEqual(node.serialize(), [2, 0])
            child = NDtree.Node(mode, -np.ones(2), np.ones(2), 0, 4, np.ones(2))
            node.children = [child, child, child, child]
            self.assertEqual(node.serialize(), [9, 2, 0, 2, 0, 2, 0, 2, 0])

    def test_node_deserialize(self):
        node = NDtree.Node(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 1, 4, np.ones(2))
        node.deserialize([9, 2, 1, 2, 2, 2, 3, 2, 4])
        self.assertTrue(node.children is not None)
        self.assertEqual(node.children[0].data_reserve_count, 1)
        self.assertEqual(node.children[1].data_reserve_count, 2)
        self.assertEqual(node.children[2].data_reserve_count, 3)
        self.assertEqual(node.children[3].data_reserve_count, 4)

    def test_node_merge(self):
        t1 = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        t1.split()
        t1.children[0].split()
        t1.children[0].children[0].data_reserve_count = 2
        t1.children[0].children[1].data_reserve_count = 2
        t1.children[0].children[2].data_reserve_count = 2
        t1.children[0].children[3].data_reserve_count = 2
        t1_total = t1.propagate_up_reserve_counts()

        t2 = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        t2.split()
        t2.children[3].split()
        t2.children[3].children[0].data_reserve_count = 3
        t2.children[3].children[1].data_reserve_count = 3
        t2.children[3].children[2].data_reserve_count = 3
        t2.children[3].children[3].data_reserve_count = 3
        t2_total = t2.propagate_up_reserve_counts()

        expected_total = t1_total + t2_total
        t = NDtree.Node(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 2, 4, np.ones(2))
        t.merge(t1)
        t.merge(t2)
        self.assertTrue(t.children[0].children is not None)
        self.assertTrue(t.children[3].children is not None)
        total = t.propagate_up_reserve_counts()
        self.assertEqual(total, expected_total)

    def test_filled_coords(self):
        t = NDtree(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4)
        t.root.deserialize([9, 2, 1, 2, 0, 2, 0, 2, 2])

        coords = t.get_filled_coords()
        true_targets = np.array([[0, 0], [1, 1], [1, 1]])
        for i in range(3):
            self.assertTrue(np.all(true_targets[i] == coords[i]))

    def test_coords_of(self):
        t = NDtree(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 1, 4)
        t.root.deserialize([9, 2, 0, 2, 0, 2, 0, 2, 0])

        coords = t.get_coords_of(np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]))
        true_targets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        for i in range(4):
            self.assertTrue(np.all(true_targets[i] == coords[i]))

    def test_min_depth(self):
        t = NDtree(NDtree.Mode.INDEX, -np.ones(2), np.ones(2), 2, 4)
        t.root.deserialize([16, 2, 1, 2, 3, 2, 1, 9, 2, 1, 2, 1, 2, 3, 2, 3])
        t.propagate_up_reserve_counts()
        self.assertEqual(t.find_min_depth_for_n_neighbors(4, np.array([[0.5, 0.5]])), 1)
        self.assertEqual(t.find_min_depth_for_n_neighbors(3, np.array([[0.5, 0.5]])), 2)

    def test_disc_insert(self):
        """
        Test if point is inserted at max depth
        """
        tree = NDtree(NDtree.Mode.DISCRETIZE, -np.ones(2), np.ones(2), 3, 4)
        tree.insert(-np.ones(2))
        self.assertTrue(len(tree.root.children[0].children[0].children[0].data) > 0)
        self.assertTrue(np.all(tree.root.children[0].children[0].children[0].data[0] == -np.ones(2)))


class TestHilberDirect(TestCase):
    def test_i2c(self):
        h = HilbertDirect(2, 3)
        n_per_dim = np.power(2, 3)
        n_max = np.power(n_per_dim, 2)
        c_low = np.zeros(2)
        c_high = np.ones(2) * n_per_dim - 1

        for i in range(n_max):
            c = h.index2coord(i)
            self.assertTrue(np.all(c >= c_low) and np.all(c <= c_high))

    def test_c2i(self):
        h = HilbertDirect(2, 3)
        n_per_dim = np.power(2, 3)
        n_max = np.power(n_per_dim, 2) - 1

        for y in range(n_per_dim):
            for x in range(n_per_dim):
                i = h.coord2index(np.array([x, y]))
                self.assertTrue(0 <= i <= n_max)

    def test_unique(self):
        h = HilbertDirect(2, 3)
        n_per_dim = np.power(2, 3)
        n_max = np.power(n_per_dim, 2)

        indices = []
        for y in range(n_per_dim):
            for x in range(n_per_dim):
                indices.append(h.coord2index(np.array([x, y])))
        u = np.unique(np.array(indices))
        self.assertEqual(len(u), n_max)


class TestProjector(TestCase):
    def test_std_proj(self):
        proj : Projector = STDProjector(1, MPI.COMM_SELF)
        data = np.array([
            [0.1, 0],
            [0.2, 5],
            [0.1, 10],
            [0.3, 20],
        ])
        proj.initialize(data)
        self.assertEqual(proj.target_dims[0], 1)
        self.assertListEqual(data[:, 1].tolist(), proj(data).flatten().tolist())

    def test_id_proj(self):
        proj : Projector = IdentityProjector()
        data = np.array([
            [0.1, 0],
            [0.2, 5],
            [0.1, 10],
            [0.3, 20],
        ])
        proj.initialize(data)
        self.assertTrue(np.all(data == proj(data)))


def f_ana(x):
    return 1 + 2 * x[:, 0] + 1 * x[:, 1] + 0.1 * x[:, 2]

rbf_config = {
    "domain_config": {
            "max_filling": 8,
            "coarsening_factor": 2,
            "n_neighbors": 10,
            "projection_type": "std",
            "projection_std_dims": 2,
        },
    "use_pu": False,
    "basis": "c6"
}
# we have 2 clusters, centered around -1/-1/0 and 1/1/0
ordered_global_x = np.array([
    [-1.5, -1.0, -0.1], [-0.5, -1.0, -0.1], [-1.0, -1.5, -0.1], [-1.0, -0.5, -0.1], [-1.0, -1.0, -0.1],
    [-1.5, -1.0,  0.1], [-0.5, -1.0,  0.1], [-1.0, -1.5,  0.1], [-1.0, -0.5,  0.1], [-1.0, -1.0,  0.1],
    [ 1.5,  1.0, -0.1], [ 0.5,  1.0, -0.1], [ 1.0,  1.5, -0.1], [ 1.0,  0.5, -0.1], [ 1.0,  1.0, -0.1],
    [ 1.5,  1.0,  0.1], [ 0.5,  1.0,  0.1], [ 1.0,  1.5,  0.1], [ 1.0,  0.5,  0.1], [ 1.0,  1.0,  0.1],
])
ordered_global_f = f_ana(ordered_global_x)
reordering = np.array([
    6, 17, 7, 12, 5, 16, 4, 1, 0, 8, 9, 13, 18, 14, 19, 15, 3, 11, 2, 10
])
ordered_global_xq = np.array([
    [-1.25, -1.25, 0.0], [-0.75, -1.25, 0.0], [-0.75, -0.75, 0.0], [-1.25, -0.75, 0.0],
    [ 1.25,  1.25, 0.0], [ 0.75,  1.25, 0.0], [ 0.75,  0.75, 0.0], [ 1.25,  0.75, 0.0],
])
reordering_q = np.array([0, 2, 4, 6, 1, 3, 5, 7])


class TestInterleavedDomain(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        config = MagicMock()
        self._domain = InterleavedDomain(config, self._comm)
        self._domain.configure(rbf_config["domain_config"])
        self._ordered_global_x = ordered_global_x
        self._ordered_global_f = ordered_global_f
        self._reordering = reordering
        self._ordered_global_xq = ordered_global_xq
        self._reordering_q = reordering_q


    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_p2p_comm(self):
        send_map = {0: [], 1: []}
        if self._rank == 0:
            send_map[0].extend([0, 1, 2])
            send_map[1].extend([3, 4, 5])
        else:
            send_map[0].extend([6, 7, 8])
            send_map[1].extend([9, 10, 11])

        local_result, inv_map = self._domain._communicate(send_map, True)

        if self._rank == 0:
            self.assertListEqual(sorted(local_result), [0, 1, 2, 6, 7, 8])
            self.assertListEqual(sorted(inv_map[0]), [0, 1, 2])
            self.assertListEqual(sorted(inv_map[1]), [6, 7, 8])
        else:
            self.assertListEqual(sorted(local_result), [3, 4, 5, 9, 10, 11])
            self.assertListEqual(sorted(inv_map[0]), [3, 4, 5])
            self.assertListEqual(sorted(inv_map[1]), [9, 10, 11])

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_normalize(self):
        self._domain.set_local_data(
            self._ordered_global_x[self._reordering][10*self._rank:10*self._rank+10],
            self._ordered_global_xq[self._reordering_q][4*self._rank:4*self._rank+4],
            self._ordered_global_f[self._reordering][10*self._rank:10*self._rank+10],
        )

        self._domain._normalize_x()

        self.assertTrue(np.all(self._domain._x_local >= -1) and np.all(self._domain._x_local <= 1))
        self.assertTrue(np.all(self._domain._x_query_local >= -1) and np.all(self._domain._x_query_local <= 1))
        self.assertTrue(np.all(self._domain._projector.target_dims == np.array([0, 1])))
        self.assertEqual(self._domain._proj_x_local.ndim, 2)
        self.assertEqual(self._domain._proj_x_query_local.ndim, 2)

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_gen_trees(self):
        self._domain.set_local_data(
            self._ordered_global_x[self._reordering][10 * self._rank:10 * self._rank + 10],
            self._ordered_global_xq[self._reordering_q][4 * self._rank:4 * self._rank + 4],
            self._ordered_global_f[self._reordering][10 * self._rank:10 * self._rank + 10],
        )

        self._domain._generate_trees()
        self._domain._tree.propagate_up_reserve_counts()
        c_list = self._domain._tree.root.children
        self.assertTrue(
            c_list[1].data_reserve_count == 0 and
            c_list[2].data_reserve_count == 0 and
            c_list[0].data_reserve_count != 0 and
            c_list[3].data_reserve_count != 0
        )
        self.assertEqual(self._domain._tree.root.data_reserve_count, 20)

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_create_partitions(self):
        self._domain.set_local_data(
            self._ordered_global_x[self._reordering][10 * self._rank:10 * self._rank + 10],
            self._ordered_global_xq[self._reordering_q][4 * self._rank:4 * self._rank + 4],
            self._ordered_global_f[self._reordering][10 * self._rank:10 * self._rank + 10],
        )
        self._domain._generate_trees()
        x, xq, f = self._domain._create_partitions()

        expected_xq = self._ordered_global_xq[4 * self._rank:4 * self._rank + 4] / self._domain._normalization[None, :]
        expected_xq_set = set()
        for i in range(len(expected_xq)):
            expected_xq_set.add(tuple(expected_xq[i].tolist()))
        for i in range(len(xq)):
            self.assertTrue(tuple(xq[i].tolist()) in expected_xq_set)


class TestRBF(TestCase):
    def setUp(self):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()
        self._rbf = RBF_PU(
            MagicMock(),
            MagicMock(),
            self._comm,
            self._rank,
            self._size
        )
        self._rbf.configure(rbf_config)

    @unittest.skipUnless(
        MPI.COMM_WORLD.Get_size() == 2, "This test only works with 2 ranks."
    )
    def test_interpolation(self):
        xq = ordered_global_xq[reordering_q][4 * self._rank:4 * self._rank + 4]
        self._rbf.set_local_data(
            ordered_global_x[reordering][10 * self._rank:10 * self._rank + 10],
            xq,
            ordered_global_f[reordering][10 * self._rank:10 * self._rank + 10],
        )

        fq = self._rbf.interpolate()
        fq_ana = f_ana(xq).reshape(-1, 1)
        norms = np.linalg.norm(fq_ana - fq, ord=2, axis=-1)
        self.assertTrue(np.allclose(norms, 0, rtol=1e-8))
