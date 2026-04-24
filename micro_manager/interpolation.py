from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union
import sys

from mpi4py import MPI
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, item):
            return self

    NearestNeighbors = Dummy

from micro_manager import Config
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.p2p import create_tag

# handle compat issue between np version 1 and 2
if int(np.version.version.split(".")[0]) > 1:
    np.alltrue = np.all


class Interpolation:
    def __init__(self, logger):

        self._logger = logger

    def get_nearest_neighbor_indices(
        self,
        coords: np.ndarray,
        inter_point: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Get local indices of the k nearest neighbors of a point.

        Parameters
        ----------
        coords : list
            List of coordinates of all points.
        inter_point : list | np.ndarray
            Coordinates of the point for which the neighbors are to be found.
        k : int
            Number of neighbors to consider.

        Returns
        ------
        neighbor_indices : np.ndarray
            Local indices of the k nearest neighbors in all local points.
        """
        if len(coords) < k:
            self._logger.log_info(
                "Number of desired neighbors k = {} is larger than the number of available neighbors {}. Resetting k = {}.".format(
                    k, len(coords), len(coords)
                )
            )
            k = len(coords)
        if NearestNeighbors.__name__ != "NearestNeighbors":
            raise RuntimeError("scipy was not imported.")
        neighbors = NearestNeighbors(n_neighbors=k).fit(coords)

        neighbor_indices = neighbors.kneighbors(
            [inter_point], return_distance=False
        ).flatten()

        return neighbor_indices

    def interpolate(self, neighbors: np.ndarray, point: np.ndarray, values):
        r"""
            Interpolate a value at a point using inverse distance weighting. (https://en.wikipedia.org/wiki/Inverse_distance_weighting)
            .. math::
                f(x) = (\sum_{i=1}^{n} \frac{f_i}{\Vert x_i - x \Vert^2}) / (\sum_{j=1}^{n} \frac{1}{\Vert x_j - x \Vert^2})

        Parameters
        ----------
        neighbors : np.ndarray
            Coordinates at which the values are known.
        point : np.ndarray
            Coordinates at which the value is to be interpolated.
        values :
            Values at the known coordinates.

        Returns
        -------
        interpol_val / summed_weights :
            Value at interpolation point.
        """
        interpol_val = 0
        summed_weights = 0
        # Iterate over all neighbors
        for inx in range(len(neighbors)):
            # Compute the squared norm of the difference between interpolation point and neighbor
            norm = np.linalg.norm(np.array(neighbors[inx]) - np.array(point)) ** 2
            # If interpolation point is already part of the data it is returned as the interpolation result
            # This avoids division by zero
            if norm < 1e-16:
                return values[inx]
            # Update interpolation value
            interpol_val += values[inx] / norm
            # Extend normalization factor
            summed_weights += 1 / norm

        return interpol_val / summed_weights


class NDtree:
    """
    This is a spatial data structure to store N-dimensional data points. Can be used either for discretization or
    spatial indexing purposes. Is based on an octtree but ported to N dimensions.
    """

    class Mode(Enum):
        DISCRETIZE = 0
        INDEX = 1

    class Node:
        def __init__(
            self,
            mode: "NDtree.Mode",
            low: np.ndarray,
            high: np.ndarray,
            max_depth: int,
            max_filling: int,
            is_bound: np.ndarray,
        ):
            """
            Constructs an NDtree node.

            Parameters
            ----------
            mode : NDtree.Mode
                The mode of operation.
            low : np.ndarray
                Lower bound of the node.
            high : np.ndarray
                Upper bound of the node.
            max_depth : int
                Remaining maximum depth of the node.
            max_filling : int
                Maximum number of points within the node, until split.
            is_bound : np.ndarray
                Boolean indicating whether the node is on the boundary.
            """
            self._mode: NDtree.Mode = mode
            self.low = low
            self.high = high
            self.max_depth = max_depth
            self.max_filling = max_filling
            self.is_bound = is_bound
            self.children: Optional[list[NDtree.Node]] = None
            self.data = []
            self.data_reserve_count = 0

        @property
        def dim(self) -> int:
            return self.low.shape[0]

        @property
        def num_max_split(self) -> int:
            return 2**self.dim

        @property
        def filling(self) -> int:
            return len(self.data)

        def clear(self) -> None:
            """
            Clears all data, but preserves node structure.
            """
            self.data.clear()
            self.data_reserve_count = 0

            if self.children is None:
                return
            for node in self.children:
                node.clear()

        def propagate_up_reserve_counts(self) -> int:
            """
            Counts the reserve counts of child nodes and returns sum.
            Used during discretization mode when all data points are in leaf nodes at max depth
            to approximate the required depth to find N neighbours.

            Returns
            -------
            reserve_count : int
                sum of child node reserve counts.
            """
            if self.children is None:
                return self.data_reserve_count

            for node in self.children:
                self.data_reserve_count += node.propagate_up_reserve_counts()

            return self.data_reserve_count

        def find_min_depth_for_n_neighbors(
            self, n: int, depth: int, p: np.ndarray
        ) -> Optional[int]:
            """
            Finds the minimum depth required to find N nearest neighbors for the given point.
            Assumes propagate_up_reserve_counts was called.

            Parameters
            ----------
            n : int
                Number of nearest neighbors.
            depth : int
                Recursion depth. Start recursion with 0.
            p : np.ndarray
                Query point.

            Returns
            -------
            min_depth : Optional[int]
                None depth cannot be found, else depth.
            """
            if self.data_reserve_count < n:
                return None
            if not self.is_within(p):
                return None
            if self.children is None:
                return depth

            tmp = [
                node.find_min_depth_for_n_neighbors(n, depth + 1, p)
                for node in self.children
            ]
            depths = []
            for d in tmp:
                if d is None:
                    continue
                depths.append(d)
            if len(depths) == 0:
                return depth

            min_depth = min(depths)
            return min_depth

        def get_filled_coords(
            self, bin_low: np.ndarray, bin_high: np.ndarray
        ) -> list[np.ndarray]:
            """
            Finds coordinates of all cells that have non-zero reserve counts. Assumes discretization mode is used.

            Parameters
            ----------
            bin_low : np.ndarray
                Lower bound of possible bins.
            bin_high : np.ndarray
                Upper bound of possible bins.

            Returns
            -------
            coords : list[np.ndarray]
                Coordinates of all cells that have non-zero reserve counts.
            """
            assert self._mode == NDtree.Mode.DISCRETIZE

            if self.children is None:
                if self.data_reserve_count == 0:
                    return []
                assert np.allclose(bin_high - bin_low, 1)
                return [bin_low] * self.data_reserve_count

            buffer = []
            for i in range(self.num_max_split):
                mask = self._idx2mask(i)
                inv_mask = np.ones_like(mask) - mask
                delta_bin_half = ((bin_high - bin_low) / 2).astype(bin_low.dtype)
                bin_low_i = bin_low + mask * delta_bin_half
                bin_high_i = bin_high - inv_mask * delta_bin_half
                buffer.extend(
                    self.children[i].get_filled_coords(
                        bin_low_i.astype(bin_low.dtype),
                        bin_high_i.astype(bin_low.dtype),
                    )
                )
            return buffer

        def split(self):
            """
            Splits node if possible and transfers data points to child nodes.
            """
            if self.children is not None:
                return
            if self.max_depth == 0:
                return

            self.children = [None] * self.num_max_split
            delta = (self.high - self.low) / 2
            for i in range(self.num_max_split):
                new_low = self._idx2coord(delta, self.low, i)
                self.children[i] = NDtree.Node(
                    self._mode,
                    new_low,
                    new_low + delta,
                    self.max_depth - 1,
                    self.max_filling,
                    self._idx2mask(i) * self.is_bound,
                )

            for p in self.data:
                self._insert_find_child_node(p)
            self.data.clear()

        def insert(self, p: np.ndarray):
            """
            Inserts data point into this node if possible, else into child node.

            Parameters
            ----------
            p : np.ndarray
                Data point to be inserted.
            """
            if self._mode == NDtree.Mode.INDEX:
                # first insert to sub nodes if available
                if self.children is not None:
                    self._insert_find_child_node(p)
                    return
                # no sub nodes
                # insert locally if possible
                if self.filling < self.max_filling:
                    self.data.append(p)
                    return
                # max filling reached, split and insert
                self.split()
                # insert local if split unsuccessful
                if self.children is None:
                    self.data.append(p)
                else:
                    self._insert_find_child_node(p)
                return

            if self._mode == NDtree.Mode.DISCRETIZE:
                # split as far as max depth allows
                self.split()
                # insert here if max depth reached
                if self.children is None:
                    self.data.append(p)
                else:
                    self._insert_find_child_node(p)

        def get_coord_of(
            self, point: np.ndarray, bin_low: np.ndarray, bin_high: np.ndarray
        ) -> np.ndarray:
            """
            Finds the cell coordinate of given point. Assumes discretization mode is used.

            Parameters
            ----------
            point : np.ndarray
                Query point.
            bin_low : np.ndarray
                Lower bound of possible bins.
            bin_high : np.ndarray
                Upper bound of possible bins.

            Returns
            -------
            coord : np.ndarray
                Coordinate of given point.
            """
            assert self._mode == NDtree.Mode.DISCRETIZE

            if self.children is None:
                return bin_low

            for i in range(self.num_max_split):
                if self.children[i].is_within(point):
                    mask = self._idx2mask(i)
                    inv_mask = np.ones_like(mask) - mask
                    delta_bin_half = (bin_high - bin_low) / 2
                    bin_low_i = bin_low + mask * delta_bin_half
                    bin_high_i = bin_high - inv_mask * delta_bin_half
                    return self.children[i].get_coord_of(point, bin_low_i, bin_high_i)

            raise RuntimeError("Failed to locate cell of point")

        def is_within(self, point: np.ndarray) -> bool:
            """
            Checks whether given point is within the bounds of this node.

            Parameters
            ----------
            point : np.ndarray
                Query point.

            Returns
            -------
            is_within : bool
                True if point is within bounds of this node.
            """
            return np.alltrue(point >= self.low) and np.alltrue(
                np.logical_or(
                    np.logical_and(self.is_bound, np.isclose(point, self.high, 1e-10)),
                    point < self.high,
                )
            )

        def get_height(self) -> int:
            """
            Returns height of this node.

            Returns
            -------
            height : int
                Height of this node.
            """
            if self.children is None:
                return 0

            heights = [node.get_height() for node in self.children]
            return max(heights) + 1

        def serialize(self) -> list[int]:
            """
            Serializes the tree in a run-length encoded format. First entry determines amount of owned entries.

            Returns
            -------
            serialized_data : list[int]
                Serialized tree.
            """
            if self.children is None:
                return [2, len(self.data)]

            result = [1]
            for node in self.children:
                c_result = node.serialize()
                result[0] += c_result[0]
                result.extend(c_result)
            return result

        def deserialize(self, serialized: list[int]) -> None:
            """
            Deserializes tree from serialized data.

            Parameters
            ----------
            serialized : list[int]
                Serialized tree.
            """
            if self.children is not None or len(self.data) > 0:
                raise RuntimeError("Deserialize called on non empty tree.")

            if serialized[0] == 2:
                self.data_reserve_count = serialized[1]
                return

            self.split()
            offset = 1
            for i in range(self.num_max_split):
                self.children[i].deserialize(
                    serialized[offset : offset + serialized[offset]]
                )
                offset += serialized[offset]

        def merge(self, other: "NDtree.Node") -> None:
            """
            Merges the other node structure and reserve counts into this node.

            Parameters
            ----------
            other : NDtree.Node
                Other node structure.
            """
            assert self._mode == NDtree.Mode.DISCRETIZE
            is_split = self.children is not None
            is_split_other = other.children is not None

            if not is_split and not is_split_other:
                self.data_reserve_count += other.data_reserve_count
                return

            if not is_split and is_split_other:
                self.split()
                for i in range(self.num_max_split):
                    self.children[i].merge(other.children[i])

            if is_split and not is_split_other:
                assert other.data_reserve_count == 0

            if is_split and is_split_other:
                for i in range(self.num_max_split):
                    self.children[i].merge(other.children[i])

        def _insert_find_child_node(self, p: np.ndarray) -> None:
            """
            Inserts the point into the correct child node.

            Parameters
            ----------
            p : np.ndarray
                Point to insert.
            """
            for i in range(self.num_max_split):
                if not self.children[i].is_within(p):
                    continue
                self.children[i].insert(p)
                return

        def _idx2mask(self, idx: int) -> np.ndarray:
            """
            Converts the given index into its corresponding binary mask.

            Parameters
            ----------
            idx : int
                Index to convert.

            Returns
            -------
            mask : np.ndarray
                Binary mask. If bit i of idx is 1, then entry i of mask if 1.
            """
            return (
                (idx & np.array([1 << i for i in range(self.dim)], dtype=np.int32)) != 0
            ).astype(np.int32)

        def _idx2coord(
            self, delta: np.ndarray, low: np.ndarray, idx: int
        ) -> np.ndarray:
            """
            Computes the new lower bound for the child node with the given index.

            Parameters
            ----------
            delta : np.ndarray
                New cell size
            low : np.ndarray
                Old lower bound.
            idx : np.ndarray
                Index of child node.

            Returns
            -------
            coord : np.ndarray
                New cell lower bound.
            """
            mask = self._idx2mask(idx).astype(dtype=delta.dtype)
            return (low + delta * mask).astype(mask.dtype)

    def __init__(
        self,
        mode: "NDtree.Mode",
        low: np.ndarray,
        high: np.ndarray,
        max_depth: int,
        max_filling: int,
    ):
        """
        Constructs a NDtree with the given parameters.
        In discretize mode, all data points are inserted into nodes at max_depth.
        In index mode, max_filling is used for insertion. When the threshold is met, nodes are split.

        Parameters
        ----------
        mode : NDtree.Mode
            Mode of operation.
        low : np.ndarray
            Lower bound of space.
        high : np.ndarray
            Upper bound of space.
        max_depth : int
            Maximum depth of the tree.
        max_filling : int
            Maximum filling of the tree.
        """
        self.root = NDtree.Node(
            mode,
            low,
            high,
            max_depth,
            max_filling,
            np.ones(low.shape[0], dtype=np.int32),
        )

    def get_filled_coords(self, height: Optional[int] = None) -> list[np.ndarray]:
        """
        Finds coordinates of all cells that have non-zero reserve counts. Assumes discretization mode is used.

        Parameters
        ----------
        height : Optional[int]
            Height of the tree. If None, will be computed.

        Returns
        -------
        coords : list[np.ndarray]
            Coordinates of all cells that have non-zero reserve counts.
        """
        if height is None:
            height = self.root.get_height()
        dtype = np.int32
        if height > 32:
            dtype = np.int64
        return self.root.get_filled_coords(
            np.zeros(self.root.dim, dtype=dtype),
            np.power(2 * np.ones(self.root.dim, dtype=dtype), height),
        )

    def get_coords_of(
        self, points: np.ndarray, height: Optional[int] = None
    ) -> np.ndarray:
        """
        Finds the cell coordinate of all given point. Assumes discretization mode is used.

        Parameters
        ----------
        points : np.ndarray
            Points to find coordinates of.
        height : Optional[int]
            Height of the tree. If None, will be computed.

        Returns
        -------
        coords : np.ndarray
            Coordinates of all points.
        """
        if height is None:
            height = self.root.get_height()
        dtype = np.int32
        if height > 32:
            dtype = np.int64
        coords = np.zeros((len(points), self.root.dim), dtype=dtype)
        c_min = (np.zeros(self.root.dim, dtype=dtype),)
        c_max = np.power(2 * np.ones(self.root.dim, dtype=dtype), height)
        for i, point in enumerate(points):
            coords[i, :] = self.root.get_coord_of(point, c_min, c_max)
        return coords

    def find_min_depth_for_n_neighbors(self, n: int, points: np.ndarray) -> int:
        """
        Finds the minimum depth of all given points to encounter n neighbors.

        Parameters
        ----------
        n : int
            Number of neighbors.
        points : np.ndarray
            Query points.

        Returns
        -------
        depth : int
            Minimum depth of all given points to encounter n neighbors.
        """
        if points.shape[0] == 0:
            return 0
        depths = np.ones(len(points)) * self.get_height()
        for idx in range(points.shape[0]):
            d = self.root.find_min_depth_for_n_neighbors(n, 0, points[idx, :])
            if d is not None:
                depths[idx] = d
        return np.min(depths)

    def propagate_up_reserve_counts(self) -> int:
        """
        Counts the reserve counts of child nodes and returns sum.
        Used during discretization mode when all data points are in leaf nodes at max depth to approximate the required depth to find N neighbours.

        Returns
        -------
        reserve_counts : int
            sum of child node reserve counts
        """
        return self.root.propagate_up_reserve_counts()

    def split(self):
        """
        Splits node if possible and transfers data points to child nodes.
        """
        return self.root.split()

    def insert(self, p: np.ndarray) -> None:
        """
        Inserts data point into this node if possible, else into child node.

        Parameters
        ----------
        p : np.ndarray
            Data point to be inserted.
        """
        return self.root.insert(p)

    def serialize(self) -> list[int]:
        """
        Serializes the tree in a run-length encoded format. First entry determines amount of owned entries.

        Returns
        -------
        serialized_data : list[int]
            Serialized tree.
        """
        return self.root.serialize()

    def deserialize(self, serialized: list[int]) -> None:
        """
        Deserializes tree from serialized data.

        Parameters
        ----------
        serialized : list[int]
            Serialized tree.
        """
        return self.root.deserialize(serialized)

    def merge(self, other: "NDtree") -> None:
        """
        Merges the other node structure and reserve counts into this node.

        Parameters
        ----------
        other : NDtree
            Other node structure.
        """
        return self.root.merge(other.root)

    def get_height(self) -> int:
        """
        Returns height of this node.

        Returns
        -------
        height : int
            Height of this node.
        """
        return self.root.get_height()

    def clear(self) -> None:
        """
        Clears all data, but preserves tree structure.
        """
        return self.root.clear()


class HilbertDirect:
    """
    Provides a bijective mapping between an N-dimensional space and 1D space based on the algorithm provided in:
    Programming the Hilbert curve by John Skilling (from the AIP Conf. Proc. 707, 381 (2004))

    Example: 5 bits for each of n=3 coordinates.
        15-bit Hilbert integer = A B C D E F G H I J K L M N O is stored
        as its Transpose                        ^
        X[0] = A D G J M                    X[2]|  7
        X[1] = B E H K N        <------->       | /X[1]
        X[2] = C F I L O                   axes |/
               high low                         0------> X[0]
    """

    def __init__(self, dim: int, bits: int):
        """
        Constructs mapping between N-dimensional space and 1D space.
        Used n bits to encode coords along one dimension.
        Therefore, 2**n - 1 is the max coord along one dimension.

        Parameters
        ----------
        dim : int
            Number of dimensions.
        bits : int
            Number of bits used per dimension.
        """
        self.dim = dim
        self.bits = bits
        self.dtype = None

        if bits <= 32:
            self.dtype = np.int32
        else:
            self.dtype = np.int64

    def index2coord(self, idx: int) -> np.ndarray:
        """
        Converts index to coordinate.

        Parameters
        ----------
        idx : int
            Index to convert.

        Returns
        -------
        coords : np.ndarray
            Coordinates of index.
        """
        X = np.zeros(self.dim, dtype=self.dtype)
        if self.bits == 0:
            return X

        # flat index to array
        pos = (self.bits * self.dim) - 1
        for b in range(self.bits):
            for n in range(self.dim):
                bit = (idx >> pos) & 1
                X[n] = X[n] | (bit << (self.bits - b - 1))
                pos = pos - 1

        # gray decode
        N = 2 << (self.bits - 1)
        tmp = X[self.dim - 1] >> 1
        i = self.dim - 1
        while i > 0:
            X[i] = X[i] ^ X[i - 1]
            i = i - 1
        X[0] = X[0] ^ tmp

        # undo excess work
        Q = 2
        while Q != N:
            P = Q - 1
            i = self.dim - 1
            while i >= 0:
                if X[i] & Q:
                    X[0] = X[0] ^ P
                else:
                    tmp = (X[0] ^ X[i]) & P
                    X[0] = X[0] ^ tmp
                    X[i] = X[i] ^ tmp
                i = i - 1
            Q = Q << 1

        return X

    def coord2index(self, coord: np.ndarray) -> int:
        """
        Converts coordinate to index.

        Parameters
        ----------
        coord : np.ndarray
            Coordinate to convert.

        Returns
        -------
        index : int
            Index of coordinate.
        """
        if self.bits == 0:
            return 0
        X = deepcopy(coord)
        M = 1 << (self.bits - 1)

        # inverse undo
        Q = M
        while Q > 1:
            P = Q - 1
            for i in range(self.dim):
                if X[i] & Q:
                    X[0] = X[0] ^ P
                else:
                    tmp = (X[0] ^ X[i]) & P
                    X[0] = X[0] ^ tmp
                    X[i] = X[i] ^ tmp
            Q = Q >> 1

        # gray encode
        for i in range(1, self.dim):
            X[i] = X[i] ^ X[i - 1]
        tmp = 0
        Q = M
        while Q > 1:
            if X[self.dim - 1] & Q:
                tmp = tmp ^ Q - 1
            Q = Q >> 1
        for i in range(self.dim):
            X[i] = X[i] ^ tmp

        # conv arrays to flat index
        result = 0
        pos = (self.bits * self.dim) - 1
        for b in range(self.bits):
            for n in range(self.dim):
                result = result | (((X[n] >> (self.bits - b - 1)) & 1) << pos)
                pos = pos - 1

        return result


class Projector(ABC):
    """
    Interface to project high-dimensional data into low-dimensional space.
    """

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        pass

    @abstractmethod
    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        pass


class STDProjector(Projector):
    """
    Projects high-dimensional data into low-dimensional space using the fields with the highest standard deviation.
    """

    def __init__(self, target_dims: int, comm: MPI.Comm):
        """
        Constructs STD projection.

        Parameters
        ----------
        target_dims : int
            Number of target dimensions.
        comm : MPI.Comm
            MPI communicator.
        """
        self.num_target_dims = target_dims
        self.target_dims = np.zeros(target_dims, dtype=np.int32)
        self.comm = comm

    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data.
        Computes target dimensions using provided data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        assert data.ndim > 1
        std = np.zeros(data.shape[-1])
        if data.shape[0] > 0:
            std = np.std(data, axis=0)
        stds = np.array(self.comm.allgather(std))
        stds = np.mean(stds, axis=0)
        self.target_dims[:] = np.sort(
            np.argsort(stds)[::-1][0 : self.num_target_dims]
        ).astype(np.int32)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        d = data
        if data.ndim == 1:
            d = d[np.newaxis, :]
        return d[:, self.target_dims]


class IdentityProjector(Projector):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data. (does nothing)

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        return data

    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data. (does nothing)

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        pass


class InterleavedDomain:
    """
    Handles n-dimensional data in an overlapping domain.
    Will de- and re-compose the distributed data to allow for domain local operations.
    """

    def __init__(self, config: Config, comm: MPI.Comm):
        """
        Constructs InterleavedDomain object.

        Parameters
        ----------
        config : Config
            Configuration object.
        comm : MPI.Comm
            MPI communicator.
        """
        self._config = config
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()

        # decomp data
        self._x_local = None  # n_points x point_dim
        self._x_query_local = None  # m_points x point_dim
        self._f_local = None  # n_points x fun_dim
        self._proj_x_local = None  # n_points x proj_dim
        self._proj_x_query_local = None  # m_points x proj_dim
        self._bound_low = None
        self._bound_high = None
        self._normalization = None
        self._shift = None
        self._max_depth = None
        self._max_filling = 8
        self._coarsening_factor = 2
        self._n_neighbors = 50
        self._tree: Optional[NDtree] = None
        self._query_tree: Optional[NDtree] = None
        self._projector: Projector = IdentityProjector()

        self._query_rank_mapping = None

    def configure(self, domain_config: dict) -> None:
        """
        Configures InterleavedDomain object to the provided settings.

        Parameters
        ----------
        domain_config : dict
            Target configuration.
        """
        self._max_filling = (
            8 if "max_filling" not in domain_config else domain_config["max_filling"]
        )
        self._coarsening_factor = (
            2
            if "coarsening_factor" not in domain_config
            else domain_config["coarsening_factor"]
        )
        self._n_neighbors = (
            50 if "n_neighbors" not in domain_config else domain_config["n_neighbors"]
        )
        if "projection_type" not in domain_config:
            self._projector = IdentityProjector()
            return

        match domain_config["projection_type"]:
            case "std":
                target_dims = (
                    1
                    if "projection_std_dims" not in domain_config
                    else domain_config["projection_std_dims"]
                )
                self._projector = STDProjector(target_dims, self._comm)
            case "identity":
                self._projector = IdentityProjector()

    def set_local_data(self, x: np.ndarray, x_: np.ndarray, f: np.ndarray) -> None:
        """
        Sets local data for interleaved domain.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        x_ : np.ndarray
            Query points.
        f : np.ndarray
            Support point function values.
        """

        def dim_extend(a):
            if a.ndim == 1:
                return a.reshape(-1, 1)
            return a

        self._x_local = dim_extend(x)
        self._x_query_local = dim_extend(x_)
        self._f_local = dim_extend(f)

    def decompose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decomposes the domain, by conceptually merging all support and query points across all rank
        and splitting the query points, s.t. each rank will have approx the same amount of query points.
        Support points alongside their function values are distributed to the respective ranks, that query points
        are surrounded by sufficient support points.

        Returns
        -------
        x : np.ndarray
            Assigned support points.
        x_ : np.ndarray
            Assigned query points.
        f : np.ndarray
            Assigned support point function values.
        """
        # if not parallel, no work to be done
        if self._size == 1:
            return self._x_local, self._x_query_local, self._f_local

        self._generate_trees()
        return self._create_partitions()

    def get_depth_filling(self) -> Tuple[int, int]:
        """
        Gets the tree properties.

        Returns
        -------
        max_depth : int
            Maximum depth of the tree.
        max_filling : int
            Maximum filling of the tree.
        """
        return self._max_depth, self._max_filling

    def reassemble(self, x_query: np.ndarray, f_query: np.ndarray) -> np.ndarray:
        """
        Reassembles the query point function values to match the configuration prior of decomposition.

        Parameters
        ----------
        x_query : np.ndarray
            Query points.
        f_query : np.ndarray
            Query point function values.

        Returns
        -------
        reassembled : np.ndarray
            Reassembled query point function values.
        """
        # if not parallel, no work to be done
        if self._size == 1:
            return f_query

        # transfer data back to original rank
        send_map = {r: [] for r in range(self._size)}
        for i in range(x_query.shape[0]):
            dst_rank = self._query_rank_mapping[tuple(x_query[i, :].tolist())]
            data = []
            data.extend(x_query[i, :].tolist())
            data.extend(f_query[i, :].tolist())
            send_map[dst_rank].append(data)
        local_data = self._communicate(send_map)
        local_data = np.array(local_data).reshape(
            -1, x_query.shape[-1] + f_query.shape[-1]
        )

        # sort data to match order of initial query input
        idx_map = {}
        for i in range(self._x_query_local.shape[0]):
            idx_map[tuple(self._x_query_local[i, :].tolist())] = i

        result = np.zeros((self._x_query_local.shape[0], f_query.shape[-1]))
        for d_idx in range(local_data.shape[0]):
            idx = idx_map[tuple(local_data[d_idx, 0 : x_query.shape[-1]].tolist())]
            result[idx, :] = local_data[d_idx, x_query.shape[-1] :]
        return result

    def _normalize_x(self) -> None:
        """
        Normalizes support and query points to fit within -1 and 1.
        """
        x_loc_min = np.ones(self._x_local.shape[-1]) * np.inf
        if self._x_local.shape[0] > 0:
            x_loc_min = np.min(self._x_local, axis=0)
        xq_loc_min = np.ones(self._x_query_local.shape[-1]) * np.inf
        if self._x_query_local.shape[0] > 0:
            xq_loc_min = np.min(self._x_query_local, axis=0)
        local_min = np.minimum(x_loc_min, xq_loc_min)
        glob_min = np.min(np.array(self._comm.allgather(local_min)), axis=0)
        self._bound_low = glob_min

        x_loc_max = -np.ones(self._x_local.shape[-1]) * np.inf
        if self._x_local.shape[0] > 0:
            x_loc_max = np.max(self._x_local, axis=0)
        xq_loc_max = -np.ones(self._x_query_local.shape[-1]) * np.inf
        if self._x_query_local.shape[0] > 0:
            xq_loc_max = np.max(self._x_query_local, axis=0)
        local_max = np.maximum(x_loc_max, xq_loc_max)
        glob_max = np.max(np.array(self._comm.allgather(local_max)), axis=0)
        self._bound_high = glob_max

        delta = glob_max - glob_min
        shift = glob_min + delta / 2.0

        self._normalization = delta / 2.0
        self._shift = shift
        self._x_local = (self._x_local - shift) / self._normalization[None, :]
        self._x_query_local = (self._x_query_local - shift) / self._normalization[
            None, :
        ]

        def eval_cond():
            return not all(
                [
                    np.alltrue(self._x_local > -1.0),
                    np.alltrue(self._x_local < 1.0),
                    np.alltrue(self._x_query_local > -1.0),
                    np.alltrue(self._x_query_local < 1.0),
                ]
            )

        glob_cond = self._comm.allgather(eval_cond())
        while any(glob_cond):
            # undo prev norm
            self._x_local = self._x_local * self._normalization[None, :]
            self._x_query_local = self._x_query_local * self._normalization[None, :]
            # retry with larger norm
            self._normalization += 1e-10
            self._x_local = self._x_local / self._normalization[None, :]
            self._x_query_local = self._x_query_local / self._normalization[None, :]
            glob_cond = self._comm.allgather(eval_cond())

        self._projector.initialize(self._x_local * self._normalization[None, :])
        self._proj_x_local = self._projector(self._x_local)
        self._proj_x_query_local = self._projector(self._x_query_local)

    def _generate_trees(self) -> None:
        """
        Generates domain decomposition trees, shares them across all ranks and constructs
        a globally valid tree used for partitioning.
        """
        self._normalize_x()

        proj_dim = self._proj_x_local.shape[1]
        low, high = -np.ones(proj_dim), np.ones(proj_dim)
        # determine max required depth
        # populate and query height
        depth_tree = NDtree(NDtree.Mode.INDEX, low, high, 32, self._max_filling)
        for n in range(self._proj_x_local.shape[0]):
            depth_tree.insert(self._proj_x_local[n, :])
        for m in range(self._proj_x_query_local.shape[0]):
            depth_tree.insert(self._proj_x_query_local[m, :])
        max_depth = np.maximum(
            self._comm.allreduce(depth_tree.get_height(), op=MPI.MAX)
            // self._coarsening_factor,
            self._coarsening_factor,
        )
        del depth_tree
        self._max_depth = max_depth

        # populate discretization trees
        tree = NDtree(NDtree.Mode.DISCRETIZE, low, high, max_depth, self._max_filling)
        for n in range(self._proj_x_local.shape[0]):
            tree.insert(self._proj_x_local[n, :])
        query_tree = NDtree(
            NDtree.Mode.DISCRETIZE, low, high, max_depth, self._max_filling
        )
        for m in range(self._proj_x_query_local.shape[0]):
            query_tree.insert(self._proj_x_query_local[m, :])

        # merge into a global tree structure
        def bcast_tree(t) -> NDtree:
            serial = t.serialize()
            serial_global = self._comm.allgather(serial)
            res = NDtree(
                NDtree.Mode.DISCRETIZE, low, high, max_depth, self._max_filling
            )
            for s in serial_global:
                other_tree = NDtree(
                    NDtree.Mode.DISCRETIZE, low, high, max_depth, self._max_filling
                )
                other_tree.deserialize(s)
                res.merge(other_tree)
            return res

        self._tree = bcast_tree(tree)
        self._query_tree = bcast_tree(query_tree)

    def _create_partitions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates partitions based on a equidistant splitting of the
        hilbert space indices of the domain decomposition trees.

        Returns
        -------
        x : np.ndarray
            Support points around new query points
        x_ : np.ndarray
            New query points
        f : np.ndarray
            Support point function values.
        """
        self._tree.propagate_up_reserve_counts()
        r_m_depth = self._tree.find_min_depth_for_n_neighbors(
            self._n_neighbors, self._proj_x_query_local
        )
        r_m_depth = self._comm.allreduce(r_m_depth, op=MPI.MAX)
        r_m_cells = np.power(2, r_m_depth)
        grid_resolution = self._tree.get_height()
        hMap = HilbertDirect(self._proj_x_local.shape[-1], grid_resolution)

        # index query points
        query_coords = self._query_tree.get_filled_coords(grid_resolution)
        query_mapping = {tuple(c.tolist()): hMap.coord2index(c) for c in query_coords}
        query_mapping_inv = {}
        for coord, idx in query_mapping.items():
            query_mapping_inv[idx] = coord
        sorted_1d_query_indices = sorted(query_mapping.values())

        # partition based on query points
        target_point_per_rank = len(sorted_1d_query_indices) // self._size
        partitions = {r: [-1, -1] for r in range(self._size)}
        last_val = sorted_1d_query_indices[0]
        start_idx = 0
        part_begin = 0
        part_idx = 0
        for i in range(1, len(sorted_1d_query_indices)):
            if sorted_1d_query_indices[i] != last_val:
                last_val = sorted_1d_query_indices[i]
                start_idx = i

            if i - part_begin + 1 < target_point_per_rank:
                continue

            # handle last partition
            if part_idx == self._size - 1:
                partitions[part_idx][0] = part_begin
                partitions[part_idx][1] = len(sorted_1d_query_indices) - 1
                part_idx = part_idx + 1
                break

            # partition has minimum size, find nearest end of current cell
            end_idx = i
            for j in range(i + 1, len(sorted_1d_query_indices)):
                if sorted_1d_query_indices[j] != last_val:
                    end_idx = j - 1
                    break
            # start is closer
            if i - start_idx < end_idx - i:
                partitions[part_idx][0] = part_begin
                partitions[part_idx][1] = start_idx - 1
                part_begin = start_idx
            # end is closer
            else:
                partitions[part_idx][0] = part_begin
                partitions[part_idx][1] = end_idx
                part_begin = end_idx + 1

            part_idx = part_idx + 1
        # last part was not used
        if (
            part_idx in partitions
            and partitions[part_idx][0] == 0
            and partitions[part_idx][1] == 0
        ):
            partitions[part_idx][0] = part_begin
            partitions[part_idx][1] = len(sorted_1d_query_indices) - 1

        # assign surrounding src domain to rank local query points
        src_domains = {r: [None, None] for r in range(self._size)}
        for rank, p_range in partitions.items():
            if -1 == p_range[0] == p_range[1]:
                continue

            # gather coords, find bounding box
            local_indices = sorted_1d_query_indices[p_range[0] : p_range[1] + 1]
            local_coords = np.array([query_mapping_inv[idx] for idx in local_indices])
            bbox_low = np.min(local_coords, axis=0)
            bbox_high = np.max(local_coords, axis=0)
            # np.where for valid bbox ranges
            low_mask = (bbox_low >= 2 * r_m_cells).astype(np.int32)
            bbox_low = low_mask * (bbox_low - 2 * r_m_cells) + (
                1 - low_mask
            ) * np.zeros_like(bbox_low)

            max_coord = np.power(2, grid_resolution) - 1
            high_mask = (bbox_high < (max_coord - 2 * r_m_cells)).astype(np.int32)
            bbox_high = high_mask * (bbox_high + 2 * r_m_cells) + (1 - high_mask) * (
                np.ones_like(bbox_high) * max_coord
            )
            src_domains[rank][0] = bbox_low
            src_domains[rank][1] = bbox_high

        # figure out which query points need to be sent where
        owned_query_coords = self._query_tree.get_coords_of(
            self._proj_x_query_local, grid_resolution
        )
        owned_query_indices = [
            query_mapping[tuple(coord.tolist())] for coord in owned_query_coords
        ]
        send_map = {r: [] for r in range(self._size)}
        for i in range(len(owned_query_indices)):
            # find owning partition
            found = False
            for rank, rank_range in partitions.items():
                if -1 == rank_range[0] == rank_range[1]:
                    continue

                if (
                    sorted_1d_query_indices[rank_range[0]] > owned_query_indices[i]
                    or sorted_1d_query_indices[rank_range[1]] < owned_query_indices[i]
                ):
                    continue

                send_map[rank].append(self._x_query_local[i, :].tolist())
                found = True
                break

            # part not found
            if not found:
                raise RuntimeError("Corresponding rank not found for query point")

        # transfer query points
        x_query_part, inv_map = self._communicate(send_map, return_inverse=True)
        x_query = np.array(x_query_part).reshape(-1, self._x_query_local.shape[-1])
        # invert query send map for later (to transfer back)
        self._query_rank_mapping = {}
        for rank, data in inv_map.items():
            data_ = np.array(data).reshape(-1, self._x_query_local.shape[-1])
            for p_idx in range(data_.shape[0]):
                self._query_rank_mapping[tuple(data_[p_idx, :].tolist())] = rank

        # figure out which source points need to be sent where
        send_map = {r: [] for r in range(self._size)}
        source_coords = self._tree.get_filled_coords(grid_resolution)
        source_mapping = {tuple(c.tolist()): hMap.coord2index(c) for c in source_coords}
        owned_source_coords = self._tree.get_coords_of(
            self._proj_x_local, grid_resolution
        )
        owned_source_indices = [
            source_mapping[tuple(coord.tolist())] for coord in owned_source_coords
        ]
        for i in range(len(owned_source_indices)):
            for rank, rank_domain in src_domains.items():
                if rank_domain[0] is None or rank_domain[1] is None:
                    continue

                if np.alltrue(rank_domain[0] <= source_coords[i]) and np.alltrue(
                    source_coords[i] <= rank_domain[1]
                ):
                    data = []
                    data.extend(self._x_local[i, :].tolist())
                    data.extend(self._f_local[i, :].tolist())
                    send_map[rank].append(data)

        # transfer source points
        xf_part = self._communicate(send_map)
        xf_part = np.array(xf_part).reshape(
            -1, self._x_local.shape[-1] + self._f_local.shape[-1]
        )
        x = xf_part[:, 0 : self._x_local.shape[-1]]
        f = xf_part[:, self._x_local.shape[-1] :]

        return x, x_query, f

    def _communicate(
        self, send_map: dict[int, list], return_inverse: bool = False
    ) -> Union[list, Tuple[list, dict[int, list]]]:
        """
        Transfers data between ranks a p2p communication according to the provided send_map.

        Parameters
        ----------
        send_map : dict[int, list]
            Mapping from destination rank to list of data to be transferred.
        return_inverse : bool
            Return inverse transfer or not.

        Returns
        -------
        comm_result : Union[list, Tuple[list, dict[int, list]]]
            If return_inverse is True, returns a list of all received data as well as a mapping from which rank
            which data was sent. Otherwise, returns only the received data list.
        """
        send_counts = [len(send_map[r]) for r in range(self._size)]
        send_counts[self._rank] = 0  # ignore local count
        glob_send_counts = self._comm.allgather(send_counts)

        send_reqs = []
        for recv_rank, data in send_map.items():
            if recv_rank == self._rank:
                continue
            if len(data) == 0:
                continue
            for d_idx, entry in enumerate(data):
                req = self._comm.isend(
                    entry, dest=recv_rank, tag=create_tag(d_idx, self._rank, recv_rank)
                )
                send_reqs.append(req)

        recv_reqs = []
        for send_rank in range(self._size):
            if send_rank == self._rank:
                continue
            if glob_send_counts[send_rank][self._rank] == 0:
                continue
            for d_idx in range(glob_send_counts[send_rank][self._rank]):
                req = self._comm.irecv(
                    None, source=send_rank, tag=create_tag(d_idx, send_rank, self._rank)
                )
                recv_reqs.append(tuple([send_rank, req]))

        MPI.Request.Waitall(send_reqs)

        result = []
        result.extend(send_map[self._rank])
        inv_map = {r: [] for r in range(self._size)}
        inv_map[self._rank].extend(send_map[self._rank])

        for source_rank, req in recv_reqs:
            data = req.wait()
            result.append(data)
            if return_inverse:
                inv_map[source_rank].append(data)

        if return_inverse:
            return result, inv_map
        else:
            return result


class RBF_PU:
    """
    Interpolates f(x) for f: R^n -> R^m using partition of unity RBF interpolant.

    The approach here does not require a support radius as data is normalized.
    """

    def __init__(
        self, config: Config, logger: Logger, comm: MPI.Comm, rank: int, size: int
    ):
        """
        Constructs the RBF_PU interpolator.
        For rank local interpolation provide MPI.COMM_SELF as comm with according rank and size.

        Parameters
        ----------
        config : Config
            Configuration object.
        logger : Logger
            Logger object.
        comm : MPI.Comm
            MPI communicator.
        rank : int
            Rank within the provided MPI communicator.
        size : int
            Size within the provided MPI communicator.
        """
        self._config = config
        self._logger = logger
        self._comm = comm
        self._rank = rank
        self._size = size

        self._domain = InterleavedDomain(config, comm)
        self._use_pu = False
        self._pu_overlap = 0.1
        self._pu_cluster_size = 50

        # RBF data
        self._phi = RBF_PU.basis_c6
        self._x = None
        self._x_query = None
        self._f = None

    def configure(self, interp_config: dict) -> None:
        """
        Configures the interpolator to the provided parameters.

        Parameters
        ----------
        interp_config : dict
            Interpolator configuration.
        """
        self._domain.configure(
            {}
            if "domain_config" not in interp_config
            else interp_config["domain_config"]
        )
        self._use_pu = (
            False if "use_pu" not in interp_config else interp_config["use_pu"]
        )
        if self._use_pu:
            self._pu_overlap = (
                0.1
                if "pu_overlap" not in interp_config
                else interp_config["pu_overlap"]
            )
            self._pu_cluster_size = (
                50
                if "pu_cluster_size" not in interp_config
                else interp_config["pu_cluster_size"]
            )
        if "basis" not in interp_config:
            return
        match interp_config["basis"]:
            case "c0":
                self._phi = RBF_PU.basis_c0
            case "c2":
                self._phi = RBF_PU.basis_c2
            case "c4":
                self._phi = RBF_PU.basis_c4
            case "c6":
                self._phi = RBF_PU.basis_c6
            case "gauss":
                eps = (
                    1.0
                    if "gauss_eps" not in interp_config
                    else interp_config["gauss_eps"]
                )
                self._phi = partial(RBF_PU.basis_gauss, eps=eps)

    def set_local_data(self, x: np.ndarray, x_: np.ndarray, f: np.ndarray) -> None:
        """
        Sets local data for interleaved domain.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        x_ : np.ndarray
            Query points.
        f : np.ndarray
            Support point function values.
        """
        self._domain.set_local_data(x, x_, f)

    def interpolate(self) -> np.ndarray:
        """
        Interpolates the function values at the set query points.

        Returns
        -------
        interp_result : np.ndarray
            Interpolated function values.
        """
        self._x, self._x_query, self._f = self._domain.decompose()

        interp = self.compute_interpolant(self._x, self._f)
        xq, fq = self.evaluate_interpolant(interp, self._x_query)

        fq_local = self._domain.reassemble(xq, fq)

        return fq_local

    # ================================
    #              RBF
    # ================================
    @property
    def compute_interpolant(self):
        if self._use_pu:
            return self.compute_rbf_pu_interpolant
        else:
            return self.compute_rbf_interpolant

    @property
    def evaluate_interpolant(self):
        if self._use_pu:
            return self.evaluate_rbf_pu_interpolant
        else:
            return self.evaluate_rbf_interpolant

    def _compute_cluster_centers(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates cluster centers based on the provided support points.
        2 cluster centers per dimension and one in the middle.

        Parameters
        ----------
        x : np.ndarray
            Support points.

        Returns
        -------
        cluster_centers : np.ndarray
            Cluster centers.
        local_min : np.ndarray
            Minimum of local points.
        local_max : np.ndarray
            Maximum of local points.
        """
        assert self._use_pu
        local_min, local_max = np.min(x, axis=0), np.max(x, axis=0)
        d4 = (local_max - local_min) / 4

        center = local_min + 2.0 * d4
        centers = np.zeros((2 * x.shape[-1] + 1, x.shape[-1]))
        centers[-1, :] = center
        for d in range(x.shape[-1]):
            mask = np.zeros_like(d4)
            mask[d] = 1

            centers[2 * d + 0, :] = center - mask * d4
            centers[2 * d + 1, :] = center + mask * d4

        return centers, local_min, local_max

    def compute_rbf_pu_interpolant(self, x, f):
        # compute r_m
        c_centers, local_min, local_max = self._compute_cluster_centers(x)
        index_tree = NDtree(
            NDtree.Mode.INDEX, local_min, local_max, *self._domain.get_depth_filling()
        )
        # TODO later
        # determine clusters
        # ignore empty clusters
        # compute local RBF interpolant for remaining clusters
        pass

    def compute_rbf_interpolant(
        self, x: np.ndarray, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs an interpolant based on the provided support points and function values.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        f : np.ndarray
            Support point function values.

        Returns
        -------
        interp_weights_high: np.ndarray
            Interpolant weights, higher order.
        interp_weights_low: np.ndarray
            Interpolant weights, lower order.
        x : np.ndarray
            Support points.
        f : np.ndarray
            Support point function values.
        """
        n_points = x.shape[0]
        src_size = x.shape[-1]
        dst_size = f.shape[-1]

        r = np.linalg.norm(x[None, :, :] - x[:, None, :], ord=2, axis=-1)
        # compute lin and const term
        b = np.zeros((src_size + 1, dst_size))
        p = np.zeros((n_points, src_size + 1))
        p[:, 0] = 1
        p[:, 1:] = x
        for k in range(dst_size):
            b[:, k] = np.linalg.lstsq(p, f[:, k], rcond=None)[0]

        a = self._phi(r)
        # compute basis weights
        w = np.zeros((dst_size, n_points))
        for k in range(dst_size):
            w[k, :] = np.linalg.solve(a, f[:, k] - np.matmul(p, b[:, k]))

        return w, b, x, f

    def evaluate_rbf_pu_interpolant(self, interp, xq):
        # eval xq for all cluster interpolants
        # compute weights
        # sum contributions
        pass

    def evaluate_rbf_interpolant(
        self,
        interp: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        xq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates the function values at the set query points.

        Parameters
        ----------
        interp : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Interpolation model as computed by compute_rbf_interpolant
        xq : np.ndarray
            Query points.

        Returns
        -------
        xq : np.ndarray
            Query points.
        fq : np.ndarray
            Query point function values.
        """
        w, b, x, f = interp

        r = np.linalg.norm(xq[None, :, :] - x[:, None, :], ord=2, axis=-1)
        contrib_basis = np.matmul(w[:, :], self._phi(r))  # f_k x eval_p
        contrib_const = b[0, :]  # f_k
        # b: p_size+1 x f_k
        # xq: eval_p x p_size
        contrib_lin = np.matmul(xq[:, :], b[1:, :]).T  # f_k x eval_p

        fq = (contrib_basis + contrib_const[:, None] + contrib_lin).T
        return xq, fq

    # ================================
    #        BASIS FUNCTIONS
    # ================================
    @staticmethod
    def basis_c0(r):
        return np.maximum(0.0, np.power(1.0 - r, 2))

    @staticmethod
    def basis_c2(r):
        return np.maximum(0.0, np.power(1.0 - r, 4)) * (4.0 * r + 1)

    @staticmethod
    def basis_c4(r):
        return (
            np.maximum(0.0, np.power(1.0 - r, 6))
            * (35.0 * np.power(r, 2) + 18.0 * r + 3.0)
            / 3.0
        )

    @staticmethod
    def basis_c6(r):
        return np.maximum(0.0, np.power(1.0 - r, 8)) * (
            32.0 * np.power(r, 3) + 25.0 * np.power(r, 2) + 8.0 * r + 1.0
        )

    @staticmethod
    def basis_gauss(r, eps):
        return np.exp(-np.power(eps * r, 2.0))
