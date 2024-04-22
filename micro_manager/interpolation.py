import numpy as np
from sklearn.neighbors import NearestNeighbors


class Interpolation:
    def __init__(self, logger):

        self._logger = logger

    def get_nearest_neighbor_indices_local(
        self,
        all_local_coords,
        inter_point,
        k: int,
        inter_point_is_neighbor: bool = False,
    ) -> np.ndarray:
        """
        Get the indices of the k nearest neighbors of a point in a list of coordinates.
        Note: It can be chosen whether the point itself is considered as a neighbor or not.
        Args:
            all_local_coords: list
                List of coordinates of all points.
            inter_point:
                Coordinates of the point for which the neighbors are to be found.
            k: int
            inter_point_is_neighbor: bool, optional
                Decide whether the interpolation point is considered as its own neighbor.
                Defaults to False.

        Returns: np.ndarray
            of indices of the k nearest neighbors.
        """
        assert (
            len(all_local_coords) > k
        ), "Desired number of neighbors must be less than the number of all available neighbors."
        if not inter_point_is_neighbor:
            neighbors = NearestNeighbors(n_neighbors=k + 1).fit(all_local_coords)

            dists, neighbor_indices = neighbors.kneighbors(
                [inter_point], return_distance=True
            )

            if np.min(dists) < 1e-10:
                argmin = np.argmin(dists)
                neighbor_indices = np.delete(neighbor_indices, argmin)
            else:
                argmax = np.argmax(dists)
                neighbor_indices = np.delete(neighbor_indices, argmax)
        else:
            neighbors = NearestNeighbors(n_neighbors=k).fit(all_local_coords)
            neighbor_indices = neighbors.kneighbors(
                [inter_point], return_distance=False
            )

        return neighbor_indices

    def inv_dist_weighted_interp(
        self, neighbors: list, point, values: list
    ) -> np.ndarray:
        """
            Interpolate a value at a point using inverse distance weighting.

        Args:
            neighbors: list
                Coordinates at which the values are known.
            point:
                Coordinates at which the value is to be interpolated.
            values: list
                Values at the known coordinates.

        Returns: nd.array
            Value at interpolation point.
        """
        interpol_val = 0
        summed_weights = 0
        for inx in range(len(neighbors)):
            norm = np.linalg.norm(np.array(neighbors[inx]) - np.array(point)) ** 2
            if norm < 1e-10:
                return values[inx]
            interpol_val += values[inx] / norm
            summed_weights += 1 / norm

        return interpol_val / summed_weights
