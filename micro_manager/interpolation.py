import numpy as np
from sklearn.neighbors import NearestNeighbors


class Interpolation:
    def __init__(self, logger):

        self._logger = logger

    def get_nearest_neighbor_indices_local(
        self,
        all_local_coords: np.ndarray,
        inter_point: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Get the indices of the k nearest neighbors of a point in a list of coordinates.
        If inter_point is part of all_local_coords, it is only considered one time less than it occurs.
        inter_point is expected to be in all_local_coords at most once.

        Parameters
        ----------
        all_local_coords : list
            List of coordinates of all points.
        inter_point : list | np.ndarray
            Coordinates of the point for which the neighbors are to be found.
        k : int
            Number of neighbors to consider.

        Returns
        ------
        neighbor_indices : np.ndarray
            Indices of the k nearest neighbors in all local points.
        """
        assert (
            len(all_local_coords) >= k
        ), "Desired number of neighbors must be less than or equal to the number of all available neighbors."
        # If the number of neighbors is larger than the number of all available neighbors, increase the number of neighbors
        # to be able to delete a neighbor if it coincides with the interpolation point.
        if len(all_local_coords) > k:
            k += 1
        neighbors = NearestNeighbors(n_neighbors=k).fit(all_local_coords)

        dists, neighbor_indices = neighbors.kneighbors(
            [inter_point], return_distance=True
        )

        # Check whether the inter_point is also part of the neighbor list and remove it.
        if np.min(dists) < 1e-16:
            argmin = np.argmin(dists)
            neighbor_indices = np.delete(neighbor_indices, argmin)
        # If point itself is not in neighbor list, remove neighbor with largest distance
        # to return the desired number of neighbors
        else:
            argmax = np.argmax(dists)
            neighbor_indices = np.delete(neighbor_indices, argmax)

        return neighbor_indices

    def inv_dist_weighted_interp(
        self, neighbors: np.ndarray, point: np.ndarray, values
    ):
        """
            Interpolate a value at a point using inverse distance weighting.
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
        # iterate over all neighbors
        for inx in range(len(neighbors)):
            # compute the squared norm of the difference between interpolation point and neighbor
            norm = np.linalg.norm(np.array(neighbors[inx]) - np.array(point)) ** 2
            # If interpolation point is already part of the data it is returned as the interpolation result
            # This avoids division by zero
            if norm < 1e-16:
                return values[inx]
            # update interpolation value
            interpol_val += values[inx] / norm
            # extend normalization factor
            summed_weights += 1 / norm

        return interpol_val / summed_weights
