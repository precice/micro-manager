import numpy as np
from sklearn.neighbors import NearestNeighbors


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
            self._logger.info(
                "Number of desired neighbors k = {} is larger than the number of available neighbors {}. Resetting k = {}.".format(
                    k, len(coords), len(coords)
                )
            )
            k = len(coords)
        neighbors = NearestNeighbors(n_neighbors=k).fit(coords)

        neighbor_indices = neighbors.kneighbors(
            [inter_point], return_distance=False
        ).flatten()

        return neighbor_indices

    def interpolate(self, neighbors: np.ndarray, point: np.ndarray, values):
        """
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
