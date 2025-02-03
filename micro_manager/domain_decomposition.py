"""
Class DomainDecomposer provides the method decompose_macro_domain which returns partitioned bounds
"""

import numpy as np


class DomainDecomposer:
    def __init__(self, rank, size) -> None:
        """
        Class constructor.

        Parameters
        ----------
        rank : int
            MPI rank.
        size : int
            Total number of MPI processes.
        """
        self._rank = rank
        self._size = size

    def decompose_macro_domain(self, macro_bounds: list, ranks_per_axis: list) -> list:
        """
        Decompose the macro domain equally among all ranks, if the Micro Manager is run in parallel.

        Parameters
        ----------
        macro_bounds : list
            List containing upper and lower bounds of the macro domain.
            Format in 2D is [x_min, x_max, y_min, y_max]
            Format in 3D is [x_min, x_max, y_min, y_max, z_min, z_max]
        ranks_per_axis : list
            List containing axis wise ranks for a parallel run
            Format in 2D is [ranks_x, ranks_y]
            Format in 3D is [ranks_x, ranks_y, ranks_z]

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        assert (
            np.prod(ranks_per_axis) == self._size
        ), "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."

        dims = len(ranks_per_axis)

        if dims == 3:
            for z in range(ranks_per_axis[2]):
                for y in range(ranks_per_axis[1]):
                    for x in range(ranks_per_axis[0]):
                        n = (
                            x
                            + y * ranks_per_axis[0]
                            + z * ranks_per_axis[0] * ranks_per_axis[1]
                        )
                        if n == self._rank:
                            rank_in_axis = [x, y, z]
        elif dims == 2:
            for y in range(ranks_per_axis[1]):
                for x in range(ranks_per_axis[0]):
                    n = x + y * ranks_per_axis[0]
                    if n == self._rank:
                        rank_in_axis = [x, y]

        dx = []
        for d in range(dims):
            dx.append(
                abs(macro_bounds[d * 2 + 1] - macro_bounds[d * 2]) / ranks_per_axis[d]
            )

        mesh_bounds = []
        for d in range(dims):
            if rank_in_axis[d] > 0:
                mesh_bounds.append(macro_bounds[d * 2] + rank_in_axis[d] * dx[d])
                mesh_bounds.append(macro_bounds[d * 2] + (rank_in_axis[d] + 1) * dx[d])
            elif rank_in_axis[d] == 0:
                mesh_bounds.append(macro_bounds[d * 2])
                mesh_bounds.append(macro_bounds[d * 2] + dx[d])

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = macro_bounds[d * 2 + 1]

        return mesh_bounds
