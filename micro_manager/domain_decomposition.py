"""
Class DomainDecomposer provides the method decompose_macro_domain which returns partitioned bounds
"""

import numpy as np


class DomainDecomposer:
    def __init__(self, logger, dims, rank, size) -> None:
        """
        Class constructor.

        Parameters
        ----------
        logger : object of logging
            Logger defined from the standard package logging.
        dims : int
            Dimensions of the problem.
        rank : int
            MPI rank.
        size : int
            Total number of MPI processes.
        """
        self._logger = logger
        self._rank = rank
        self._size = size
        self._dims = dims

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
        assert np.prod(
            ranks_per_axis) == self._size, "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."

        dx = []
        for d in range(self._dims):
            dx.append(abs(macro_bounds[d * 2 + 1] - macro_bounds[d * 2]) / ranks_per_axis[d])

        rank_in_axis: list[int] = [0] * self._dims
        if ranks_per_axis[0] == 1:  # if serial in x axis
            rank_in_axis[0] = 0
        else:
            rank_in_axis[0] = self._rank % ranks_per_axis[0]  # x axis

        if self._dims == 2:
            if ranks_per_axis[1] == 1:  # if serial in y axis
                rank_in_axis[1] = 0
            else:
                rank_in_axis[1] = int(self._rank / ranks_per_axis[0])  # y axis
        elif self._dims == 3:
            if ranks_per_axis[2] == 1:  # if serial in z axis
                rank_in_axis[2] = 0
            else:
                rank_in_axis[2] = int(self._rank / (ranks_per_axis[0] * ranks_per_axis[1]))  # z axis

            if ranks_per_axis[1] == 1:  # if serial in y axis
                rank_in_axis[1] = 0
            else:
                rank_in_axis[1] = (self._rank - ranks_per_axis[0] * ranks_per_axis[1]
                                   * rank_in_axis[2]) % ranks_per_axis[2]  # y axis

        mesh_bounds = []
        for d in range(self._dims):
            if rank_in_axis[d] > 0:
                mesh_bounds.append(macro_bounds[d * 2] + rank_in_axis[d] * dx[d])
                mesh_bounds.append(macro_bounds[d * 2] + (rank_in_axis[d] + 1) * dx[d])
            elif rank_in_axis[d] == 0:
                mesh_bounds.append(macro_bounds[d * 2])
                mesh_bounds.append(macro_bounds[d * 2] + dx[d])

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = macro_bounds[d * 2 + 1]

        self._logger.info("Bounding box limits are {}".format(mesh_bounds))

        return mesh_bounds
