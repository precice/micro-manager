"""
Functionality to partition the macro domain according to the user provided partitions in each axis
"""

import numpy as np

class DomainDecomposer:
    def __init__(self, logger, interface, rank, size) -> None:
        self._logger = logger
        self._interface = interface
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
            Format in 2D is [x_min, x_max, y_min, y_max, z_min, z_max]
        ranks_per_axis : list
            List containing axis wise ranks for a parallel run
            Format in 2D is [ranks_x, ranks_y]
            Format in 2D is [ranks_x, ranks_y, ranks_z]

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        assert np.prod(
            ranks_per_axis) == self._size, "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."

        dims = self._interface.get_dimensions()

        dx = []
        for d in range(dims):
            dx.append(abs(macro_bounds[d * 2 + 1] - macro_bounds[d * 2]) / ranks_per_axis[d])

        rank_in_axis: list[int] = [None] * dims
        if ranks_per_axis[0] == 1:
            rank_in_axis[0] = 0
        else:
            rank_in_axis[0] = self._rank % ranks_per_axis[0]  # x axis
        if dims == 2:
            if ranks_per_axis[1] == 1:
                rank_in_axis[1] = 0
            else:
                rank_in_axis[1] = int(self._rank / ranks_per_axis[0])  # y axis
        elif dims == 3:
            if ranks_per_axis[2] == 1:
                rank_in_axis[2] = 0
            else:
                rank_in_axis[2] = int(self._rank / (ranks_per_axis[0] * ranks_per_axis[1]))  # z axis

            if ranks_per_axis[1] == 1:
                rank_in_axis[1] = 0
            else:
                rank_in_axis[1] = (self._rank - ranks_per_axis[0] * ranks_per_axis[1]
                                    * rank_in_axis[2]) % ranks_per_axis[0]  # y axis

        print(rank_in_axis)

        mesh_bounds = []
        for d in range(dims):
            mesh_bounds.append(dx[d] * rank_in_axis[d])
            mesh_bounds.append(dx[d] * (rank_in_axis[d] + 1))

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = macro_bounds[d * 2 + 1]

        print(mesh_bounds)

        self._logger.info("Bounding box limits are {}".format(mesh_bounds))

        return mesh_bounds