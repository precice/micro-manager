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
        if np.prod(ranks_per_axis) != self._size:
            raise ValueError(
                "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."
            )

        dims = len(ranks_per_axis)

        rank_in_axis = None
        if dims == 3:
            for x in range(ranks_per_axis[0]):
                for y in range(ranks_per_axis[1]):
                    for z in range(ranks_per_axis[2]):
                        n = (
                            y
                            + z * ranks_per_axis[1]
                            + x * ranks_per_axis[1] * ranks_per_axis[2]
                        )
                        if n == self._rank:
                            rank_in_axis = [x, y, z]
                            break
                    if rank_in_axis is not None:
                        break
                if rank_in_axis is not None:
                    break
        elif dims == 2:
            for x in range(ranks_per_axis[0]):
                for y in range(ranks_per_axis[1]):
                    n = x + y * ranks_per_axis[0]
                    if n == self._rank:
                        rank_in_axis = [x, y]
                        break
                if rank_in_axis is not None:
                    break
        else:
            raise ValueError(
                f"Unsupported number of dimensions: {dims}. Only 2D and 3D domains are supported."
            )

        if rank_in_axis is None:
            raise RuntimeError(
                f"Could not determine rank position in domain decomposition for rank {self._rank}"
            )

        dx = []
        for d in range(dims):
            dx.append(
                abs(macro_bounds[d * 2 + 1] - macro_bounds[d * 2]) / ranks_per_axis[d]
            )

        mesh_bounds = []
        for d in range(dims):
            mesh_bounds.append(macro_bounds[d * 2] + rank_in_axis[d] * dx[d])
            mesh_bounds.append(macro_bounds[d * 2] + (rank_in_axis[d] + 1) * dx[d])

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = macro_bounds[d * 2 + 1]

        return mesh_bounds

    def decompose_micro_simulations(
        self, macro_bounds: list, ranks_per_axis: list, macro_coords: np.ndarray
    ) -> tuple[int, list]:
        """
        Decompose the micro simulations among all ranks based on their positions in the macro domain.

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
        macro_coords : numpy.ndarray
            The coordinates associated to the IDs and corresponding data values (dim * size)

        Returns
        -------
        micro_sims_on_rank : int
            Number of micro simulations assigned to this rank.
        macro_coords_on_this_rank : list
            List of macro coordinates assigned to this rank.
        """
        local_mesh_bounds = self.decompose_macro_domain(macro_bounds, ranks_per_axis)

        macro_coords_on_this_rank = []

        micro_sims_on_rank = 0
        for position in macro_coords:
            inside = True
            for d in range(len(ranks_per_axis)):
                if not (
                    position[d] >= local_mesh_bounds[d * 2]
                    and position[d] <= local_mesh_bounds[d * 2 + 1]
                ):
                    inside = False
                    break
            if inside:
                macro_coords_on_this_rank.append(position)
                micro_sims_on_rank += 1

        return micro_sims_on_rank, macro_coords_on_this_rank
