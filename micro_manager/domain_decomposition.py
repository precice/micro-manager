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

    def get_local_mesh_bounds(self, macro_bounds: list, ranks_per_axis: list) -> list:
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
        else:
            raise ValueError("Domain decomposition only supports 2D and 3D cases.")

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

    def get_local_sims_and_macro_coords(
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
        local_mesh_bounds = self.get_local_mesh_bounds(macro_bounds, ranks_per_axis)

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

    def filter_duplicate_coords(
        self,
        all_coords: list,
        all_ids: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter out vertex coordinates that are already owned by a lower-ranked rank.

        When a macro-point lies exactly on the boundary between two rank bounding
        boxes, preCICE returns it to both ranks. This function ensures every vertex
        is processed by exactly one rank — the lowest-ranked rank that received it —
        while preserving the preCICE ID-coord pairing.

        Parameters
        ----------
        all_coords : list
            List of numpy arrays, one per rank, containing vertex coordinates.
        all_ids : list
            List of arrays, one per rank, containing preCICE vertex IDs.

        Returns
        -------
        filtered_coords : numpy.ndarray
            Vertex coordinates with duplicates removed.
        filtered_ids : numpy.ndarray
            preCICE vertex IDs corresponding to the filtered coordinates.
        """
        mesh_vertex_coords = np.array(all_coords[self._rank])
        mesh_vertex_ids = np.array(all_ids[self._rank])

        seen_coords = set()
        keep_mask = np.ones(len(mesh_vertex_coords), dtype=bool)

        for rank in range(self._size):
            for i, coord in enumerate(all_coords[rank]):
                coord_key = tuple(np.round(coord, decimals=10))
                if rank < self._rank:
                    # Mark coords already claimed by earlier ranks
                    seen_coords.add(coord_key)
                elif rank == self._rank:
                    # Only keep coords not already claimed by earlier ranks
                    if coord_key in seen_coords:
                        keep_mask[i] = False
                    else:
                        seen_coords.add(coord_key)

        return mesh_vertex_coords[keep_mask], mesh_vertex_ids[keep_mask]
