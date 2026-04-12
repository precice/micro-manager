"""
Class DomainDecomposer provides the method decompose_macro_domain which returns partitioned bounds
"""

import numpy as np
from micro_manager.config import Config
from typing import Callable


class DomainDecomposer:
    def __init__(self, configurator: Config, rank: int, size: int) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        rank : int
            MPI rank.
        size : int
            Total number of MPI processes.
        """
        self._rank = rank
        self._size = size

        self._ranks_per_axis = (
            configurator.get_ranks_per_axis()
        )  # Check if ranks per axis is provided in the configuration file for parallel runs

        self._dims = len(self._ranks_per_axis)

        self._is_minimum_access_region_size_specified = False

        self._decomposition_type = configurator.get_decomposition_type()

        self._macro_bounds = configurator.get_macro_domain_bounds()

        self._get_local_mesh_bounds = self._get_local_mesh_bounds_variant()

        self._minimum_access_region_size: list = (
            configurator.get_minimum_access_region_size()
        )
        if self._minimum_access_region_size:  # if list is not empty
            self._is_minimum_access_region_size_specified = True

    def get_local_mesh_bounds(self) -> list:
        """
        Get the local mesh bounds for this rank based on the domain decomposition type specified in the configuration file.

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        return self._get_local_mesh_bounds()

    def _get_uniform_local_mesh_bounds(self) -> list:
        """
        Decompose the macro domain equally among all ranks, if the Micro Manager is run in parallel.

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        if np.prod(self._ranks_per_axis) != self._size:
            raise ValueError(
                "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."
            )

        if self._dims == 3:
            for z in range(self._ranks_per_axis[2]):
                for y in range(self._ranks_per_axis[1]):
                    for x in range(self._ranks_per_axis[0]):
                        n = (
                            x
                            + y * self._ranks_per_axis[0]
                            + z * self._ranks_per_axis[0] * self._ranks_per_axis[1]
                        )
                        if n == self._rank:
                            rank_in_axis = [x, y, z]
        elif self._dims == 2:
            for y in range(self._ranks_per_axis[1]):
                for x in range(self._ranks_per_axis[0]):
                    n = x + y * self._ranks_per_axis[0]
                    if n == self._rank:
                        rank_in_axis = [x, y]
        else:
            raise ValueError("Domain decomposition only supports 2D and 3D cases.")

        dx = []
        for d in range(self._dims):
            dx.append(
                abs(self._macro_bounds[d * 2 + 1] - self._macro_bounds[d * 2])
                / self._ranks_per_axis[d]
            )

        mesh_bounds = []
        for d in range(self._dims):
            if rank_in_axis[d] > 0:
                mesh_bounds.append(self._macro_bounds[d * 2] + rank_in_axis[d] * dx[d])
                mesh_bounds.append(
                    self._macro_bounds[d * 2] + (rank_in_axis[d] + 1) * dx[d]
                )
            elif rank_in_axis[d] == 0:
                mesh_bounds.append(self._macro_bounds[d * 2])
                mesh_bounds.append(self._macro_bounds[d * 2] + dx[d])

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == self._ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = self._macro_bounds[d * 2 + 1]

        return mesh_bounds

    def _get_nonuniform_local_mesh_bounds(self) -> list:
        """
        Decompose the macro domain among all ranks with an non-uniform distribution, if the Micro Manager is run in parallel.
        The non-uniform distribution is based on a geometric progression, where the size of the local mesh bounds increases
        by a factor of 2 in each rank. This is just one of many possible non-uniform domain decompositions.

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        if np.prod(self._ranks_per_axis) != self._size:
            raise ValueError(
                "Total number of processors provided in the Micro Manager configuration and in the MPI execution command do not match."
            )

        if self._dims == 3:
            for z in range(self._ranks_per_axis[2]):
                for y in range(self._ranks_per_axis[1]):
                    for x in range(self._ranks_per_axis[0]):
                        n = (
                            x
                            + y * self._ranks_per_axis[0]
                            + z * self._ranks_per_axis[0] * self._ranks_per_axis[1]
                        )
                        if n == self._rank:
                            rank_in_axis = [x, y, z]
        elif self._dims == 2:
            for y in range(self._ranks_per_axis[1]):
                for x in range(self._ranks_per_axis[0]):
                    n = x + y * self._ranks_per_axis[0]
                    if n == self._rank:
                        rank_in_axis = [x, y]
        else:
            raise ValueError("Domain decomposition only supports 2D and 3D cases.")

        dx: list = []
        for d in range(self._dims):
            dx.append(np.zeros(self._ranks_per_axis[d]))
            for rank in range(self._ranks_per_axis[d]):
                if rank == 0:
                    base_dx = abs(
                        self._macro_bounds[d * 2 + 1] - self._macro_bounds[d * 2]
                    ) / (2 ** self._ranks_per_axis[d] - 1)
                    if self._is_minimum_access_region_size_specified:
                        if base_dx < self._minimum_access_region_size[d]:
                            base_dx = self._minimum_access_region_size[d]

                    dx[d][rank] = base_dx
                else:
                    dx[d][rank] = 2 * dx[d][rank - 1]

        mesh_bounds = []
        for d in range(self._dims):
            rank = rank_in_axis[d]
            if rank > 0:
                min_bound = self._macro_bounds[d * 2] + sum(dx[d][:rank])
                mesh_bounds.append(min_bound)
                mesh_bounds.append(min_bound + dx[d][rank])
            elif rank == 0:
                mesh_bounds.append(self._macro_bounds[d * 2])
                mesh_bounds.append(self._macro_bounds[d * 2] + dx[d][rank])

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == self._ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = self._macro_bounds[d * 2 + 1]

        return mesh_bounds

    def _get_local_mesh_bounds_variant(self) -> Callable:
        """
        Get uniform or nonuniform variant of calculating local mesh bounds

        Returns
        -------
        get_local_mesh_bounds_variant : function
            Function to calculate local mesh bounds based on the decomposition type specified in the configuration file.
        """
        if self._decomposition_type == "uniform":
            return self._get_uniform_local_mesh_bounds
        elif self._decomposition_type == "nonuniform":
            return self._get_nonuniform_local_mesh_bounds
        else:
            raise ValueError(
                "Decomposition type can be either 'uniform' or 'nonuniform'."
            )

    def get_local_sims_and_macro_coords(
        self, macro_coords: np.ndarray
    ) -> tuple[int, list[np.ndarray]]:
        """
        Decompose the micro simulations among all ranks based on their positions in the macro domain.
        Parameters
        ----------
        macro_coords : numpy.ndarray
            Array containing the coordinates of the macro points corresponding to the micro simulations.

        Returns
        -------
        micro_sims_on_rank : int
            Number of micro simulations pertaining to this rank.
        macro_coords_on_this_rank : list of numpy.ndarray
            List of coordinates of the macro points pertaining to this rank.
        """
        local_mesh_bounds = self.get_local_mesh_bounds()

        macro_coords_on_this_rank = []

        micro_sims_on_rank = 0
        for position in macro_coords:
            inside = True
            for d in range(self._dims):
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
