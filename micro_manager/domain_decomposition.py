"""
Class DomainDecomposer provides the method decompose_macro_domain which returns partitioned bounds
"""
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import brentq

from micro_manager.config import Config
from micro_manager.tools.mpi_handler import MPIHandler, MPI
from micro_manager.tools.logging_wrapper import Logger


class DomainDecomposer(ABC):
    def __init__(self, config: Config, mpi: MPIHandler, log: Logger) -> None:
        self._mpi: MPIHandler = mpi
        self._log: Logger = log
        # Check if ranks per axis is provided in the configuration file for parallel runs
        self._ranks_per_axis: List[int] = config.ranks_per_axis()
        self._dims: int = len(self._ranks_per_axis)
        self._macro_bounds: List[float] = config.macro_domain_bounds()

        # initial checks
        if self._dims not in [2, 3]:
            raise ValueError("Domain decomposition only supports 2D and 3D cases.")
        if self._dims * 2 != len(self._macro_bounds):
            raise ValueError("Provided macro mesh bounds are of incorrect dimension")
        if self._dims != len(self._ranks_per_axis):
            raise ValueError(
                "Provided ranks combination is of incorrect dimension "
                "and does not match the dimensions of the macro mesh."
            )

    @abstractmethod
    def get_mesh_bounds(self) -> List[float]:
        """
        Get the local mesh bounds for this rank based on the domain decomposition type
        specified in the configuration file.

        Returns
        -------
        mesh_bounds : List[float]
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        pass

    @abstractmethod
    def partition(
        self,
        vertex_coords: List[np.ndarray],
        vertex_ids: List[int],
        access_region: List[float],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Decompose the micro simulations among all ranks based on their positions in the macro domain.

        Parameters
        ----------
        vertex_coords : List[np.ndarray]
            Array containing the coordinates of the macro points corresponding to the micro simulations.
        vertex_ids : List[int]
            Array containing IDs associated with the coordinates.
        access_region : List[float]
            Either the mesh_bounds as returned by get_mesh_bounds or the macro_domain.

        Returns
        -------
        local_vertex_coords : List[np.ndarray]
            List of coordinates of the macro points pertaining to this rank.
        local_vertex_ids : List[int]
            List of corresponding vertex IDs
        """
        pass

    def filter_duplicates(
        self,
        global_vertex_coords: List[List[np.ndarray]],
        global_vertex_ids: List[List[int]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Filter out vertex coordinates that are already owned by a lower-ranked rank.

        When a macro-point lies exactly on the boundary between two rank bounding
        boxes, preCICE returns it to both ranks. This function ensures every vertex
        is processed by exactly one rank — the lowest-ranked rank that received it —
        while preserving the preCICE ID-coord pairing.

        Parameters
        ----------
        global_vertex_coords : List[List[np.ndarray]]
            Lists of rank local vertex coordinates, one per rank.
        global_vertex_ids : List[List[int]]
            Lists of rank local vertex IDs (preCICE), one per rank.

        Returns
        -------
        filtered_coords : List[np.ndarray]
            Local vertex coordinates with duplicates removed.
        filtered_ids : List[np.ndarray]
            Local preCICE vertex IDs corresponding to the filtered coordinates.
        """
        mesh_vertex_coords = np.array(global_vertex_coords[self._mpi.rank])
        mesh_vertex_ids = np.array(global_vertex_ids[self._mpi.rank])

        seen_coords = set()
        keep_mask = np.ones(len(mesh_vertex_coords), dtype=bool)

        for rank in range(self._mpi.size):
            for i, coord in enumerate(global_vertex_coords[rank]):
                coord_key = tuple(np.round(coord, decimals=10))
                if rank < self._mpi.rank:
                    # Mark coords already claimed by earlier ranks
                    seen_coords.add(coord_key)
                elif rank == self._mpi.rank:
                    # Only keep coords not already claimed by earlier ranks
                    if coord_key in seen_coords:
                        keep_mask[i] = False
                    else:
                        seen_coords.add(coord_key)

        return mesh_vertex_coords[keep_mask], mesh_vertex_ids[keep_mask]

    def finalize(
        self, local_vertex_coords: List[np.ndarray]
    ) -> Tuple[int, int, List[int]]:
        """
        Prints decomposition statistics of all ranks and computes local and global counts.

        Parameters
        ----------
        local_vertex_coords : List[np.ndarray]
            Vertex coords after partitioning.

        Returns
        -------
        local_num_sims : int
            Local number of simulations
        global_num_sims : int
            Global number of simulations
        sims_per_rank : List[int]
        """
        local_num_sims = len(local_vertex_coords)
        nms_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
        # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
        # simulations have been created by previous ranks, so that it can set
        # the correct global IDs
        self._mpi.comm.Allgatherv(np.array(local_num_sims), nms_all_ranks)

        max_nms = np.max(nms_all_ranks)
        min_nms = np.min(nms_all_ranks)

        if max_nms != min_nms:
            # if the number of maximum and minimum micro simulations per rank are different
            self._log.log_info_rank_zero(
                "The following ranks have the maximum number of micro simulations "
                f"({max_nms}): {np.where(nms_all_ranks == max_nms)[0]}"
            )
            self._log.log_info_rank_zero(
                "The following ranks have the minimum number of micro simulations "
                f"({min_nms}): {np.where(nms_all_ranks == min_nms)[0]}"
            )
        else:
            # if the number of maximum and minimum micro simulations per rank are the same
            self._log.log_info_rank_zero(
                f"All ranks have the same number of micro simulations: {max_nms}"
            )

        # Get global number of micro simulations
        global_num_sims = np.sum(nms_all_ranks)
        self._log.log_info_rank_zero(
            f"Total number of micro simulations: {global_num_sims}"
        )

        return local_num_sims, global_num_sims, nms_all_ranks


class NoOpDecomp(DomainDecomposer):
    """
    Performs no decomposition. Assigns full domain to local rank.
    """

    def __init__(self, config: Config, mpi: MPIHandler, log: Logger) -> None:
        super().__init__(config, mpi, log)
        self._bounds: List[float] = config.macro_domain_bounds()

    def get_mesh_bounds(self) -> List[float]:
        return self._bounds

    def partition(
        self,
        vertex_coords: List[np.ndarray],
        vertex_ids: List[int],
        access_region: List[float],
    ) -> Tuple[List[np.ndarray], List[int]]:
        if len(vertex_ids) == 0:
            raise RuntimeError(
                "The macro mesh has no vertices in the specified access region."
            )

        return vertex_coords, vertex_ids


class GridDecomp(DomainDecomposer, ABC):
    def __init__(self, config: Config, mpi: MPIHandler, log: Logger) -> None:
        super().__init__(config, mpi, log)

    def partition(
        self,
        vertex_coords: List[np.ndarray],
        vertex_ids: List[int],
        access_region: List[float],
    ) -> Tuple[List[np.ndarray], List[int]]:
        mesh_bounds: List[float] = self.get_mesh_bounds()

        # Apply filtering if the access region is equal to the local mesh bounds
        # Filtering is required as in this case duplicates can arise due to numerical issues.
        if np.all(np.array(mesh_bounds) == np.array(access_region)):
            # Gather all vertex coords and IDs from all ranks onto all ranks,
            # filter out coords already claimed by lower-ranked ranks.
            # When load balancing, all ranks receive all coords. No duplicates can arise.
            # TODO: Avoid the allgather by smartly selecting the relevant coordinates
            global_vertex_coords: List[List[np.ndarray]] = self._mpi.comm.allgather(
                vertex_coords
            )
            global_vertex_ids: List[List[int]] = self._mpi.comm.allgather(vertex_ids)
            vertex_coords, vertex_ids = self.filter_duplicates(
                global_vertex_coords, global_vertex_ids
            )

        if len(vertex_ids) == 0:
            self._log.log_warning(
                "The access region of this rank has no macro-scale vertices. "
                "This rank will not have any micro simulations. "
                "To avoid this, change the domain decomposition"
            )

        # perform actual partitioning
        local_coords = []
        local_ids = []

        for idx, coord in enumerate(vertex_coords):
            inside = True
            for d in range(self._dims):
                if not (
                    coord[d] >= mesh_bounds[d * 2]
                    and coord[d] <= mesh_bounds[d * 2 + 1]
                ):
                    inside = False
                    break
            if inside:
                local_coords.append(coord)
                local_ids.append(vertex_ids[idx])

        return local_coords, local_ids

    def _calc_rank_in_axis(self) -> List[int]:
        if np.prod(self._ranks_per_axis) != self._mpi.size:
            raise ValueError(
                "Total number of processors provided in the Micro Manager "
                "configuration and in the MPI execution command do not match."
            )

        rank_in_axis: Optional[List[int]] = None
        # force ranks_per_axis to be 3D for 2D case with value 0
        if self._dims == 2:
            self._ranks_per_axis.append(0)

        for z in range(self._ranks_per_axis[2]):
            for y in range(self._ranks_per_axis[1]):
                for x in range(self._ranks_per_axis[0]):
                    n = (
                        x
                        + y * self._ranks_per_axis[0]
                        + z * self._ranks_per_axis[0] * self._ranks_per_axis[1]
                    )
                    if n == self._mpi.rank:
                        rank_in_axis = [x, y, z]

        if rank_in_axis is None:
            raise ValueError("Provided invalid values for ranks per axis.")
        # extract data for 2D case and restore ranks_per_axis
        if self._dims == 2:
            rank_in_axis = rank_in_axis[:-1]
            self._ranks_per_axis.pop()

        return rank_in_axis


class UniformGridDecomp(GridDecomp):
    def __init__(self, config: Config, mpi: MPIHandler, log: Logger) -> None:
        super().__init__(config, mpi, log)

    def get_mesh_bounds(self) -> List[float]:
        """
        Decompose the macro domain equally among all ranks, if the Micro Manager is run in parallel.

        Returns
        -------
        mesh_bounds : List[float]
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        rank_in_axis: List[int] = self._calc_rank_in_axis()

        mesh_bounds: List[int] = []
        for d in range(self._dims):
            dx = (
                abs(self._macro_bounds[d * 2 + 1] - self._macro_bounds[d * 2])
                / self._ranks_per_axis[d]
            )

            if rank_in_axis[d] > 0:
                mesh_bounds.append(self._macro_bounds[d * 2] + rank_in_axis[d] * dx)
                mesh_bounds.append(
                    self._macro_bounds[d * 2] + (rank_in_axis[d] + 1) * dx
                )
            elif rank_in_axis[d] == 0:
                mesh_bounds.append(self._macro_bounds[d * 2])
                mesh_bounds.append(self._macro_bounds[d * 2] + dx)

            # Adjust the maximum bound to be exactly the domain size
            if rank_in_axis[d] + 1 == self._ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = self._macro_bounds[d * 2 + 1]

        return mesh_bounds


class NonUniformGridDecomp(GridDecomp):
    def __init__(self, config: Config, mpi: MPIHandler, log: Logger) -> None:
        super().__init__(config, mpi, log)

        self._minimum_access_region_size: List[
            int
        ] = config.minimum_access_region_size()
        self._has_minimum_access_region_size = len(self._minimum_access_region_size) > 0

    def get_mesh_bounds(self) -> List[float]:
        """
        Decompose the macro domain among all ranks with an non-uniform distribution, if the Micro Manager is run in parallel.
        The non-uniform distribution is based on a geometric progression, where the size of the local mesh bounds increases
        by a factor of 2 in each rank. This is just one of many possible non-uniform domain decompositions.

        Returns
        -------
        mesh_bounds : List[float
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        rank_in_axis: List[int] = self._calc_rank_in_axis()

        mesh_bounds = []
        multiplier = 2  # factor by which the local mesh bounds increase in each rank. 2 means geometric progression.
        for d in range(self._dims):
            macro_bounds_diff = abs(
                self._macro_bounds[d * 2 + 1] - self._macro_bounds[d * 2]
            )

            dx0 = (
                macro_bounds_diff
                * (multiplier - 1)
                / (multiplier ** self._ranks_per_axis[d] - 1)
            )

            if self._has_minimum_access_region_size:
                if dx0 < self._minimum_access_region_size[d]:
                    dx0 = self._minimum_access_region_size[d]
                    n_ranks = self._ranks_per_axis[d]

                    def _geom_sum_residual(r):
                        return dx0 * (r**n_ranks - 1) / (r - 1) - macro_bounds_diff

                    # Find upper bracket where residual is positive
                    r_upper = 2.0
                    while _geom_sum_residual(r_upper) <= 0:
                        r_upper *= 2.0

                    # When the minimum access region size is specified,
                    # the multiplier is numerically calculated such that the sum of
                    # the geometric progression of the local mesh bounds equals
                    # the macro domain size in that axis.
                    multiplier = brentq(_geom_sum_residual, 1.0 + 1e-12, r_upper)

            dx = np.zeros(self._ranks_per_axis[d])

            for rank in range(self._ranks_per_axis[d]):
                if rank == 0:
                    dx[rank] = dx0
                else:
                    dx[rank] = multiplier * dx[rank - 1]

            rank = rank_in_axis[d]
            if rank == 0:
                mesh_bounds.append(self._macro_bounds[d * 2])
                mesh_bounds.append(self._macro_bounds[d * 2] + dx[rank])

            if rank > 0:
                min_bound = self._macro_bounds[d * 2] + sum(dx[:rank])
                mesh_bounds.append(min_bound)
                mesh_bounds.append(min_bound + dx[rank])

            # Adjust the maximum bound of the access region of ranks on the boundary to be exactly the upper bound of the macro domain
            if rank_in_axis[d] + 1 == self._ranks_per_axis[d]:
                mesh_bounds[d * 2 + 1] = self._macro_bounds[d * 2 + 1]

        return mesh_bounds


def create_domain_decomposer(
    config: Config, mpi: MPIHandler, log: Logger
) -> DomainDecomposer:
    """
    Creates a decomposition object according to the current configuration.

    Parameters
    ----------
    config : Config
        configuration object
    mpi : MPIHandler
        MPIHandler object
    log : Logger
        Logger object

    Returns
    -------
    domain_decomposer : DomainDecomposer
        decomposition object
    """
    if not mpi.is_parallel():
        return NoOpDecomp(config, mpi, log)

    decomp_type = config.decomposition_type()
    match decomp_type:
        case "uniform":
            return UniformGridDecomp(config, mpi, log)
        case "nonuniform":
            return NonUniformGridDecomp(config, mpi, log)

    raise ValueError(f"Unknown decomposition type: {decomp_type}")
