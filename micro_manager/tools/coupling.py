from micro_manager.config import Config
from micro_manager.simulation_container import SimulationContainer

import precice as p
import numpy as np
from typing import List, Dict, Any, Optional
from mpi4py import MPI


class CouplingHandler:
    """
    Manages all coupling aspects. Acts as an interface to preCICE and contains
    all preCICE relevant data.
    """

    def __init__(self, config: Config, rank: int, size: int, comm: MPI.Comm, simulation_container: SimulationContainer):
        """
        Creates a new CouplingHandler instance in which a connection to preCICE is established.

        Parameters
        ----------
        config : Config
            configuration object
        rank : int
            rank within the communicator
        size : int
            size within the communicator
        comm : MPI.Comm
            MPI communicator object
        simulation_container : SimulationContainer
            Simulation container object.
        """
        self._rank : int = rank
        self._size : int = size
        self._comm : MPI.Comm = comm

        self._micro_dt : float = config.micro_dt()
        self._macro_mesh_name : str = config.macro_mesh_name()
        # Data names of data to output to the snapshot database
        self._write_data_names : List[str] = config.write_data_names()
        # Data names of data to read as input parameter to the simulations
        self._read_data_names : List[str] = config.read_data_names()

        # Define the preCICE Participant
        self._participant : p.Participant = p.Participant(
            "Micro-Manager",
            config.precice_config_file_name(),
            rank,
            size,
        )
        self._sim_container : SimulationContainer = simulation_container

        self._access_region : List[List[float]] = []
        # Based on the access region, this rank is associated with a subset of
        # IDs and coordinates of the global macro mesh within preCICE.
        self._mesh_vertex_ids : List[int] = []
        self._mesh_vertex_coords : List[np.ndarray] = []
        self._global_mesh_vertex_coords : List[np.ndarray] = []
        self._mesh_dims : int = self._participant.get_mesh_dimensions(self._macro_mesh_name)

    @property
    def write_data_names(self) -> List[str]:
        return self._write_data_names
    @property
    def read_data_names(self) -> List[str]:
        return self._read_data_names
    @property
    def macro_mesh_name(self) -> str:
        return self._macro_mesh_name
    @property
    def registered_vertex_ids(self) -> List[int]:
        return self._mesh_vertex_ids
    @property
    def registered_vertex_coords(self) -> List[np.ndarray]:
        return self._mesh_vertex_coords
    @property
    def global_vertex_coords(self) -> List[np.ndarray]:
        return self._global_mesh_vertex_coords
    @property
    def dt(self) -> float:
        return float(np.minimum(self._micro_dt, self._participant.get_max_time_step_size()))
    @property
    def micro_dt(self) -> float:
        return self._micro_dt
    @property
    def participant(self):
        return self._participant
    @property
    def mesh_dims(self) -> int:
        return self._mesh_dims
    @property
    def num_registered_vertices(self) -> int:
        return len(self._mesh_vertex_coords)

    def set_access_region(self, access_bounds: List[List[float]]) -> None:
        """
        Sets the region in which the local rank accesses the mesh.

        Parameters
        ----------
        access_bounds : List[List[float]]
            List of lower and upper bound.
        """
        self._access_region.clear()
        self._access_region.extend(access_bounds)

        self._participant.set_mesh_access_region(
            self._macro_mesh_name,
            self._access_region,
        )

    def override_registered_vertices(self, coords: List[np.ndarray], ids: List[int]) -> None:
        """
        Sets the registered vertex IDs and coordinates to the provided parameters. Used for example
        after some postprocessing (filtering).

        Parameters
        ----------
        coords : List[np.ndarray]
            List of coordinates.
        ids : List[int]
            List of IDs
        """
        self._mesh_vertex_coords.clear()
        self._mesh_vertex_coords.extend(coords)
        self._mesh_vertex_ids.clear()
        self._mesh_vertex_ids.extend(ids)

    def load_global_vertex_coords(self) -> None:
        """
        Broadcasts all coordinates between all ranks.
        """
        glob_coords = self._comm.allgather(self._mesh_vertex_coords)
        self._global_mesh_vertex_coords.extend(
            np.array(glob_coords).reshape((-1, self.mesh_dims))
        )

    def apply_access_region(self) -> None:
        """
        Reads the IDs and coordinates defined by the access region.
        """
        ids, coords = self._participant.get_mesh_vertex_ids_and_coordinates(self._macro_mesh_name)
        self._mesh_vertex_ids.extend(ids)
        self._mesh_vertex_coords.extend(coords)

    def initialize(self) -> None:
        """
        Initializes the preCICE participant.
        """
        self._participant.initialize()

    def finalize(self) -> None:
        """
        Finalizes the preCICE participant.
        """
        self._participant.finalize()

    def is_ongoing(self) -> bool:
        """
        Returns whether the preCICE coupling is ongoing.

        Returns
        -------
        ongoing : bool
            True if the preCICE coupling is ongoing.
        """
        return self._participant.is_coupling_ongoing()

    def is_time_window_complete(self) -> bool:
        """
        Checks if the current coupling time window is completed.

        Returns
        -------
        is_complete : bool
            True if time window is completed.
        """
        return self._participant.is_time_window_complete()

    def requires_writing_checkpoint(self) -> bool:
        """
        Checks if a checkpoint should be written.

        Returns
        -------
        should_write : bool
            True if the checkpoint should be written.
        """
        return self._participant.requires_writing_checkpoint()

    def requires_reading_checkpoint(self) -> bool:
        """
        Checks if a checkpoint should be read.

        Returns
        -------
        should_read : bool
            True if the checkpoint should be read.
        """
        return self._participant.requires_reading_checkpoint()

    def advance(self, dt: Optional[float]=None) -> None:
        """
        Attempts to advance the coupling to the next time step.

        Parameters
        ----------
        dt : Optional[float]
            Time step size.
        """
        dt = dt or self.dt
        self._participant.advance(dt)

    def read_from_precice(self, dt: Optional[float]=None, read_buffer: Optional[Dict[str, List[Any]]]=None) -> List[Dict[str, Any]]:
        """
        Read data from preCICE.

        Parameters
        ----------
        dt : Optional[float]
            Time step size at which data is to be read from preCICE.
        read_buffer : Optional[Dict[str, List[Any]]]
            Buffer in which fields from preCICE are read into.

        Returns
        -------
        local_read_data : List[Dict[str, Any]]
            List of dicts in which keys are names of data being read and the values are the data from preCICE.
        """
        read_data: Dict[str, List[Any]] = read_buffer or {}
        read_data.clear()
        read_data.update({name:[] for name in self._read_data_names})
        read_vertex_ids : List[int] = self._sim_container.local_gids
        dt = dt or self.dt

        for name in self._read_data_names:
            read_data.update(
                {
                    name: self._participant.read_data(
                        self._macro_mesh_name, name, read_vertex_ids, dt
                    )
                }
            )

        return [dict(zip(read_data, t)) for t in zip(*read_data.values())]

    def write_to_precice(self, data: List[Dict[str, Any]]) -> None:
        """
        Write data to preCICE.

        Parameters
        ----------
        data : list
            List of dicts in which keys are names of data and the values are the data to be written to preCICE.
        """
        write_vertex_ids: List[int] = self._sim_container.local_gids
        data_dict: Dict[str, List[Any]] = {dname: [] for dname in self._write_data_names}

        for d in data:
            for dname in self._write_data_names:
                data_dict[dname].append(d[dname])

        if self._sim_container.empty():
            write_vertex_ids = []

        for dname in self._write_data_names:
            self._participant.write_data(
                self._macro_mesh_name,
                dname,
                write_vertex_ids,
                data_dict[dname],
            )