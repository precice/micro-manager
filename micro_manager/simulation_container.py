from micro_simulation import MicroSimulationInterface

from typing import Optional, Any, List, Dict, Set, Iterable, Tuple
import numpy as np


class SimulationContainer:
    EntryType = Tuple[
            Optional[MicroSimulationInterface],
            Dict[str, Optional[Any]],
            int,
            np.ndarray,
        ]

    def __init__(self):
        """
        Constructs SimContainer. When model adaptivity is active, state storage needs to capture states of all models.

        Parameters
        ----------

        """
        self._sims : List[Optional[MicroSimulationInterface]] = []
        # we store one state for each potential model, associated by its name
        self._sim_checkpoints : List[Dict[str, Optional[Any]]] = []
        self._sim_gids : List[int] = []
        self._sim_gids_set : Set[int] = set()
        self._sim_coords : List[np.ndarray] = []
        self._global_num_sims : int = 0

    def initialize(self, glob_num_sims: int, local_num_sims: int, local_gids: List[int], local_coords: List[np.ndarray]) -> None:
        """
        Initializes buffers to appropriate sizes.

        Parameters
        ----------
        glob_num_sims : int
            global number of simulations
        local_num_sims : int
            number of local simulations on this rank
        local_gids : List[int]
            gids on this rank
        local_coords : List[np.ndarray]
            coords of simulation on this rank
        """
        self._global_num_sims = glob_num_sims
        self._sims = [None] * local_num_sims
        self._sim_checkpoints = [{} for _ in range(local_num_sims)]
        self._sim_gids = local_gids.copy()
        self._sim_gids_set = set(self._sim_gids)
        self._sim_coords = local_coords.copy()

    def is_sim_on_rank(self, gid: int) -> bool:
        """
        Checks if a simulation given by its gid is on this rank.

        Parameters
        ----------
        gid : int
           simulation gid

        Returns
        -------
        is_on_rank : bool
            True if simulation on this rank
        """
        return gid in self._sim_gids_set

    @property
    def global_num_sims(self) -> int:
        """
        Returns the global number of simulations.

        Returns
        -------
        global_num_sims : int
            global number of simulations
        """
        return self._global_num_sims

    @property
    def local_num_sims(self) -> int:
        """
        Returns the local number of simulations.

        Returns
        -------
        local_num_sims : int
            local number of simulations
        """
        return len(self._sims)

    @property
    def local_gids(self) -> List[int]:
        """
        Returns a reference to the local list of gids on this rank.
        Has the same order as the simulations stored in this container.

        Returns
        -------
        local_gids : List[int]
            gids on this rank
        """
        return self._sim_gids

    @property
    def local_coords(self) -> List[np.ndarray]:
        """
        Returns a reference to the local list of simulation coordinates on this rank.
        Has the same order as the simulations stored in this container.

        Returns
        -------
        local_coords : List[np.ndarray]
            coords on this rank
        """
        return self._sim_coords

    @property
    def range_lid(self) -> Iterable[int]:
        """
        Returns an iterator that yields all current local IDs.

        Returns
        -------
        lid_range : Iterable[int]
            lid iterator
        """
        return range(self.local_num_sims)

    @property
    def range_gid(self) -> Iterable[int]:
        """
        Returns an iterator that yields all global IDs.

        Returns
        -------
        gid_range : Iterable[int]
            gid iterator
        """
        return range(self.global_num_sims)

    def write_checkpoints(self, only_none: bool = False) -> None:
        """
        Writes checkpoints for respective simulations. If only_none is True, only checkpoints are written for
        simulations that do not have checkpoints yet.

        Parameters
        ----------
        only_none : bool
            Should only checkpoints be written for simulations without checkpoints.
        """
        for lid in self.range_lid:
            if self._sims[lid] is None:
                continue

            sim = self._sims[lid]
            state_dict = self._sim_checkpoints[lid]
            key = f"{sim.name}-state"

            if only_none and key in state_dict:
                continue

            state_dict[key] = sim.get_state()

    def load_checkpoints(self) -> None:
        """
        Resets the state of all simulations to the state stored in checkpoints.
        """
        for lid in self.range_lid:
            if self._sims[lid] is None:
                continue

            sim = self._sims[lid]
            state_dict = self._sim_checkpoints[lid]
            key = f"{sim.name}-state"
            if key not in state_dict:
                continue

            sim.set_state(state_dict[key])

    def clear_checkpoints(self) -> None:
        """
        Deletes all stored checkpoints.
        """
        for lid in self.range_lid:
            self._sim_checkpoints[lid].clear()

    def get_state(self, lid: int) -> Dict[str, Any]:
        """
        Gets the state of the simulation at the given local id.

        Parameters
        ----------
        lid : int
            Local id of the simulation.

        Returns
        -------
        state : Dict[str, Any]
            State of the simulation across all models.
        """
        sim = self._sims[lid]
        assert sim is not None

        state_dict = self._sim_checkpoints[lid]
        key = f"{sim.name}-state"
        state_dict[key] = sim.get_state()

        return state_dict

    def set_state(self, lid: int, state: Dict[str, Any]) -> None:
        """
        Sets the state of the simulation at the given local id to the provided state.

        Parameters
        ----------
        lid : int
            Local id of the simulation.
        state : Dict[str, Any]
            State of the simulation across all models.
        """
        state_dict = self._sim_checkpoints[lid]
        state_dict.update(state)

        sim = self._sims[lid]
        assert sim is not None
        key = f"{sim.name}-state"

        sim.set_state(state_dict[key])

    def __len__(self) -> int:
        """
        Returns the number of local simulations.

        Returns
        -------
        num_sims : int
            Number of local simulations.
        """
        return len(self._sims)

    def __iter__(self) -> Iterable[EntryType]:
        """
        Iterates over all entries in this container. Each entry is a tuple of a potential
        simulation instance, its checkpoint data, the GID and spatial coordinate.

        Returns
        -------
        iterator : Iterable[Tuple[
            Optional[MicroSimulationInterface],
            Dict[str, Optional[Any]],
            int,
            np.ndarray,
        ]]
        """
        return zip(
            self._sims,
            self._sim_checkpoints,
            self._sim_gids,
            self._sim_coords,
        )

    def __getitem__(self, lid: int) -> Optional[MicroSimulationInterface]:
        """
        Returns the simulation instance specified by its LID.

        Parameters
        ----------
        lid : int
           Local id of the simulation.

        Returns
        -------
        sim : Optional[MicroSimulationInterface]
        """
        return self._sims[lid]

    def __setitem__(self, lid: int, sim: Optional[MicroSimulationInterface]) -> None:
        """
        Writes access to simulation storage.
        This method can only access the simulation list.

        Parameters
        ----------
        lid : int
            Local id of the simulation.
        sim : Optional[MicroSimulationInterface]
            Simulation instance to be stored at lid
        """
        self._sims[lid] = sim

    def remove_sim(self, lid: int) -> None:
        """
        Removes the simulation at the given local id.

        Parameters
        ----------
        lid : int
            Local id of the simulation.
        """
        if self._sims[lid] is not None:
            self._sims[lid].destroy()
        del self._sims[lid]
        del self._sim_checkpoints[lid]
        gid = self._sim_gids[lid]
        del self._sim_gids[lid]
        del self._sim_coords[lid]
        self._sim_gids_set.remove(gid)

    def add_sim(self, gid: int, sim: Optional[MicroSimulationInterface], coord: np.ndarray) -> int:
        """
        Adds the provided simulation to this container.

        Parameters
        ----------
        gid : int
            gid of the simulation
        sim : Optional[MicroSimulationInterface]
            Simulation to add.
        coord : np.ndarray
            Coordinate of the simulation.

        Returns
        -------
        lid : int
            local id of the added simulation.
        """
        self._sims.append(sim)
        self._sim_gids.append(gid)
        self._sim_gids_set.add(gid)
        self._sim_checkpoints.append({})
        self._sim_coords.append(coord)

        # return lid of sim
        return len(self._sims) - 1

    def empty(self) -> bool:
        """
        Checks if the container is empty.

        Returns
        -------
        empty : bool
            True if the container is empty, False otherwise.
        """
        return len(self._sims) == 0