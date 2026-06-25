from copy import deepcopy
from typing import Hashable, Set, Dict, Any, Optional, List, Union
import sys
import numpy as np

from micro_manager.simulation_container import SimulationContainer
from micro_manager.micro_simulation import MicroSimulationInterface
from micro_manager.interpolation import Interpolator
from micro_manager.config import Config
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler


class CrashHandler:
    """
    CrashHandler manages and tracks simulation crashes.
    If enabled, it can interpolate results of crashed simulations.
    """

    def __init__(
        self, logger: Logger, mpi: MPIHandler, sim_container: SimulationContainer
    ):
        """
        Constructs an empty crash handler.

        Parameters
        ----------
        logger : Logger
            Logger object.
        mpi : MPIHandler
            MPIHandler object.
        sim_container : SimulationContainer
            Simulation container object.
        """
        self._logger: Logger = logger
        self._mpi: MPIHandler = mpi
        self._sim_container: SimulationContainer = sim_container
        self._interp_id: Hashable = 0

        self._enable_interp: bool = False
        self._crashed_sims: Set[int] = set()

        self._crash_threshold: float = 0.2

    def initialize(self, config: Config) -> None:
        """
        Initializes crash interpolation.

        Parameters
        ----------
        config : Config
            Configuration object.
        """
        self._interp_id = config.crashed_sim_interpolation_id()
        self._crash_threshold = config.crashed_sim_interpolation_threshold()
        self._enable_interp = config.enable_crashed_sim_interpolation()

        if self._enable_interp:
            if not Interpolator.is_id_valid(self._interp_id):
                raise ValueError("Invalid interpolation id in CrashHandler.")

    def solve_micro_safe(
        self,
        lid: int,
        sim: MicroSimulationInterface,
        micro_input: Dict[str, Any],
        dt: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Provides a safe environment and attempts to solve the provided micro simulation.

        Parameters
        ----------
        lid : int
            LID of the micro simulation.
        sim : MicroSimulationInterface
            Micro Simulation object to be solved.
        micro_input : Dict[str, Any]
            Input to solve method.
        dt : float
            Current delta time.

        Returns
        -------
        solve_result : Optional[Dict[str, Any]]
            Micro Simulation solve call result. None on failure.
        """
        result: Optional[Dict[str, Any]] = None

        try:
            result = sim.solve(micro_input, dt)
        except Exception as e:
            self._logger.log_error(
                "Micro simulation at macro coordinates {} has experienced an error. "
                "See next entry on this rank for error message.".format(
                    self._sim_container.local_coords[lid]
                )
            )
            self._logger.log_error(e)
            self.register_sim_crash(self._sim_container.local_gids[lid])

        return result

    def interpolate_outputs(
        self,
        selected_lids: List[int],
        query_lids: Set[int],
        sim_inputs: List[Dict[str, Any]],
        sim_outputs: List[Optional[Dict[str, Any]]],
    ) -> None:
        """
        If interpolation is enabled, interpolates outputs for crashed simulations.
        Otherwise, only checks if crashed have occurred.

        Parameters
        ----------
        selected_lids : List[int]
            LIDs that may be considered as interpolation support points.
        query_lids : List[int]
            LIDs for which outputs should be interpolated.
        sim_inputs : List[Dict[str, Any]]
            List of simulation inputs. (of all local simulations)
        sim_outputs : List[Optional[Dict[str, Any]]]
            List of simulation outputs. (of all local simulations)
        """
        if not self._enable_interp:
            if self.get_glob_num_crashes() > 0:
                self._logger.log_error(
                    "Exiting simulation after micro simulation crash."
                )
                sys.exit()
            return

        interp = Interpolator.get_instance(self._interp_id)
        support_lids: List[int] = [
            lid
            for lid in selected_lids
            if lid not in query_lids and sim_outputs[lid] is not None
        ]
        if len(support_lids) <= interp.get_min_support_size():
            self._logger.log_error(
                "Exiting simulation after micro simulation crash. Not enough data for interpolation."
            )
            sys.exit()

        np_inputs = self._conv_to_ndarray(sim_inputs)
        np_outputs = self._conv_to_ndarray(sim_outputs)
        interp.set_local_data(
            np_inputs[support_lids, :],
            np_inputs[list(query_lids), :],
            np_outputs[support_lids, :],
        )
        interp_outputs = interp.interpolate()

        for interp_idx, lid in enumerate(query_lids):
            result = deepcopy(sim_outputs[support_lids[0]])
            offset = 0
            for key, val in result.items():
                size = self._load_size(val)
                extracted_val = interp_outputs[interp_idx, offset : offset + size]
                if size == 1 and not (
                    isinstance(val, np.ndarray) or isinstance(val, list)
                ):
                    extracted_val = type(val)(extracted_val[0])
                result[key] = extracted_val
                offset += size

            sim_outputs[lid] = result

    def has_sim_crashed(self, gid: int) -> bool:
        """
        Checks if the simulation given by its GID has crashed.

        Parameters
        ----------
        gid : int
            GID of simulation.

        Returns
        -------
        has_sim_crashed : bool
            True if simulation has crashed, False otherwise.
        """
        return gid in self._crashed_sims

    def register_sim_crash(self, gid: int) -> None:
        """
        Marks the simulation given by its GID as crashed.

        Parameters
        ----------
        gid : int
            GID of simulation.
        """
        self._crashed_sims.add(gid)

    def reset(self) -> None:
        """
        Resets the crash information.
        Sets all simulations as not crashed.
        """
        self._crashed_sims.clear()

    def get_glob_num_crashes(self) -> int:
        """
        Gets the global amount of crashed simulations.

        Returns
        -------
        global_num_crashes : int
            Global number of simulation crashes.
        """
        crashed_sims_on_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
        self._mpi.comm.Allgather(
            np.int64(len(self._crashed_sims)), crashed_sims_on_all_ranks
        )
        return np.sum(crashed_sims_on_all_ranks)

    def check_crash_ratio(self) -> None:
        """
        Checks if the ratio of crashed simulations is within the valid range.
        If not, then the Micro Manager is terminated.
        """
        if not self._enable_interp:
            return

        crash_ratio = self.get_glob_num_crashes() / self._sim_container.global_num_sims
        if crash_ratio > self._crash_threshold:
            self._logger.log_info(
                "{:.1%} of the micro simulations have crashed exceeding the threshold of {:.1%}. "
                "Exiting simulation.".format(crash_ratio, self._crash_threshold)
            )
            sys.exit()

    @staticmethod
    def _load_size(val: Union[np.ndarray, List, float, int]) -> int:
        """
        Returns the vector size of the provided value.
        Scalars are interpreted as a vector of size 1.

        Parameters
        ----------
        val : Union[np.ndarray, List, float, int]
            Value to be checked.

        Returns
        -------
        size : int
            vector size of the provided value.
        """
        if isinstance(val, np.ndarray) or isinstance(val, list):
            return len(val)
        return 1

    @staticmethod
    def _conv_to_ndarray(data_list: List[Optional[Dict[str, Any]]]) -> np.ndarray:
        """
        Converts the list of length N containing data dicts into a np.ndarray of shape NxM
        by merging all entries of the respective dicts into a vector of length M.
        Missing dicts, i.e. None values, are converted to a zero vector.

        Parameters
        ----------
        data_list : List[Optional[Dict[str, Any]]]
            Data to be converted

        Returns
        -------
        np_data : np.ndarray
            Data as np.ndarray.
        """
        # collect output dimensions, assuming each dict follows the same structure
        entry: Dict[str, Any] = next(d for d in data_list if d is not None)
        entry_size = sum([CrashHandler._load_size(val) for val in entry.values()])
        entry_count = len(data_list)

        # alloc buffer and populate
        buffer = np.zeros(shape=(entry_count, entry_size), dtype=np.float64)
        for d_idx in range(entry_count):
            if data_list[d_idx] is None:
                continue
            entry = data_list[d_idx]
            offset = 0
            for val in entry.values():
                size = CrashHandler._load_size(val)
                buffer[d_idx, offset : offset + size] = val
                offset += size
        return buffer
