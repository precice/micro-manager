from copy import deepcopy
from typing import Hashable, Set, Dict, Any, Optional, List
import sys
import numpy as np

from micro_manager.simulation_container import SimulationContainer
from micro_manager.micro_simulation import MicroSimulationInterface
from micro_manager.interpolation import Interpolator
from micro_manager.config import Config
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler, MPI


class CrashHandler:
    def __init__(
        self, logger: Logger, mpi: MPIHandler, sim_container: SimulationContainer
    ):
        self._logger: Logger = logger
        self._mpi: MPIHandler = mpi
        self._sim_container: SimulationContainer = sim_container
        self._interp_id: Hashable = 0

        self._enable_interp: bool = False
        self._crashed_sims: Set[int] = set()

        self._crash_threshold: float = 0.2

    def initialize(self, config: Config) -> None:
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

    def _load_size(self, val):
        if isinstance(val, np.ndarray) or isinstance(val, list):
            return len(val)
        return 1

    def _conv_to_ndarray(self, data_list: List[Optional[Dict[str, Any]]]) -> np.ndarray:
        # collect output dimensions, assuming each dict follows the same structure
        entry: Dict[str, Any] = next(d for d in data_list if d is not None)
        entry_size = sum([self._load_size(val) for val in entry.values()])
        entry_count = len(data_list)

        # alloc buffer and populate
        buffer = np.zeros(shape=(entry_count, entry_size), dtype=np.float64)
        for d_idx in range(entry_count):
            if data_list[d_idx] is None:
                continue
            entry = data_list[d_idx]
            offset = 0
            for val in entry.values():
                size = self._load_size(val)
                buffer[d_idx, offset : offset + size] = val
                offset += size
        return buffer

    def interpolate_outputs(
        self,
        unset_output_lids: List[int],
        sim_inputs: List[Dict[str, Any]],
        sim_outputs: List[Optional[Dict[str, Any]]],
    ) -> None:
        if not self._enable_interp:
            if self.get_glob_num_crashes() > 0:
                self._logger.log_error(
                    "Exiting simulation after micro simulation crash."
                )
                sys.exit()
            return

        interp = Interpolator.get_instance(self._interp_id)
        avail_count = self._sim_container.local_num_sims - len(unset_output_lids)
        req_count = interp.get_min_support_size()
        if avail_count <= req_count:
            self._logger.log_error(
                "Exiting simulation after micro simulation crash. Not enough data for interpolation."
            )
            sys.exit()

        np_inputs = self._conv_to_ndarray(sim_inputs)
        np_outputs = self._conv_to_ndarray(sim_outputs)
        set_output_lids = list(
            set(self._sim_container.range_lid).difference(unset_output_lids)
        )
        interp.set_local_data(
            np_inputs[set_output_lids, :],
            np_inputs[unset_output_lids, :],
            np_outputs[set_output_lids, :],
        )
        interp_outputs = interp.interpolate()

        for lid in unset_output_lids:
            result = deepcopy(sim_outputs[set_output_lids[0]])
            offset = 0
            for key, val in result.items():
                size = self._load_size(val)
                extracted_val = interp_outputs[lid, offset : offset + size]
                if size == 1:
                    extracted_val = type(val)(extracted_val[0])
                result[key] = extracted_val
                offset += size

            sim_outputs[lid] = result

    def has_sim_crashed(self, gid: int) -> bool:
        return gid in self._crashed_sims

    def register_sim_crash(self, gid: int) -> None:
        self._crashed_sims.add(gid)

    def reset(self):
        self._crashed_sims.clear()

    def get_glob_num_crashes(self) -> int:
        crashed_sims_on_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
        self._mpi.comm.Allgather(len(self._crashed_sims), crashed_sims_on_all_ranks)
        return np.sum(crashed_sims_on_all_ranks)

    def check_crash_ratio(self):
        if not self._enable_interp:
            return

        crash_ratio = self.get_glob_num_crashes() / self._sim_container.global_num_sims
        if crash_ratio > self._crash_threshold:
            self._logger.log_info(
                "{:.1%} of the micro simulations have crashed exceeding the threshold of {:.1%}. "
                "Exiting simulation.".format(crash_ratio, self._crash_threshold)
            )
            sys.exit()
