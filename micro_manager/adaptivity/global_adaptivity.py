"""
Class GlobalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
from copy import deepcopy
from typing import Dict, List, Any
import numpy as np

from .adaptivity import AdaptivityCalculator
from micro_manager.config import Config
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler, MPI
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager
from micro_manager.interpolation import RBF_PU
from micro_manager.simulation_container import SimulationContainer


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
        self,
        config: Config,
        sim_container: SimulationContainer,
        participant,
        base_logger: Logger,
        mpi: MPIHandler,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        config : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        sim_container : SimulationContainer
            Simulation container object.
        participant : object of class Participant
            Object of the class Participant using which the preCICE API is called.
        base_logger : object of class Logger
            Logger object to log messages.
        mpi : MPIHandler
            MPIHandler object.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object of class ModelManager
            Handles instantiation of the micro simulation.
        """
        super().__init__(
            config,
            sim_container.global_num_sims,
            sim_container,
            micro_problem_cls,
            model_manager,
            base_logger,
            mpi,
        )

        self._interpolation = RBF_PU(base_logger, mpi)
        self._precice_participant = participant

        buffer_size = self._sim_container.global_num_sims ** 2
        array_buffer, mpi_node = self._mpi.create_node_buffer(MPI.FLOAT, buffer_size)
        self._mpi_node = mpi_node

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        # This matrix is modified in place via the function update_similarity_dists
        self._similarity_dists: np.ndarray = np.ndarray(
            buffer=array_buffer,
            dtype="f",
            shape=(
                self._sim_container.global_num_sims,
                self._sim_container.global_num_sims,
            ),
        )

        if self._mpi_node.rank == 0:
            # Initialize the similarity distances to zero
            self._similarity_dists.fill(0.0)

    def compute_adaptivity(
        self,
        dt: float,
        data_for_adaptivity: dict,
    ) -> None:
        """
        Compute adaptivity globally based on similarity distances and micro simulation states

        Parameters
        ----------
        dt : float
            Current time step of the macro-micro coupled problem
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        data_for_adaptivity : dict
            Dictionary with keys as names of data to be used in the similarity calculation, and values as the respective data for the micro simulations
        """
        for name in data_for_adaptivity.keys():
            if name not in self._adaptivity_data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._adaptivity_data_names.keys()
                    )
                )

        # Gather adaptivity data from all ranks
        global_data_for_adaptivity = dict()
        for name in data_for_adaptivity.keys():
            data_as_list = self._mpi.comm.allgather(data_for_adaptivity[name])
            global_ids_as_list = self._mpi.comm.allgather(self._sim_container.local_gids)
            global_data_for_adaptivity[name] = [0] * self._sim_container.global_num_sims
            for i, gids_list in enumerate(global_ids_as_list):
                count = 0
                for gid in gids_list:
                    global_data_for_adaptivity[name][gid] = data_as_list[i][count]
                    count += 1

            # global_data_for_adaptivity[name] = np.concatenate(global_data_for_adaptivity[name], axis=0)
            global_data_for_adaptivity[name] = np.array(
                global_data_for_adaptivity[name]
            )

            # global_data_for_adaptivity[name] = np.concatenate((data_as_list[:]), axis=0)

        if (
            self._mpi_node.rank == 0
        ):  # Only the first rank in the node updates the similarity distances
            self._update_similarity_dists(dt, global_data_for_adaptivity)

        self._mpi_node.comm.Barrier()  # Wait for the similarity distances to be updated on all ranks of the node

        self._max_similarity_dist = np.amax(self._similarity_dists)

        self._update_active_sims()

        self._update_inactive_sims()

        self._associate_inactive_to_active()

    def get_active_sim_local_ids(self) -> np.ndarray:
        """
        Get the local ids of active simulations on this rank.

        Returns
        -------
        numpy array
            1D array of active simulation ids
        """
        active_sim_ids = []
        for gid in self._sim_container.local_gids:
            if self._is_sim_active[gid]:
                active_sim_ids.append(self._sim_container.local_gids.index(gid))

        return np.array(active_sim_ids)

    def get_inactive_sim_local_ids(self) -> np.ndarray:
        """
        Get the local ids of inactive simulations on this rank.

        Returns
        -------
        numpy array
            1D array of inactive simulation ids
        """
        inactive_sim_ids = []
        for gid in self._sim_container.local_gids:
            if not self._is_sim_active[gid]:
                inactive_sim_ids.append(self._sim_container.local_gids.index(gid))

        return np.array(inactive_sim_ids)

    def get_active_sim_global_ids(self) -> np.ndarray:
        """
        Get the global ids of active simulations on this rank.

        Returns
        -------
        numpy array
            1D array of active simulation ids
        """
        active_sim_ids = []
        for gid in self._sim_container.local_gids:
            if self._is_sim_active[gid]:
                active_sim_ids.append(gid)

        return np.array(active_sim_ids)

    def get_inactive_sim_global_ids(self) -> np.ndarray:
        """
        Get the global ids of inactive simulations on this rank.

        Returns
        -------
        numpy array
            1D array of inactive simulation ids
        """
        inactive_sim_ids = []
        for gid in self._sim_container.local_gids:
            if not self._is_sim_active[gid]:
                inactive_sim_ids.append(gid)

        return np.array(inactive_sim_ids)

    def get_full_field_micro_output(
        self, micro_input: list, micro_output: list
    ) -> list:
        """
        Get the full field micro output from active simulations to inactive simulations.

        Parameters
        ----------
        micro_input : list
            List of dicts containing the input data for each simulation.
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.

        Returns
        -------
        micro_output : list
            List of dicts having individual output of each simulation. Active and inactive simulation outputs are entered.
        """
        self._precice_participant.start_profiling_section(
            "micro_manager.global_adaptivity.get_full_field_micro_output"
        )

        micro_sims_output = deepcopy(micro_output)
        num_active = np.sum(self._is_sim_active)
        if num_active == self._is_sim_active.shape[0]:
            self._precice_participant.stop_last_profiling_section()
            return micro_sims_output

        self._communicate_micro_output(micro_sims_output)
        if num_active <= self._interp_min:
            self._precice_participant.stop_last_profiling_section()
            return micro_sims_output

        self._interpolate_output(micro_input, micro_sims_output)

        self._precice_participant.stop_last_profiling_section()

        return micro_sims_output

    def log_metrics(self, n: int) -> None:
        """
        Log the following metrics:

        Local metrics:
        - Time window at which the metrics are logged
        - Number of active simulations
        - Number of inactive simulations
        - Ranks to which inactive simulations on this rank are associated

        Global metrics:
        - Time window at which the metrics are logged
        - Global number of active simulations
        - Global number of inactive simulations
        - Average number of active simulations
        - Average number of inactive simulations
        - Maximum number of active simulations
        - Maximum number of inactive simulations

        Parameters
        ----------
        n : int
            Time step count at which the metrics are logged
        """
        active_sims_on_this_rank = 0
        inactive_sims_on_this_rank = 0
        for gid in self._sim_container.local_gids:
            if self._is_sim_active[gid]:
                active_sims_on_this_rank += 1
            else:
                inactive_sims_on_this_rank += 1

        if (
            self._adaptivity_output_type == "all"
            or self._adaptivity_output_type == "local"
        ):
            ranks_of_sims = self._sim_container.get_ranks_of_sims()

            assoc_ranks = []  # Ranks to which inactive sims on this rank are associated
            for gid in self._sim_container.local_gids:
                if not self._is_sim_active[gid]:
                    assoc_rank = int(ranks_of_sims[self._sim_is_associated_to[gid]])
                    if not assoc_rank in assoc_ranks:
                        assoc_ranks.append(assoc_rank)

            self._metrics_logger.log_info(
                "{}|{}|{}|{}".format(
                    n,
                    active_sims_on_this_rank,
                    inactive_sims_on_this_rank,
                    assoc_ranks,
                )
            )

        if (
            self._adaptivity_output_type == "all"
            or self._adaptivity_output_type == "global"
        ):
            active_sims_rankwise = self._mpi.comm.gather(active_sims_on_this_rank, root=0)
            inactive_sims_rankwise = self._mpi.comm.gather(
                inactive_sims_on_this_rank, root=0
            )

            if self._mpi.rank == 0:
                size = self._mpi.size

                self._global_metrics_logger.log_info(
                    "{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
                        n,
                        sum(active_sims_rankwise),
                        sum(inactive_sims_rankwise),
                        sum(active_sims_rankwise) / size,
                        sum(inactive_sims_rankwise) / size,
                        max(active_sims_rankwise),
                        active_sims_rankwise.index(max(active_sims_rankwise)),
                        max(inactive_sims_rankwise),
                        inactive_sims_rankwise.index(max(inactive_sims_rankwise)),
                    )
                )

    def _update_active_sims(self) -> None:
        """
        Update set of active micro simulations.
        Pairs of active simulations (A, B) are compared and if found to be similar, B is deactivated.
        """
        self._coarse_tol = (
            self._coarse_const * self._refine_const * self._max_similarity_dist
        )

        active_gids_this_rank = self.get_active_sim_global_ids()
        # Gather global ids of active sims from all ranks
        active_gids_all_ranks = self._mpi.comm.allgather(active_gids_this_rank.tolist())

        active_gids_to_iterate = []
        # Iterate over global ids of active sims in a round-robin fashion across ranks
        while any(len(gid_list) > 0 for gid_list in active_gids_all_ranks):
            for gid_list in active_gids_all_ranks:
                if gid_list:  # if the list of global ids is not empty
                    # Pick the first global id on every rank and add it to the list which is later iterated over
                    active_gids_to_iterate.append(gid_list[0])
                    # Remove the picked global id from the rank list
                    gid_list.pop(0)

        # Update the set of active micro sims
        active_gids_to_check = active_gids_to_iterate.copy()
        for gid in active_gids_to_iterate:
            if self._check_for_deactivation(gid, active_gids_to_check):
                self._is_sim_active[gid] = False
                self._just_deactivated.append(gid)
                # Remove deactivated gid from further checks
                active_gids_to_check.remove(gid)

    def _communicate_micro_output(
        self,
        micro_output: list,
    ) -> None:
        """
        Communicate micro output from active simulation to their associated inactive simulations.
        Process to process (p2p) communication is done.

        Parameters
        ----------
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.
        """
        # Keys are global IDs of active simulations associated to inactive
        # simulations on this rank. Values are global IDs of the inactive
        # simulations.
        active_to_inactive_map: Dict[int, list] = dict()

        inactive_gids = self.get_inactive_sim_global_ids()

        for gid in inactive_gids:
            assoc_active_gid = self._sim_is_associated_to[gid]
            # Gather global IDs of associated active simulations not on this rank
            if not self._sim_container.is_sim_on_rank(assoc_active_gid):
                if assoc_active_gid in active_to_inactive_map:
                    active_to_inactive_map[assoc_active_gid].append(gid)
                else:
                    active_to_inactive_map[assoc_active_gid] = [gid]
            else:  # If associated active simulation is on this rank, copy the output directly
                lid = self._sim_container.local_gids.index(gid)
                assoc_active_lid = self._sim_container.local_gids.index(
                    assoc_active_gid
                )
                micro_output[lid] = deepcopy(micro_output[assoc_active_lid])

        assoc_active_gids = list(active_to_inactive_map.keys())

        send_map = self._create_comm_map(assoc_active_gids, [(self._sim_container.local_gids[lid], micro_output[lid]) for lid in self._sim_container.range_lid])
        recv_data = self._mpi.exchange(send_map)

        # Add received output of active sims to inactive sims on this rank
        for gid, data in recv_data:
            for inactive_gid in active_to_inactive_map[assoc_active_gids[gid]]:
                inactive_lid = self._sim_container.local_gids.index(inactive_gid)
                micro_output[inactive_lid] = deepcopy(data)

    def _update_inactive_sims(self) -> None:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.
        """
        self._ref_tol = self._refine_const * self._max_similarity_dist

        _sim_is_associated_to_updated = np.copy(self._sim_is_associated_to)

        # -------------------- Global computation on every rank ----------------------------
        # Check inactive simulations for activation and collect IDs of those to be activated
        active_gids_all_ranks = np.where(self._is_sim_active)[0]
        inactive_gids_all_ranks = np.where(self._is_sim_active == False)[0]
        to_be_activated_gids = []  # Global IDs to be activated

        for gid in inactive_gids_all_ranks:
            if self._check_for_activation(gid, active_gids_all_ranks):
                self._is_sim_active[gid] = True
                # Active sim cannot have an associated sim
                _sim_is_associated_to_updated[gid] = -2
                if gid not in self._just_deactivated:
                    # Add the newly activated gid to active_gids_all_ranks for further checks
                    active_gids_all_ranks = np.append(active_gids_all_ranks, gid)
                    # Collect the global IDs to be activated on this rank
                    if self._sim_container.is_sim_on_rank(gid):
                        to_be_activated_gids.append(gid)
        # ----------------------------------------------------------------------------------

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        # Only handle activation of simulations on this rank
        for gid in to_be_activated_gids:
            to_be_activated_lid = self._sim_container.local_gids.index(gid)
            self._sim_container[to_be_activated_lid] = self._model_manager.get_instance(
                gid, self._micro_problem_cls
            )
            assoc_active_gid = self._sim_is_associated_to[gid]

            # Associated active simulation is on the same rank
            if self._sim_container.is_sim_on_rank(assoc_active_gid):
                assoc_active_lid = self._sim_container.local_gids.index(
                    assoc_active_gid
                )
                state = self._sim_container.get_sim_state(assoc_active_lid)
                self._sim_container.set_sim_state(to_be_activated_lid, state)
            else:  # Associated active simulation is not on this rank
                if assoc_active_gid in to_be_activated_map:
                    to_be_activated_map[assoc_active_gid].append(to_be_activated_lid)
                else:
                    to_be_activated_map[assoc_active_gid] = [to_be_activated_lid]

        self._precice_participant.start_profiling_section(
            "micro_manager.global_adaptivity.update_inactive_sims.communication"
        )

        sim_states_and_global_ids = []
        for lid in self._sim_container.range_lid:
            sim = self._sim_container[lid]
            gid = self._sim_container.local_gids[lid]
            if sim == 0 or sim is None:
                sim_states_and_global_ids.append((None, gid))
            else:
                sim_states_and_global_ids.append(
                    (self._sim_container.get_sim_state(lid), gid)
                )

        send_map = self._create_comm_map(list(to_be_activated_map.keys()), sim_states_and_global_ids)
        recv_data = self._mpi.exchange(send_map)

        # Use received micro sims to activate the required simulations
        for state, gid in recv_data:
            local_ids = to_be_activated_map[gid]
            for lid in local_ids:
                # Create the micro simulation object and set its state
                self._sim_container[lid] = self._model_manager.get_instance(
                    self._sim_container.local_gids[lid], self._micro_problem_cls
                )
                self._sim_container.set_sim_state(lid, state)

        # Delete the micro simulation object if it is inactive
        for gid in self._sim_container.local_gids:
            if not self._is_sim_active[gid]:
                lid = self._sim_container.local_gids.index(gid)
                if self._sim_container[lid] is None:
                    continue
                # Release resources now, especially for remote simulation instance.
                # If left to garbage collector this might lead to a race condition.
                # Releasing with call to sim.destroy(), afterwards reference in sim
                # sim list can be removed.
                self._sim_container[lid].destroy()
                self._sim_container[lid] = None

        self._precice_participant.stop_last_profiling_section()

        self._sim_is_associated_to = np.copy(_sim_is_associated_to_updated)

    def _create_comm_map(self, requested_gids: List[int], lid_range_data: List[Any]) -> Dict[int, List[Any]]:
        send_map_local = {gid: self._mpi.rank for gid in requested_gids}
        send_map_global = self._mpi.comm.allgather(send_map_local)
        send_map = self._mpi.create_empty_exchange_map()
        for map in send_map_global:
            for gid, rnk in map.items():
                if not self._sim_container.is_sim_on_rank(gid):
                    continue
                lid = self._sim_container.local_gids.index(gid)
                send_map[rnk].append(lid_range_data[lid])
        return send_map
