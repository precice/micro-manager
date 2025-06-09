"""
Class GlobalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import hashlib
import importlib
from copy import deepcopy
from typing import Dict
import numpy as np
from mpi4py import MPI

from .adaptivity import AdaptivityCalculator
from ..micro_simulation import create_simulation_class


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
        self,
        configurator,
        global_number_of_sims: int,
        global_ids: list,
        participant,
        rank: int,
        comm,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        global_number_of_sims : int
            Total number of simulations in the macro-micro coupled problem.
        global_ids : list
            List of global IDs of simulations living on this rank.
        participant : object of class Participant
            Object of the class Participant using which the preCICE API is called.
        rank : int
            MPI rank.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        is_load_balancing : bool
            Flag to indicate if load balancing is to be done.
        """
        super().__init__(configurator, rank, global_number_of_sims)
        self._global_number_of_sims = global_number_of_sims
        self._global_ids = global_ids
        self._comm = comm

        local_number_of_sims = len(global_ids)

        # Create a map of micro simulation global IDs and the ranks on which they are
        micro_sims_on_this_rank = np.zeros(local_number_of_sims, dtype=np.intc)
        for i in range(local_number_of_sims):
            micro_sims_on_this_rank[i] = self._rank

        rank_of_sim = self._get_ranks_of_sims()

        self._is_sim_on_this_rank = [False] * global_number_of_sims  # DECLARATION
        for i in range(global_number_of_sims):
            if rank_of_sim[i] == self._rank:
                self._is_sim_on_this_rank[i] = True

        self._updating_inactive_sims = self._get_update_inactive_sims_variant()

        self._precice_participant = participant

        self._metrics_logger.log_info("n,n active,n inactive,assoc ranks")

    def compute_adaptivity(
        self,
        dt: float,
        micro_sims: list,
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
        self._precice_participant.start_profiling_section(
            "global_adaptivity.compute_adaptivity"
        )

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
            data_as_list = self._comm.allgather(data_for_adaptivity[name])
            # NOTE: Data type restricted to float32 to save memory. Remove this restriction if higher precision is needed.
            global_data_for_adaptivity[name] = np.concatenate(
                (data_as_list[:]), axis=0, dtype=np.float32
            )

        self._update_similarity_dists(dt, global_data_for_adaptivity)

        self._update_active_sims()

        self._updating_inactive_sims(micro_sims)

        self._associate_inactive_to_active()

        self._precice_participant.stop_last_profiling_section()

        self._precice_participant.stop_last_profiling_section()

    def get_active_sim_local_ids(self) -> np.ndarray:
        """
        Get the local ids of active simulations on this rank.

        Returns
        -------
        numpy array
            1D array of active simulation ids
        """
        active_sim_ids = []
        for global_id in self._global_ids:
            if self._is_sim_active[global_id]:
                active_sim_ids.append(self._global_ids.index(global_id))

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
        for global_id in self._global_ids:
            if not self._is_sim_active[global_id]:
                inactive_sim_ids.append(self._global_ids.index(global_id))

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
        for global_id in self._global_ids:
            if self._is_sim_active[global_id]:
                active_sim_ids.append(global_id)

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
        for global_id in self._global_ids:
            if not self._is_sim_active[global_id]:
                inactive_sim_ids.append(global_id)

        return np.array(inactive_sim_ids)

    def get_full_field_micro_output(self, micro_output: list) -> list:
        """
        Get the full field micro output from active simulations to inactive simulations.

        Parameters
        ----------
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.

        Returns
        -------
        micro_output : list
            List of dicts having individual output of each simulation. Active and inactive simulation outputs are entered.
        """
        self._precice_participant.start_profiling_section(
            "global_adaptivity.get_full_field_micro_output"
        )

        micro_sims_output = deepcopy(micro_output)
        self._communicate_micro_output(micro_sims_output)

        self._precice_participant.stop_last_profiling_section()

        return micro_sims_output

    def log_metrics(self, n: int) -> None:
        """
        Log the following metrics:

        Metrics on this rank:
        - Time window at which the metrics are logged
        - Number of active simulations
        - Number of inactive simulations
        - Ranks to which inactive simulations on this rank are associated

        Global metrics:
        - Time window at which the metrics are logged
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
        for global_id in self._global_ids:
            if self._is_sim_active[global_id]:
                active_sims_on_this_rank += 1
            else:
                inactive_sims_on_this_rank += 1

        ranks_of_sims = self._get_ranks_of_sims()

        assoc_ranks = []  # Ranks to which inactive sims on this rank are associated
        for global_id in self._global_ids:
            if not self._is_sim_active[global_id]:
                assoc_rank = int(ranks_of_sims[self._sim_is_associated_to[global_id]])
                if not assoc_rank in assoc_ranks:
                    assoc_ranks.append(assoc_rank)

        self._metrics_logger.log_info(
            "{},{},{},{}".format(
                n,
                active_sims_on_this_rank,
                inactive_sims_on_this_rank,
                assoc_ranks,
            )
        )

        active_sims_rankwise = self._comm.gather(active_sims_on_this_rank, root=0)
        inactive_sims_rankwise = self._comm.gather(inactive_sims_on_this_rank, root=0)

        if self._rank == 0:
            size = self._comm.Get_size()

            self._global_metrics_logger.log_info(
                "{},{},{},{},{}".format(
                    n,
                    sum(active_sims_rankwise) / size,
                    sum(inactive_sims_rankwise) / size,
                    max(active_sims_rankwise),
                    max(inactive_sims_rankwise),
                )
            )

    def write_checkpoint(self) -> None:
        """
        Write checkpoint.
        """
        self._similarity_dists_cp = np.copy(self._similarity_dists)
        self._is_sim_active_cp = np.copy(self._is_sim_active)
        self._sim_is_associated_to_cp = np.copy(self._sim_is_associated_to)

    def read_checkpoint(self) -> None:
        """
        Read checkpoint.
        """
        self._similarity_dists = np.copy(self._similarity_dists_cp)
        del self._similarity_dists_cp

        self._is_sim_active = np.copy(self._is_sim_active_cp)
        del self._is_sim_active_cp

        self._sim_is_associated_to = np.copy(self._sim_is_associated_to_cp)
        del self._sim_is_associated_to_cp

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

        for global_id in self._global_ids:
            if not self._is_sim_active[global_id]:
                assoc_active_gid = self._sim_is_associated_to[global_id]
                # Gather global IDs of associated active simulations not on this rank
                if not self._is_sim_on_this_rank[assoc_active_gid]:
                    if assoc_active_gid in active_to_inactive_map:
                        active_to_inactive_map[assoc_active_gid].append(global_id)
                    else:
                        active_to_inactive_map[assoc_active_gid] = [global_id]
                else:  # If associated active simulation is on this rank, copy the output directly
                    local_id = self._global_ids.index(global_id)
                    micro_output[local_id] = deepcopy(
                        micro_output[self._global_ids.index(assoc_active_gid)]
                    )

        assoc_active_ids = list(active_to_inactive_map.keys())

        recv_reqs = self._p2p_comm(assoc_active_ids, micro_output)

        # Add received output of active sims to inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            output = req.wait()
            for global_id in active_to_inactive_map[assoc_active_ids[count]]:
                local_id = self._global_ids.index(global_id)
                micro_output[local_id] = deepcopy(output)

    def _update_inactive_sims(self, micro_sims: list) -> None:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._ref_tol = self._refine_const * self._max_similarity_dist

        _sim_is_associated_to_updated = np.copy(self._sim_is_associated_to)

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(self._is_sim_active.size):
            if not self._is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, self._is_sim_active):
                    self._is_sim_active[i] = True
                    _sim_is_associated_to_updated[
                        global_id
                    ] = -2  # Active sim cannot have an associated sim
                    if self._is_sim_on_this_rank[i] and i not in self._just_deactivated:
                        to_be_activated_ids.append(i)

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        # Only handle activation of simulations on this rank -- LOCAL SCOPE HERE ON
        for global_id in to_be_activated_ids:
            if self._is_sim_on_this_rank[global_id]:
                to_be_activated_local_id = self._global_ids.index(global_id)
                assoc_active_id = self._sim_is_associated_to[global_id]

                if self._is_sim_on_this_rank[
                    assoc_active_id
                ]:  # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id].set_state(
                        micro_sims[assoc_active_local_id].get_state()
                    )
                else:  # Associated active simulation is not on this rank
                    if assoc_active_id in to_be_activated_map:
                        to_be_activated_map[assoc_active_id].append(
                            to_be_activated_local_id
                        )
                    else:
                        to_be_activated_map[assoc_active_id] = [
                            to_be_activated_local_id
                        ]

        sim_states_and_global_ids = []
        for sim in micro_sims:
            sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(
            list(to_be_activated_map.keys()), sim_states_and_global_ids
        )

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, global_id = req.wait()
            local_ids = to_be_activated_map[global_id]
            for local_id in local_ids:
                micro_sims[local_id].set_state(state)

        self._sim_is_associated_to = _sim_is_associated_to_updated
        del _sim_is_associated_to_updated

    def _update_inactive_sims_lazy_init(self, micro_sims: list) -> None:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._ref_tol = self._refine_const * self._max_similarity_dist

        _sim_is_associated_to_updated = np.copy(self._sim_is_associated_to)

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(self._is_sim_active.size):
            if not self._is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, self._is_sim_active):
                    self._is_sim_active[i] = True
                    _sim_is_associated_to_updated[
                        i
                    ] = -2  # Active sim cannot have an associated sim
                    if self._is_sim_on_this_rank[i] and i not in self._just_deactivated:
                        to_be_activated_ids.append(i)

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        local_sim_is_associated_to = self._sim_is_associated_to[
            self._global_ids[0] : self._global_ids[-1] + 1
        ]

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        for i in to_be_activated_ids:
            # Only handle activation of simulations on this rank -- LOCAL SCOPE HERE ON
            if self._is_sim_on_this_rank[i]:
                to_be_activated_local_id = self._global_ids.index(i)
                if (
                    micro_sims[to_be_activated_local_id] == 0
                ):  # 0 indicates that the micro simulation object has not been created yet
                    micro_problem = getattr(
                        importlib.import_module(
                            self._micro_file_name, "MicroSimulation"
                        ),
                        "MicroSimulation",
                    )
                    micro_sims[to_be_activated_local_id] = create_simulation_class(
                        micro_problem
                    )(i)
                assoc_active_id = local_sim_is_associated_to[to_be_activated_local_id]

                if self._is_sim_on_this_rank[
                    assoc_active_id
                ]:  # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id].set_state(
                        micro_sims[assoc_active_local_id].get_state()
                    )
                else:  # Associated active simulation is not on this rank
                    if assoc_active_id in to_be_activated_map:
                        to_be_activated_map[assoc_active_id].append(
                            to_be_activated_local_id
                        )
                    else:
                        to_be_activated_map[assoc_active_id] = [
                            to_be_activated_local_id
                        ]

        # TODO: could be moved to before the lazy initialization above
        sim_states_and_global_ids = []
        for local_id, sim in enumerate(micro_sims):
            if sim == 0:
                sim_states_and_global_ids.append((None, self._global_ids[local_id]))
            else:
                sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(
            list(to_be_activated_map.keys()), sim_states_and_global_ids
        )

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, global_id = req.wait()
            local_ids = to_be_activated_map[global_id]
            for local_id in local_ids:
                micro_sims[local_id].set_state(state)

        self._sim_is_associated_to = _sim_is_associated_to_updated
        del _sim_is_associated_to_updated

    def _get_update_inactive_sims_variant(self):
        """
        Get the variant of the function _update_inactive_sims.

        Returns
        -------
        function
            Function which updates the set of inactive micro simulations.
        """
        if self._lazy_init:
            return self._update_inactive_sims_lazy_init
        else:
            return self._update_inactive_sims

    def _create_tag(self, sim_id: int, src_rank: int, dest_rank: int) -> int:
        """
        For a given simulations ID, source rank, and destination rank, a unique tag is created.

        Parameters
        ----------
        sim_id : int
            Global ID of a simulation.
        src_rank : int
            Rank on which the simulation lives
        dest_rank : int
            Rank to which data of a simulation is to be sent to.

        Returns
        -------
        tag : int
            Unique tag.
        """
        send_hashtag = hashlib.sha256()
        send_hashtag.update(
            (str(src_rank) + str(sim_id) + str(dest_rank)).encode("utf-8")
        )
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _p2p_comm(self, assoc_active_ids: list, data: list) -> list:
        """
        Handle process to process communication for a given set of associated active IDs and data.

        Parameters
        ----------
        assoc_active_ids : list
            Global IDs of active simulations which are not on this rank and are associated to
            the inactive simulations on this rank.
        data : list
            Complete data from which parts are to be sent and received.

        Returns
        -------
        recv_reqs : list
            List of MPI requests of receive operations.
        """
        rank_of_sim = self._get_ranks_of_sims()

        send_map_local: Dict[
            int, int
        ] = dict()  # keys are global IDs, values are rank to send to
        send_map: Dict[
            int, list
        ] = (
            dict()
        )  # keys are global IDs of sims to send, values are ranks to send the sims to
        recv_map: Dict[
            int, int
        ] = dict()  # keys are global IDs to receive, values are ranks to receive from

        for i in assoc_active_ids:
            # Add simulation and its rank to receive map
            recv_map[i] = rank_of_sim[i]
            # Add simulation and this rank to local sending map
            send_map_local[i] = self._rank

        # Gather information about which sims to send where, from the sending perspective
        send_map_list = self._comm.allgather(send_map_local)

        for d in send_map_list:
            for i, rank in d.items():
                if self._is_sim_on_this_rank[i]:
                    if i in send_map:
                        send_map[i].append(rank)
                    else:
                        send_map[i] = [rank]

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_ranks in send_map.items():
            local_id = self._global_ids.index(global_id)
            for send_rank in send_ranks:
                tag = self._create_tag(global_id, self._rank, send_rank)
                req = self._comm.isend(data[local_id], dest=send_rank, tag=tag)
                send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in recv_map.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            bufsize = (
                1 << 30
            )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
            req = self._comm.irecv(bufsize, source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        return recv_reqs

    def _get_ranks_of_sims(self) -> np.ndarray:
        """
        Get the ranks of all simulations.

        Returns
        -------
        ranks_of_sims : np.ndarray
            Array of ranks on which simulations exist.
        """
        local_gids_to_rank = dict()
        for gid in self._global_ids:
            local_gids_to_rank[gid] = self._rank

        ranks_maps_as_list = self._comm.allgather(local_gids_to_rank)

        ranks_of_sims = np.zeros(self._global_number_of_sims, dtype=np.intc)
        for ranks_map in ranks_maps_as_list:
            for gid, rank in ranks_map.items():
                ranks_of_sims[gid] = rank

        return ranks_of_sims
