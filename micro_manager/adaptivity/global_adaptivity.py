"""
Class GlobalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import hashlib
from copy import deepcopy
import sys
from typing import Dict
import numpy as np
from mpi4py import MPI

from .adaptivity import AdaptivityCalculator
from micro_manager.config import Config
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
        self,
        configurator: Config,
        global_number_of_sims: int,
        global_ids: list,
        participant,
        base_logger: Logger,
        rank: int,
        comm: MPI.Comm,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
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
        base_logger : object of class Logger
            Logger object to log messages.
        rank : int
            MPI rank.
        comm : MPI.Comm
            Communicator for MPI.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object of class ModelManager
            Handles instantiation of the micro simulation.
        """
        super().__init__(
            configurator,
            global_number_of_sims,
            micro_problem_cls,
            model_manager,
            base_logger,
            rank,
        )
        self._global_number_of_sims = global_number_of_sims
        self._global_ids = global_ids
        self._comm = comm

        rank_of_sim = self._get_ranks_of_sims()

        self._is_sim_on_this_rank = [False] * global_number_of_sims  # DECLARATION
        for gid in range(global_number_of_sims):
            if rank_of_sim[gid] == self._rank:
                self._is_sim_on_this_rank[gid] = True

        self._precice_participant = participant

        self._comm_node = comm.Split_type(MPI.COMM_TYPE_SHARED)

        self._MPI_local_rank = self._comm_node.Get_rank()

        # Size of data type
        itemsize = MPI.FLOAT.Get_size()

        if (
            self._MPI_local_rank == 0
        ):  # Only the first rank in the node allocates the shared memory
            nbytes = (
                self._global_number_of_sims * self._global_number_of_sims * itemsize
            )
        else:
            nbytes = 0

        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=self._comm_node)

        # Get the buffer on the local rank 0
        buffer, itemsize = win.Shared_query(0)

        if itemsize != MPI.FLOAT.Get_size():
            raise RuntimeError("Item size mismatch in shared memory.")

        # Create a numpy array from the buffer
        array_buffer = np.array(buffer, dtype="B", copy=False)

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        # This matrix is modified in place via the function update_similarity_dists
        self._similarity_dists: np.ndarray = np.ndarray(
            buffer=array_buffer,
            dtype="f",
            shape=(self._global_number_of_sims, self._global_number_of_sims),
        )

        if self._MPI_local_rank == 0:
            # Initialize the similarity distances to zero
            self._similarity_dists.fill(0.0)

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
            global_ids_as_list = self._comm.allgather(self._global_ids)
            global_data_for_adaptivity[name] = [0] * self._global_number_of_sims
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
            self._MPI_local_rank == 0
        ):  # Only the first rank in the node updates the similarity distances
            self._update_similarity_dists(dt, global_data_for_adaptivity)

        self._comm_node.Barrier()  # Wait for the similarity distances to be updated on all ranks of the node

        self._max_similarity_dist = np.amax(self._similarity_dists)

        self._update_active_sims()

        self._update_inactive_sims(micro_sims)

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
        for gid in self._global_ids:
            if self._is_sim_active[gid]:
                active_sim_ids.append(self._global_ids.index(gid))

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
        for gid in self._global_ids:
            if not self._is_sim_active[gid]:
                inactive_sim_ids.append(self._global_ids.index(gid))

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
        for gid in self._global_ids:
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
        for gid in self._global_ids:
            if not self._is_sim_active[gid]:
                inactive_sim_ids.append(gid)

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
            "micro_manager.global_adaptivity.get_full_field_micro_output"
        )

        micro_sims_output = deepcopy(micro_output)
        self._communicate_micro_output(micro_sims_output)

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
        for gid in self._global_ids:
            if self._is_sim_active[gid]:
                active_sims_on_this_rank += 1
            else:
                inactive_sims_on_this_rank += 1

        if (
            self._adaptivity_output_type == "all"
            or self._adaptivity_output_type == "local"
        ):
            ranks_of_sims = self._get_ranks_of_sims()

            assoc_ranks = []  # Ranks to which inactive sims on this rank are associated
            for gid in self._global_ids:
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
            active_sims_rankwise = self._comm.gather(active_sims_on_this_rank, root=0)
            inactive_sims_rankwise = self._comm.gather(
                inactive_sims_on_this_rank, root=0
            )

            if self._rank == 0:
                size = self._comm.Get_size()

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
        if self._max_similarity_dist == 0.0:
            self._base_logger.log_warning(
                "All similarity distances are zero, which means all the data for adaptivity is the same. Coarsening tolerance will be manually set to minimum float number."
            )
            self._coarse_tol = sys.float_info.min
        else:
            self._coarse_tol = (
                self._coarse_const * self._refine_const * self._max_similarity_dist
            )

        active_gids_this_rank = self.get_active_sim_global_ids()
        # Gather global ids of active sims from all ranks
        active_gids_all_ranks = self._comm.allgather(active_gids_this_rank.tolist())

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
            if not self._is_sim_on_this_rank[assoc_active_gid]:
                if assoc_active_gid in active_to_inactive_map:
                    active_to_inactive_map[assoc_active_gid].append(gid)
                else:
                    active_to_inactive_map[assoc_active_gid] = [gid]
            else:  # If associated active simulation is on this rank, copy the output directly
                lid = self._global_ids.index(gid)
                assoc_active_lid = self._global_ids.index(assoc_active_gid)
                micro_output[lid] = deepcopy(micro_output[assoc_active_lid])

        assoc_active_gids = list(active_to_inactive_map.keys())

        recv_reqs = self._p2p_comm(assoc_active_gids, micro_output)

        # Add received output of active sims to inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            output = req.wait()
            for gid in active_to_inactive_map[assoc_active_gids[count]]:
                lid = self._global_ids.index(gid)
                micro_output[lid] = deepcopy(output)

    def _update_inactive_sims(self, micro_sims: list) -> None:
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
                    if self._is_sim_on_this_rank[gid]:
                        to_be_activated_gids.append(gid)
        # ----------------------------------------------------------------------------------

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        # Only handle activation of simulations on this rank
        for gid in to_be_activated_gids:
            to_be_activated_lid = self._global_ids.index(gid)
            micro_sims[to_be_activated_lid] = self._model_manager.get_instance(
                gid, self._micro_problem_cls
            )
            assoc_active_gid = self._sim_is_associated_to[gid]

            if self._is_sim_on_this_rank[
                assoc_active_gid
            ]:  # Associated active simulation is on the same rank
                assoc_active_lid = self._global_ids.index(assoc_active_gid)
                micro_sims[to_be_activated_lid].set_state(
                    micro_sims[assoc_active_lid].get_state()
                )
            else:  # Associated active simulation is not on this rank
                if assoc_active_gid in to_be_activated_map:
                    to_be_activated_map[assoc_active_gid].append(to_be_activated_lid)
                else:
                    to_be_activated_map[assoc_active_gid] = [to_be_activated_lid]

        self._precice_participant.start_profiling_section(
            "micro_manager.global_adaptivity.update_inactive_sims.communication"
        )

        sim_states_and_global_ids = []
        for lid, sim in enumerate(micro_sims):
            if sim == 0:
                sim_states_and_global_ids.append((None, self._global_ids[lid]))
            else:
                sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(
            list(to_be_activated_map.keys()), sim_states_and_global_ids
        )

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, gid = req.wait()
            local_ids = to_be_activated_map[gid]
            for lid in local_ids:
                # Create the micro simulation object and set its state
                micro_sims[lid] = self._model_manager.get_instance(
                    self._global_ids[lid], self._micro_problem_cls
                )
                micro_sims[lid].set_state(state)

        # Delete the micro simulation object if it is inactive
        for gid in self._global_ids:
            if not self._is_sim_active[gid]:
                lid = self._global_ids.index(gid)
                micro_sims[lid] = 0

        self._precice_participant.stop_last_profiling_section()

        self._sim_is_associated_to = np.copy(_sim_is_associated_to_updated)

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
        for gid, send_ranks in send_map.items():
            lid = self._global_ids.index(gid)
            for send_rank in send_ranks:
                tag = self._create_tag(gid, self._rank, send_rank)
                req = self._comm.isend(data[lid], dest=send_rank, tag=tag)
                send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for gid, recv_rank in recv_map.items():
            tag = self._create_tag(gid, recv_rank, self._rank)
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
        gids_to_rank = dict()
        for gid in self._global_ids:
            gids_to_rank[gid] = self._rank

        ranks_maps_as_list = self._comm.allgather(gids_to_rank)

        ranks_of_sims = np.zeros(self._global_number_of_sims, dtype=np.intc)
        for ranks_map in ranks_maps_as_list:
            for gid, rank in ranks_map.items():
                ranks_of_sims[gid] = rank

        return ranks_of_sims
