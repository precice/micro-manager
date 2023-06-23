"""
Functionality for adaptive control of micro simulations in a global way (all-to-all comparison of micro simulations)
"""
import sys
import numpy as np
import hashlib
from copy import deepcopy
from math import exp
from mpi4py import MPI
from .adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    """
    This class provides functionality to compute adaptivity globally, i.e. by comparing micro simulation from all processes.
    All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
    """
    def __init__(
            self,
            configurator,
            logger,
            is_sim_on_this_rank: list,
            rank_of_sim,
            global_ids: list,
            comm,
            rank: int) -> None:
        super().__init__(configurator, logger)
        self._is_sim_on_this_rank = is_sim_on_this_rank
        self._rank_of_sim = rank_of_sim
        self._global_ids = global_ids
        self._comm = comm
        self._rank = rank

    def compute_adaptivity(
            self,
            dt: float,
            micro_sims: list,
            similarity_dists_nm1: np.ndarray,
            micro_sim_states_nm1: np.ndarray,
            data_for_adaptivity: dict) -> tuple:
        """
        Compute adaptivity globally based on similarity distances and micro simulation states

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        similarity_dists_nm1 : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states_nm1 : numpy array
            1D array having state (active or inactive) of each micro simulation on this rank
        data_for_adaptivity : dict
            Dictionary with keys as names of data to be used in the similarity calculation, and values as the respective data for the micro simulations
            
        Results
        -------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        """
        # Gather adaptivity data from all ranks
        global_data_for_adaptivity = dict()
        for name in self._adaptivity_data_names.keys():
            data_as_list = self._comm.allgather(data_for_adaptivity[name])
            global_data_for_adaptivity[name] = np.concatenate((data_as_list[:]), axis=0)

        # Multiply old similarity distance by history term to get current distances
        similarity_dists_n = exp(-self._hist_param * dt) * similarity_dists_nm1

        for name in self._adaptivity_data_names.keys():
            # Similarity distance matrix is calculated globally on every rank
            similarity_dists_n = self._get_similarity_dists(
                dt, similarity_dists_n, global_data_for_adaptivity[name])

        micro_sim_states_n = self._update_active_sims(
            similarity_dists_n, micro_sim_states_nm1, micro_sims)

        micro_sim_states_n = self._update_inactive_sims(
            similarity_dists_n, micro_sim_states_nm1, micro_sims)

        self._associate_inactive_to_active(
            similarity_dists_n, micro_sim_states_n, micro_sims)

        self._logger.info(
            "Number of active micro simulations = {}".format(
                np.count_nonzero(
                    micro_sim_states_n == 1)))
        self._logger.info(
            "Number of inactive micro simulations = {}".format(
                np.count_nonzero(
                    micro_sim_states_n == 0)))

        return similarity_dists_n, micro_sim_states_n

    def communicate_micro_output(self, global_ids: list, micro_sims: list, micro_sim_states: np.ndarray, micro_output: list) -> list:
        """
        Communicate micro output from active simulation to their associated inactive simulations. P2P communication is done.

        Parameters
        ----------
        global_ids : list
            Global IDs of simulations on this rank.
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        micro_sim_states_nm1 : numpy array
            1D array having state (active or inactive) of each micro simulation on this rank
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.
        
        Returns
        -------
        _micro_output : list
            List of dicts having individual output of each simulation. Inactive simulation entries are now filled.
        """
        _micro_output = np.copy(micro_output)

        inactive_local_ids = np.where(micro_sim_states[self._global_ids[0]:self._global_ids[-1]+1] == 0)[0]

        assoc_active_ids = []
        for i in inactive_local_ids:
            assoc_active_id = micro_sims[i].get_associated_active_id()
            if not self._is_sim_on_this_rank[assoc_active_id]:  # Gather global IDs of associated active simulations not on this rank for communication
                assoc_active_ids.append(micro_sims[i].get_associated_active_id())
            else:  # If associated active simulation is on this rank, copy the output directly
                _micro_output[i] = _micro_output[self._global_ids.index(assoc_active_id)]

        send_sims_from_this_rank, recv_sims_from_ranks = self._get_send_and_recv_maps_for_activation(assoc_active_ids)

        recv_reqs = self._p2p_comm(send_sims_from_this_rank, recv_sims_from_ranks, _micro_output)

        global_ids_of_recv_data = list(recv_sims_from_ranks.keys())
        # Use received micro sims to activate the currently inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            local_id = global_ids.index(global_ids_of_recv_data[count])
            _micro_output[local_id] = req.wait()

        #print("_micro_output at the END: {}".format(_micro_output))

        return _micro_output

    def _update_active_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> np.ndarray:
        """
        Update set of active micro simulations. Active micro simulations are compared to each other
        and if found similar, one of them is deactivated.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        Returns
        -------
        _micro_sim_states : numpy array
            Updated 1D array having state (active or inactive) of each micro simulation
        """
        self._coarse_tol = self._coarse_const * self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Update the set of active micro sims
        for i in range(_micro_sim_states.size):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    if self._is_sim_on_this_rank[i]:
                        micro_sims[self._global_ids.index(i)].deactivate()
                    _micro_sim_states[i] = 0

        return _micro_sim_states

    def _update_inactive_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> np.ndarray:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations

        Returns
        -------
        _micro_sim_states : numpy array
            Updated 1D array having state (active or inactive) of each micro simulation
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(_micro_sim_states.size):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1
                    if self._is_sim_on_this_rank[i]:
                        to_be_activated_ids.append(i)

        # Keys are global IDs of active sims not on this rank, values are tuples of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map = dict()
        assoc_global_ids_not_on_this_rank = []

        for i in to_be_activated_ids:
            # Handle activation of simulations on this rank
            if self._is_sim_on_this_rank[i]:
                to_be_activated_local_id = self._global_ids.index(i)
                assoc_active_id = micro_sims[to_be_activated_local_id].get_associated_active_id()

                # Kill the inactive micro sim object
                micro_sims[to_be_activated_local_id] = None

                if self._is_sim_on_this_rank[assoc_active_id]: # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id] = deepcopy(micro_sims[assoc_active_local_id])
                    micro_sims[to_be_activated_local_id].set_global_id(assoc_active_id)
                else: # Associated active simulation is not on this rank
                    if isinstance(to_be_activated_map[assoc_active_id], list):
                        to_be_activated_map[assoc_active_id].append((to_be_activated_local_id, i))
                    else:
                        to_be_activated_map[assoc_active_id] = (to_be_activated_local_id, i)

                    assoc_global_ids_not_on_this_rank.append(assoc_active_id)

        send_sims_from_this_rank, recv_sims_from_ranks = self._get_send_and_recv_maps_for_activation(
            assoc_global_ids_not_on_this_rank)

        recv_reqs = self._p2p_comm(send_sims_from_this_rank, recv_sims_from_ranks, micro_sims)

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            active_sim = req.wait()
            inactive_sims_data = to_be_activated_map[active_sim.get_global_id()]
            for inactive_sim_data in inactive_sims_data:
                local_id = inactive_sim_data[0]
                global_id = inactive_sim_data[1]
                micro_sims[local_id] = deepcopy(active_sim)
                micro_sims[local_id].set_global_id(global_id)

        return _micro_sim_states

    def _associate_inactive_to_active(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> None:
        """
        Associate inactive micro simulations to most similar active micro simulation.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        #print("Rank {}: micro_sim_states: {}".format(self._rank, micro_sim_states))

        active_ids = np.where(micro_sim_states == 1)[0]
        inactive_ids = np.where(micro_sim_states == 0)[0]

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_ids:
            dist_min = sys.float_info.max
            for active_id in active_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = similarity_dists[inactive_id, active_id]

            if self._is_sim_on_this_rank[inactive_id]:
                micro_sims[self._global_ids.index(inactive_id)].is_associated_to_active_sim(associated_active_id)

    def _get_send_and_recv_maps_for_activation(self, assoc_active_ids: list) -> tuple:
        """
        This function created sending and receiving maps for p2p communication.

        Parameters
        ----------
        assoc_active_ids : list
            Global IDs of active simulations which are not on this rank and are associated to the inactive simulations on this rank

        Returns
        -------
        send_sims_from_this_rank : dict
            Dict where the keys are global IDs of simulations to send and values are ranks to send the simulations to
        recv_sims_from_ranks : dict
            Dict where keys are global IDs of simulations to receive and values are ranks to receive the simulations from
        """
        send_sims_to_ranks_local = dict()  # keys are global IDs, values are rank to send to
        send_sims_to_ranks = dict()  # keys are global IDs of sims to send, values are ranks to send the sims to
        recv_sim_from_rank = dict()  # keys are global IDs to receive, values are ranks to receive from
    
        for i in assoc_active_ids:
            # Add simulation and its rank to receive map
            recv_sim_from_rank[i] = self._rank_of_sim[i]
            # Add simulation and this rank to local sending map
            send_sims_to_ranks_local[i] = self._rank

        print("Rank: {} recv_sim_from_rank: {}, send_sims_to_ranks_local: {}".format(self._rank, recv_sim_from_rank, send_sims_to_ranks_local))
        # ----- Gather information about which sims to send where, from the sending perspective -----
        send_sims_to_ranks_list = self._comm.allgather(send_sims_to_ranks_local)

        print("Rank: {} send_sims_to_ranks_list: {}".format(self._rank, send_sims_to_ranks_list))

        for d in send_sims_to_ranks_list:
            for i, rank in d.items():
                if self._is_sim_on_this_rank[i]:
                    if isinstance(send_sims_to_ranks[i], list):
                        send_sims_to_ranks[i].append(rank)
                    else:
                        send_sims_to_ranks[i] = [rank]

        return send_sims_to_ranks, recv_sim_from_rank

    def _create_tag(self, sim_id, src_rank, dest_rank):
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode('utf-8'))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _p2p_comm(self, send_map, recv_map, data):
        """
        """
        # Asynchronous send operations
        send_reqs = []
        for global_id, send_rank in send_map.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            local_id = self._global_ids.index(global_id)
            req = self._comm.isend(data[local_id], dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in recv_map.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

        # Wait for all non-blocking communications to complete
        MPI.Request.Waitall(send_reqs)

        return recv_reqs
