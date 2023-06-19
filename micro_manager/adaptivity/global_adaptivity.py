"""
Functionality for adaptive control of micro simulations in a global way (all-to-all comparison of micro simulations)
"""
import sys
import numpy as np
import hashlib
from copy import deepcopy
from .adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
            self,
            configurator,
            micro_sim_is_on_rank: list,
            global_ids: list,
            comm,
            rank: int) -> None:
        super().__init__(configurator)
        self._sim_is_on_this_rank = micro_sim_is_on_rank
        self._global_ids = global_ids
        self._comm = comm
        self._rank = rank
        # keys are active global IDs, values are lists of local and global IDs of associated inactive sims on this rank
        self._active_to_inactive_map = dict()
        self._send_sims_from_this_rank = dict()  # keys are global IDs of sims to send, values are ranks to send the sims to
        self._recv_sims_from_ranks = dict()  # keys are global IDs to receive, values are ranks to receive from

    def update_active_sims(
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
        global_number_of_sims = _micro_sim_states.size

        # Update the set of active micro sims
        for i in range(global_number_of_sims):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    if self._sim_is_on_this_rank[i]:
                        local_id = self._global_ids.index(i)
                        micro_sims[local_id].deactivate()
                    _micro_sim_states[i] = 0

        return _micro_sim_states

    def update_inactive_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list):
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point
        number_of_sims = _micro_sim_states.size

        send_sims_to_ranks_local = dict()  # keys are global IDs, values are rank to send to

        # Create a map between inactive sims on this rank and their globally associated simulations
        for global_id in range(number_of_sims):
            if not _micro_sim_states[global_id]:  # if id is inactive
                if self._check_for_activation(global_id, similarity_dists, _micro_sim_states):
                    _micro_sim_states[global_id] = 1

        inactive_local_ids = np.where(micro_sim_states[self._global_ids[0]:self._global_ids[-1]] == 0)[0]

        for inactive_local_id in inactive_local_ids:
            assoc_active_global_id = micro_sims[inactive_local_id].get_associated_active_id()

            # Kill the inactive micro sim object
            micro_sims[inactive_local_id] = None

            # Get the rank on which the associated active simulation is
            recv_rank = self._sim_is_on_this_rank[assoc_active_global_id]

            # If associated simulation is on this rank, copy it directly
            if recv_rank == self._rank:
                assoc_active_local_id = self._global_ids.index(assoc_active_global_id)
                micro_sims[inactive_local_id] = deepcopy(micro_sims[assoc_active_local_id])
                micro_sims[inactive_local_id].set_global_id(assoc_active_global_id)
            else:
                # Add simulation and its rank to receive map
                self._recv_sims_from_ranks[assoc_active_global_id] = recv_rank
                # Add simulation and this rank to local sending map
                send_sims_to_ranks_local[assoc_active_global_id] = [self._rank]

        # ----- Gather information about which sims to send where, from the sending perspective -----
        send_sims_to_ranks_list = self._comm.allgather(send_sims_to_ranks_local)

        for d in send_sims_to_ranks_list:
            for global_id, rank in d.items():
                if self._sim_is_on_this_rank[global_id] == self._rank:
                    self._send_sims_from_this_rank[global_id].append(rank)
        # ----------

        recv_reqs = self._p2p_comm(self._send_sims_from_this_rank, self._recv_sims_from_ranks, micro_sims)

        # Use received micro sims to activate the currently inactive sims on this rank
        for req in recv_reqs:
            active_sim = req.wait()
            global_id = active_sim.get_global_id()
            inactive_sims_data = self._active_to_inactive_map[global_id]
            for inactive_sim_data in inactive_sims_data:
                local_id = inactive_sim_data[0]
                global_id = inactive_sim_data[1]
                micro_sims[local_id] = deepcopy(active_sim)
                micro_sims[local_id].set_global_id(global_id)

        return _micro_sim_states

    def associate_inactive_to_active(
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
        self._active_to_inactive_map.clear()  # Clear the dictionary for new entires in this timestep

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

            if self._sim_is_on_this_rank[inactive_id]:
                local_id = self._global_ids.index(inactive_id)
                micro_sims[local_id].is_associated_to_active_sim(associated_active_id)

                if isinstance(self._active_to_inactive_map[associated_active_id], list):
                    self._active_to_inactive_map[associated_active_id].append([local_id, inactive_id])
                else:
                    self._active_to_inactive_map[associated_active_id] = [[local_id, inactive_id]]

    def communicate_micro_output(self, global_ids, micro_output):
        """
        """
        _micro_output = np.copy(micro_output)

        recv_reqs = self._p2p_comm(self._send_sims_from_this_rank, self._recv_sims_from_ranks, _micro_output)

        global_ids_of_recv_data = list(self._recv_sims_from_ranks.keys())
        # Use received micro sims to activate the currently inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            local_id = global_ids.index(global_ids_of_recv_data(count))
            _micro_output[local_id] = req.wait()

        return _micro_output

    def _create_tag(self, sim_id, src_rank, dest_rank):
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode('utf-8'))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _p2p_comm(self, send_map, recv_map, data):
        """
        """
        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in recv_map.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_rank in send_map.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            local_id = self._global_ids.index(global_id)
            req = self._comm.isend(data[local_id], dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Wait for all non-blocking communications to complete
        self._comm.waitall(send_reqs)

        return recv_reqs
