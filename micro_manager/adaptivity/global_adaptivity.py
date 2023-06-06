"""
Functionality for adaptive control of micro simulations in a global way (all-to-all comparison of micro simulations)
"""
import sys
import numpy as np
import hashlib
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
            self,
            configurator,
            micro_sim_is_on_rank: list,
            comm,
            rank: int) -> None:
        super().__init__(configurator)
        self._micro_sim_is_on_rank = micro_sim_is_on_rank
        self._comm = comm
        self._rank = rank
        self._send_sims_from_this_rank = dict()
        self._recv_sims_from_ranks = dict()  # keys are global IDs, values are ranks to receive from

    def _create_tag(self, sim_id, src_rank, dest_rank):
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode('utf-8'))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def update_inactive_micro_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list):
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point
        number_of_sims = _micro_sim_states.size

        # keys are associated global active IDs, values are global inactive IDs which are to be activated, and
        inactive_to_associated_active_map = dict()
        send_sims_to_ranks_local = dict()  # keys are global IDs, values are rank to send to

        # Update the set of inactive micro sims
        for i in range(number_of_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    if i in self._global_ids_of_local_sims:  # Local scope from here on
                        associated_active_global_id = micro_sims[i].get_associated_active_global_id()

                        # Store the associated link in the map
                        inactive_to_associated_active_map[associated_active_global_id] = [
                            i, micro_sims[i].get_local_id(), micro_sims[i].get_global_id()]

                        # Effectively kill the micro sim object associated to the inactive ID
                        micro_sims[i] = None

                        # Get the rank on which the associated active simulation is
                        recv_rank = self._micro_sim_is_on_rank[associated_active_global_id]

                        # If simulation is to be copied from this rank, just do it directly
                        if recv_rank == self._rank:
                            local_id = self._global_ids_of_local_sims.index(associated_active_global_id)
                            micro_sims[i] = deepcopy(micro_sims[local_id])
                        else:
                            # Gather information about which sims to receive from where
                            self._recv_sims_from_ranks[associated_active_global_id] = recv_rank
                            # Gather information about which sims to send where, but from the receiving perspective
                            send_sims_to_ranks_local[associated_active_global_id] = self._rank

        # ----- Gather information about which sims to send where, from the sending perspective -----
        send_sims_to_ranks_list = self._comm.allgather(send_sims_to_ranks_local)

        for d in send_sims_to_ranks_list:
            for global_id, rank in d.items():
                if self._micro_sim_is_on_rank[global_id] == self._rank:
                    self._send_sims_from_this_rank[global_id] = rank
        # ----------

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in self._recv_sims_from_ranks.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_rank in self._send_sims_from_this_rank.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            req = self._comm.isend(micro_sims[global_id], dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Wait for all non-blocking communications to complete
        self._comm.waitall(send_reqs)

        # Use received micro sims to activate the currently inactive sims on this rank
        for req in recv_reqs:
            active_sim = req.wait()
            global_id = active_sim.get_global_id()
            inactive_sim_data = inactive_to_associated_active_map[global_id]
            micro_sims[inactive_sim_data[0]] = deepcopy(active_sim)
            micro_sims[inactive_sim_data[0]].set_local_id(inactive_sim_data[1])
            micro_sims[inactive_sim_data[0]].set_global_id(inactive_sim_data[2])

        return _micro_sim_states

    def communicate_data(self, global_ids, micro_output):
        """
        """
        _micro_output = np.copy(micro_output)

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in self._recv_sims_from_ranks.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_rank in self._send_sims_from_this_rank.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            local_id = global_ids.index(global_id)
            req = self._comm.isend(_micro_output[local_id], dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Wait for all non-blocking communications to complete
        self._comm.waitall(send_reqs)

        global_ids_of_recv_data = list(self._recv_sims_from_ranks.keys())
        # Use received micro sims to activate the currently inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            local_id = global_ids.index(global_ids_of_recv_data(count))
            _micro_output[local_id] = req.wait()

        return _micro_output
