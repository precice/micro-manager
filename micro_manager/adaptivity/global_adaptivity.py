"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import hashlib
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, global_ids: list, number_of_global_sims: int, global_ids_of_local_sims: list, micro_sim_is_on_rank: list, comm, rank: int) -> None:
        super().__init__(configurator, global_ids)
        self._number_of_global_sims = number_of_global_sims
        self._micro_sim_is_on_rank = micro_sim_is_on_rank
        self._global_ids_of_local_sims = global_ids_of_local_sims
        self._comm = comm
        self._rank = rank

    def update_active_micro_sims(
        self,
        similarity_dists: np.ndarray,
        micro_sim_states: np.ndarray,
        micro_sims: list) -> np.ndarray:
        """
        """
        self._coarse_tol = self._coarse_const * self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point
        # Update the set of active micro sims
        for i in range(self._number_of_global_sims):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 0
                    if i in self._global_ids_of_local_sims:
                        micro_sims[i].deactivate()

        return _micro_sim_states

    def update_inactive_micro_sims(
        self,
        similarity_dists: np.ndarray,
        micro_sim_states: np.ndarray,
        micro_sims: list) -> np.ndarray:
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Update the set of inactive micro sims
        global_ids_of_sims_to_receive = []
        recv_reqs = []
        for i in range(self._number_of_global_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    if i in self._global_ids_of_local_sims:  # Local scope from here on
                        associated_active_global_id = micro_sims[i].get_associated_active_global_id()

                        # Effectively kill the micro sim object associated to the inactive ID
                        micro_sims[i] = None

                        recv_rank = self._micro_sim_is_on_rank[associated_active_global_id]
                        
                        if recv_rank == self._rank:
                            local_id = self._global_ids_of_local_sims.index(associated_active_global_id)
                            micro_sims[i] = deepcopy(micro_sims[local_id])
                        else:
                            hash_tag = hashlib.sha256()
                            hash_tag.update((str(self._rank) + str(associated_active_global_id) + str(recv_rank)).encode('utf-8'))
                            tag = int(hash_tag.hexdigest()[:6], base=16)
                            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

                            global_ids_of_sims_to_receive.append(associated_active_global_id)

        for active_global_id in global_ids_of_sims_to_receive:
            if self._rank == self._micro_sim_is_on_rank[active_global_id]:
                send_hashtag = hashlib.sha256()
                send_hashtag.update((str(self._rank) + str(neigh)).encode('utf-8'))
                send_tag = int(send_hashtag.hexdigest()[:6], base=16)
                req = comm.isend(unowned_gids, dest=neigh, tag=send_tag)
                send_reqs.append(req)


        return _micro_sim_states