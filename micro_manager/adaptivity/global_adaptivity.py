"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import hashlib
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, global_ids, number_of_global_sims) -> None:
        super().__init__(configurator, global_ids)
        self._number_of_global_sims = number_of_global_sims

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
        comm,
        rank,
        similarity_dists: np.ndarray,
        micro_sim_states: np.ndarray,
        micro_sims: list) -> np.ndarray:
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Update the set of inactive micro sims
        for i in range(self._number_of_global_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    if i in self._global_ids_of_local_sims:
                        associated_active_global_id = micro_sims[i].get_associated_active_global_id()

                        # Effectively kill the micro sim object associated to the inactive ID
                        micro_sims[i] = None

                        recv_rank = self._micro_sim_is_on_rank[associated_active_global_id]
                        
                        recv_reqs = []
                        if recv_rank == rank:
                            local_id = self._global_ids_of_local_sims.index(associated_active_global_id)
                            self._micro_sims[i] = deepcopy(self._micro_sims[local_id])
                        else:
                            hash_tag = hashlib.sha256()
                            hash_tag.update((str(rank) + str(associated_active_global_id) + str(recv_rank)).encode('utf-8'))
                            tag = int(hash_tag.hexdigest()[:6], base=16)
                            recv_reqs.append(comm.irecv(source=recv_rank, tag=tag))

        return _micro_sim_states