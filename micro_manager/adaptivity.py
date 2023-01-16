"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import sys


class AdaptiveController:
    def __init__(self, configurator) -> None:
        # Names of data to be used for adaptivity computation
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()

    def get_active_and_inactive_ids(self, micro_sim_states):
        """
        
        """
        active_sims_local_ids = []
        inactive_sims_local_ids = []
        for id in range(micro_sim_states.size):
            if micro_sim_states[id] == 1:
                active_sims_local_ids.append(id)
            elif micro_sim_states[id] == 0:
                inactive_sims_local_ids.append(id)
        
        return active_sims_local_ids, inactive_sims_local_ids

    def get_similarity_dists(self, dt, similarity_dists, data):
        """
        Calculate metric which determines if two micro simulations are similar enough to have one of them deactivated.

        Parameters
        ----------

        Returns
        -------
        similarity_dists : numpy array
        """
        _similarity_dists = np.copy(similarity_dists)
        n_sims, _ = _similarity_dists.shape

        if data.ndim == 1:
            dim = 0
        elif data.ndim == 2:
            _, dim = data.shape

        counter_1 = 0
        for id_1 in range(n_sims):
            counter_2 = 0
            for id_2 in range(n_sims):
                data_diff = 0
                if id_1 != id_2:
                    if dim:
                        for d in range(dim):
                            data_diff += abs(data[counter_1, d] - data[counter_2, d])
                    else:
                        data_diff = abs(data[counter_1] - data[counter_2])

                    _similarity_dists[id_1, id_2] += dt * data_diff
                else:
                    _similarity_dists[id_1, id_2] = 0
                counter_2 += 1
            counter_1 += 1

        return _similarity_dists

    def update_active_micro_sims(self, similarity_dists, micro_sim_states, micro_sims):
        coarse_tol = self._coarse_const * self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)
        n_sims, _ = _micro_sim_states.shape

        # Update the set of active micro sims
        for id_1 in range(n_sims):
            if _micro_sim_states[id_1]:  # if id_1 sim is active
                for id_2 in range(n_sims):
                    if _micro_sim_states[id_2]:  # if id_2 is active
                        if id_1 != id_2:  # don't compare active sim to itself
                            # If active sim is similar to another active sim,
                            # deactivate it
                            if similarity_dists[id_1, id_2] < coarse_tol:
                                micro_sims[id_1].deactivate()
                                _micro_sim_states[id_1] = 0
                                break

        return _micro_sim_states

    def update_inactive_micro_sims(self, similarity_dists, micro_sim_states, micro_sims):
        ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)
        n_sims, _ = _micro_sim_states.shape

        if not np.any(_micro_sim_states):
            _micro_sim_states[0] = 1  # If all sims are inactive, activate the first one (a random choice)

        # Update the set of inactive micro sims
        for id_1 in range(n_sims):
            dists = []
            if not _micro_sim_states[id_1]:  # if id_1 is inactive
                for id_2 in range(n_sims):
                    if _micro_sim_states[id_2]:  # if id_2 is active
                        dists.append(similarity_dists[id_1, id_2])
                # If inactive sim is not similar to any active sim, activate it
                if min(dists) > ref_tol:
                    micro_sims[id_1].activate()
                    _micro_sim_states[id_1] = 1

        return _micro_sim_states

    def associate_inactive_to_active(self, similarity_dists, micro_sim_states, micro_sims):
        """
        
        """
        active_sim_ids, inactive_sim_ids = self.get_active_and_inactive_ids(micro_sim_states)
        
        # Associate inactive micro sims to active micro sims
        for id_1 in inactive_sim_ids:
            dist_min = sys.float_info.max
            for id_2 in active_sim_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[id_1, id_2] < dist_min:
                    micro_id = id_2
                    dist_min = similarity_dists[id_1, id_2]
            micro_sims[id_1].is_most_similar_to(micro_id)
