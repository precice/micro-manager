"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
from math import exp

class AdaptiveController:
    def __init__(self, configurator) -> None:
        # Names of data to be used for adaptivity computation
        self._hist_param = configurator.get_adaptivity_hist_param()
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()

    def get_similarity_dists(self, dt, similarity_dists_nm1, data):
        """

        Returns
        -------

        """
        if data.ndim == 1:
            number_of_micro_sims = len(data)
            dim = 0
        elif data.ndim == 2:
            number_of_micro_sims, dim = data.shape

        micro_ids = list(range(number_of_micro_sims))
        similarity_dists = np.zeros((number_of_micro_sims, number_of_micro_sims))
        for id_1 in micro_ids:
            for id_2 in micro_ids:
                if id_1 != id_2:
                    if dim:
                        for d in dim:
                            data_diff += abs(data[id_1, d] - data[id_2, d])
                    else:
                        data_diff = abs(data[id_1] - data[id_2])
                    
                    similarity_dists[id_1, id_2] = exp(-self._hist_param * self._dt) * similarity_dists_nm1[id_1, id_2] + dt * data_diff)
                else:
                    similarity_dists[id_1, id_2] = 0.0

        return similarity_dists

    def update_active_micro_simulations(self, similarity_dists, micro_sim_states):
        ref_tol = self._refine_const * np.amax(similarity_dists)
        coarse_tol = self._coarse_const * ref_tol

        number_of_micro_sims, _ = similarity_dists.shape

        # Update the set of active micro sims
        for id_1 in range(number_of_micro_sims):
            if micro_sim_states[id_1]:  # if id_1 sim is active
                for id_2 in range(number_of_micro_sims):
                    if micro_sim_states[id_2]:  # if id_2 is active
                        if id_1 != id_2:
                            # If active sim is similar to another active sim,
                            # deactivate it
                            if similarity_dists[id_1, id_2] < coarse_tol:
                                print("sim {} and sim {} are similar, so deactivating {}".format(id_1, id_2, id_1))
                                self._micro_sims[id_1].deactivate()
                                micro_sim_states[id_1] = 0
                                break

        return micro_sim_states

    def update_inactive_micro_simulations(self, similarity_dists, micro_sim_states):
        ref_tol = self._refine_const * np.amax(similarity_dists)

        number_of_micro_sims, _ = similarity_dists.shape

        # Update the set of inactive micro sims
        dists = []
        for id_1 in range(number_of_micro_sims):
            if not micro_sim_states[id_1]:
                for id_2 in range(number_of_micro_sims):
                    if micro_sim_states[id_2]:
                        dists.append(similarity_dists[id_1, id_2])
                # If inactive sim is not similar to any active sim, activate it
                if min(dists) > ref_tol:
                    self._micro_sims[id_1].activate()
                dists = []

    def associate_inactive_to_active(self, similarity_dists, micro_sims):
        # Associate inactive micro sims to active micro sims
        micro_id = 0
        for id_1 in self._inactive_ids:
            dist_min = 100
            for id_2 in self._active_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[id_1, id_2] < dist_min:
                    micro_id = id_2
                    dist_min = similarity_dists[id_1, id_2]
            micro_sims[id_1].is_most_similar_to(micro_id)
