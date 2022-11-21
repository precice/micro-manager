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

        if data.ndim == 1:
            number_of_micro_sims = len(data)
            dim = 0
        elif data.ndim == 2:
            number_of_micro_sims, dim = data.shape

        micro_ids = list(range(number_of_micro_sims))
        for id_1 in micro_ids:
            for id_2 in micro_ids:
                data_diff = 0
                if id_1 != id_2:
                    if dim:
                        for d in range(dim):
                            data_diff += abs(data[id_1, d] - data[id_2, d])
                    else:
                        data_diff = abs(data[id_1] - data[id_2])

                    _similarity_dists[id_1, id_2] += dt * data_diff
                else:
                    _similarity_dists[id_1, id_2] = 0.0

        return _similarity_dists

    def update_active_micro_sims(self, similarity_dists, micro_sim_states, micro_sims):
        ref_tol = self._refine_const * np.amax(similarity_dists)
        coarse_tol = self._coarse_const * ref_tol

        number_of_micro_sims, _ = similarity_dists.shape

        _micro_sim_states = np.copy(micro_sim_states)

        # Update the set of active micro sims
        for id_1 in range(number_of_micro_sims):
            if _micro_sim_states[id_1]:  # if id_1 sim is active
                for id_2 in range(number_of_micro_sims):
                    if _micro_sim_states[id_2]:  # if id_2 is active
                        if id_1 != id_2:  # don't compare active sim to itself
                            # If active sim is similar to another active sim,
                            # deactivate it
                            if similarity_dists[id_1, id_2] < coarse_tol:
                                print("sim {} and sim {} are similar, so deactivating {}".format(id_1, id_2, id_1))
                                micro_sims[id_1].deactivate()
                                _micro_sim_states[id_1] = 0
                                break

        print("Micro sim states after activation update = {}".format(_micro_sim_states))

        return _micro_sim_states

    def update_inactive_micro_sims(self, similarity_dists, micro_sim_states, micro_sims):
        ref_tol = self._refine_const * np.amax(similarity_dists)

        number_of_micro_sims, _ = similarity_dists.shape

        _micro_sim_states = np.copy(micro_sim_states)

        if not np.any(_micro_sim_states):
            _micro_sim_states[0] = 1  # If all sims are inactive, activate the first one (a random choice)

        # Update the set of inactive micro sims
        for id_1 in range(number_of_micro_sims):
            dists = []
            if not _micro_sim_states[id_1]:  # if id_1 is inactive
                for id_2 in range(number_of_micro_sims):
                    if _micro_sim_states[id_2]:  # if id_2 is active
                        dists.append(similarity_dists[id_1, id_2])
                # If inactive sim is not similar to any active sim, activate it
                print("For inactive id {}, similarity dists are = {}, and ref_tol is {}".format(id_1, dists, ref_tol))
                if min(dists) > ref_tol:
                    micro_sims[id_1].activate()
                    print("Activate {}".format(id_1))
                    _micro_sim_states[id_1] = 1

        print("Micro sim states after inactivation update = {}".format(_micro_sim_states))

        return _micro_sim_states

    def associate_inactive_to_active(self, similarity_dists, micro_sim_states, micro_sims):
        # Associate inactive micro sims to active micro sims
        micro_id = 0
        active_sim_indices = np.where(micro_sim_states == 1)[0]
        inactive_sim_indices = np.where(micro_sim_states == 0)[0]

        print("Active sim indices in association = {}".format(active_sim_indices))
        print("Inactive sim indices in association = {}".format(inactive_sim_indices))

        for id_1 in inactive_sim_indices:
            dist_min = sys.float_info.max
            for id_2 in active_sim_indices:
                # Find most similar active sim for every inactive sim
                if similarity_dists[id_1, id_2] < dist_min:
                    micro_id = id_2
                    dist_min = similarity_dists[id_1, id_2]
            micro_sims[id_1].is_most_similar_to(micro_id)
