"""

"""
import sys
import numpy as np
from copy import deepcopy
from adaptivity import AdaptivityCalculator

class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, number_of_local_sims, global_ids) -> None:
        super().__init__(configurator)
        self._number_of_local_sims = number_of_local_sims
        self._micro_sim_global_ids = global_ids

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
        for i in range(self._number_of_local_sims):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    micro_sims[i].deactivate()
                    _micro_sim_states[i] = 0

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
        for i in range(self._number_of_local_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    associated_active_local_id = micro_sims[i].get_associated_active_local_id()

                    # Effectively kill the micro sim object associated to the inactive ID
                    micro_sims[i] = None

                    # Make a copy of the associated active micro sim object
                    micro_sims[i] = deepcopy(micro_sims[associated_active_local_id])
        
        return _micro_sim_states
    
    def associate_inactive_to_active(
        self,
        similarity_dists: np.ndarray,
        micro_sim_states: np.ndarray,
        micro_sims: list) -> list:
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
        active_sim_ids = np.where(micro_sim_states == 1)[0]
        inactive_sim_ids = np.where(micro_sim_states == 0)[0]

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_sim_ids:
            dist_min = sys.float_info.max
            for active_id in active_sim_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = similarity_dists[inactive_id, active_id]

            micro_sims[inactive_id].is_associated_to_active_sim(associated_active_id, self._micro_sim_global_ids[associated_active_id])
