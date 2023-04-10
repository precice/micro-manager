"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import sys
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, number_of_global_sims, global_ids) -> None:
        super().__init__(configurator)
        self._number_of_global_sims = number_of_global_sims
        self._global_sim_ids = global_ids

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
                    micro_sims[i].deactivate()
                    _micro_sim_states[i] = 0
