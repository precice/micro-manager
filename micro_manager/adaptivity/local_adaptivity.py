"""
Functionality for adaptive control of micro simulations locally within a rank (or the entire domain if the Micro Manager is run in serial)
"""
import numpy as np
from copy import deepcopy
from .adaptivity import AdaptivityCalculator


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator) -> None:
        super().__init__(configurator)

    def update_inactive_sims(
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
        number_of_sims = _micro_sim_states.size

        # Update the set of inactive micro sims
        for i in range(number_of_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    associated_active_local_id = micro_sims[i].get_associated_active_local_id()

                    # Get local and global ID of inactive simulation, to set it to the copied simulation later
                    local_id = micro_sims[i].get_local_id()
                    global_id = micro_sims[i].get_global_id()

                    # Copy state from associated active simulation with get_state and
                    # set_state if available else deepcopy
                    if hasattr(micro_sims[associated_active_local_id], 'get_state') and \
                            hasattr(micro_sims[associated_active_local_id], 'set_state'):
                        micro_sims[i].set_state(*micro_sims[associated_active_local_id].get_state())
                    else:
                        micro_sims[i] = None
                        micro_sims[i] = deepcopy(micro_sims[associated_active_local_id])
                        micro_sims[i].set_local_id(local_id)
                        micro_sims[i].set_global_id(global_id)
                    _micro_sim_states[i] = 1

        return _micro_sim_states
