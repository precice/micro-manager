"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import sys
from copy import deepcopy


class AdaptivityCalculator:
    def __init__(self, configurator) -> None:
        # Names of data to be used for adaptivity computation
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._adaptivity_type = configurator.get_adaptivity_type()
        self._number_of_local_sims = 0
        self._number_of_global_sims = 0
        self._coarse_tol = 0.0
        self._ref_tol = 0.0

    def set_number_of_sims(self, number_of_local_sims, number_of_global_sims: int) -> None:
        """
        Setting number of simulations for the AdaptiveController object.

        Parameters
        ----------
        number_of_sims : int
            Number of micro simulations
        """
        self._number_of_local_sims = number_of_local_sims
        self._number_of_global_sims = number_of_global_sims

    def get_similarity_dists(self, dt: float, similarity_dists: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate metric which determines if two micro simulations are similar enough to have one of them deactivated.

        Parameters
        ----------
        dt : float
            Timestep
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        _similarity_dists = np.copy(similarity_dists)

        if data.ndim == 1:
            dim = 0
        elif data.ndim == 2:
            _, dim = data.shape

        number_of_sims, _ = _similarity_dists.shape

        for counter_1, id_1 in enumerate(range(number_of_sims)):
            for counter_2, id_2 in enumerate(range(number_of_sims)):
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

        return _similarity_dists

    def update_active_micro_sims_locally(
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

    def update_active_micro_sims_globally(
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



    def update_inactive_micro_sims_locally(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> np.ndarray:
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Update the set of inactive micro sims
        for i in range(self._number_of_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    associated_active_local_id = micro_sims[i].get_associated_active_local_id()

                    # Effectively kill the micro sim object associated to the inactive ID
                    micro_sims[i] = None

                    # Make a copy of the associated active micro sim object
                    micro_sims[i] = deepcopy(self._micro_sims[associated_active_local_id])
        
        return _micro_sim_states



    def _check_for_activation(
            self,
            inactive_id: int,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray) -> bool:
        """
        Function to check if an inactive simulation needs to be activated

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        """
        active_sim_ids = np.where(micro_sim_states == 1)[0]

        dists = similarity_dists[inactive_id, active_sim_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(
            self,
            active_id: int,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray) -> bool:
        """
        Function to check if an active simulation needs to be deactivated

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        """
        active_sim_ids = np.where(micro_sim_states == 1)[0]

        for active_id_2 in active_sim_ids:
            if active_id != active_id_2:  # don't compare active sim to itself
                # If active sim is similar to another active sim, deactivate it
                if similarity_dists[active_id, active_id_2] < self._coarse_tol:
                    return True
        return False

    def associate_inactive_to_active_locally(
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

            micro_sims[inactive_id].is_associated_to(associated_active_id)
