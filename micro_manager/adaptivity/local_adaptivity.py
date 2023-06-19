"""
Functionality for adaptive control of micro simulations locally within a rank (or the entire domain if the Micro Manager is run in serial)
"""
import sys
import numpy as np
from copy import deepcopy
from math import exp
from .adaptivity import AdaptivityCalculator


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, logger) -> None:
        super().__init__(configurator, logger)

    def compute_adaptivity(
            self,
            dt,
            micro_sims,
            similarity_dists_nm1: np.ndarray,
            micro_sim_states_nm1: np.ndarray,
            data_for_adaptivity: dict):
        """
        Compute adaptivity locally (within a rank) based on similarity distances and micro simulation states

        Parameters
        ----------
        similarity_dists_nm1 : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states_nm1 : numpy array
            1D array having state (active or inactive) of each micro simulation on this rank

        Results
        -------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        """
        # Multiply old similarity distance by history term to get current distances
        similarity_dists_n = exp(-self._hist_param * dt) * similarity_dists_nm1

        for name, _ in self._adaptivity_data_names.items():
            # For global adaptivity, similarity distance matrix is calculated globally on every rank
            similarity_dists_n = self._get_similarity_dists(
                dt, similarity_dists_n, data_for_adaptivity[name])

        # Operation done globally if global adaptivity is chosen
        micro_sim_states_n = self._update_active_sims(
            similarity_dists_n, micro_sim_states_nm1, micro_sims)

        micro_sim_states_n = self._update_inactive_sims(
            similarity_dists_n, micro_sim_states_nm1, micro_sims)

        self._associate_inactive_to_active(
            similarity_dists_n, micro_sim_states_n, micro_sims)

        self._logger.info(
            "Number of active micro simulations = {}".format(
                np.count_nonzero(
                    micro_sim_states_n == 1)))
        self._logger.info(
            "Number of inactive micro simulations = {}".format(
                np.count_nonzero(
                    micro_sim_states_n == 0)))

        return similarity_dists_n, micro_sim_states_n

    def _update_active_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> np.ndarray:
        """
        Update set of active micro simulations. Active micro simulations are compared to each other
        and if found similar, one of them is deactivated.
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
        self._coarse_tol = self._coarse_const * self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point
        local_number_of_sims = _micro_sim_states.size

        # Update the set of active micro sims
        for i in range(local_number_of_sims):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    micro_sims[i].deactivate()
                    _micro_sim_states[i] = 0

        return _micro_sim_states

    def _update_inactive_sims(
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
                    associated_active_local_id = micro_sims[i].get_associated_active_id()

                    # Get local and global ID of inactive simulation, to set it to the copied simulation later
                    global_id = micro_sims[i].get_global_id()

                    # Copy state from associated active simulation with get_state and
                    # set_state if available else deepcopy
                    if hasattr(micro_sims[associated_active_local_id], 'get_state') and \
                            hasattr(micro_sims[associated_active_local_id], 'set_state'):
                        micro_sims[i].set_state(*micro_sims[associated_active_local_id].get_state())
                    else:
                        micro_sims[i] = None
                        micro_sims[i] = deepcopy(micro_sims[associated_active_local_id])
                        micro_sims[i].set_global_id(global_id)
                    _micro_sim_states[i] = 1

        return _micro_sim_states

    def _associate_inactive_to_active(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> None:
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

        active_ids = np.where(micro_sim_states == 1)[0]
        inactive_ids = np.where(micro_sim_states == 0)[0]

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_ids:
            dist_min = sys.float_info.max
            for active_id in active_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = similarity_dists[inactive_id, active_id]

            micro_sims[inactive_id].is_associated_to_active_sim(associated_active_id)
