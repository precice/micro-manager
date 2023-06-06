"""
Functionality for adaptive initialization and control of micro simulations
"""
import sys
import numpy as np
from typing import Callable


class AdaptivityCalculator:
    def __init__(self, configurator) -> None:
        # Names of data to be used for adaptivity computation
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._adaptivity_type = configurator.get_adaptivity_type()
        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._similarity_measure = self._get_similarity_measure(configurator.get_adaptivity_similarity_measure())

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
            # If the adaptivity-data is a scalar for each simulation,
            # expand the dimension to make it a 2D array to unify the calculation.
            # The axis is later reduced with a norm.
            data = np.expand_dims(data, axis=1)

        data_diff = self._similarity_measure(data)
        _similarity_dists += dt * data_diff

        return _similarity_dists

    def update_active_micro_sims(
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
        number_of_sims = _micro_sim_states.size

        # Update the set of active micro sims
        for i in range(number_of_sims):
            if _micro_sim_states[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _micro_sim_states):
                    micro_sims[i].deactivate()
                    _micro_sim_states[i] = 0

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

            micro_sims[inactive_id].is_associated_to_active_sim(
                micro_sims[associated_active_id].get_local_id(), micro_sims[associated_active_id].get_global_id())

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

    def _get_similarity_measure(self, similarity_measure: str) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get similarity measure to be used for similarity calculation

        Parameters
        ----------
        similarity_measure : str
            String specifying the similarity measure to be used

        Returns
        -------
        similarity_measure : function
            Function to be used for similarity calculation. Takes data as input and returns similarity measure
        """
        if similarity_measure == 'L1':
            return self._l1
        elif similarity_measure == 'L2':
            return self._l2
        elif similarity_measure == 'L1rel':
            return self._l1rel
        elif similarity_measure == 'L2rel':
            return self._l2rel
        else:
            raise ValueError(
                'Similarity measure not supported. Currently supported similarity measures are "L1", "L2", "L1rel", "L2rel".')

    def _l1(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate L1 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        return np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=1, axis=-1)

    def _l2(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate L2 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        return np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=2, axis=-1)

    def _l1rel(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate L1 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        pointwise_diff = data[np.newaxis, :] - data[:, np.newaxis]
        # divide by data to get relative difference
        # divide i,j by max(data[i],data[j]) to get relative difference
        relative = np.nan_to_num((pointwise_diff / np.maximum(data[np.newaxis, :], data[:, np.newaxis])))
        return np.linalg.norm(relative, ord=1, axis=-1)

    def _l2rel(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate L2 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        pointwise_diff = data[np.newaxis, :] - data[:, np.newaxis]
        # divide by data to get relative difference
        # divide i,j by max(data[i],data[j]) to get relative difference
        relative = np.nan_to_num((pointwise_diff / np.maximum(data[np.newaxis, :], data[:, np.newaxis])))
        return np.linalg.norm(relative, ord=2, axis=-1)
