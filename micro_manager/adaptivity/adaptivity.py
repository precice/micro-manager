"""
Functionality for adaptive initialization and control of micro simulations
"""
import sys
import numpy as np
from math import exp
from typing import Callable


class AdaptivityCalculator:
    def __init__(self, configurator, logger) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        logger : Logger defined from the standard package logging
        """
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._hist_param = configurator.get_adaptivity_hist_param()
        self._adaptivity_data_names = configurator.get_data_for_adaptivity()
        self._adaptivity_type = configurator.get_adaptivity_type()

        self._logger = logger

        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._similarity_measure = self._get_similarity_measure(configurator.get_adaptivity_similarity_measure())

    def _get_similarity_dists(self, dt: float, similarity_dists: np.ndarray, data: dict) -> np.ndarray:
        """
        Calculate metric which determines if two micro simulations are similar enough to have one of them deactivated.

        Parameters
        ----------
        dt : float
            Current time step
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        data : dict
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        _similarity_dists = np.copy(similarity_dists)

        data_diff = np.zeros_like(_similarity_dists)
        for name in self._adaptivity_data_names:
            data_vals = data[name]
            if data_vals.ndim == 1:
                # If the adaptivity-data is a scalar for each simulation,
                # expand the dimension to make it a 2D array to unify the calculation.
                # The axis is later reduced with a norm.
                data_vals = np.expand_dims(data_vals, axis=1)

            data_diff += self._similarity_measure(data_vals)

        return exp(-self._hist_param * dt) * _similarity_dists + dt * data_diff

    def _update_active_sims(
            self,
            similarity_dists: np.ndarray,
            is_sim_active: np.ndarray) -> np.ndarray:
        """
        Update set of active micro simulations. Active micro simulations are compared to each other
        and if found similar, one of them is deactivated.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        _is_sim_active : numpy array
            Updated 1D array having state (active or inactive) of each micro simulation
        """
        self._coarse_tol = self._coarse_const * self._refine_const * np.amax(similarity_dists)

        _is_sim_active = np.copy(is_sim_active)  # Input is_sim_active is not longer used after this point

        # Update the set of active micro sims
        for i in range(_is_sim_active.size):
            if _is_sim_active[i]:  # if sim is active
                if self._check_for_deactivation(i, similarity_dists, _is_sim_active):
                    _is_sim_active[i] = False

        return _is_sim_active

    def _associate_inactive_to_active(
            self,
            similarity_dists: np.ndarray,
            is_sim_active: np.ndarray,
            sim_is_associated_to: np.ndarray) -> np.ndarray:
        """
        Associate inactive micro simulations to most similar active micro simulation.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None

        Returns
        -------
        _sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        """
        active_ids = np.where(is_sim_active)[0]
        inactive_ids = np.where(is_sim_active == False)[0]

        _sim_is_associated_to = np.copy(sim_is_associated_to)

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_ids:
            dist_min = sys.float_info.max
            for active_id in active_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = similarity_dists[inactive_id, active_id]

            _sim_is_associated_to[inactive_id] = associated_active_id

        return _sim_is_associated_to

    def _check_for_activation(
            self,
            inactive_id: int,
            similarity_dists: np.ndarray,
            is_sim_active: np.ndarray) -> bool:
        """
        Check if an inactive simulation needs to be activated.

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation.
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair.
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation.

        Return
        ------
        tag : bool
            True if the inactive simulation needs to be activated, False otherwise.
        """
        active_sim_ids = np.where(is_sim_active)[0]

        dists = similarity_dists[inactive_id, active_sim_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(
            self,
            active_id: int,
            similarity_dists: np.ndarray,
            is_sim_active: np.ndarray) -> bool:
        """
        Check if an active simulation needs to be deactivated.

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation.
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair.
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation.

        Return
        ------
        tag : bool
            True if the active simulation needs to be deactivated, False otherwise.
        """
        active_sim_ids = np.where(is_sim_active)[0]

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
