"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np


class AdaptivityCalculator:
    def __init__(self, configurator, global_ids) -> None:
        # Names of data to be used for adaptivity computation
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._adaptivity_type = configurator.get_adaptivity_type()
        self._coarse_tol = 0.0
        self._ref_tol = 0.0
        # Use set to make the "in" functionality faster for large lists
        self._global_ids_of_local_sims = global_ids

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
            data = np.expand_dims(data, axis=1)

        data_diff = self._similarity_measure(data)
        _similarity_dists += dt * data_diff

        return _similarity_dists

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

    def _get_similarity_measure(self, similarity_measure):
        """
        Set similarity measure to be used for similarity calculation

        Parameters
        ----------
        similarity_measure : str
            String specifying the similarity measure to be used
        """
        if similarity_measure == 'L1':
            return lambda data: np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=1, axis=-1)
        elif similarity_measure == 'L2':
            return lambda data: np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=2, axis=-1)
        elif similarity_measure == 'L1rel':
            def l1rel(data):
                pointwise_diff = data[np.newaxis, :] - data[:, np.newaxis]
                # divide by data to get relative difference
                # transpose to divide row i by data[i].
                relative = np.nan_to_num((pointwise_diff.transpose(1, 0, 2) / data).transpose(1, 0, 2))
                return np.linalg.norm(relative, ord=1, axis=-1)
            return l1rel
        elif similarity_measure == 'L2rel':
            def l2rel(data):
                pointwise_diff = data[np.newaxis, :] - data[:, np.newaxis]
                # divide by data to get relative difference
                # transpose to divide row i by data[i]
                relative = np.nan_to_num((pointwise_diff.transpose(1, 0, 2) / data).transpose(1, 0, 2))
                return np.linalg.norm(relative, ord=2, axis=-1)
            return l2rel
        else:
            raise ValueError(
                'Similarity measure not supported. Currently supported similarity measures are "L1", "L2", "L1rel", "L2rel".')
