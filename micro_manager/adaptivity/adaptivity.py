"""
Functionality for adaptive initialization and control of micro simulations
"""

from math import exp
from typing import Callable
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.config import Config
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager

import numpy as np


class AdaptivityCalculator:
    def __init__(
        self,
        configurator: Config,
        nsims: int,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
        base_logger: Logger,
        rank: int,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        nsims : int
            Number of micro simulations.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object
            Handles instantiation of micro simulation.
        base_logger : object of class Logger
            Logger object to log messages.
        rank : int
            Rank of the MPI communicator.
        """
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._hist_param = configurator.get_adaptivity_hist_param()
        self._adaptivity_data_names = configurator.get_data_for_adaptivity()
        self._adaptivity_type = configurator.get_adaptivity_type()
        self._adaptivity_output_type = configurator.get_adaptivity_output_type()

        self._micro_problem_cls = micro_problem_cls
        self._model_manager = model_manager

        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._rank = rank
        self._base_logger = base_logger

        self._max_similarity_dist = 0.0

        self._nsims = nsims

        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        # This array is modified in place via the function update_active_sims and update_inactive_sims
        self._is_sim_active = np.array([True] * self._nsims, dtype=np.bool_)

        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        # This array is modified in place via the function associate_inactive_to_active
        self._sim_is_associated_to = np.full((self._nsims), -2, dtype=np.intc)

        self._just_deactivated: list[int] = []

        self._similarity_measure = self._get_similarity_measure(
            configurator.get_adaptivity_similarity_measure()
        )

        output_dir = configurator.get_output_dir()

        if output_dir is not None:
            metrics_output_dir = output_dir + "/adaptivity-metrics"
        else:
            metrics_output_dir = "adaptivity-metrics"

        if self._rank == 0 and (
            self._adaptivity_output_type == "global"
            or self._adaptivity_output_type == "all"
        ):
            self._global_metrics_logger = Logger(
                "global-metrics-logger",
                metrics_output_dir + "-global.csv",
                rank,
                csv_logger=True,
            )

            self._global_metrics_logger.log_info(
                "n|n active|n inactive|avg active|avg inactive|max active|rank of max active|max inactive|rank of max inactive"
            )

        if (
            self._adaptivity_output_type == "local"
            or self._adaptivity_output_type == "all"
        ):
            self._metrics_logger = Logger(
                "metrics-logger",
                metrics_output_dir + "-" + str(rank) + ".csv",
                rank,
                csv_logger=True,
            )

            self._metrics_logger.log_info("n|n active|n inactive|assoc ranks")

    def _update_similarity_dists(self, dt: float, data: dict) -> None:
        """
        Calculate metric which determines if two micro simulations are similar enough to have one of them deactivated.

        Parameters
        ----------
        dt : float
            Current time step
        data : dict
            Data to be used in similarity distance calculation
        """
        # Update similarity distances without copying
        self._similarity_dists *= exp(-self._hist_param * dt)

        for name in data.keys():
            data_vals = np.array(data[name])
            if data_vals.ndim == 1:
                # If the adaptivity data is a scalar for each simulation,
                # expand the dimension to make it a 2D array to unify the calculation.
                # The axis is later reduced with a norm.
                data_vals = np.expand_dims(data_vals, axis=1)

            self._similarity_measure(data_vals)

        self._similarity_dists *= dt

    def _associate_inactive_to_active(self) -> None:
        """
        Associate inactive micro simulations to most similar active micro simulation.
        """
        active_ids = np.where(self._is_sim_active)[0]
        inactive_ids = np.where(self._is_sim_active == False)[0]

        # Start with a large distance to trigger the search for the most similar active sim
        # Add the +1 for the case when the similarity distance matrix is zeros
        dist_min_start_value = self._max_similarity_dist + 1

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_ids:
            # Begin with a large distance to trigger the search for the most similar active sim
            dist_min = dist_min_start_value
            for active_id in active_ids:
                flat_idx = self._map2dto1d([inactive_id, active_id])
                # Find most similar active sim for every inactive sim
                if self._similarity_dists[flat_idx] < dist_min:
                    associated_active_id = active_id
                    dist_min = self._similarity_dists[flat_idx]

            self._sim_is_associated_to[inactive_id] = associated_active_id

    def _check_for_activation(self, inactive_id: int, active_ids: np.ndarray) -> bool:
        """
        Check if an inactive simulation needs to be activated.

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation.
        active_ids : numpy array
            1D array having IDs of active micro simulations.
        Return
        ------
        tag : bool
            True if the inactive simulation needs to be activated, False otherwise.
        """
        flat_ids = [
            self._map2dto1d([inactive_id, active_id]) for active_id in active_ids
        ]
        dists = self._similarity_dists[flat_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(self, active_id: int, active_ids: list) -> bool:
        """
        Check if an active simulation needs to be deactivated.

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation.
        active_ids : list
            List having IDs of active micro simulations.

        Return
        ------
        tag : bool
            True if the active simulation needs to be deactivated, False otherwise.
        """
        for active_id_2 in active_ids:
            if active_id != active_id_2:  # don't compare active sim to itself
                flat_idx = self._map2dto1d([active_id, active_id_2])
                # If active sim is similar to another active sim, deactivate it
                if self._similarity_dists[flat_idx] < self._coarse_tol:
                    return True
        return False

    def _get_similarity_measure(
        self, similarity_measure: str
    ) -> Callable[[np.ndarray], None]:
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
        if similarity_measure == "L1":
            return self._l1
        elif similarity_measure == "L2":
            return self._l2
        elif similarity_measure == "L1rel":
            return self._l1rel
        elif similarity_measure == "L2rel":
            return self._l2rel
        else:
            raise ValueError(
                'Similarity measure not supported. Currently supported similarity measures are "L1", "L2", "L1rel", "L2rel".'
            )

    def _l1(self, data: np.ndarray) -> None:
        """
        Calculate L1 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation
        """
        for i in range(1, self._nsims):
            for j in range(i):
                p_diff = np.abs(data[i] - data[j])
                flat_idx = self._map2dto1d([i, j])
                self._similarity_dists[flat_idx] += np.linalg.norm(p_diff, ord=1)

    def _l2(self, data: np.ndarray) -> None:
        """
        Calculate L2 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation
        """
        for i in range(1, self._nsims):
            for j in range(i):
                p_diff = np.abs(data[i] - data[j])
                flat_idx = self._map2dto1d([i, j])
                self._similarity_dists[flat_idx] += np.linalg.norm(p_diff, ord=2)

    def _l1rel(self, data: np.ndarray) -> None:
        """
        Calculate L1 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation
        """
        eps = np.finfo(np.float64).eps
        for i in range(1, self._nsims):
            for j in range(i):
                p_diff = np.abs(data[i] - data[j])
                denom = np.maximum(np.abs(data[i]), np.abs(data[j]))
                rel_diff = p_diff / np.maximum(denom, eps)
                flat_idx = self._map2dto1d([i, j])
                self._similarity_dists[flat_idx] += np.linalg.norm(rel_diff, ord=1)

    def _l2rel(self, data: np.ndarray) -> None:
        """
        Calculate L2 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation
        """
        eps = np.finfo(np.float64).eps
        for i in range(1, self._nsims):
            for j in range(i):
                p_diff = np.abs(data[i] - data[j])
                denom = np.maximum(np.abs(data[i]), np.abs(data[j]))
                rel_diff = p_diff / np.maximum(denom, eps)
                flat_idx = self._map2dto1d([i, j])
                self._similarity_dists[flat_idx] += np.linalg.norm(rel_diff, ord=2)

    def _map1dto2d(self, flat_idx: int) -> tuple:
        """
        Map a 1D index to 2D coordinates in strictly lower triangular part.

        Parameters
        ----------
        flat_idx : int
            1D index to be mapped

        Returns
        -------
        coord : tuple
            2D coordinates (i, j) corresponding to the 1D index, where i > j
        """
        # Find row i: solve i(i-1)/2 <= flat_idx < i(i+1)/2
        i = 1
        while i * (i + 1) // 2 <= flat_idx:
            i += 1
        j = flat_idx - i * (i - 1) // 2
        return (i, j)

    def _map2dto1d(self, coords: list) -> int:
        """
        Map 2D coordinates to a 1D index in strictly lower triangular storage.

        Parameters
        ----------
        coords : list
            2D coordinates [i, j] to be mapped, where i > j

        Returns
        -------
        flat_idx : int
            1D index corresponding to the 2D coordinates in strictly lower triangular part
        """
        i, j = coords[0], coords[1]
        if i <= j:
            # For symmetric matrix access, swap if needed
            i, j = j, i
        if i <= j:
            raise ValueError(
                f"Invalid indices for lower triangular: ({i}, {j}). Need i > j."
            )
        return i * (i - 1) // 2 + j
