"""
Functionality for adaptive initialization and control of micro simulations
"""
import sys
from math import exp
from typing import Callable
from warnings import warn
import importlib
from micro_manager.tools.logging_wrapper import Logger

import numpy as np


class AdaptivityCalculator:
    def __init__(self, configurator, rank, nsims) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        rank : int
            Rank of the MPI communicator.
        nsims : int
            Number of micro simulations.
        """
        self._refine_const = configurator.get_adaptivity_refining_const()
        self._coarse_const = configurator.get_adaptivity_coarsening_const()
        self._hist_param = configurator.get_adaptivity_hist_param()
        self._adaptivity_data_names = configurator.get_data_for_adaptivity()
        self._adaptivity_type = configurator.get_adaptivity_type()
        self._adaptivity_output_type = configurator.get_adaptivity_output_type()

        self._micro_problem = getattr(
            importlib.import_module(
                configurator.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation",
        )

        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._rank = rank

        self._max_similarity_dist = 0.0

        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        # This array is modified in place via the function update_active_sims and update_inactive_sims
        self._is_sim_active = np.array([True] * nsims, dtype=np.bool_)

        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        # This array is modified in place via the function associate_inactive_to_active
        self._sim_is_associated_to = np.full((nsims), -2, dtype=np.intc)

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
                "n,avg active,avg inactive,max active,max inactive"
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

            self._similarity_dists += dt * self._similarity_measure(data_vals)

    def _update_active_sims(self) -> None:
        """
        Update set of active micro simulations. Active micro simulations are compared to each other
        and if found similar, one of them is deactivated.
        """
        if self._max_similarity_dist == 0.0:
            warn(
                "All similarity distances are zero, probably because all the data for adaptivity is the same. Coarsening tolerance will be manually set to minimum float number."
            )
            self._coarse_tol = sys.float_info.min
        else:
            self._coarse_tol = (
                self._coarse_const * self._refine_const * self._max_similarity_dist
            )

        # Update the set of active micro sims
        for i in range(self._is_sim_active.size):
            if self._is_sim_active[i]:  # if sim is active
                if self._check_for_deactivation(i, self._is_sim_active):
                    self._is_sim_active[i] = False
                    self._just_deactivated.append(i)

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
                # Find most similar active sim for every inactive sim
                if self._similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = self._similarity_dists[inactive_id, active_id]

            self._sim_is_associated_to[inactive_id] = associated_active_id

    def _check_for_activation(
        self, inactive_id: int, is_sim_active: np.ndarray
    ) -> bool:
        """
        Check if an inactive simulation needs to be activated.

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation.
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation.

        Return
        ------
        tag : bool
            True if the inactive simulation needs to be activated, False otherwise.
        """
        active_sim_ids = np.where(is_sim_active)[0]

        dists = self._similarity_dists[inactive_id, active_sim_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(
        self, active_id: int, is_sim_active: np.ndarray
    ) -> bool:
        """
        Check if an active simulation needs to be deactivated.

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation.
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
                if self._similarity_dists[active_id, active_id_2] < self._coarse_tol:
                    return True
        return False

    def _get_similarity_measure(
        self, similarity_measure: str
    ) -> Callable[[np.ndarray], np.ndarray]:
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
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

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
        # divide i,j by max(abs(data[i]),abs(data[j])) to get relative difference
        relative = np.nan_to_num(
            (
                pointwise_diff
                / np.maximum(
                    np.absolute(data[np.newaxis, :]), np.absolute(data[:, np.newaxis])
                )
            )
        )
        return np.linalg.norm(relative, ord=1, axis=-1)

    def _l2rel(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate L2 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

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
        # divide i,j by max(abs(data[i]),abs(data[j])) to get relative difference
        relative = np.nan_to_num(
            (
                pointwise_diff
                / np.maximum(
                    np.absolute(data[np.newaxis, :]), np.absolute(data[:, np.newaxis])
                )
            )
        )
        return np.linalg.norm(relative, ord=2, axis=-1)
