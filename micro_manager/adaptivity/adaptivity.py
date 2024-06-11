"""
Functionality for adaptive initialization and control of micro simulations
"""
import sys
import os
import numpy as np
from math import exp
from typing import Callable
import re
import xml.etree.ElementTree as ET
from warnings import warn

import numpy as np


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
        self._config_file_name = configurator.get_config_file_name()

        self._logger = logger

        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._similarity_measure = self._get_similarity_measure(
            configurator.get_adaptivity_similarity_measure()
        )

    def _get_similarity_dists(
        self, dt: float, similarity_dists: np.ndarray, data: dict
    ) -> np.ndarray:
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
        for name in data.keys():
            data_vals = data[name]
            if data_vals.ndim == 1:
                # If the adaptivity-data is a scalar for each simulation,
                # expand the dimension to make it a 2D array to unify the calculation.
                # The axis is later reduced with a norm.
                data_vals = np.expand_dims(data_vals, axis=1)

            data_diff += self._similarity_measure(data_vals)

        return exp(-self._hist_param * dt) * _similarity_dists + dt * data_diff

    def _get_distance_weight_power(self) -> float:
        """
        Calculate distance weights from convergence status: if the residual is still reducing fast, it tends to have less miceo simulations, otherwise more micro simulations.
        If no convergence status is found, the default value 0.5 is returned.

        Returns
        -------
        distance_weight : float
        """
        # Read the XML file as text
        with open(self._config_file_name, "r") as xml_file:
            xml_data = xml_file.read()

        unique_names = [
            "absolute-convergence-measure",
            "relative-convergence-measure",
            "residual-relative-convergence-measure",
        ]

        # Initialize lists to store the found attributes
        data_values = []
        limit_values = []

        for unique_name in unique_names:
            patteren = f'<{unique_name} limit="([^"]+)" data="([^"]+)" mesh="([^"]+)"'
            matches = re.finditer(patteren, xml_data)
            for match in matches:
                data_values.append(match.group(2))
                limit_values.append(match.group(1))

        # read convergence value from precice-Mysolver-convergence.log file
        # Initialize lists to store the extracted values
        last_convergence_values = []
        last_sec_convergence_values = []
        power_value = 0.5

        file_path = None
        file_name_suffix = "-convergence.log"

        # Search for the file in the current directory and its subdirectories
        for root, _, files in os.walk(os.getcwd()):
            for file_name in files:
                if file_name.endswith(file_name_suffix):
                    file_path = os.path.join(root, file_name)
                    break

        if file_path:
            with open(file_path, "r") as file:
                lines = file.readlines()
            if len(lines) < 3:
                print("File does not contain enough lines.")
            else:
                # Read the header line and last two lines of the file
                header_line = lines[0].strip().split()
                last_line = lines[-1].strip().split()
                last_second_line = lines[-2].strip().split()
                # Extract the convergence values from the last line and the last second line, store in two arraries
                for i in range(0, len(header_line)):
                    last_convergence_values.append(float(last_line[i]))
                    last_sec_convergence_values.append(float(last_second_line[i]))
                for data in data_values:
                    for element in header_line:
                        if data in element:
                            index_convergence = header_line.index(element)
                            index_config = data_values.index(data)
                            last_convergence_values[index_convergence] = max(
                                last_convergence_values[index_convergence],
                                float(limit_values[index_config]),
                            )  # if current residual is smaller than the limit, set this value to limit to avoid that one over-converged value balances the other not converged values in the following multiplication
                            last_sec_convergence_values[index_convergence] = max(
                                last_sec_convergence_values[index_convergence],
                                float(limit_values[index_config]),
                            )

                # Calculate the power value based on the convergence values
                if last_convergence_values[0] != last_sec_convergence_values[0]:
                    power_value = 0.5  # first iteration in each time step
                else:
                    rel_diff = (
                        np.log10(np.prod(np.array(last_convergence_values[2:])))
                        - np.log10(np.prod(np.array(last_sec_convergence_values[2:])))
                    ) / np.log10(
                        np.prod(np.array(last_convergence_values[2:]))
                    )  # the rel_diff is (lg(convergence values at the last iteration)-lg(convergence values at the second last iteration))/lg(convergence values at the last iteration)
                    # TODO: test insensely
                    power_value = min(
                        max(abs(np.log10(abs(rel_diff))), 0.5), 1.0
                    )  # limit the power value in between 0.5 and 1.0 (motivated by experiments)

        else:
            print(
                "Convergence file is not found in the current directory or its subdirectories."
            )

        self._logger.info("power value: {} ".format(power_value))

        return power_value

    def _get_deactivate_distance(
        self, similarity_dists: np.ndarray, is_sim_active: np.ndarray
    ) -> float:
        """
        Get maximum gap between ascending distances between all active micro simulations.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        max_gap : float
            Maximum gap between active micro simulations
        """
        _similarity_dists = np.copy(similarity_dists)
        active_ids = np.where(is_sim_active)[0]
        max_gap = 0.0
        similarity_dists_active = _similarity_dists[active_ids, :][:, active_ids]
        deactivate_distance = 0.0
        # power=self._get_distance_weight_power()
        power = 0.5

        # sort the distances between active sims in ascending order
        similarity_dists_active = np.sort(similarity_dists_active, axis=None)
        similarity_dists_active = np.append(
            similarity_dists_active, np.amax(_similarity_dists)
        )
        # print(f"similarity_dists_active: {similarity_dists_active}") # observe the distance distribution
        # get the maximum gap between ascending distances
        for i in range(1, similarity_dists_active.size):
            measure = (
                similarity_dists_active[i] - similarity_dists_active[i - 1]
            ) * np.power(similarity_dists_active[i], power)
            if measure > max_gap:
                max_gap = measure
                deactivate_distance = similarity_dists_active[
                    i
                ]  # get the distance value at the right end of the gap
        # print(f"deactivate_distance: {deactivate_distance}")
        return deactivate_distance

    def _get_activate_distance(
        self, similarity_dists: np.ndarray, is_sim_active: np.ndarray
    ) -> float:
        """
        Get maximum gap between ascending distances from inactive to active micro simulations.

        Parameters
        ----------
        similarity_dists : numpy array
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        max_gap : float
            Maximum gap between active micro simulations
        """
        _similarity_dists = np.copy(similarity_dists)
        inactive_ids = np.where(is_sim_active == False)[0]
        active_ids = np.where(is_sim_active)[0]
        max_gap = 0.0
        activate_distance = 0.0
        similarity_dists_inactive_active = np.zeros(inactive_ids.size)
        # power=self._get_distance_weight_power()
        power = 0.5

        # get minimum distance between active and inactive sims for each inactive sim
        for i in range(inactive_ids.size):
            similarity_dists_inactive_active[i] = np.amin(
                _similarity_dists[inactive_ids[i], active_ids]
            )

        # sort the distances between active sims in ascending order
        similarity_dists_inactive_active = np.sort(
            similarity_dists_inactive_active, axis=None
        )
        similarity_dists_inactive_active = np.append(
            similarity_dists_inactive_active, np.amax(_similarity_dists)
        )
        # print(f"similarity_dists_inactive: {similarity_dists_inactive_active}")

        # get the maximum gap between ascending distances
        for i in range(1, similarity_dists_inactive_active.size):
            measure = (
                similarity_dists_inactive_active[i]
                - similarity_dists_inactive_active[i - 1]
            ) * np.power(similarity_dists_inactive_active[i], power)
            if measure > max_gap:
                max_gap = measure
                activate_distance = similarity_dists_inactive_active[i - 1]
        # print(f"activate_distance: {activate_distance}")
        return activate_distance

    def _update_active_sims(
        self, similarity_dists: np.ndarray, is_sim_active: np.ndarray
    ) -> np.ndarray:
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
        max_similarity_dist = np.amax(similarity_dists)

        if max_similarity_dist == 0.0:
            warn(
                "All similarity distances are zero, probably because all the data for adaptivity is the same. Coarsening tolerance will be manually set to minimum float number."
            )
            self._coarse_tol = sys.float_info.min
        else:
            self._coarse_tol = (
                self._coarse_const * self._refine_const * max_similarity_dist
            )

        _is_sim_active = np.copy(
            is_sim_active
        )  # Input is_sim_active is not longer used after this point

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
        sim_is_associated_to: np.ndarray,
    ) -> np.ndarray:
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
        self, inactive_id: int, similarity_dists: np.ndarray, is_sim_active: np.ndarray
    ) -> bool:
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
        self, active_id: int, similarity_dists: np.ndarray, is_sim_active: np.ndarray
    ) -> bool:
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
        relative = np.nan_to_num(
            (pointwise_diff / np.maximum(data[np.newaxis, :], data[:, np.newaxis]))
        )
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
        relative = np.nan_to_num(
            (pointwise_diff / np.maximum(data[np.newaxis, :], data[:, np.newaxis]))
        )
        return np.linalg.norm(relative, ord=2, axis=-1)
