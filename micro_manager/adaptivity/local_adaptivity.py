"""
Class LocalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a local way. If the Micro Manager is run in parallel, simulations on one rank are compared to
each other. A global comparison is not done.
"""
import numpy as np
from copy import deepcopy

from .adaptivity import AdaptivityCalculator


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, logger) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        logger : object of logging
            Logger defined from the standard package logging
        """
        super().__init__(configurator, logger)

    def compute_adaptivity(
        self,
        dt,
        micro_sims,
        adaptivity_data_nm1: list,
        data_for_adaptivity: dict,
    ) -> tuple:
        """
        Compute adaptivity locally (within a rank).

        Parameters
        ----------
        dt : float
            Current time step
        micro_sims : list
            List containing simulation objects
        adaptivity_data_nm1 : list
            List of numpy arrays: similarity_dists (2D array having similarity distances between each micro simulation pair), is_sim_active (1D array having state (active or inactive) of each micro simulation), sim_is_associated_to (1D array with values of associated simulations of inactive simulations. Active simulations have None)
        data_for_adaptivity : dict
            A dictionary containing the names of the data to be used in adaptivity as keys and information on whether
            the data are scalar or vector as values.

        Returns
        -------
        list
            List of numpy arrays: similarity_dists (2D array having similarity distances between each micro simulation pair), is_sim_active (1D array having state (active or inactive) of each micro simulation), sim_is_associated_to (1D array with values of associated simulations of inactive simulations. Active simulations have None)
        """
        for name in data_for_adaptivity.keys():
            if name not in self._adaptivity_data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._adaptivity_data_names.keys()
                    )
                )

        similarity_dists = self._get_similarity_dists(
            dt, adaptivity_data_nm1[0], data_for_adaptivity
        )

        # Operation done globally if global adaptivity is chosen
        is_sim_active = self._update_active_sims(
            similarity_dists, adaptivity_data_nm1[1]
        )

        is_sim_active, sim_is_associated_to = self._update_inactive_sims(
            similarity_dists, is_sim_active, adaptivity_data_nm1[2], micro_sims
        )

        sim_is_associated_to = self._associate_inactive_to_active(
            similarity_dists, is_sim_active, sim_is_associated_to
        )

        self._logger.info(
            "{} active simulations, {} inactive simulations".format(
                np.count_nonzero(is_sim_active),
                np.count_nonzero(is_sim_active == False),
            )
        )

        return [similarity_dists, is_sim_active, sim_is_associated_to]

    def get_active_sim_ids(self, is_sim_active) -> np.ndarray:
        """
        Get the ids of active simulations.

        Parameters
        ----------
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        numpy array
            1D array of active simulation ids
        """
        return np.where(is_sim_active)[0]

    def get_inactive_sim_ids(self, is_sim_active: np.array) -> np.ndarray:
        """
        Get the ids of inactive simulations.

        Parameters
        ----------
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        numpy array
            1D array of inactive simulation ids
        """
        return np.where(is_sim_active == False)[0]

    def get_full_field_micro_output(
        self, adaptivity_data: list, micro_output: list
    ) -> list:
        """
        Get the full field micro output from active simulations to inactive simulations.

        Parameters
        ----------
        adaptivity_data : list
            List of numpy arrays:
                similarity_dists (2D array having similarity distances between each micro simulation pair)
                is_sim_active (1D array having state (active or inactive) of each micro simulation)
                sim_is_associated_to (1D array with values of associated simulations of inactive simulations. Active simulations have None)
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.

        Returns
        -------
        micro_output : list
            List of dicts having individual output of each simulation. Active and inactive simulation outputs are entered.
        """
        micro_sims_output = deepcopy(micro_output)

        sim_is_associated_to = adaptivity_data[2]

        inactive_sim_ids = self.get_inactive_sim_ids(adaptivity_data[1])

        for inactive_id in inactive_sim_ids:
            micro_sims_output[inactive_id] = deepcopy(
                micro_sims_output[sim_is_associated_to[inactive_id]]
            )

        return micro_sims_output

    def _update_inactive_sims(
        self,
        similarity_dists: np.ndarray,
        is_sim_active: np.ndarray,
        sim_is_associated_to: np.ndarray,
        micro_sims: list,
    ) -> tuple:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair.
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation.
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None.
        micro_sims : list
            List containing micro simulation objects.

        Returns
        -------
        _is_sim_active : numpy array
            Updated 1D array having state (active or inactive) of each micro simulation
        _sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _is_sim_active = np.copy(
            is_sim_active
        )  # Input is_sim_active is not longer used after this point
        _sim_is_associated_to = np.copy(sim_is_associated_to)

        # Update the set of inactive micro sims
        for i in range(_is_sim_active.size):
            if not _is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _is_sim_active):
                    associated_active_local_id = _sim_is_associated_to[i]
                    micro_sims[i].set_state(
                        micro_sims[associated_active_local_id].get_state()
                    )
                    _is_sim_active[i] = True
                    _sim_is_associated_to[
                        i
                    ] = -2  # Active sim cannot have an associated sim

        return _is_sim_active, _sim_is_associated_to
