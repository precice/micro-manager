"""
Class LocalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a local way. If the Micro Manager is run in parallel, simulations on one rank are compared to
each other. A global comparison is not done.
"""
import numpy as np
import importlib
from copy import deepcopy

from .adaptivity import AdaptivityCalculator
from ..micro_simulation import create_simulation_class


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, rank, comm, num_sims) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        rank : int
            Rank of the current MPI process.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        num_sims : int
            Number of micro simulations.
        """
        super().__init__(configurator, rank)
        self._comm = comm

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        self._similarity_dists = np.zeros((num_sims, num_sims))

        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        self._is_sim_active = np.array([True] * num_sims)

        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        self._sim_is_associated_to = np.full((num_sims), -2, dtype=np.intc)

        # Copies of variables for checkpointing
        self._similarity_dists_cp = None
        self._is_sim_active_cp = None
        self._sim_is_associated_to_cp = None

        self._updating_inactive_sims = self._get_update_inactive_sims_variant()

        self._metrics_logger.log_info("n,n active,n inactive")

    def compute_adaptivity(
        self,
        dt,
        micro_sims,
        data_for_adaptivity: dict,
    ) -> None:
        """
        Compute adaptivity locally (within a rank).

        Parameters
        ----------
        dt : float
            Current time step
        micro_sims : list
            List containing simulation objects
        data_for_adaptivity : dict
            A dictionary containing the names of the data to be used in adaptivity as keys and information on whether
            the data are scalar or vector as values.

        """
        for name in data_for_adaptivity.keys():
            if name not in self._adaptivity_data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._adaptivity_data_names.keys()
                    )
                )

        similarity_dists = self._get_similarity_dists(
            dt, self._similarity_dists, data_for_adaptivity
        )

        is_sim_active = self._update_active_sims(similarity_dists, self._is_sim_active)

        is_sim_active, sim_is_associated_to = self._updating_inactive_sims(
            similarity_dists, is_sim_active, self._sim_is_associated_to, micro_sims
        )

        sim_is_associated_to = self._associate_inactive_to_active(
            similarity_dists, is_sim_active, sim_is_associated_to
        )

        # Update member variables
        self._similarity_dists = similarity_dists
        self._is_sim_active = is_sim_active
        self._sim_is_associated_to = sim_is_associated_to

    def get_active_sim_ids(self) -> np.ndarray:
        """
        Get the ids of active simulations.

        Returns
        -------
        numpy array
            1D array of active simulation ids
        """
        return np.where(self._is_sim_active)[0]

    def get_inactive_sim_ids(self) -> np.ndarray:
        """
        Get the ids of inactive simulations.

        Returns
        -------
        numpy array
            1D array of inactive simulation ids
        """
        return np.where(self._is_sim_active == False)[0]

    def get_full_field_micro_output(self, micro_output: list) -> list:
        """
        Get the full field micro output from active simulations to inactive simulations.

        Parameters
        ----------
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.

        Returns
        -------
        micro_output : list
            List of dicts having individual output of each simulation. Active and inactive simulation outputs are entered.
        """
        micro_sims_output = deepcopy(micro_output)

        inactive_sim_ids = self.get_inactive_sim_ids()

        for inactive_id in inactive_sim_ids:
            micro_sims_output[inactive_id] = deepcopy(
                micro_sims_output[self._sim_is_associated_to[inactive_id]]
            )

        return micro_sims_output

    def log_metrics(self, n: int) -> None:
        """
        Log the following metrics:

        Metrics on this rank:
        - Time window at which the metrics are logged
        - Number of active simulations
        - Number of inactive simulations

        Global metrics:
        - Average number of active simulations per rank
        - Average number of inactive simulations per rank
        - Maximum number of active simulations on a rank
        - Maximum number of inactive simulations on a rank

        Parameters
        ----------
        n : int
            Current time step
        """
        active_sims_on_this_rank = 0
        inactive_sims_on_this_rank = 0
        for local_id in range(self._is_sim_active.size):
            if self._is_sim_active[local_id]:
                active_sims_on_this_rank += 1
            else:
                inactive_sims_on_this_rank += 1

        self._metrics_logger.log_info(
            "{},{},{}".format(
                n,
                active_sims_on_this_rank,
                inactive_sims_on_this_rank,
            )
        )

        active_sims_rankwise = self._comm.gather(active_sims_on_this_rank, root=0)
        inactive_sims_rankwise = self._comm.gather(inactive_sims_on_this_rank, root=0)

        if self._rank == 0:
            size = self._comm.Get_size()

            self._global_metrics_logger.log_info_rank_zero(
                "{},{},{},{},{}".format(
                    n,
                    sum(active_sims_rankwise) / size,
                    sum(inactive_sims_rankwise) / size,
                    max(active_sims_rankwise),
                    max(inactive_sims_rankwise),
                )
            )

    def write_checkpoint(self) -> None:
        """
        Write checkpoint.
        """
        self._similarity_dists_cp = np.copy(self._similarity_dists)
        self._is_sim_active_cp = np.copy(self._is_sim_active)
        self._sim_is_associated_to_cp = np.copy(self._sim_is_associated_to)

    def read_checkpoint(self) -> None:
        """
        Read checkpoint.
        """
        self._similarity_dists = np.copy(self._similarity_dists_cp)
        self._is_sim_active = np.copy(self._is_sim_active_cp)
        self._sim_is_associated_to = np.copy(self._sim_is_associated_to_cp)

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
                    _is_sim_active[i] = True
                    if i not in self._just_deactivated:
                        associated_active_local_id = _sim_is_associated_to[i]
                        micro_sims[i].set_state(
                            micro_sims[associated_active_local_id].get_state()
                        )
                    _sim_is_associated_to[
                        i
                    ] = -2  # Active sim cannot have an associated sim

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        return _is_sim_active, _sim_is_associated_to

    def _update_inactive_sims_lazy_init(
        self,
        similarity_dists: np.ndarray,
        is_sim_active: np.ndarray,
        sim_is_associated_to: np.ndarray,
        micro_sims: list,
    ) -> tuple:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.

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
                    _is_sim_active[i] = True
                    if i not in self._just_deactivated:
                        associated_active_local_id = _sim_is_associated_to[i]
                        if (
                            micro_sims[i] == 0
                        ):  # 0 indicates that the micro simulation object has not been created yet
                            micro_problem = getattr(
                                importlib.import_module(
                                    self._micro_file_name, "MicroSimulation"
                                ),
                                "MicroSimulation",
                            )
                            micro_sims[i] = create_simulation_class(micro_problem)(i)
                        micro_sims[i].set_state(
                            micro_sims[associated_active_local_id].get_state()
                        )
                    _sim_is_associated_to[
                        i
                    ] = -2  # Active sim cannot have an associated sim

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        return _is_sim_active, _sim_is_associated_to

    def _get_update_inactive_sims_variant(self):
        """
        Get the variant of the function _update_inactive_sims.

        Returns
        -------
        function
            Function which updates the set of inactive micro simulations.
        """
        if self._lazy_init:
            return self._update_inactive_sims_lazy_init
        else:
            return self._update_inactive_sims
