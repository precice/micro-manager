"""
Class LocalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a local way. If the Micro Manager is run in parallel, simulations on one rank are compared to
each other. A global comparison is not done.
"""
import numpy as np
from copy import deepcopy

from .adaptivity import AdaptivityCalculator
from ..micro_simulation import create_simulation_class


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(self, configurator, num_sims, participant, rank, comm_world) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        num_sims : int
            Number of micro simulations.
        participant : object of class Participant
            Object of the class Participant using which the preCICE API is called.
        rank : int
            Rank of the current MPI process.
        comm_world : MPI.COMM_WORLD
            Global communicator of MPI.
        """
        super().__init__(configurator, rank, num_sims)
        self._comm_world = comm_world

        if (
            self._adaptivity_output_type == "all"
            or self._adaptivity_output_type == "local"
        ):
            self._metrics_logger.log_info("n,n active,n inactive")

        self._precice_participant = participant

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        # This matrix is modified in place via the function update_similarity_dists
        self._similarity_dists = np.zeros((num_sims, num_sims))

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
        self._precice_participant.start_profiling_section(
            "local_adaptivity.compute_adaptivity"
        )

        for name in data_for_adaptivity.keys():
            if name not in self._adaptivity_data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._adaptivity_data_names.keys()
                    )
                )

        self._update_similarity_dists(dt, data_for_adaptivity)

        self._max_similarity_dist = np.amax(self._similarity_dists)

        self._update_active_sims()

        self._update_inactive_sims(micro_sims)

        self._associate_inactive_to_active()

        self._precice_participant.stop_last_profiling_section()

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

        Local metrics:
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

        if (
            self._adaptivity_output_type == "all"
            or self._adaptivity_output_type == "local"
        ):
            self._metrics_logger.log_info(
                "{},{},{}".format(
                    n,
                    active_sims_on_this_rank,
                    inactive_sims_on_this_rank,
                )
            )

        if (
            self._adaptivity_output_type == "global"
            or self._adaptivity_output_type == "all"
        ):
            active_sims_rankwise = self._comm_world.gather(
                active_sims_on_this_rank, root=0
            )
            inactive_sims_rankwise = self._comm_world.gather(
                inactive_sims_on_this_rank, root=0
            )

            if self._rank == 0:
                size = self._comm_world.Get_size()

                self._global_metrics_logger.log_info_rank_zero(
                    "{},{},{},{},{}".format(
                        n,
                        sum(active_sims_rankwise) / size,
                        sum(inactive_sims_rankwise) / size,
                        max(active_sims_rankwise),
                        max(inactive_sims_rankwise),
                    )
                )

    def _update_inactive_sims(self, micro_sims: list) -> None:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.

        Parameters
        ----------
        micro_sims : list
            List containing micro simulation objects.
        """
        self._ref_tol = self._refine_const * self._max_similarity_dist

        to_be_activated_ids = []
        # Update the set of inactive micro sims
        for i in range(self._is_sim_active.size):
            if not self._is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, self._is_sim_active):
                    self._is_sim_active[i] = True
                    if i not in self._just_deactivated:
                        to_be_activated_ids.append(i)

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        # Update the set of inactive micro sims
        for i in to_be_activated_ids:
            associated_active_id = self._sim_is_associated_to[i]
            micro_sims[i] = create_simulation_class(self._micro_problem)(i)
            micro_sims[i].set_state(micro_sims[associated_active_id].get_state())
            self._sim_is_associated_to[
                i
            ] = -2  # Active sim cannot have an associated sim

        # Delete the inactive micro simulations which have not been activated
        for i in range(self._is_sim_active.size):
            if not self._is_sim_active[i]:
                micro_sims[i] = 0
