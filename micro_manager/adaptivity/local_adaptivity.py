"""
Class LocalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a local way. If the Micro Manager is run in parallel, simulations on one rank are compared to
each other. A global comparison is not done.
"""
from typing import Optional, List, Dict, Any

import numpy as np
from copy import deepcopy

from .adaptivity import AdaptivityCalculator
from micro_manager.config import Config
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler, MPI, MPIHandlerRankLocal
from micro_manager.tools.profiling import Profiler
from micro_manager.model_manager import ModelManager
from micro_manager.interpolation import RBF_PU
from micro_manager.simulation_container import SimulationContainer


class LocalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
        self,
        config: Config,
        sim_container: SimulationContainer,
        profiler: Profiler,
        base_logger: Logger,
        mpi: MPIHandler,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        config : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        sim_container : SimulationContainer
            Simulation container object.
        profiler : Profiler
            Profiler object.
        base_logger : object of class Logger
            Logger object to log messages.
        mpi : MPIHandler
            MPI handler object.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object of class ModelManager
            Handles instantiation of micro simulation.
        """
        super().__init__(
            config,
            sim_container.local_num_sims,
            sim_container,
            profiler,
            micro_problem_cls,
            model_manager,
            base_logger,
            mpi,
        )
        # using local handler to only perform local interpolation
        self._interpolation = RBF_PU(
            base_logger,
            MPIHandlerRankLocal,
        )

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        # This matrix is modified in place via the function update_similarity_dists
        self._similarity_dists = np.zeros(
            (sim_container.local_num_sims, sim_container.local_num_sims)
        )
        self._max_similarity_dist_local: float = 0.0

    def compute(self, dt: float) -> None:
        """
        Compute adaptivity locally (within a rank).

        Parameters
        ----------
        dt : float
            Current time step
        """
        for name in self._data_for_adaptivity.keys():
            if name not in self._data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._data_names
                    )
                )

        self._update_similarity_dists(dt, self._data_for_adaptivity)
        self._max_similarity_dist_local = np.amax(self._similarity_dists)
        # Gather maximum similarity distance from every rank, and use the global maximum distance
        self._max_similarity_dist = self._mpi.comm.allreduce(
            self._max_similarity_dist_local, op=MPI.MAX
        )

        self._update_active_sims()

        self._update_inactive_sims()

        self._associate_inactive_to_active()

    def get_active_lids(self) -> List[int]:
        """
        Get the local ids of active simulations on this rank.

        Returns
        -------
        active_lids : List[int]
            List of active simulation LIDs
        """
        return np.where(self._is_sim_active)[0].tolist()

    def get_active_gids(self) -> List[int]:
        """
        Get the global ids of active simulations on this rank.

        For local adaptivity, global ids are same as local ids.

        Returns
        -------
        active_gids : List[int]
            List of active simulation GIDs
        """
        active_gids: List[int] = []
        for lid, gid in enumerate(self._sim_container.local_gids):
            if self._is_sim_active[lid]:
                active_gids.append(gid)

        return active_gids

    def get_inactive_lids(self) -> List[int]:
        """
        Get the local ids of inactive simulations on this rank.

        Returns
        -------
        inactive_lids : List[int]
            List of inactive simulation LIDs
        """
        return np.where(self._is_sim_active == False)[0].tolist()

    def get_inactive_gids(self) -> List[int]:
        """
        Get the global ids of inactive simulations on this rank.

        For local adaptivity, global ids are same as local ids.

        Returns
        -------
        inactive_gids : List[int]
            List of inactive simulation GIDs
        """
        inactive_gids: List[int] = []
        for lid, gid in enumerate(self._sim_container.local_gids):
            if not self._is_sim_active[lid]:
                inactive_gids.append(gid)

        return inactive_gids

    def get_full_field_micro_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_output: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Get the full field micro output from active simulations to inactive simulations.

        Parameters
        ----------
        micro_input : List[Dict[str, Any]]
            List of dicts containing the input data for each simulation.
        micro_output : List[Optional[Dict[str, Any]]]
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.

        Returns
        -------
        micro_output : List[Dict[str, Any]]
            List of dicts having individual output of each simulation. Active and inactive simulation outputs are entered.
        """
        micro_sims_output = deepcopy(micro_output)

        inactive_lids = self.get_inactive_lids()

        for inactive_lid in inactive_lids:
            micro_sims_output[inactive_lid] = deepcopy(
                micro_sims_output[self._sim_is_associated_to[inactive_lid]]
            )
        self._interpolate_output(micro_input, micro_sims_output)

        return micro_sims_output

    def log_metrics(self, n: int) -> None:
        """
        Log the following metrics:

        Local metrics:
        - Time window at which the metrics are logged
        - Number of active simulations
        - Number of inactive simulations

        Global metrics:
        - Time window at which the metrics are logged
        - Global number of active simulations
        - Global number of inactive simulations
        - Average number of active simulations
        - Average number of inactive simulations
        - Maximum number of active simulations
        - Maximum number of inactive simulations

        Parameters
        ----------
        n : int
            Time step count at which the metrics are logged
        """
        if n % self._output_n != 0:
            return
        if self._logger_local_metrics is None or self._logger_global_metrics is None:
            return

        active_sims_on_this_rank = 0
        inactive_sims_on_this_rank = 0
        for local_id in range(self._is_sim_active.size):
            if self._is_sim_active[local_id]:
                active_sims_on_this_rank += 1
            else:
                inactive_sims_on_this_rank += 1

        if self._logger_local_metrics is not None:
            self._logger_local_metrics.log_info(
                "{}|{}|{}".format(
                    n,
                    active_sims_on_this_rank,
                    inactive_sims_on_this_rank,
                )
            )

        if self._logger_global_metrics is not None:
            active_sims_rankwise = self._mpi.comm.gather(
                active_sims_on_this_rank, root=0
            )
            inactive_sims_rankwise = self._mpi.comm.gather(
                inactive_sims_on_this_rank, root=0
            )

            if self._mpi.rank == 0:
                size = self._mpi.size

                self._logger_global_metrics.log_info(
                    "{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
                        n,
                        sum(active_sims_rankwise),
                        sum(inactive_sims_rankwise),
                        sum(active_sims_rankwise) / size,
                        sum(inactive_sims_rankwise) / size,
                        max(active_sims_rankwise),
                        active_sims_rankwise.index(max(active_sims_rankwise)),
                        max(inactive_sims_rankwise),
                        inactive_sims_rankwise.index(max(inactive_sims_rankwise)),
                    )
                )

    def _update_active_sims(self) -> None:
        """
        Update set of active micro simulations. Active micro simulations are compared to each other
        and if found similar, one of them is deactivated.
        """
        self._coarse_tol = (
            self._coarse_const * self._refine_const * self._max_similarity_dist
        )

        active_lids = self.get_active_lids()
        active_lids_to_check = active_lids.copy()

        # Update the set of active micro sims
        for lid in active_lids:
            if self._check_for_deactivation(lid, active_lids_to_check):
                self._is_sim_active[lid] = False
                self._just_deactivated.append(lid)
                # Remove deactivated gid from further checks
                active_lids_to_check.remove(lid)

    def _update_inactive_sims(self) -> None:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones
        and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.
        """
        self._ref_tol = self._refine_const * self._max_similarity_dist

        active_lids = self.get_active_lids()
        inactive_lids = self.get_inactive_lids()

        to_be_activated_lids: List[int] = []
        # Update the set of inactive micro sims
        for lid in inactive_lids:
            if self._check_for_activation(lid, active_lids):
                self._is_sim_active[lid] = True
                if lid not in self._just_deactivated:
                    to_be_activated_lids.append(lid)
                    # Add the newly activated lid to active_lids for further checks
                    active_lids = np.append(active_lids, lid)

        self._just_deactivated.clear()  # Clear the list of sims deactivated in this step

        # Update the set of inactive micro sims
        for lid in to_be_activated_lids:
            associated_active_id = self._sim_is_associated_to[lid]
            self._sim_container[lid] = self._model_manager.get_instance(
                lid, self._micro_problem_cls
            )
            state = self._sim_container.get_sim_state(associated_active_id)
            self._sim_container.set_sim_state(lid, state)
            # Active sim cannot have an associated sim
            del self._sim_is_associated_to[lid]

        # Delete the inactive micro simulations which have not been activated
        for lid in range(self._is_sim_active.size):
            if not self._is_sim_active[lid] and self._sim_container[lid] is not None:
                # Release resources now, especially for remote simulation instance.
                # If left to garbage collector this might lead to a race condition.
                # Releasing with call to sim.destroy(), afterwards reference in sim
                # sim list can be removed.
                self._sim_container[lid].destroy()
                self._sim_container[lid] = None
