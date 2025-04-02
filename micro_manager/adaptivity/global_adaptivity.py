"""
Class GlobalAdaptivityCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import hashlib
import importlib
from copy import deepcopy
from typing import Dict

import numpy as np
from mpi4py import MPI

from .adaptivity import AdaptivityCalculator
from ..micro_simulation import create_simulation_class


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
        self,
        configurator,
        global_number_of_sims: int,
        global_ids: list,
        rank: int,
        comm,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        global_number_of_sims : int
            Total number of simulations in the macro-micro coupled problem.
        global_ids : list
            List of global IDs of simulations living on this rank.
        rank : int
            MPI rank.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        """
        super().__init__(configurator, rank)
        self._global_number_of_sims = global_number_of_sims
        self._global_ids = global_ids
        self._comm = comm
        self._rank = rank

        # similarity_dists: 2D array having similarity distances between each micro simulation pair
        self._similarity_dists = np.zeros(
            (global_number_of_sims, global_number_of_sims)
        )

        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        self._is_sim_active = np.array([True] * global_number_of_sims)

        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        self._sim_is_associated_to = np.full((global_number_of_sims), -2, dtype=np.intc)

        local_number_of_sims = len(global_ids)

        # Create a map of micro simulation global IDs and the ranks on which they are
        micro_sims_on_this_rank = np.zeros(local_number_of_sims, dtype=np.intc)
        for i in range(local_number_of_sims):
            micro_sims_on_this_rank[i] = self._rank

        rank_of_sim = self._get_ranks_of_sims()

        self._is_sim_on_this_rank = [False] * global_number_of_sims  # DECLARATION
        for i in range(global_number_of_sims):
            if rank_of_sim[i] == self._rank:
                self._is_sim_on_this_rank[i] = True

        # Copies of variables for checkpointing
        self._similarity_dists_cp = None
        self._is_sim_active_cp = None
        self._sim_is_associated_to_cp = None

        self._updating_inactive_sims = self._get_update_inactive_sims_variant()

    def compute_adaptivity(
        self,
        dt: float,
        micro_sims: list,
        data_for_adaptivity: dict,
    ) -> None:
        """
        Compute adaptivity globally based on similarity distances and micro simulation states

        Parameters
        ----------
        dt : float
            Current time step of the macro-micro coupled problem
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        data_for_adaptivity : dict
            Dictionary with keys as names of data to be used in the similarity calculation, and values as the respective data for the micro simulations
        """
        for name in data_for_adaptivity.keys():
            if name not in self._adaptivity_data_names:
                raise ValueError(
                    "Data for adaptivity must be one of the following: {}".format(
                        self._adaptivity_data_names.keys()
                    )
                )

        # Gather adaptivity data from all ranks
        global_data_for_adaptivity = dict()
        for name in data_for_adaptivity.keys():
            data_as_list = self._comm.allgather(data_for_adaptivity[name])
            global_data_for_adaptivity[name] = np.concatenate((data_as_list[:]), axis=0)

        similarity_dists = self._get_similarity_dists(
            dt, self._similarity_dists, global_data_for_adaptivity
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
        return np.where(
            self._is_sim_active[self._global_ids[0] : self._global_ids[-1] + 1]
        )[0]

    def get_inactive_sim_ids(self) -> np.ndarray:
        """
        Get the ids of inactive simulations.

        Returns
        -------
        numpy array
            1D array of inactive simulation ids
        """
        return np.where(
            self._is_sim_active[self._global_ids[0] : self._global_ids[-1] + 1] == False
        )[0]

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
        self._communicate_micro_output(
            self._is_sim_active, self._sim_is_associated_to, micro_sims_output
        )

        return micro_sims_output

    def log_metrics(self, n: int) -> None:
        """
        Log metrics for global adaptivity.

        Parameters
        ----------
        n : int
            Time step count at which the metrics are logged
        """
        global_active_sims = np.count_nonzero(self._is_sim_active)
        global_inactive_sims = np.count_nonzero(self._is_sim_active == False)

        self._metrics_logger.log_info_rank_zero(
            "{},{},{},{},{}".format(
                n,
                np.mean(global_active_sims),
                np.mean(global_inactive_sims),
                np.max(global_active_sims),
                np.max(global_inactive_sims),
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

    def _communicate_micro_output(
        self,
        is_sim_active: np.ndarray,
        sim_is_associated_to: np.ndarray,
        micro_output: list,
    ) -> None:
        """
        Communicate micro output from active simulation to their associated inactive simulations.
        Process to process (p2p) communication is done.

        Parameters
        ----------
        is_sim_active : np.ndarray
            1D array having state (active or inactive) of each micro simulation
        sim_is_associated_to : np.ndarray
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.
        """

        inactive_local_ids = np.where(
            is_sim_active[self._global_ids[0] : self._global_ids[-1] + 1] == False
        )[0]

        local_sim_is_associated_to = sim_is_associated_to[
            self._global_ids[0] : self._global_ids[-1] + 1
        ]

        # Keys are global IDs of active simulations associated to inactive
        # simulations on this rank. Values are global IDs of the inactive
        # simulations.
        active_to_inactive_map: Dict[int, list] = dict()

        for i in inactive_local_ids:
            assoc_active_id = local_sim_is_associated_to[i]
            # Gather global IDs of associated active simulations not on this rank
            if not self._is_sim_on_this_rank[assoc_active_id]:
                if assoc_active_id in active_to_inactive_map:
                    active_to_inactive_map[assoc_active_id].append(i)
                else:
                    active_to_inactive_map[assoc_active_id] = [i]
            else:  # If associated active simulation is on this rank, copy the output directly
                micro_output[i] = deepcopy(
                    micro_output[self._global_ids.index(assoc_active_id)]
                )

        assoc_active_ids = list(active_to_inactive_map.keys())

        recv_reqs = self._p2p_comm(assoc_active_ids, micro_output)

        # Add received output of active sims to inactive sims on this rank
        for count, req in enumerate(recv_reqs):
            output = req.wait()
            for local_id in active_to_inactive_map[assoc_active_ids[count]]:
                micro_output[local_id] = deepcopy(output)

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
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations

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
        _sim_is_associated_to_updated = np.copy(sim_is_associated_to)

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(_is_sim_active.size):
            if not _is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _is_sim_active):
                    _is_sim_active[i] = True
                    _sim_is_associated_to_updated[
                        i
                    ] = -2  # Active sim cannot have an associated sim
                    if self._is_sim_on_this_rank[i]:
                        to_be_activated_ids.append(i)

        local_sim_is_associated_to = _sim_is_associated_to[
            self._global_ids[0] : self._global_ids[-1] + 1
        ]

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        for i in to_be_activated_ids:
            # Only handle activation of simulations on this rank -- LOCAL SCOPE HERE ON
            if self._is_sim_on_this_rank[i]:
                to_be_activated_local_id = self._global_ids.index(i)
                assoc_active_id = local_sim_is_associated_to[to_be_activated_local_id]

                if self._is_sim_on_this_rank[
                    assoc_active_id
                ]:  # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id].set_state(
                        micro_sims[assoc_active_local_id].get_state()
                    )
                else:  # Associated active simulation is not on this rank
                    if assoc_active_id in to_be_activated_map:
                        to_be_activated_map[assoc_active_id].append(
                            to_be_activated_local_id
                        )
                    else:
                        to_be_activated_map[assoc_active_id] = [
                            to_be_activated_local_id
                        ]

        sim_states_and_global_ids = []
        for sim in micro_sims:
            sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(
            list(to_be_activated_map.keys()), sim_states_and_global_ids
        )

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, global_id = req.wait()
            local_ids = to_be_activated_map[global_id]
            for local_id in local_ids:
                micro_sims[local_id].set_state(state)

        return _is_sim_active, _sim_is_associated_to_updated

    def _update_inactive_sims_lazy_init(
        self,
        similarity_dists: np.ndarray,
        is_sim_active: np.ndarray,
        sim_is_associated_to: np.ndarray,
        micro_sims: list,
    ) -> tuple:
        """
        Update set of inactive micro simulations. Each inactive micro simulation is compared to all active ones and if it is not similar to any of them, it is activated.

        If a micro simulation which has been inactive since the start of the simulation is activated for the
        first time, the simulation object is created and initialized.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations

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
        _sim_is_associated_to_updated = np.copy(sim_is_associated_to)

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(_is_sim_active.size):
            if not _is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _is_sim_active):
                    _is_sim_active[i] = True
                    _sim_is_associated_to_updated[
                        i
                    ] = -2  # Active sim cannot have an associated sim
                    if self._is_sim_on_this_rank[i]:
                        to_be_activated_ids.append(i)

        local_sim_is_associated_to = _sim_is_associated_to[
            self._global_ids[0] : self._global_ids[-1] + 1
        ]

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        for i in to_be_activated_ids:
            # Only handle activation of simulations on this rank -- LOCAL SCOPE HERE ON
            if self._is_sim_on_this_rank[i]:
                to_be_activated_local_id = self._global_ids.index(i)
                if (
                    micro_sims[to_be_activated_local_id] == 0
                ):  # 0 indicates that the micro simulation object has not been created yet
                    micro_problem = getattr(
                        importlib.import_module(
                            self._micro_file_name, "MicroSimulation"
                        ),
                        "MicroSimulation",
                    )
                    micro_sims[to_be_activated_local_id] = create_simulation_class(
                        micro_problem
                    )(i)
                assoc_active_id = local_sim_is_associated_to[to_be_activated_local_id]

                if self._is_sim_on_this_rank[
                    assoc_active_id
                ]:  # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id].set_state(
                        micro_sims[assoc_active_local_id].get_state()
                    )
                else:  # Associated active simulation is not on this rank
                    if assoc_active_id in to_be_activated_map:
                        to_be_activated_map[assoc_active_id].append(
                            to_be_activated_local_id
                        )
                    else:
                        to_be_activated_map[assoc_active_id] = [
                            to_be_activated_local_id
                        ]

        # TODO: could be moved to before the lazy initialization above
        sim_states_and_global_ids = []
        for local_id, sim in enumerate(micro_sims):
            if sim == 0:
                sim_states_and_global_ids.append((None, self._global_ids[local_id]))
            else:
                sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(
            list(to_be_activated_map.keys()), sim_states_and_global_ids
        )

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, global_id = req.wait()
            local_ids = to_be_activated_map[global_id]
            for local_id in local_ids:
                micro_sims[local_id].set_state(state)

        return _is_sim_active, _sim_is_associated_to_updated

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

    def _create_tag(self, sim_id: int, src_rank: int, dest_rank: int) -> int:
        """
        For a given simulations ID, source rank, and destination rank, a unique tag is created.

        Parameters
        ----------
        sim_id : int
            Global ID of a simulation.
        src_rank : int
            Rank on which the simulation lives
        dest_rank : int
            Rank to which data of a simulation is to be sent to.

        Returns
        -------
        tag : int
            Unique tag.
        """
        send_hashtag = hashlib.sha256()
        send_hashtag.update(
            (str(src_rank) + str(sim_id) + str(dest_rank)).encode("utf-8")
        )
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _p2p_comm(self, assoc_active_ids: list, data: list) -> list:
        """
        Handle process to process communication for a given set of associated active IDs and data.

        Parameters
        ----------
        assoc_active_ids : list
            Global IDs of active simulations which are not on this rank and are associated to
            the inactive simulations on this rank.
        data : list
            Complete data from which parts are to be sent and received.

        Returns
        -------
        recv_reqs : list
            List of MPI requests of receive operations.
        """
        rank_of_sim = self._get_ranks_of_sims()

        send_map_local: Dict[
            int, int
        ] = dict()  # keys are global IDs, values are rank to send to
        send_map: Dict[
            int, list
        ] = (
            dict()
        )  # keys are global IDs of sims to send, values are ranks to send the sims to
        recv_map: Dict[
            int, int
        ] = dict()  # keys are global IDs to receive, values are ranks to receive from

        for i in assoc_active_ids:
            # Add simulation and its rank to receive map
            recv_map[i] = rank_of_sim[i]
            # Add simulation and this rank to local sending map
            send_map_local[i] = self._rank

        # Gather information about which sims to send where, from the sending perspective
        send_map_list = self._comm.allgather(send_map_local)

        for d in send_map_list:
            for i, rank in d.items():
                if self._is_sim_on_this_rank[i]:
                    if i in send_map:
                        send_map[i].append(rank)
                    else:
                        send_map[i] = [rank]

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_ranks in send_map.items():
            local_id = self._global_ids.index(global_id)
            for send_rank in send_ranks:
                tag = self._create_tag(global_id, self._rank, send_rank)
                req = self._comm.isend(data[local_id], dest=send_rank, tag=tag)
                send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in recv_map.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            bufsize = (
                1 << 30
            )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
            req = self._comm.irecv(bufsize, source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        return recv_reqs

    def _get_ranks_of_sims(self) -> np.ndarray:
        """
        Get the ranks of all simulations.
        Returns
        -------
        ranks_of_sims : np.ndarray
            Array of ranks on which simulations exist.
        """
        local_gids_to_rank = dict()
        for gid in self._global_ids:
            local_gids_to_rank[gid] = self._rank

        ranks_maps_as_list = self._comm.allgather(local_gids_to_rank)

        ranks_of_sims = np.zeros(self._global_number_of_sims, dtype=np.intc)
        for ranks_map in ranks_maps_as_list:
            for gid, rank in ranks_map.items():
                ranks_of_sims[gid] = rank

        return ranks_of_sims
