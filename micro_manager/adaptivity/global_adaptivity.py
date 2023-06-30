"""
Functionality for adaptive control of micro simulations in a global way (all-to-all comparison of micro simulations)
"""
import numpy as np
import hashlib
from copy import deepcopy
from mpi4py import MPI
from typing import Dict
from .adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    """
    This class provides functionality to compute adaptivity globally, i.e. by comparing micro simulation from all processes.
    All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
    """
    def __init__(
            self,
            configurator,
            logger,
            is_sim_on_this_rank: list,
            rank_of_sim: np.ndarray,
            global_ids: list,
            comm,
            rank: int) -> None:
        super().__init__(configurator, logger)
        self._is_sim_on_this_rank = is_sim_on_this_rank
        self._rank_of_sim = rank_of_sim
        self._global_ids = global_ids
        self._comm = comm
        self._rank = rank

    def compute_adaptivity(
            self,
            dt: float,
            micro_sims: list,
            similarity_dists_nm1: np.ndarray,
            is_sim_active_nm1: np.ndarray,
            sim_is_associated_to_nm1: np.ndarray,
            data_for_adaptivity: dict) -> tuple:
        """
        Compute adaptivity globally based on similarity distances and micro simulation states

        Parameters
        ----------
        dt : float
            TODO
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        similarity_dists_nm1 : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active_nm1 : numpy array
            1D array having state (active or inactive) of each micro simulation on this rank
        sim_is_associated_to_nm1 : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        data_for_adaptivity : dict
            Dictionary with keys as names of data to be used in the similarity calculation, and values as the respective data for the micro simulations

        Results
        -------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation
        """
        # Gather adaptivity data from all ranks
        global_data_for_adaptivity = dict()
        for name in self._adaptivity_data_names.keys():
            data_as_list = self._comm.allgather(data_for_adaptivity[name])
            global_data_for_adaptivity[name] = np.concatenate((data_as_list[:]), axis=0)

        # Similarity distance matrix is calculated globally on every rank
        similarity_dists = self._get_similarity_dists(dt, similarity_dists_nm1, global_data_for_adaptivity)

        is_sim_active = self._update_active_sims(similarity_dists, is_sim_active_nm1)

        is_sim_active, sim_is_associated_to = self._update_inactive_sims(
            similarity_dists, is_sim_active_nm1, sim_is_associated_to_nm1, micro_sims)

        sim_is_associated_to = self._associate_inactive_to_active(
            similarity_dists, is_sim_active, sim_is_associated_to)

        self._logger.info(
            "{} active simulations, {} inactive simulations".format(
                np.count_nonzero(
                    is_sim_active[self._global_ids[0]:self._global_ids[-1] + 1]),
                np.count_nonzero(
                    is_sim_active[self._global_ids[0]:self._global_ids[-1] + 1] == False)))

        return similarity_dists, is_sim_active, sim_is_associated_to

    def communicate_micro_output(
            self,
            is_sim_active: np.ndarray,
            sim_is_associated_to: np.ndarray,
            micro_output: list) -> None:
        """
        Communicate micro output from active simulation to their associated inactive simulations. P2P communication is done.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation on this rank
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        micro_output : list
            List of dicts having individual output of each simulation. Only the active simulation outputs are entered.
        """
        inactive_local_ids = np.where(is_sim_active[self._global_ids[0]:self._global_ids[-1] + 1] == False)[0]

        local_sim_is_associated_to = sim_is_associated_to[self._global_ids[0]:self._global_ids[-1] + 1]

        # Keys are global IDs of active simulations associated to inactive
        # simulations on this rank. Values are global IDs of the inactive
        # simulations.
        active_to_inactive_map: Dict[int, list] = dict()

        for i in inactive_local_ids:
            assoc_active_id = local_sim_is_associated_to[i]
            # Gather global IDs of associated active simulations not on this rank for communication
            if not self._is_sim_on_this_rank[assoc_active_id]:
                if assoc_active_id in active_to_inactive_map:
                    active_to_inactive_map[assoc_active_id].append(i)
                else:
                    active_to_inactive_map[assoc_active_id] = [i]
            else:  # If associated active simulation is on this rank, copy the output directly
                micro_output[i] = deepcopy(micro_output[self._global_ids.index(assoc_active_id)])

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
            micro_sims: list) -> tuple:
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

        _is_sim_active = np.copy(is_sim_active)  # Input is_sim_active is not longer used after this point
        _sim_is_associated_to = np.copy(sim_is_associated_to)

        # Check inactive simulations for activation and collect IDs of those to be activated
        to_be_activated_ids = []  # Global IDs to be activated
        for i in range(_is_sim_active.size):
            if not _is_sim_active[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _is_sim_active):
                    _is_sim_active[i] = True
                    if self._is_sim_on_this_rank[i]:
                        to_be_activated_ids.append(i)

        print("is_sim_active: {}, to_be_activated: {}".format(_is_sim_active, to_be_activated_ids))

        local_sim_is_associated_to = _sim_is_associated_to[self._global_ids[0]:self._global_ids[-1] + 1]

        # Keys are global IDs of active sims not on this rank, values are lists of local and
        # global IDs of inactive sims associated to the active sims which are on this rank
        to_be_activated_map: Dict[int, list] = dict()

        for i in to_be_activated_ids:
            # Only handle activation of simulations on this rank -- LOCAL SCOPE HERE ON
            if self._is_sim_on_this_rank[i]:
                to_be_activated_local_id = self._global_ids.index(i)
                assoc_active_id = local_sim_is_associated_to[to_be_activated_local_id]

                if self._is_sim_on_this_rank[assoc_active_id]:  # Associated active simulation is on the same rank
                    assoc_active_local_id = self._global_ids.index(assoc_active_id)
                    micro_sims[to_be_activated_local_id].set_state(micro_sims[assoc_active_local_id].get_state())
                    _sim_is_associated_to[i] = -2  # Active sim cannot have an associated sim
                else:  # Associated active simulation is not on this rank
                    if assoc_active_id in to_be_activated_map:
                        to_be_activated_map[assoc_active_id].append(to_be_activated_local_id)
                    else:
                        to_be_activated_map[assoc_active_id] = [to_be_activated_local_id]

        sim_states_and_global_ids = []
        for sim in micro_sims:
            sim_states_and_global_ids.append((sim.get_state(), sim.get_global_id()))

        recv_reqs = self._p2p_comm(list(to_be_activated_map.keys()), sim_states_and_global_ids)

        # Use received micro sims to activate the required simulations
        for req in recv_reqs:
            state, global_id = req.wait()
            local_ids = to_be_activated_map[global_id]
            for local_id in local_ids:
                micro_sims[local_id].set_state(state)
                _sim_is_associated_to[self._global_ids[local_id]] = -2  # Active sim cannot have an associated sim

        return _is_sim_active, _sim_is_associated_to

    def _create_tag(self, sim_id, src_rank, dest_rank):
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode('utf-8'))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _p2p_comm(self, assoc_active_ids: list, data: list) -> list:
        """
        This function created sending and receiving maps for p2p communication.

        Parameters
        ----------
        assoc_active_ids : list
            Global IDs of active simulations which are not on this rank and are associated to the inactive simulations on this rank
        """
        send_map_local: Dict[int, int] = dict()  # keys are global IDs, values are rank to send to
        send_map: Dict[int, list] = dict()  # keys are global IDs of sims to send, values are ranks to send the sims to
        recv_map: Dict[int, int] = dict()  # keys are global IDs to receive, values are ranks to receive from

        for i in assoc_active_ids:
            # Add simulation and its rank to receive map
            recv_map[i] = self._rank_of_sim[i]
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
            req = self._comm.irecv(source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        return recv_reqs
