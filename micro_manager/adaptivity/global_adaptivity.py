"""
Functionality for adaptive control of micro simulations in a global way (all-to-all comparison of micro simulations)
"""
import sys
import numpy as np
import hashlib
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    def __init__(
            self,
            configurator,
            global_ids: list,
            number_of_sims: int,
            micro_sim_is_on_rank: list,
            comm,
            rank: int) -> None:
        super().__init__(configurator, global_ids, number_of_sims)
        self._micro_sim_is_on_rank = micro_sim_is_on_rank
        self._comm = comm
        self._rank = rank

    def _create_tag(self, sim_id, src_rank, dest_rank):
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode('utf-8'))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def update_inactive_micro_sims(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> np.ndarray:
        """
        """
        self._ref_tol = self._refine_const * np.amax(similarity_dists)

        _micro_sim_states = np.copy(micro_sim_states)  # Input micro_sim_states is not longer used after this point

        # Update the set of inactive micro sims
        send_sims_to_ranks_local = dict()  # keys are global IDs, values are rank to send to
        recv_sims_from_ranks = dict()  # keys are global IDs, values are ranks to receive from
        for i in range(self._number_of_sims):
            if not _micro_sim_states[i]:  # if id is inactive
                if self._check_for_activation(i, similarity_dists, _micro_sim_states):
                    _micro_sim_states[i] = 1

                    if i in self._global_ids_of_local_sims:  # Local scope from here on
                        associated_active_global_id = micro_sims[i].get_associated_active_global_id()

                        # Effectively kill the micro sim object associated to the inactive ID
                        micro_sims[i] = None

                        # Get the rank on which the associated active simulation is
                        recv_rank = self._micro_sim_is_on_rank[associated_active_global_id]

                        # If simulation is to be copied from this rank, just do it directly
                        if recv_rank == self._rank:
                            local_id = self._global_ids_of_local_sims.index(associated_active_global_id)
                            micro_sims[i] = deepcopy(micro_sims[local_id])
                        else:
                            # Gather information about which sims to receive from where
                            recv_sims_from_ranks[associated_active_global_id] = recv_rank
                            # Gather information about which sims to send where, but from the receiving perspective
                            send_sims_to_ranks_local[associated_active_global_id] = self._rank

        # ----- Gather information about which sims to send where, and now from the sending perspective -----
        send_sims_to_ranks = self._comm.allgather(send_sims_to_ranks_local)

        send_sims_from_this_rank = dict()
        for global_id, rank in send_sims_to_ranks.items():
            if self._micro_sim_is_on_rank[global_id] == self._rank:
                send_sims_from_this_rank[global_id] = rank
        # ----------

        # Asynchronous receive operations
        recv_reqs = []
        for global_id, recv_rank in recv_sims_from_ranks.items():
            tag = self._create_tag(global_id, recv_rank, self._rank)
            recv_reqs.append(self._comm.irecv(source=recv_rank, tag=tag))

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_rank in send_sims_from_this_rank.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            req = self._comm.isend(micro_sims[self._global_ids_of_local_sims.index(global_id)], dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Wait for all non-blocking communications to complete
        self._comm.waitall(send_reqs)

        # Attach received data into the existing FEniCS style data array
        counter = 0
        for recv_gid in recv_pts.keys():
            recv_lid = int(np.where(fenics_gids == recv_gid)[0][0])
            coupling_data[tuple(fenics_coords[recv_lid])] = recv_reqs[counter].wait()
            counter += 1

        return _micro_sim_states

    def associate_inactive_to_active(
            self,
            similarity_dists: np.ndarray,
            micro_sim_states: np.ndarray,
            micro_sims: list) -> list:
        """
        Associate inactive micro simulations to most similar active micro simulation.

        Parameters
        ----------
        similarity_dists : numpy array
            2D array having similarity distances between each micro simulation pair
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        active_sim_ids = np.where(micro_sim_states == 1)[0]
        inactive_sim_ids = np.where(micro_sim_states == 0)[0]

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_sim_ids:
            dist_min = sys.float_info.max
            for active_id in active_sim_ids:
                # Find most similar active sim for every inactive sim
                if similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = similarity_dists[inactive_id, active_id]

            micro_sims[inactive_id].is_associated_to_active_sim(
                associated_active_id, self._global_ids_of_local_sims[associated_active_id])