"""
Class GlobalAdaptivityLBCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done, along with dynamic load balancing.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import importlib
from typing import Dict

import numpy as np
from mpi4py import MPI

from .global_adaptivity import GlobalAdaptivityCalculator

from micro_manager.tools.misc import divide_in_parts
from micro_manager.micro_simulation import create_simulation_class


class GlobalAdaptivityLBCalculator(GlobalAdaptivityCalculator):
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
        logger : object of logging
            Logger defined from the standard package logging
        global_number_of_sims : int
            Total number of simulations in the macro-micro coupled problem.
        global_ids : list
            List of global IDs of simulations living on this rank.
        rank : int
            MPI rank.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        """
        super().__init__(configurator, global_number_of_sims, global_ids, rank, comm)

        self._micro_problem = getattr(
            importlib.import_module(
                configurator.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation",
        )

        self._global_number_of_sims = global_number_of_sims
        self._local_number_of_sims = len(global_ids)

    def redistribute_sims(self, micro_sims: list) -> None:
        """
        Redistribute simulations among ranks to balance compute load.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._redistribute_active_sims(micro_sims)
        self._redistribute_inactive_sims(micro_sims)

    def _get_ranks_of_sims(self) -> np.ndarray:
        """
        Get the ranks of the simulations.

        Returns
        -------
        ranks_of_sim : np.ndarray
            Array of ranks on which simulations exist.
        """
        # Create a map of micro simulation global IDs and the ranks on which they are
        micro_sims_on_this_rank = np.zeros(self._local_number_of_sims, dtype=np.intc)
        for i in range(self._local_number_of_sims):
            micro_sims_on_this_rank[i] = self._rank

        ranks_of_sim = np.zeros(self._global_number_of_sims, dtype=np.intc)

        self._comm.Allgatherv(micro_sims_on_this_rank, ranks_of_sim)

        return ranks_of_sim

    def _redistribute_active_sims(self, micro_sims: list) -> None:
        """
        Redistribute simulations among ranks.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        ranks_of_sim = self._get_ranks_of_sims()

        current_ranks_of_active_sims = []
        for i in np.where(self._is_sim_active)[0]:
            current_ranks_of_active_sims.append(ranks_of_sim[i])

        # Divide the total number of active simulation as equally as possible among all ranks
        active_sims_per_rank = divide_in_parts(
            np.count_nonzero(self._is_sim_active), self._comm.size
        )

        counter = 0
        # Get the ranks to which active simulations are to be located at
        new_ranks_of_active_sims = np.copy(current_ranks_of_active_sims)
        for rank, n in enumerate(active_sims_per_rank):
            for i in range(n):
                new_ranks_of_active_sims[counter] = rank
                counter += 1

        send_map: Dict[
            int, list
        ] = (
            dict()
        )  # keys are global IDs of sim states to send, values are ranks to send the sims to
        recv_map: Dict[
            int, int
        ] = (
            dict()
        )  # keys are global IDs of sim states to receive, values are ranks to receive from

        for i in range(self._global_number_of_sims):
            if current_ranks_of_active_sims[i] != new_ranks_of_active_sims[i]:
                if current_ranks_of_active_sims[i] == self._rank:
                    send_map[i] = new_ranks_of_active_sims[i]
                if new_ranks_of_active_sims[i] == self._rank:
                    recv_map[i] = current_ranks_of_active_sims[i]

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_ranks in send_map.items():
            for send_rank in send_ranks:
                tag = self._create_tag(global_id, self._rank, send_rank)
                req = self._comm.isend(
                    micro_sims[global_id].get_state(), dest=send_rank, tag=tag
                )
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

        # Delete the micro simulations which no longer exist on this rank
        for global_id in send_map.keys():
            micro_sims[global_id] = None
            self._local_number_of_sims -= 1

        # Create micro simulations and set them to the received states
        for req in recv_reqs:
            output = req.wait()
            for global_id in recv_map.keys():
                micro_sims[global_id] = create_simulation_class(self._micro_problem)(
                    global_id
                )
                micro_sims[global_id].set_state(output)
                self._local_number_of_sims += 1

    def _redistribute_inactive_sims(self, micro_sims):
        """ """
        ranks_of_sims = self._get_ranks_of_sims()

        current_ranks_of_active_sims = []
        for i in np.where(self._is_sim_active)[0]:
            current_ranks_of_active_sims.append(ranks_of_sims[i])

        current_ranks_of_inactive_sims = []
        for i in np.where(self._is_sim_active)[0] == False:
            current_ranks_of_inactive_sims.append(ranks_of_sims[i])

        associated_inactive_sims = dict()
        for active_gid in range(len(current_ranks_of_active_sims)):
            associated_inactive_sims[active_gid] = [
                i for i, x in enumerate(self._sim_is_associated_to) if x == active_gid
            ]

        new_ranks_of_inactive_sims = np.copy(current_ranks_of_inactive_sims)
        for active_gid, assoc_inactive_gids in associated_inactive_sims.items():
            if current_ranks_of_active_sims[active_gid] != self._rank:
                for inactive_id in assoc_inactive_gids:
                    new_ranks_of_inactive_sims[
                        inactive_id
                    ] = current_ranks_of_active_sims[active_gid]

        send_map: Dict[
            int, list
        ] = (
            dict()
        )  # keys are global IDs of sim states to send, values are ranks to send the sims to
        recv_map: Dict[
            int, int
        ] = (
            dict()
        )  # keys are global IDs of sim states to receive, values are ranks to receive from

        for i in range(self._global_number_of_sims):
            if current_ranks_of_inactive_sims[i] != new_ranks_of_inactive_sims[i]:
                if current_ranks_of_active_sims[i] == self._rank:
                    send_map[i] = new_ranks_of_inactive_sims[i]
                if new_ranks_of_inactive_sims[i] == self._rank:
                    recv_map[i] = current_ranks_of_inactive_sims[i]

        # Asynchronous send operations
        send_reqs = []
        for global_id, send_ranks in send_map.items():
            for send_rank in send_ranks:
                tag = self._create_tag(global_id, self._rank, send_rank)
                req = self._comm.isend(
                    micro_sims[global_id].get_state(), dest=send_rank, tag=tag
                )
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
