"""
Class GlobalAdaptivityLBCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done, along with dynamic load balancing.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import numpy as np
from mpi4py import MPI
import math

from .global_adaptivity import GlobalAdaptivityCalculator
from micro_manager.config import Config
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.model_manager import ModelManager


class GlobalAdaptivityLBCalculator(GlobalAdaptivityCalculator):
    def __init__(
        self,
        configurator: Config,
        global_number_of_sims: int,
        global_ids: list,
        participant,
        base_logger: Logger,
        rank: int,
        comm: MPI.Comm,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
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
        participant : object of class Participant
            Object of the class Participant using which the preCICE API is called.
        base_logger : object of class Logger
            Logger to log to terminal.
        rank : int
            MPI rank.
        comm : MPI.Comm
            Communicator for MPI.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object of class ModelManager
            Handles instantiation of the micro simulation.
        """
        super().__init__(
            configurator,
            global_number_of_sims,
            global_ids,
            participant,
            base_logger,
            rank,
            comm,
            micro_problem_cls,
            model_manager,
        )

        self._base_logger = base_logger

        self._threshold = configurator.get_load_balancing_threshold()

        self._balance_inactive_sims = configurator.balance_inactive_sims()

        self._nothing_to_balance = False

        self._precice_participant = participant

    def redistribute_sims(self, micro_sims: list) -> None:
        """
        Redistribute simulations among ranks to balance compute load.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._nothing_to_balance = False

        self._precice_participant.start_profiling_section(
            "global_adaptivity_lb.redistributing_sims"
        )

        self._redistribute_active_sims(micro_sims)

        if (not self._nothing_to_balance) and self._balance_inactive_sims:
            self._redistribute_inactive_sims(micro_sims)

        self._precice_participant.stop_last_profiling_section()

    def _redistribute_active_sims(self, micro_sims: list) -> None:
        """
        Redistribute active simulations as per the configured load balancing settings.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        avg_active_sims = np.count_nonzero(self._is_sim_active) / self._comm.size

        active_sims_local_ids = self.get_active_sim_local_ids()

        n_active_sims_local = active_sims_local_ids.size

        send_sims = 0  # Sims that this rank wants to send
        recv_sims = 0  # Sims that this rank wants to receive

        f_avg_active_sims = math.floor(avg_active_sims - self._threshold)
        c_avg_active_sims = math.ceil(avg_active_sims + self._threshold)

        if f_avg_active_sims == c_avg_active_sims:
            if n_active_sims_local < avg_active_sims:
                recv_sims = int(avg_active_sims) - n_active_sims_local
            elif n_active_sims_local > avg_active_sims:
                send_sims = n_active_sims_local - int(avg_active_sims)
        else:
            if n_active_sims_local < f_avg_active_sims:
                recv_sims = f_avg_active_sims - n_active_sims_local
            elif n_active_sims_local == f_avg_active_sims:
                recv_sims += 1
            elif n_active_sims_local > c_avg_active_sims:
                send_sims = n_active_sims_local - c_avg_active_sims
            elif n_active_sims_local == c_avg_active_sims:
                send_sims += 1

        # Number of active sims that each rank wants to send and receive
        global_send_sims = self._comm.allgather(send_sims)
        global_recv_sims = self._comm.allgather(recv_sims)

        n_global_send_sims = sum(global_send_sims)
        n_global_recv_sims = sum(global_recv_sims)

        if n_global_send_sims == 0 and n_global_recv_sims == 0:
            self._nothing_to_balance = True
            self._base_logger.log_warning_rank_zero(
                "It appears that the micro simulations are already fairly balanced. No load balancing will be done. Try changing the threshold value to induce load balancing."
            )
            return

        if n_global_send_sims < n_global_recv_sims:
            excess_recv_sims = n_global_recv_sims - n_global_send_sims
            while excess_recv_sims > 0:
                for i, e in enumerate(global_recv_sims):
                    if e > 0:
                        # Remove the excess receive request from the rank
                        global_recv_sims[i] -= 1

                        excess_recv_sims -= 1

                        if excess_recv_sims == 0:
                            break
        elif n_global_send_sims > n_global_recv_sims:
            excess_send_sims = n_global_send_sims - n_global_recv_sims
            while excess_send_sims > 0:
                for i, e in enumerate(global_send_sims):
                    if e > 0:
                        # Remove the excess send request
                        global_send_sims[i] -= 1

                        excess_send_sims -= 1

                        if excess_send_sims == 0:
                            break

        send_map, recv_map = self._get_communication_maps(
            global_send_sims, global_recv_sims
        )

        self._move_active_sims(micro_sims, send_map, recv_map)

    def _redistribute_inactive_sims(self, micro_sims: list) -> None:
        """
        Redistribute inactive simulations based on where the associated active simulations are.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        # Dict of
        # keys: global IDs of sim states to send from this rank
        # values: ranks to send the sims to
        send_map: dict[int, int] = dict()

        # Dict of
        # keys: global IDs of sim states to receive on this rank
        # values: are ranks to receive from
        recv_map: dict[int, int] = dict()

        ranks_of_sims = self._get_ranks_of_sims()

        global_ids_of_inactive_sims = np.where(self._is_sim_active == False)[0]

        for gid in global_ids_of_inactive_sims:
            assoc_active_gid = self._sim_is_associated_to[gid]
            rank_of_inactive_sim = ranks_of_sims[gid]
            rank_of_assoc_active_sim = ranks_of_sims[assoc_active_gid]
            if rank_of_inactive_sim != rank_of_assoc_active_sim:
                if rank_of_inactive_sim == self._rank:
                    send_map[gid] = rank_of_assoc_active_sim
                if rank_of_assoc_active_sim == self._rank:
                    recv_map[gid] = rank_of_inactive_sim

        self._move_inactive_sims(micro_sims, send_map, recv_map)

    def _get_communication_maps(
        self, global_send_sims: list, global_recv_sims: list
    ) -> tuple:
        """
        Create dictionaries which map global IDs of simulations to ranks for sending and receiving.

        Parameters
        ----------
        global_send_sims : list
            Number of simulations that each rank sends.
        global_recv_sims : list
            Number of simulations that each rank receives.

        Returns
        -------
        tuple of dicts
            send_map : dict
                keys are global IDs of sim states to send, values are ranks to send the sims to
            recv_map : dict
                keys are global IDs of sim states to receive, values are ranks to receive from
        """
        active_sims_global_ids = list(self.get_active_sim_global_ids())

        rank_wise_global_ids_of_active_sims = self._comm.allgather(
            active_sims_global_ids
        )

        # Keys are ranks sending sims, values are lists of tuples: (list of global IDs to send, the rank to send them to)
        global_send_map: dict[int, list] = dict()

        # Keys are ranks receiving sims, values are lists of tuples: (list of global IDs to receive, the rank to receive them from)
        global_recv_map: dict[int, list] = dict()

        for rank in [i for i, e in enumerate(global_send_sims) if e != 0]:
            global_send_map[rank] = []

        for rank in [i for i, e in enumerate(global_recv_sims) if e != 0]:
            global_recv_map[rank] = []

        send_ranks = list(global_send_map.keys())
        recv_ranks = list(global_recv_map.keys())

        count = 0
        recv_rank = recv_ranks[count]

        for send_rank in send_ranks:
            sims = global_send_sims[send_rank]
            while sims > 0:
                if global_recv_sims[recv_rank] <= sims:
                    # Get the global IDs to move
                    global_ids_of_sims_to_move = rank_wise_global_ids_of_active_sims[
                        send_rank
                    ][0 : int(global_recv_sims[recv_rank])]

                    global_send_map[send_rank].append(
                        (global_ids_of_sims_to_move, recv_rank)
                    )

                    global_recv_map[recv_rank].append(
                        (global_ids_of_sims_to_move, send_rank)
                    )

                    sims -= global_recv_sims[recv_rank]

                    # Remove the global IDs which are already mapped for moving
                    del rank_wise_global_ids_of_active_sims[send_rank][
                        0 : int(global_recv_sims[recv_rank])
                    ]

                    if count < len(recv_ranks) - 1:
                        count += 1
                        recv_rank = recv_ranks[count]

                elif global_recv_sims[recv_rank] > sims:
                    # Get the global IDs to move
                    global_ids_of_sims_to_move = rank_wise_global_ids_of_active_sims[
                        send_rank
                    ][0 : int(sims)]

                    global_send_map[send_rank].append(
                        (global_ids_of_sims_to_move, recv_rank)
                    )

                    global_recv_map[recv_rank].append(
                        (global_ids_of_sims_to_move, send_rank)
                    )

                    global_recv_sims[recv_rank] -= sims

                    # Remove the global IDs which are already mapped for moving
                    del rank_wise_global_ids_of_active_sims[send_rank][0 : int(sims)]

                    sims = 0

        # keys are global IDs of sim states to send, values are ranks to send the sims to
        send_map: dict[int, int] = dict()

        # keys are global IDs of sim states to receive, values are ranks to receive from
        recv_map: dict[int, int] = dict()

        if self._rank in global_send_map:
            for send_info in global_send_map[self._rank]:
                send_rank = send_info[1]
                for gid in send_info[0]:
                    send_map[gid] = send_rank

        if self._rank in global_recv_map:
            for recv_info in global_recv_map[self._rank]:
                recv_rank = recv_info[1]
                for gid in recv_info[0]:
                    recv_map[gid] = recv_rank

        return send_map, recv_map

    def _move_active_sims(
        self, micro_sims: list, send_map: dict, recv_map: dict
    ) -> None:
        """
        Move active micro simulations between ranks.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        send_map : dict
            keys are global IDs of sim states to send, values are ranks to send the sims to
        recv_map : dict
            keys are global IDs of sim states to receive, values are ranks to receive from
        """
        # Asynchronous send operations
        send_reqs = []
        for gid, send_rank in send_map.items():
            tag = self._create_tag(gid, self._rank, send_rank)
            lid = self._global_ids.index(gid)
            req = self._comm.isend(
                (micro_sims[lid].get_state(), gid), dest=send_rank, tag=tag
            )
            send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for gid, recv_rank in recv_map.items():
            tag = self._create_tag(gid, recv_rank, self._rank)
            bufsize = (
                1 << 30
            )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
            req = self._comm.irecv(bufsize, source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        # Delete the active simulations which no longer exist on this rank
        for gid in send_map.keys():
            lid = self._global_ids.index(gid)
            del micro_sims[lid]
            self._global_ids.remove(gid)
            self._is_sim_on_this_rank[gid] = False

        # Create simulations and set them to the received states
        for req in recv_reqs:
            output, gid = req.wait()
            micro_sims.append(
                self._model_manager.get_instance(gid, self._micro_problem_cls)
            )
            micro_sims[-1].set_state(output)
            self._global_ids.append(gid)
            self._is_sim_on_this_rank[gid] = True

    def _move_inactive_sims(
        self, micro_sims: list, send_map: dict, recv_map: dict
    ) -> None:
        """
        Move inactive micro simulation states between ranks.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        send_map : dict
            keys are global IDs of sim states to send, values are ranks to send the sims to
        recv_map : dict
            keys are global IDs of sim states to receive, values are ranks to receive from
        """
        # Asynchronous send operations
        send_reqs = []
        for gid, send_rank in send_map.items():
            tag = self._create_tag(gid, self._rank, send_rank)
            lid = self._global_ids.index(gid)
            req = self._comm.isend((gid), dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for gid, recv_rank in recv_map.items():
            tag = self._create_tag(gid, recv_rank, self._rank)
            bufsize = (
                1 << 30
            )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
            req = self._comm.irecv(bufsize, source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        # Delete the inactive simulations which no longer exist on this rank
        for gid in send_map.keys():
            lid = self._global_ids.index(gid)
            del micro_sims[lid]
            self._global_ids.remove(gid)
            self._is_sim_on_this_rank[gid] = False

        # Add inactive simulations in the data structure
        for req in recv_reqs:
            gid = req.wait()
            micro_sims.append(0)
            self._global_ids.append(gid)
            self._is_sim_on_this_rank[gid] = True
