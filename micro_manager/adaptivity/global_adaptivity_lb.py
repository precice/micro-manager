"""
Class GlobalAdaptivityLBCalculator provides methods to adaptively control of micro simulations
in a global way. If the Micro Manager is run in parallel, an all-to-all comparison of simulations
on each rank is done, along with dynamic load balancing.

Note: All ID variables used in the methods of this class are global IDs, unless they have *local* in their name.
"""
import importlib
import numpy as np
from mpi4py import MPI
import math

from .global_adaptivity import GlobalAdaptivityCalculator

from micro_manager.micro_simulation import create_simulation_class


class GlobalAdaptivityLBCalculator(GlobalAdaptivityCalculator):
    def __init__(
        self,
        configurator,
        global_number_of_sims: int,
        global_ids: list,
        participant,
        logger,
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
        participant : object of class Participant
            Object of the class Participant using which the preCICE API is called.
        logger : object of class Logger
            Logger to log to terminal.
        rank : int
            MPI rank.
        comm : MPI.COMM_WORLD
            Global communicator of MPI.
        """
        super().__init__(
            configurator,
            global_number_of_sims,
            global_ids,
            participant,
            rank,
            comm,
        )

        self._micro_problem = getattr(
            importlib.import_module(
                configurator.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation",
        )

        self._base_logger = logger

        self._local_number_of_sims = len(global_ids)

        self._threshold = configurator.get_load_balancing_threshold()

        self._is_load_balancing_done_in_two_steps = (
            configurator.is_load_balancing_two_step()
        )

        if self._threshold > 0 and self._is_load_balancing_done_in_two_steps:
            self._is_load_balancing_done_in_two_steps = False
            self._base_logger.log_warning_rank_zero(
                "Threshold is not zero, so two step load balancing is disabled."
            )

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

        self._redistribute_active_sims(micro_sims)

        if (not self._nothing_to_balance) and self._balance_inactive_sims:
            self._redistribute_inactive_sims(micro_sims)

    def _redistribute_active_sims(self, micro_sims: list) -> None:
        """
        Redistribute active simulations as per the configured load balancing settings.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._precice_participant.start_profiling_section(
            "load_balancing._redistribute_active_sims"
        )

        avg_active_sims = np.count_nonzero(self._is_sim_active) / self._comm.size

        n_active_sims_local = np.count_nonzero(
            self._is_sim_active[self._global_ids[0] : self._global_ids[-1] + 1]
        )

        send_sims = 0
        recv_sims = 0
        psend_sims = 0
        precv_sims = 0

        f_avg_active_sims = math.floor(avg_active_sims) - self._threshold
        c_avg_active_sims = math.ceil(avg_active_sims) + self._threshold

        if n_active_sims_local == f_avg_active_sims:
            # Simulations to potentially receive
            precv_sims = c_avg_active_sims - n_active_sims_local
        elif n_active_sims_local < f_avg_active_sims:
            # Simulations to receive
            recv_sims = f_avg_active_sims - n_active_sims_local
        elif n_active_sims_local == c_avg_active_sims:
            # Simulations to potentially send
            psend_sims = n_active_sims_local - f_avg_active_sims
        elif n_active_sims_local > c_avg_active_sims:
            # Simulations to send
            send_sims = n_active_sims_local - c_avg_active_sims

        # Number of active sims that each rank wants to send and receive
        global_send_sims = self._comm_world.allgather(send_sims)
        global_recv_sims = self._comm_world.allgather(recv_sims)

        # Number of active sims that each rank potentially wants to send and receive
        global_psend_sims = self._comm_world.allgather(psend_sims)
        global_precv_sims = self._comm_world.allgather(precv_sims)

        n_global_send_sims = sum(global_send_sims)
        n_global_recv_sims = sum(global_recv_sims)

        if n_global_send_sims == 0 or n_global_recv_sims == 0:
            self._nothing_to_balance = True
            self._base_logger.log_warning_rank_zero(
                "It appears that the micro simulations are already fairly balanced. No load balancing will be done. Try changing the threshold value to provoke load balancing."
            )
            return

        if n_global_send_sims < n_global_recv_sims:
            excess_recv_sims = n_global_recv_sims - n_global_send_sims
            while excess_recv_sims > 0:
                for i, e in enumerate(global_recv_sims):
                    if e > 0:
                        # Remove the excess receive request from the rank
                        global_recv_sims[i] -= 1

                        # Add the excess request to the potential receive requests
                        global_precv_sims[i] += 1

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

                        # Add the excess request to the potential send requests
                        global_psend_sims[i] += 1

                        excess_send_sims -= 1

                        if excess_send_sims == 0:
                            break

        send_map, recv_map = self._get_communication_maps(
            global_send_sims, global_recv_sims
        )

        self._communicate_micro_sims(micro_sims, send_map, recv_map)

        if self._is_load_balancing_done_in_two_steps:
            if sum(global_psend_sims) != 0 and sum(global_precv_sims) != 0:
                send_map, recv_map = self._get_communication_maps(
                    global_psend_sims, global_precv_sims
                )

                self._communicate_micro_sims(micro_sims, send_map, recv_map)
            else:
                self._base_logger.log_warning_rank_zero(
                    "No load balancing was done in the second step because the micro simulations are already almost perfectly balanced."
                )

        self._precice_participant.stop_last_profiling_section()

    def _redistribute_inactive_sims(self, micro_sims: list) -> None:
        """
        Redistribute inactive simulations based on where the associated active simulations are.

        Parameters
        ----------
        micro_sims : list
            List of objects of class MicroProblem, which are the micro simulations
        """
        self._precice_participant.start_profiling_section(
            "load_balancing._redistribute_inactive_sims"
        )

        ranks_of_sims = self._get_ranks_of_sims()

        global_ids_of_inactive_sims = np.where(self._is_sim_active == False)[0]

        current_ranks_of_inactive_sims = []
        new_ranks_of_inactive_sims = []
        for inactive_gid in global_ids_of_inactive_sims:
            assoc_active_gid = self._sim_is_associated_to[inactive_gid]

            current_ranks_of_inactive_sims.append(ranks_of_sims[inactive_gid])

            new_ranks_of_inactive_sims.append(ranks_of_sims[assoc_active_gid])

        # Dict of
        # keys: global IDs of sim states to send from this rank
        # values: ranks to send the sims to
        send_map: dict[int, int] = dict()

        # Dict of
        # keys: global IDs of sim states to receive on this rank
        # values: are ranks to receive from
        recv_map: dict[int, int] = dict()

        for i in range(np.count_nonzero(self._is_sim_active == False)):
            if current_ranks_of_inactive_sims[i] != new_ranks_of_inactive_sims[i]:
                inactive_gid = global_ids_of_inactive_sims[i]
                if current_ranks_of_inactive_sims[i] == self._rank:
                    send_map[inactive_gid] = new_ranks_of_inactive_sims[i]
                if new_ranks_of_inactive_sims[i] == self._rank:
                    recv_map[inactive_gid] = current_ranks_of_inactive_sims[i]

        self._communicate_micro_sims(micro_sims, send_map, recv_map)

        self._precice_participant.stop_last_profiling_section()

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
        global_ids_of_active_sims_local = []
        for global_id in self._global_ids:
            if self._is_sim_active[global_id] == True:
                global_ids_of_active_sims_local.append(global_id)

        rank_wise_global_ids_of_active_sims = self._comm_world.allgather(
            global_ids_of_active_sims_local
        )

        # Keys are ranks sending sims, values are lists of tuples: (list of global IDs to send, the rank to send them to)
        global_send_map: dict[int, list] = dict()

        # Keys are ranks, values are lists of tuples: (list of global IDs to receive, the rank to receive them from)
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
                    ][0 : global_recv_sims[recv_rank]]

                    global_send_map[send_rank].append(
                        (global_ids_of_sims_to_move, recv_rank)
                    )

                    global_recv_map[recv_rank].append(
                        (global_ids_of_sims_to_move, send_rank)
                    )

                    sims -= global_recv_sims[recv_rank]

                    # Remove the global IDs which are already mapped for moving
                    del rank_wise_global_ids_of_active_sims[send_rank][
                        0 : global_recv_sims[recv_rank]
                    ]

                    if count < len(recv_ranks) - 1:
                        count += 1
                        recv_rank = recv_ranks[count]

                elif global_recv_sims[recv_rank] > sims:
                    # Get the global IDs to move
                    global_ids_of_sims_to_move = rank_wise_global_ids_of_active_sims[
                        send_rank
                    ][0:sims]

                    global_send_map[send_rank].append((sims, recv_rank))

                    global_recv_map[recv_rank].append((sims, send_rank))

                    global_recv_sims[recv_rank] -= sims

                    # Remove the global IDs which are already mapped for moving
                    del rank_wise_global_ids_of_active_sims[send_rank][0:sims]

                    sims = 0

        # keys are global IDs of sim states to send, values are ranks to send the sims to
        send_map: dict[int, int] = dict()

        # keys are global IDs of sim states to receive, values are ranks to receive from
        recv_map: dict[int, int] = dict()

        if self._rank in global_send_map:
            for send_info in global_send_map[self._rank]:
                send_rank = send_info[1]
                for global_id in send_info[0]:
                    send_map[global_id] = send_rank

        if self._rank in global_recv_map:
            for recv_info in global_recv_map[self._rank]:
                recv_rank = recv_info[1]
                for global_id in recv_info[0]:
                    recv_map[global_id] = recv_rank

        return send_map, recv_map

    def _communicate_micro_sims(
        self, micro_sims: list, send_map: dict, recv_map: dict
    ) -> None:
        """
        Communicate micro simulation states between ranks to balance the load of solving micro simulations.

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
        for global_id, send_rank in send_map.items():
            tag = self._create_tag(global_id, self._rank, send_rank)
            local_id = self._global_ids.index(global_id)
            req = self._comm.isend(
                (micro_sims[local_id].get_state(), global_id), dest=send_rank, tag=tag
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
            local_id = self._global_ids.index(global_id)
            del micro_sims[local_id]
            self._local_number_of_sims -= 1
            self._global_ids.remove(global_id)
            self._is_sim_on_this_rank[global_id] = False

        # Create micro simulations and set them to the received states
        for req in recv_reqs:
            output, global_id = req.wait()
            micro_sims.append(create_simulation_class(self._micro_problem)(global_id))
            micro_sims[-1].set_state(output)
            self._local_number_of_sims += 1
            self._global_ids.append(global_id)
            self._is_sim_on_this_rank[global_id] = True
