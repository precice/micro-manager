import time
from typing import Optional

from mpi4py import MPI
import numpy as np

from precice import Participant

from micro_manager.config import Config
from micro_manager.adaptivity.adaptivity import AdaptivityCalculator
from micro_manager.model_manager import ModelManager
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.p2p import create_tag, get_ranks_of_sims
from micro_manager.simulation_container import SimulationContainer


class LoadBalancer:
    def __init__(
        self,
        precice_participant: Participant,
        model_manager: ModelManager,
        adaptivity_controller: Optional[AdaptivityCalculator],
        sim_container: SimulationContainer,
        log: Logger,
        config: Config,
        comm: MPI.Comm,
        rank: int,
    ):
        """
        Constructs LoadBalancer. If load balancing is disabled, this will return a dummy instance
        in which balancing request become NOOPs.

        Parameters
        ----------
        precice_participant: Participant
            preCICE participant object from coupling
        model_manager: ModelManager
            model manager object to construct instances
        adaptivity_controller: Optional[AdaptivityCalculator]
            handles adaptivity calculation if provided
        log: Logger
            logger object
        config: Config
            configuration object
        comm: MPI.Comm
            used MPI communicator
        rank: int
            local rank
        """
        self._enabled = config.enable_load_balancing()
        self._precice_participant = precice_participant
        self._model_manager = model_manager
        self._adaptivity_controller = adaptivity_controller
        self._sim_container = sim_container
        self._log = log
        self._config = config
        self._comm = comm
        self._rank = rank

        self._threshold = None  # provided by sub-cls
        self._balance_metric_local = dict()
        self._balance_metric_global = np.zeros(self._sim_container.global_num_sims)
        self._partition_impl = self.get_partition_impl(
            config.load_balancing_partitioning()
        )

        if (
            self._enabled
            and adaptivity_controller is not None
            and type(adaptivity_controller).__name__ != "GlobalAdaptivityCalculator"
        ):
            raise NotImplementedError(
                "Adaptivity must be GlobalAdaptivity for Load Balancing"
            )

    def balance(self):
        """
        Requests load balancing. If LoadBalancing is disabled, returns immediately.
        """
        if not self._enabled:
            return

        # self._precice_participant.start_profiling_section("micro_manager.solve.load_balancing.redistribute")
        if np.allclose(self._balance_metric_global, 0):
            self._balance_metric_global = self._balance_metric_global + 1
        self._redistribute()
        # self._precice_participant.stop_last_profiling_section()

    def pre_sim_solve(self, gid: int):
        """
        Notify load balancer that the micro simulation with the provided gid will start to run its solve method.
        """
        self._balance_metric_local[gid] = time.time()

    def post_sim_solve(self, gid: int):
        """
        Notify load balancer that the micro simulation with the provided gid has finished its solve method.
        """
        self._balance_metric_local[gid] = time.time() - self._balance_metric_local[gid]

    def update(self):
        """
        Needs to be called after all micro simulations have finished their solve method.
        Updates the load balancing metric and shares it globally.
        """

        # used to distribute balancing metric
        self._balance_metric_global[:] = 0
        tmp = self._comm.allgather(self._balance_metric_local)
        for d in tmp:
            for gid, val in d.items():
                self._balance_metric_global[gid] = val

        # clear local buffer
        self._balance_metric_local.clear()

    # ==================
    #    PARTITIONING
    # ==================
    def get_partition_impl(self, name: str):
        """
        Provides the selected partitioning algorithm.

        Parameters
        ----------
        name: str
            partitioning algorithm name

        Returns
        -------
        function: callable
            selected partitioning algorithm
        """
        if name == "lpt":
            return self.partition_lpt
        else:
            return self.partition_dummy

    def partition_lpt(self, n_parts: int, current_partitioning: np.ndarray):
        """
        Partitions the recorded workload using the Longest-processing-time-first scheduling algorithm.
        For more details see: https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling

        Parameters
        ----------
        n_parts: int
            number of partitions (number of ranks)
        current_partitioning: np.ndarray
            array of assignments of each work item to one partition < n_parts

        Returns
        -------
        partitioning: np.ndarray
            output of LPT algorithm
        workload: np.ndarray
            workload per partition
        """
        sorted_workload_indices = np.argsort(self._balance_metric_global)[
            ::-1
        ]  # descending
        workload_per_partition = np.zeros(n_parts)
        assignment = np.zeros(self._sim_container.global_num_sims, dtype=np.int32)

        for idx in sorted_workload_indices:
            # get current smallest partition
            p = np.argmin(workload_per_partition)
            # assign next largest work package
            assignment[idx] = p
            # update partition work load
            workload_per_partition[p] += self._balance_metric_global[idx]

        return assignment, workload_per_partition

    def partition_dummy(self, n_parts: int, current_partitioning: np.ndarray):
        """
        WARNING: Do not use this! This is only a dummy implementation that sends
        the entire workload to the first partition. All others are empty.

        Parameters
        ----------
        n_parts: int
            number of partitions (number of ranks)
        current_partitioning: np.ndarray
            array of assignments of each work item to one partition < n_parts

        Returns
        -------
        partition: np.ndarray
            output of LPT algorithm
        workload: np.ndarray
            workload per partition
        """
        # do not use this, just an example -> will send all to rank 0
        workload_per_partition = np.zeros(n_parts)
        workload_per_partition[0] = np.sum(self._balance_metric_global)
        return (
            np.zeros(self._sim_container.global_num_sims, dtype=np.int32),
            workload_per_partition,
        )

    # ==================
    #      HELPERS
    # ==================
    def _redistribute(self) -> None:
        """
        Main implementation of load balancing. First computes the new partitioning.
        Then send/receives micro simulations accordingly.
        """
        self._precice_participant.start_profiling_section(
            "micro_manager.solve.load_balancing.init"
        )
        current_partitioning = get_ranks_of_sims(
            self._sim_container.local_gids,
            self._rank,
            self._comm,
            self._sim_container.global_num_sims,
        )
        # self._precice_participant.start_profiling_section("micro_manager.solve.load_balancing.init.partition")
        target_partitioning, work_loads = self._partition_impl(
            self._comm.size, current_partitioning
        )
        # self._precice_participant.stop_last_profiling_section()
        send_map, recv_map = self._get_communication_maps(
            current_partitioning, target_partitioning
        )
        inactive_gids = self._get_global_inactive_gids()
        self._precice_participant.stop_last_profiling_section()

        self._precice_participant.start_profiling_section(
            "micro_manager.solve.load_balancing.comm"
        )
        self._exchange_sims(send_map, recv_map, {gid: True for gid in inactive_gids})
        self._precice_participant.stop_last_profiling_section()

        sims_per_rank = self._comm.gather(len(self._sim_container), 0)
        self._log.log_info_rank_zero(
            f"Load Balancing Number of Simulations per Rank: {sims_per_rank}"
        )

    def _get_communication_maps(
        self, current_partitioning: np.ndarray, target_partitioning: np.ndarray
    ) -> tuple:
        """
        Create dictionaries which map global IDs of simulations to ranks for sending and receiving.

        Parameters
        ----------
        current_partitioning : np.ndarray
            Current assignment of simulations
        target_partitioning : np.ndarray
            Target assignment of simulations

        Returns
        -------
        tuple of dicts
            send_map : dict
                keys are global IDs, values are target ranks
            recv_map : dict
                keys are global IDs, values are source ranks
        """
        send_map = dict()
        recv_map = dict()

        for gid, target_rank in enumerate(target_partitioning):
            if current_partitioning[gid] == target_rank:
                continue
            if current_partitioning[gid] == self._rank:
                send_map[gid] = target_rank
                continue
            if target_rank == self._rank:
                recv_map[gid] = current_partitioning[gid]

        return send_map, recv_map

    def _get_global_active_gids(self):
        """
        Get global IDs of all active gids. This is based on local ids.

        Returns
        -------
        active_gids: list[int]
            list of global active gids
        """
        # local count
        active_gid_arr = None
        if self._adaptivity_controller is not None:
            active_gid_arr = [
                self._sim_container.local_gids[i]
                for i in self._adaptivity_controller.get_active_sim_local_ids()
            ]
        else:
            active_gid_arr = self._sim_container.local_gids

        # bcast to all and merge
        tmp = self._comm.allgather(active_gid_arr)
        global_active_gids = []
        for l in tmp:
            global_active_gids.extend(l)
        return global_active_gids

    def _get_global_inactive_gids(self):
        """
        Get global IDs of all inactive gids. This is based on local ids.

        Returns
        -------
        inactive_gids: list[int]
            list of global inactive gids
        """
        global_active_gids = set(self._get_global_active_gids())
        global_inactive_gids = set(
            np.arange(self._sim_container.global_num_sims)
        ).difference(global_active_gids)
        return list(global_inactive_gids)

    def _exchange_sims(self, send_map, recv_map, inactive_map={}):
        """
        Move active micro simulations between ranks.
        Sends state+gid if simulation is active, None+gid otherwise.

        Parameters
        ----------
        send_map : dict
            keys are global IDs of sim states to send, values are ranks to send the sims to
        recv_map : dict
            keys are global IDs of sim states to receive, values are ranks to receive from
        inactive_map : dict
            keys are global IDs of inactive sim states, values are bool
        """
        # Asynchronous send operations
        send_reqs = []
        for gid, send_rank in send_map.items():
            tag = create_tag(gid, self._rank, send_rank)
            lid = self._sim_container.local_gids.index(gid)

            # prepare send data
            is_inactive = inactive_map[gid] if gid in inactive_map else False
            cls_name = None if is_inactive else self._sim_container[lid].name
            coord = self._sim_container.local_coords[lid]
            is_stateless = (
                None if is_inactive else self._model_manager.is_stateless(cls_name)
            )
            send_data = (
                is_inactive,
                is_stateless,
                None
                if is_stateless or is_inactive
                else self._sim_container.get_state(lid),
                cls_name,
                gid,
                coord,
            )

            req = self._comm.isend(send_data, dest=send_rank, tag=tag)
            send_reqs.append(req)

        # Asynchronous receive operations
        recv_reqs = []
        for gid, recv_rank in recv_map.items():
            tag = create_tag(gid, recv_rank, self._rank)
            bufsize = (
                1 << 30
            )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
            req = self._comm.irecv(bufsize, source=recv_rank, tag=tag)
            recv_reqs.append(req)

        # Wait for all non-blocking communication to complete
        MPI.Request.Waitall(send_reqs)

        # Delete the simulations which no longer exist on this rank
        for gid in send_map.keys():
            lid = self._sim_container.local_gids.index(gid)
            self._sim_container.remove_sim(lid)

        # Create simulations and set them to the received states
        for req in recv_reqs:
            is_inactive, is_stateless, state, cls_name, gid, coord = req.wait()
            sim = None
            if not is_inactive:
                sim = self._model_manager.get_instance_by_name(gid, cls_name)
            lid = self._sim_container.add_sim(gid, sim, coord)

            if not is_stateless and state is not None:
                self._sim_container.set_state(lid, state)


class ActiveBalancer(LoadBalancer):
    """
    ActiveBalancer will attempt to balance the number of active micro simulations between ranks.
    """

    def __init__(
        self,
        precice_participant: Participant,
        model_manager: ModelManager,
        adaptivity_controller: Optional[AdaptivityCalculator],
        sim_container: SimulationContainer,
        log: Logger,
        config: Config,
        comm: MPI.Comm,
        rank: int,
    ):
        super().__init__(
            precice_participant,
            model_manager,
            adaptivity_controller,
            sim_container,
            log,
            config,
            comm,
            rank,
        )
        self._partition_impl = lambda a, b: (None, None)
        self._threshold = config.load_balancing_threshold()
        self._balance_inactive_sims = config.enable_load_balancing_inactive()
        self._bypass_skip = False  # used for testing
        self._bypass_active = False  # used for testing

        if adaptivity_controller is None:
            raise ValueError(
                "Active Count balancing requires GlobalAdaptivityCalculator"
            )

    def pre_sim_solve(self, gid):
        pass

    def post_sim_solve(self, gid):
        pass

    def update(self):
        pass

    def _get_active_exchange_counts(self):
        avg_active_sims = (
            np.count_nonzero(self._adaptivity_controller._is_sim_active)
            / self._comm.size
        )
        f_avg_active_sims = np.floor(avg_active_sims - self._threshold)
        c_avg_active_sims = np.ceil(avg_active_sims + self._threshold)

        active_sims_local_ids = self._adaptivity_controller.get_active_sim_local_ids()
        n_active_sims_local = len(active_sims_local_ids)
        send_sims = 0  # Sims that this rank wants to send
        recv_sims = 0  # Sims that this rank wants to receive

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

        return (
            global_send_sims,
            global_recv_sims,
            n_global_send_sims,
            n_global_recv_sims,
        )

    def _get_active_comm_maps(self, global_send_sims: list, global_recv_sims: list):
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
        active_sims_global_ids = list(
            self._adaptivity_controller.get_active_sim_global_ids()
        )
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

    def _get_inactive_comm_maps(self):
        send_map: dict[int, int] = dict()
        recv_map: dict[int, int] = dict()
        ranks_of_sims = get_ranks_of_sims(
            self._sim_container.local_gids,
            self._rank,
            self._comm,
            self._sim_container.global_num_sims,
        )
        global_ids_of_inactive_sims = self._get_global_inactive_gids()

        for gid in global_ids_of_inactive_sims:
            assoc_active_gid = self._adaptivity_controller._sim_is_associated_to[gid]
            rank_of_inactive_sim = ranks_of_sims[gid]
            rank_of_assoc_active_sim = ranks_of_sims[assoc_active_gid]
            if rank_of_inactive_sim != rank_of_assoc_active_sim:
                if rank_of_inactive_sim == self._rank:
                    send_map[gid] = rank_of_assoc_active_sim
                if rank_of_assoc_active_sim == self._rank:
                    recv_map[gid] = rank_of_inactive_sim

        return send_map, recv_map

    @staticmethod
    def _correct_active_exchange_data(
        global_send_sims, global_recv_sims, n_global_send_sims, n_global_recv_sims
    ):
        if n_global_send_sims < n_global_recv_sims:
            excess_recv_sims = n_global_recv_sims - n_global_send_sims
            while excess_recv_sims > 0:
                for i, e in enumerate(global_recv_sims):
                    if e <= 0:
                        continue
                    # Remove the excess receive request from the rank
                    global_recv_sims[i] -= 1
                    excess_recv_sims -= 1
                    if excess_recv_sims == 0:
                        break
        elif n_global_send_sims > n_global_recv_sims:
            excess_send_sims = n_global_send_sims - n_global_recv_sims
            while excess_send_sims > 0:
                for i, e in enumerate(global_send_sims):
                    if e <= 0:
                        continue
                    # Remove the excess send request
                    global_send_sims[i] -= 1
                    excess_send_sims -= 1
                    if excess_send_sims == 0:
                        break

    def _get_communication_maps(self, *args, **kwargs):
        send_map: dict[int, int] = dict()
        recv_map: dict[int, int] = dict()

        if not self._bypass_active:
            (
                global_send_sims,
                global_recv_sims,
                n_global_send_sims,
                n_global_recv_sims,
            ) = self._get_active_exchange_counts()

            if (
                n_global_send_sims == 0 or n_global_recv_sims == 0
            ) and not self._bypass_skip:
                self._log.log_warning_rank_zero(
                    "It appears that the micro simulations are already fairly balanced. No load balancing will be done. Try changing the threshold value to induce load balancing."
                )
                return send_map, recv_map
            if n_global_send_sims != 0 or n_global_recv_sims != 0:
                ActiveBalancer._correct_active_exchange_data(
                    global_send_sims,
                    global_recv_sims,
                    n_global_send_sims,
                    n_global_recv_sims,
                )
                send_map_active, recv_map_active = self._get_active_comm_maps(
                    global_send_sims, global_recv_sims
                )
                send_map.update(send_map_active)
                recv_map.update(recv_map_active)

        # if requested, also balance inactive simulations if there was a change in active simulations
        if self._balance_inactive_sims:
            send_map_inactive, recv_map_inactive = self._get_inactive_comm_maps()
            send_map.update(send_map_inactive)
            recv_map.update(recv_map_inactive)

        return send_map, recv_map


def create_load_balancer(
    precice_participant: Participant,
    model_manager: ModelManager,
    adaptivity_controller: Optional[AdaptivityCalculator],
    sim_container: SimulationContainer,
    log: Logger,
    config: Config,
    comm: MPI.Comm,
    rank: int,
) -> LoadBalancer:
    lb_type = config.load_balancing_type()

    if lb_type == "time":
        lb_cls = LoadBalancer
    elif lb_type == "active":
        lb_cls = ActiveBalancer
    else:
        raise RuntimeError(f"Unknown load balancing type: {lb_type}")

    return lb_cls(
        precice_participant,
        model_manager,
        adaptivity_controller,
        sim_container,
        log,
        config,
        comm,
        rank,
    )
