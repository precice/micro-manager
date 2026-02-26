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


class LoadBalancer:
    def __init__(
        self,
        precice_participant: Participant,
        model_manager: ModelManager,
        adaptivity_controller: Optional[AdaptivityCalculator],
        state_loader: callable,
        state_setter: callable,
        log: Logger,
        config: Config,
        sim_list: list,
        global_ids: list,
        global_number_of_sims: int,
        comm: MPI.Comm,
        rank: int,
    ):
        """
        Constructs LoadBalancer.

        Parameters
        ----------
        precice_participant: Participant
            preCICE participant object from coupling
        model_manager: ModelManager
            model manager object to construct instances
        adaptivity_controller: Optional[AdaptivityCalculator]
            handles adaptivity calculation if provided
        state_loader: callable
            loads state from micro simulation
        state_setter: callable
            sets state of micro simulation
        log: Logger
            logger object
        config: Config
            configuration object
        sim_list: list
            list of simulation objects
        global_ids: list
            list of global ids on this rank
        global_number_of_sims: int
            total number of simulations in this run
        comm: MPI.Comm
            used MPI communicator
        rank: int
            local rank
        """
        self._enabled = config.turn_on_load_balancing()
        self._precice_participant = precice_participant
        self._model_manager = model_manager
        self._adaptivity_controller = adaptivity_controller
        self._state_loader = state_loader
        self._state_setter = state_setter
        self._log = log
        self._config = config
        self._sim_list = sim_list
        self._global_ids = global_ids
        self._global_number_of_sims = global_number_of_sims
        self._comm = comm
        self._rank = rank

        self._threshold = None  # provided by sub-cls
        self._balance_metric_local = dict()
        self._balance_metric_global = np.zeros(global_number_of_sims)
        self._partition_impl = self.get_partition_impl(
            config.get_load_balancing_partitioning()
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

    def pre_sim_solve(self, gid):
        """
        Notify load balancer that the micro simulation with the provided gid will start to run its solve method.
        """
        self._balance_metric_local[gid] = time.time()

    def post_sim_solve(self, gid):
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
    def get_partition_impl(self, name):
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

    def partition_lpt(self, n_parts, current_partitioning):
        """
        Partitions the recorded workload using the LPT algorithm.

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
        assignment = np.zeros(self._global_number_of_sims, dtype=np.int32)

        for idx in sorted_workload_indices:
            # get current smallest partition
            p = np.argmin(workload_per_partition)
            # assign next largest work package
            assignment[idx] = p
            # update partition work load
            workload_per_partition[p] += self._balance_metric_global[idx]

        return assignment, workload_per_partition

    def partition_dummy(self, n_parts, current_partitioning):
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
            np.zeros(self._global_number_of_sims, dtype=np.int32),
            workload_per_partition,
        )

    # ==================
    #      HELPERS
    # ==================
    def _redistribute(self):
        self._precice_participant.start_profiling_section(
            "micro_manager.solve.load_balancing.init"
        )
        current_partitioning = get_ranks_of_sims(
            self._global_ids, self._rank, self._comm, self._global_number_of_sims
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

        self._log.log_info(f"Load Balancing: new sim count={len(self._sim_list)} ")

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
                self._global_ids[i]
                for i in self._adaptivity_controller.get_active_sim_local_ids()
            ]
        else:
            active_gid_arr = self._global_ids

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
        global_inactive_gids = set(np.arange(self._global_number_of_sims)).difference(global_active_gids)
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
            lid = self._global_ids.index(gid)

            # prepare send data
            is_inactive = inactive_map[gid] if gid in inactive_map else False
            cls_name = None if is_inactive else self._sim_list[lid].name
            is_stateless = None if is_inactive else self._model_manager.is_stateless(cls_name)
            send_data = (
                is_inactive,
                is_stateless,
                None if is_stateless or is_inactive else self._state_loader(self._sim_list[lid]),
                cls_name,
                gid,
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
            lid = self._global_ids.index(gid)
            is_active = gid not in inactive_map
            if is_active:
                self._sim_list[lid].destroy()
            del self._sim_list[lid]
            self._global_ids.remove(gid)
            if self._adaptivity_controller is not None:
                self._adaptivity_controller.set_is_on_rank(gid, False)

        # Create simulations and set them to the received states
        for req in recv_reqs:
            is_inactive, is_stateless, state, cls_name, gid = req.wait()
            self._sim_list.append(
                None if is_inactive else
                self._model_manager.get_instance_by_name(gid, cls_name)
            )
            if not is_stateless and state is not None:
                self._state_setter(self._sim_list[-1], state)
            self._global_ids.append(gid)
            if self._adaptivity_controller is not None:
                self._adaptivity_controller.set_is_on_rank(gid, True)


class CountLB(LoadBalancer):
    """
    CountLB will attempt to balance the number of micro simulations between ranks.
    If Adaptivity is to be used, then only GlobalAdaptivity is supported.
    In that case, only active simulation counts are considered for rebalancing.
    """

    def __init__(
        self,
        precice_participant,
        model_manager: ModelManager,
        adaptivity_controller: Optional,
        state_loader: callable,
        state_setter: callable,
        log: Logger,
        config: Config,
        sim_list: list,
        global_ids: list,
        global_number_of_sims: int,
        comm: MPI.Comm,
        rank: int,
    ):
        super().__init__(
            precice_participant,
            model_manager,
            adaptivity_controller,
            state_loader,
            state_setter,
            log,
            config,
            sim_list,
            global_ids,
            global_number_of_sims,
            comm,
            rank,
        )

    def pre_sim_solve(self, gid):
        pass

    def post_sim_solve(self, gid):
        pass

    def update(self):
        # used to distribute balancing metric
        self._balance_metric_global[:] = 0
        self._balance_metric_global[np.array(self._get_global_active_gids())] = 1
