import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Set

import numpy as np

from micro_manager.config import Config
from micro_manager.adaptivity.adaptivity_interface import AdaptivityInterface
from micro_manager.model_manager import ModelManager
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler
from micro_manager.tools.profiling import Profiler
from micro_manager.simulation_container import SimulationContainer


@dataclass
class LoadBalancerSimData:
    """
    Captures the full simulation state required to transfer between ranks.
    """

    is_inactive: bool
    is_stateless: Optional[bool]
    state: Optional[Any]
    cls_name: Optional[str]
    gid: int
    coord: np.ndarray
    active_steps: int
    assoc_gid: Optional[int]


class LoadBalancer(ABC):
    def __init__(
        self,
        model_manager: ModelManager,
        sim_container: SimulationContainer,
        log: Logger,
        mpi: MPIHandler,
    ):
        """
        Constructs LoadBalancer.

        Parameters
        ----------
        model_manager: ModelManager
            model manager object to construct instances
        sim_container: SimulationContainer
            simulation container object
        log: Logger
            logger object
        mpi: MPIHandler
            MPIHandler object.
        """
        self._log: Logger = log
        self._mpi: MPIHandler = mpi
        self._model_manager: ModelManager = model_manager
        self._sim_container: SimulationContainer = sim_container
        self._profiler: Profiler = None
        self._adaptivity_controller: AdaptivityInterface = None

    def initialize(self, profiler: Profiler, adaptivity: AdaptivityInterface):
        """
        Initializes remaining fields of LoadBalancer.
        Used to initializes objects that were not available during construct time.

        Parameters
        ----------
        profiler : Profiler
            Profiler object
        adaptivity : AdaptivityInterface
            Adaptivity interface object
        """
        self._profiler = profiler
        self._adaptivity_controller = adaptivity

    @abstractmethod
    def balance(self, n: int) -> bool:
        """
        Requests load balancing.

        Parameters
        ----------
        n : int
            Current time step.

        Returns
        -------
        performed_lb : bool
            Did the load balancing perform any changes?
        """
        pass

    @abstractmethod
    def pre_sim_solve(self, gid: int) -> None:
        """
        Notify load balancer that the micro simulation with the provided gid will start to run its solve method.

        Parameters
        ----------
        gid : int
            Simulation GID
        """
        pass

    @abstractmethod
    def post_sim_solve(self, gid: int) -> None:
        """
        Notify load balancer that the micro simulation with the provided gid has finished its solve method.

        Parameters
        ----------
        gid : int
            Simulation GID
        """
        pass

    @abstractmethod
    def update(self):
        """
        Needs to be called after all micro simulations have finished their solve method.
        Notifies the load balancing implementation that all micro simulations were solved
        and were marked with pre_sim_solve and post_sim_solve.
        If required, the internal state should be synchronized globally to prepare for a
        subsequent load balancing call.
        """
        pass

    def postprocess_sims(self, sim_outputs: List[Dict[str, Any]]) -> None:
        """
        Attaches rank information to simulation outputs.

        Parameters
        ----------
        sim_outputs: List[Dict[str, Any]]
            Rank local simulation outputs.
        """
        for lid in self._sim_container.range_lid:
            sim_outputs[lid]["rank_of_sim"] = self._mpi.rank

    def _gather_send_data(
        self, gid: int, inactive_set: Set[int] = set()
    ) -> LoadBalancerSimData:
        """
        Collects all required data to exchange the specified simulation between ranks.

        Parameters
        ----------
        gid: int
            GID of simulation data should be gathered for.
        inactive_set: Set[int]
            Set of inactive simulations give by their GIDs

        Returns
        -------
        entry: LoadBalancerSimData
            exchange data
        """
        lid = self._sim_container.local_gids.index(gid)

        # prepare send data
        is_inactive = gid in inactive_set
        coord = self._sim_container.local_coords[lid]
        cls_name = None
        is_stateless = None
        state = None
        active_dict = self._adaptivity_controller.get_active_steps()
        active_steps = active_dict[gid]
        del active_dict[gid]
        assoc_gid = None

        if not is_inactive:
            cls_name = self._sim_container[lid].name
            is_stateless = self._model_manager.is_stateless(cls_name)
            state = None if is_stateless else self._sim_container.get_sim_state(lid)
        else:
            assoc_gid = self._adaptivity_controller.get_associated_map()[gid]
            del self._adaptivity_controller.get_associated_map()[gid]

        return LoadBalancerSimData(
            is_inactive,
            is_stateless,
            state,
            cls_name,
            gid,
            coord,
            active_steps,
            assoc_gid,
        )

    def _exchange_sims(self, send_map: Dict[int, List[LoadBalancerSimData]]) -> None:
        """
        Move active micro simulations between ranks.
        Sends state+gid if simulation is active, None+gid otherwise.

        Parameters
        ----------
        send_map : Dict[int, List[LoadBalancerSimData]]
            keys are target ranks, values are lists of data entries to be sent
        """
        recv_sims: List[LoadBalancerSimData] = self._mpi.exchange(send_map)

        # Delete the simulations which no longer exist on this rank
        for send_list in send_map.values():
            for entry in send_list:
                lid = self._sim_container.local_gids.index(entry.gid)
                self._sim_container.remove_sim(lid)

        # Create simulations and set them to the received states
        for entry in recv_sims:
            sim = None
            if not entry.is_inactive:
                assert entry.cls_name is not None
                sim = self._model_manager.get_instance_by_name(
                    entry.gid, entry.cls_name
                )
            else:
                self._adaptivity_controller.get_associated_map()[
                    entry.gid
                ] = entry.assoc_gid
            self._adaptivity_controller.get_active_steps()[
                entry.gid
            ] = entry.active_steps
            lid = self._sim_container.add_sim(entry.gid, sim, entry.coord)

            if not entry.is_stateless and entry.state is not None:
                self._sim_container.set_sim_state(lid, entry.state)

    def _get_global_active_gids(self) -> Set[int]:
        """
        Get global IDs of all active gids. This is based on local ids.

        Returns
        -------
        active_gids: Set[int]
            set of global active gids
        """
        # local count
        active_gids = self._adaptivity_controller.get_active_gids()

        # bcast to all and merge
        tmp = self._mpi.comm.allgather(active_gids)
        global_active_gids = list()
        for l in tmp:
            global_active_gids.extend(l)
        return set(global_active_gids)

    def _get_global_inactive_gids(
        self, active_gids: Optional[Set[int]] = None
    ) -> Set[int]:
        """
        Get global IDs of all inactive gids. This is based on local ids.

        Parameters
        ----------
        active_gids: Set[int]
            set of global active gids, if omitted, will be recomputed.

        Returns
        -------
        inactive_gids: Set[int]
            set of global inactive gids
        """
        global_active_gids = (
            self._get_global_active_gids() if active_gids is None else active_gids
        )
        global_inactive_gids = set(
            np.arange(self._sim_container.global_num_sims)
        ).difference(global_active_gids)
        return global_inactive_gids


class NoOpBalancer(LoadBalancer):
    def __init__(
        self,
        model_manager: ModelManager,
        sim_container: SimulationContainer,
        log: Logger,
        mpi: MPIHandler,
    ):
        super().__init__(model_manager, sim_container, log, mpi)

    def balance(self, n: int) -> bool:
        return False

    def pre_sim_solve(self, gid: int) -> None:
        pass

    def post_sim_solve(self, gid: int) -> None:
        pass

    def update(self):
        pass

    def postprocess_sims(self, sim_outputs: List[Dict[str, Any]]) -> None:
        pass


class TimeBalancer(LoadBalancer):
    def __init__(
        self,
        model_manager: ModelManager,
        sim_container: SimulationContainer,
        log: Logger,
        mpi: MPIHandler,
        config: Config,
    ):
        super().__init__(model_manager, sim_container, log, mpi)

        self._load_balancing_n = config.load_balancing_n()
        self._balance_metric_local: Dict[int, float] = dict()
        self._balance_metric_global: np.ndarray = np.empty(1)
        self._partition_impl = self.get_partition_impl(
            config.load_balancing_partitioning()
        )

    def initialize(self, profiler: Profiler, adaptivity: AdaptivityInterface):
        super().initialize(profiler, adaptivity)

        self._balance_metric_global = np.zeros(self._sim_container.global_num_sims)
        if type(adaptivity).__name__ not in [
            "GlobalAdaptivityCalculator",
            "NoOpAdaptivity",
        ]:
            raise NotImplementedError(
                "To use load balancing, adaptivity must be turned off, or the global variant must be used."
            )

    def balance(self, n: int) -> bool:
        """
        Requests load balancing.
        First computes the new partitioning.
        Then send/receives micro simulations accordingly.

        Parameters
        ----------
        n : int
            Current time step.

        Returns
        -------
        performed_lb : bool
            Did the load balancing perform any changes?
        """
        # balance in first and second step, then every M steps
        if n > 0:
            n -= 1
        if n % self._load_balancing_n != 0:
            return False

        if np.allclose(self._balance_metric_global, 0):
            self._balance_metric_global = self._balance_metric_global + 1

        gids_old = set(self._sim_container.local_gids.copy())
        with self._profiler.measure("micro_manager.solve.load_balancing.init"):
            current_partitioning = self._sim_container.get_ranks_of_sims()
            target_partitioning, work_loads = self._partition_impl(
                self._mpi.comm.size, current_partitioning
            )
            active_gids = self._get_global_active_gids()
            inactive_gids = self._get_global_inactive_gids(active_gids)
            send_map = self._get_communication_map(
                current_partitioning, target_partitioning, active_gids, inactive_gids
            )

        with self._profiler.measure("micro_manager.solve.load_balancing.comm"):
            self._exchange_sims(send_map)

        sims_per_rank = self._mpi.comm.gather(len(self._sim_container), 0)
        self._log.log_info_rank_zero(
            f"Load Balancing Number of Simulations per Rank: {sims_per_rank}"
        )
        gids_new = set(self._sim_container.local_gids.copy())
        return gids_new != gids_old

    def pre_sim_solve(self, gid: int):
        self._balance_metric_local[gid] = time.time()

    def post_sim_solve(self, gid: int):
        self._balance_metric_local[gid] = time.time() - self._balance_metric_local[gid]

    def update(self):
        # used to distribute balancing metric
        self._balance_metric_global[:] = 0
        tmp = self._mpi.comm.allgather(self._balance_metric_local)
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

    def partition_lpt(self, n_parts: int, current_partitioning: Dict[int, int]):
        """
        Partitions the recorded workload using the Longest-processing-time-first scheduling algorithm.
        For more details see: https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling

        Parameters
        ----------
        n_parts: int
            number of partitions (number of ranks)
        current_partitioning: Dict[int, int]
            mapping of each work item to one partition < n_parts

        Returns
        -------
        partitioning: Dict[int, int]
            output of LPT algorithm
        workload: np.ndarray
            workload per partition
        """
        sorted_workload_indices = np.argsort(self._balance_metric_global)[
            ::-1
        ]  # descending
        workload_per_partition = np.zeros(n_parts)
        assignment = dict()

        for idx in sorted_workload_indices:
            # get current smallest partition
            p = np.argmin(workload_per_partition)
            # assign next largest work package
            assignment[idx] = int(p)
            # update partition work load
            workload_per_partition[p] += self._balance_metric_global[idx]

        return assignment, workload_per_partition

    def partition_dummy(self, n_parts: int, current_partitioning: Dict[int, int]):
        """
        WARNING: Do not use this! This is only a dummy implementation that sends
        the entire workload to the first partition. All others are empty.

        Parameters
        ----------
        n_parts: int
            number of partitions (number of ranks)
        current_partitioning: Dict[int, int]
            mapping of each work item to one partition < n_parts

        Returns
        -------
        partition: Dict[int, int]
            output of LPT algorithm
        workload: np.ndarray
            workload per partition
        """
        # do not use this, just an example -> will send all to rank 0
        workload_per_partition = np.zeros(n_parts)
        workload_per_partition[0] = np.sum(self._balance_metric_global)
        return (
            {gid: 0 for gid in self._sim_container.range_gid},
            workload_per_partition,
        )

    # ==================
    #      HELPERS
    # ==================
    def _get_communication_map(
        self,
        current_partitioning: Dict[int, int],
        target_partitioning: Dict[int, int],
        active_set: Set[int],
        inactive_set: Set[int],
    ) -> Dict[int, List[LoadBalancerSimData]]:
        """
        Create dictionaries which map global IDs of simulations to ranks for sending and receiving.

        Parameters
        ----------
        current_partitioning : Dict[int, int]
            Current assignment of simulations
        target_partitioning : Dict[int, int]
            Target assignment of simulations
        inactive_set: Set[int]
            Set of inactive simulations give by their GIDs

        Returns
        -------
        send_map : Dict[int, List[LoadBalancerSimData]]
            Keys are the target ranks. Values are lists of data entries to be transferred.
        """
        send_map = self._mpi.create_empty_exchange_map()

        for gid, target_rank in target_partitioning.items():
            if current_partitioning[gid] == target_rank:
                continue
            if current_partitioning[gid] == self._mpi.rank:
                entry = self._gather_send_data(int(gid), inactive_set)
                send_map[target_rank].append(entry)
                continue

        return send_map


class ActiveBalancer(LoadBalancer):
    """
    ActiveBalancer will attempt to balance the number of active micro simulations between ranks.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        sim_container: SimulationContainer,
        log: Logger,
        mpi: MPIHandler,
        config: Config,
    ):
        super().__init__(
            model_manager,
            sim_container,
            log,
            mpi,
        )
        self._load_balancing_n = config.load_balancing_n()
        self._threshold = config.load_balancing_threshold()
        self._balance_inactive_sims = config.enable_load_balancing_inactive()
        self._bypass_skip = False  # used for testing
        self._bypass_active = False  # used for testing

    def initialize(self, profiler: Profiler, adaptivity: AdaptivityInterface):
        super().initialize(profiler, adaptivity)

        if type(adaptivity).__name__ != "GlobalAdaptivityCalculator":
            raise ValueError(
                "Active Count balancing requires GlobalAdaptivityCalculator"
            )

    def balance(self, n: int) -> bool:
        """
        Requests load balancing.
        First computes the new partitioning.
        Then send/receives micro simulations accordingly.

        Parameters
        ----------
        n : int
            Current time step.

        Returns
        -------
        performed_lb : bool
            Did the load balancing perform any changes?
        """
        # balance in first and second step, then every M steps
        if n > 0:
            n -= 1
        if n % self._load_balancing_n != 0:
            return False

        gids_old = set(self._sim_container.local_gids.copy())
        with self._profiler.measure("micro_manager.solve.load_balancing.init"):
            inactive_gids = self._get_global_inactive_gids()
            send_map = self._get_communication_map(inactive_gids)

        with self._profiler.measure("micro_manager.solve.load_balancing.comm"):
            self._exchange_sims(send_map)

        sims_per_rank = self._mpi.comm.gather(len(self._sim_container), 0)
        self._log.log_info_rank_zero(
            f"Load Balancing Number of Simulations per Rank: {sims_per_rank}"
        )
        gids_new = set(self._sim_container.local_gids.copy())
        return gids_new != gids_old

    def pre_sim_solve(self, gid):
        pass

    def post_sim_solve(self, gid):
        pass

    def update(self):
        pass

    def _get_active_exchange_counts(self):
        avg_active_sims = len(self._get_global_active_gids()) / self._mpi.comm.size
        f_avg_active_sims = np.floor(avg_active_sims - self._threshold)
        c_avg_active_sims = np.ceil(avg_active_sims + self._threshold)

        active_sims_local_ids = self._adaptivity_controller.get_active_lids()
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
        global_send_sims = self._mpi.comm.allgather(send_sims)
        global_recv_sims = self._mpi.comm.allgather(recv_sims)

        n_global_send_sims = sum(global_send_sims)
        n_global_recv_sims = sum(global_recv_sims)

        return (
            global_send_sims,
            global_recv_sims,
            n_global_send_sims,
            n_global_recv_sims,
        )

    def _get_active_comm_maps(
        self,
        global_send_sims: list,
        global_recv_sims: list,
    ):
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
        active_sims_global_ids = self._adaptivity_controller.get_active_gids()
        rank_wise_global_ids_of_active_sims = self._mpi.comm.allgather(
            active_sims_global_ids
        )

        # Keys are ranks sending sims, values are lists of tuples: (list of global IDs to send, the rank to send them to)
        global_send_map: Dict[int, List[int]] = dict()

        # Keys are ranks receiving sims, values are lists of tuples: (list of global IDs to receive, the rank to receive them from)
        global_recv_map: Dict[int, List[int]] = dict()

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

        if self._mpi.rank in global_send_map:
            for send_info in global_send_map[self._mpi.rank]:
                send_rank = send_info[1]
                for gid in send_info[0]:
                    send_map[gid] = send_rank

        if self._mpi.rank in global_recv_map:
            for recv_info in global_recv_map[self._mpi.rank]:
                recv_rank = recv_info[1]
                for gid in recv_info[0]:
                    recv_map[gid] = recv_rank

        return send_map, recv_map

    def _get_inactive_comm_maps(self, inactive_gids: set):
        send_map: dict[int, int] = dict()
        recv_map: dict[int, int] = dict()
        ranks_of_sims = self._sim_container.get_ranks_of_sims()
        global_ids_of_inactive_sims = inactive_gids

        for gid in global_ids_of_inactive_sims:
            assoc_active_gid = self._adaptivity_controller.get_associated_map()[gid]
            rank_of_inactive_sim = ranks_of_sims[gid]
            rank_of_assoc_active_sim = ranks_of_sims[assoc_active_gid]
            if rank_of_inactive_sim != rank_of_assoc_active_sim:
                if rank_of_inactive_sim == self._mpi.rank:
                    send_map[gid] = rank_of_assoc_active_sim
                if rank_of_assoc_active_sim == self._mpi.rank:
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

    def _get_communication_map(self, inactive_set: Set[int]):
        """
        Create dictionaries which map global IDs of simulations to ranks for sending and receiving.

        Parameters
        ----------
        inactive_set: Set[int]
            Set of inactive simulations give by their GIDs

        Returns
        -------
        send_map : Dict[int, List[LoadBalancerSimData]]
            Keys are the target ranks. Values are lists of data entries to be transferred.
        """
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
                return self._mpi.create_empty_exchange_map()
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
            send_map_inactive, recv_map_inactive = self._get_inactive_comm_maps(
                inactive_set
            )
            send_map.update(send_map_inactive)
            recv_map.update(recv_map_inactive)

        # translate to new format
        adapted_send_map = self._mpi.create_empty_exchange_map()
        for gid, target_rank in send_map.items():
            entry = self._gather_send_data(gid, inactive_set)
            adapted_send_map[target_rank].append(entry)

        return adapted_send_map


def create_load_balancer(
    model_manager: ModelManager,
    sim_container: SimulationContainer,
    log: Logger,
    config: Config,
    mpi: MPIHandler,
) -> LoadBalancer:
    if not config.enable_load_balancing():
        return NoOpBalancer(model_manager, sim_container, log, mpi)

    lb_type = config.load_balancing_type()
    if lb_type == "time":
        lb_cls = TimeBalancer
    elif lb_type == "active":
        lb_cls = ActiveBalancer
    else:
        raise RuntimeError(f"Unknown load balancing type: {lb_type}")

    return lb_cls(
        model_manager,
        sim_container,
        log,
        mpi,
        config,
    )
