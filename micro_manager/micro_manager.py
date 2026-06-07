#!/usr/bin/env python3
"""
Micro Manager is a tool to initialize and adaptively control micro simulations and couple them via preCICE to a macro simulation.
This files the class MicroManager which has the following callable public methods:

- solve
- initialize

Upon execution, an object of the class MicroManager is created using a given JSON file,
and the initialize and solve methods are called.

Detailed documentation: https://precice.org/tooling-micro-manager-overview.html
"""

import os
import sys
from typing import Callable
import numpy as np
from psutil import Process
import csv
import subprocess
from functools import partial

from .simulation_container import SimulationContainer
from .model_manager import ModelManager
from .micro_manager_base import MicroManager

from .adaptivity.model_adaptivity import ModelAdaptivity
from .adaptivity.adaptivity_selection import create_adaptivity_calculator

from .domain_decomposition import DomainDecomposer, create_domain_decomposer
from .tasking.connection import spawn_local_workers
from .tools.logging_wrapper import Logger
from .load_balancing import create_load_balancer
from .tools.mpi_handler import MPIHandler, MPI
from .tools.coupling import CouplingHandler
from .tools.profiling import Profiler

try:
    from .interpolation import Interpolation
except ImportError:
    Interpolation = None

sys.path.append(os.getcwd())


class MicroManagerCoupling(MicroManager):
    def __init__(self, config_file: str, log_file: str = "") -> None:
        """
        Constructor.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (provided by the user).
        """
        super().__init__(config_file)
        self._mpi = MPIHandler(MPI.COMM_WORLD)

        self._log_file = log_file
        self._logger = Logger(__name__, log_file, self._mpi.rank)

        self._config.set_logger(self._logger)
        self._config.read_json_micro_manager()

        self._memory_usage_output_type = self._config.memory_usage_output_type()

        self._memory_usage_output_n = self._config.memory_usage_output_n()

        self._output_dir = self._config.output_dir()

        if self._output_dir is not None:
            self._output_dir = os.path.abspath(self._output_dir) + "/"
            subprocess.run(["mkdir", "-p", self._output_dir])  # Create output directory
        else:
            self._output_dir = os.path.abspath(os.getcwd()) + "/"

        self._sim_container = SimulationContainer(self._mpi)

        self._coupling: CouplingHandler = CouplingHandler(
            self._config,
            self._mpi,
            self._sim_container,
        )
        self._profiler: Profiler = Profiler(self._coupling.participant)

        # Parameter for interpolation in case of a simulation crash
        self._interpolate_crashed_sims = self._config.enable_crashed_sim_interpolation()
        if self._interpolate_crashed_sims:
            if Interpolation is None:
                self._logger.log_info_rank_zero(
                    "Interpolation is turned off as the required package is not installed."
                )
                self._interpolate_crashed_sims = False
            else:
                # TODO: Make these parameters configurable
                self._crash_threshold = 0.2
                self._number_of_nearest_neighbors = 4

        self._micro_n_out = self._config.micro_output_n()

        self._lazy_init = self._config.enable_adaptivity_lazy_init()

        self._is_adaptivity_on = self._config.enable_adaptivity()

        if self._is_adaptivity_on:
            self._data_for_adaptivity: dict[str, list] = dict()

            self._adaptivity_data_names = self._config.data_for_adaptivity()

            # Names of macro data to be used for adaptivity computation
            self._adaptivity_macro_data_names: list = []

            # Names of micro data to be used for adaptivity computation
            self._adaptivity_micro_data_names: list = []
            for name in self._adaptivity_data_names:
                if name in self._coupling.read_data_names:
                    self._adaptivity_macro_data_names.append(name)
                if name in self._coupling.write_data_names:
                    self._adaptivity_micro_data_names.append(name)

            self._adaptivity_in_every_implicit_step = (
                self._config.enable_adaptivity_each_implicit_iteration()
            )

        self._adaptivity_n = self._config.adaptivity_n()

        self._adaptivity_output_type = self._config.adaptivity_output_type()

        self._adaptivity_output_n = self._config.adaptivity_output_n()

        self._is_model_adaptivity_on = self._config.enable_model_adaptivity()

        self._t = 0  # global time
        self._n = 0  # sim-step

        self._model_manager = ModelManager(self._logger)
        self._conn = None
        self._is_load_balancing = (
            self._config.enable_load_balancing() and self._mpi.is_parallel()
        )
        self._load_balancing_n = self._config.load_balancing_n()
        self.load_balancing = None

    # **************
    # Public methods
    # **************

    def solve(self) -> None:
        """
        Solve the problem using preCICE.
        - Handle checkpointing is implicit coupling is done.
        - Read data from preCICE, solve micro simulations, and write data to preCICE
        - If adaptivity is on, compute micro simulations adaptively.
        """
        self._t = self._n = 0
        mem_usage: list = []
        mem_usage_n = []

        process = Process()

        micro_sim_solve = self._get_solve_variant()
        # call _solve_micro_simulations or _solve_micro_simulations_with_adaptivity internally
        # should use ModelAdaptivity methods to coordinate

        first_iteration = True
        lb_counter = -1

        if self._is_adaptivity_on:
            # Log initial adaptivity metrics
            self._adaptivity_controller.log_metrics(self._n)

        while self._coupling.is_ongoing():
            dt = self._coupling.dt

            if self._is_adaptivity_on:
                if (self._adaptivity_in_every_implicit_step or first_iteration) and (
                    self._n % self._adaptivity_n == 0
                ):
                    with self._profiler.measure(
                        "micro_manager.solve.adaptivity_computation"
                    ):
                        self._adaptivity_controller.compute_adaptivity(
                            dt,
                            self._data_for_adaptivity,
                        )
                        for (
                            gid
                        ) in self._adaptivity_controller.get_active_sim_global_ids():
                            self._micro_sims_active_steps[gid] += 1

                        # Write a checkpoint if a simulation is just activated.
                        # This checkpoint will be asynchronous to the checkpoints written at the start of the time window.
                        self._sim_container.write_checkpoints(only_none=True)

            # handle load balancing, in first iteration all sims are assumed to have same cost
            performed_lb = False
            if lb_counter % self._load_balancing_n == 0 or first_iteration:
                # self._participant.start_profiling_section("micro_manager.solve.load_balancing")
                self.load_balancing.balance()

                # Reset simulation state checkpoints after load balancing
                self._sim_container.clear_checkpoints()
                if self._is_adaptivity_on:
                    for name in self._adaptivity_data_names:
                        self._data_for_adaptivity[name] = [
                            0
                        ] * self._sim_container.local_num_sims
                # Reset simulation crash state information after load balancing
                self._has_sim_crashed = [False] * self._sim_container.local_num_sims
                # self._participant.stop_last_profiling_section()
                performed_lb = True
            lb_counter += 1

            if self._coupling.requires_writing_checkpoint() or performed_lb:
                self._sim_container.write_checkpoints()

            read_buffer = {} if self._is_adaptivity_on else None
            micro_sims_input = self._coupling.read_from_precice(read_buffer=read_buffer)
            # TODO: will be removed in adaptivity refactoring
            if self._is_adaptivity_on and read_buffer is not None:
                for name in read_buffer.keys():
                    if name in self._adaptivity_macro_data_names:
                        self._data_for_adaptivity[name] = read_buffer[name]

            with self._profiler.measure("micro_manager.solve.solve_micro_simulations"):
                micro_sims_output = micro_sim_solve(micro_sims_input, dt)
                self.load_balancing.update()

            if self._is_load_balancing:
                for i in range(self._sim_container.local_num_sims):
                    micro_sims_output[i]["rank_of_sim"] = self._mpi.rank

            # Check if more than a certain percentage of the micro simulations have crashed and terminate if threshold is exceeded
            if self._interpolate_crashed_sims:
                crashed_sims_on_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
                self._mpi.comm.Allgather(
                    np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
                )

                if self._mpi.is_parallel():
                    crash_ratio = (
                        np.sum(crashed_sims_on_all_ranks)
                        / self._sim_container.global_num_sims
                    )
                else:
                    crash_ratio = np.sum(self._has_sim_crashed) / len(
                        self._has_sim_crashed
                    )

                if crash_ratio > self._crash_threshold:
                    self._logger.log_info(
                        "{:.1%} of the micro simulations have crashed exceeding the threshold of {:.1%}. "
                        "Exiting simulation.".format(crash_ratio, self._crash_threshold)
                    )
                    sys.exit()

            self._coupling.write_to_precice(micro_sims_output)
            self._coupling.advance(dt)

            # Revert micro simulations to their last checkpoints if required
            if self._coupling.requires_reading_checkpoint():
                self._sim_container.load_checkpoints()
                first_iteration = False

            # Time window has converged, now micro output can be generated
            if self._coupling.is_time_window_complete():
                self._t += dt  # Update time to the end of the time window
                self._n += 1  # Update time step to the end of the time window

                if self._micro_sims_have_output:
                    if self._n % self._micro_n_out == 0:
                        for lid in self._sim_container.range_lid:
                            sim = self._sim_container[lid]
                            if sim is not None:
                                sim.output()

                if (
                    self._is_adaptivity_on
                    and self._adaptivity_output_type
                    and (self._n % self._adaptivity_output_n == 0)
                ):
                    self._adaptivity_controller.log_metrics(self._n)

                if self._memory_usage_output_type and (
                    self._n % self._memory_usage_output_n == 0 or self._n == 1
                ):
                    mem_usage.append(process.memory_info().rss / 1024**2)
                    mem_usage_n.append(self._n)

                self._logger.log_info_rank_zero(
                    "Time window {} converged.".format(self._n)
                )

                # Reset first iteration flag for the next time window
                first_iteration = True

        # Final memory usage logging at the end of the simulation if not already logged at the end of the last time window
        if (
            self._memory_usage_output_type
            and self._n % self._memory_usage_output_n != 0
        ):
            mem_usage.append(process.memory_info().rss / 1024**2)
            mem_usage_n.append(self._n)

        # Final adaptivity metrics logging at the end of the simulation if not already logged at the end of the last time window
        if (
            self._is_adaptivity_on
            and self._adaptivity_output_type
            and self._n % self._adaptivity_output_n != 0
        ):
            self._adaptivity_controller.log_metrics(self._n)

        if (
            self._memory_usage_output_type == "all"
            or self._memory_usage_output_type == "local"
        ):
            mem_usage_output_file = (
                self._output_dir + "peak_mem_usage_" + str(self._mpi.rank) + ".csv"
            )
            with open(mem_usage_output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time window", "RSS (MB)"])
                for i, rss_mb in enumerate(mem_usage):
                    writer.writerow([mem_usage_n[i], rss_mb])

        if (
            self._memory_usage_output_type == "all"
            or self._memory_usage_output_type == "global"
        ):
            mem_usage = np.array(
                mem_usage
            )  # Convert to numpy array for collective Gather operation
            global_mem_usage = None
            if self._mpi.rank == 0:
                global_mem_usage = np.empty(
                    [self._mpi.size, len(mem_usage)], dtype=np.float64
                )

            self._mpi.comm.Gather(mem_usage, global_mem_usage, root=0)

            if self._mpi.rank == 0:
                avg_mem_usage = np.zeros((len(mem_usage)))
                for t in range(len(mem_usage)):
                    rank_wise_mem_usage = 0
                    for r in range(self._mpi.size):
                        rank_wise_mem_usage += global_mem_usage[r][t]
                    avg_mem_usage[t] = rank_wise_mem_usage / self._mpi.size

                mem_usage_output_file = (
                    self._output_dir + "global_avg_peak_mem_usage.csv"
                )
                with open(mem_usage_output_file, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Time window", "RSS (MB)"])
                    for i, rss_mb in enumerate(avg_mem_usage):
                        writer.writerow([mem_usage_n[i], rss_mb])

        if self._conn is not None:
            self._conn.close()
        self._coupling.finalize()

    def initialize(self) -> None:
        """
        Initialize the Micro Manager by performing the following tasks:
        - Decompose the domain if the Micro Manager is executed in parallel.
        - Initialize preCICE.
        - Gets the macro mesh information from preCICE.
        - Create all micro simulation objects and initialize them if an initialize() method is available.
        - If required, write initial data to preCICE.
        """
        with self._profiler.measure("micro_manager.initialize.direct_access"):
            # Decompose the macro-domain and set the mesh access region for each partition in preCICE
            domain_decomposer: DomainDecomposer = create_domain_decomposer(
                self._config, self._mpi, self._logger
            )
            access_region = self._config.macro_domain_bounds()
            if not self._is_load_balancing:
                access_region = domain_decomposer.get_mesh_bounds()
            self._coupling.set_access_region(access_region)

        # initialize preCICE
        self._coupling.initialize()

        self._profiler.begin("micro_manager.initialize.initialize_micro_sims")
        self._coupling.load_access_region()

        local_macro_coords, local_macro_ids = domain_decomposer.partition(
            self._coupling.registered_vertex_coords,
            self._coupling.registered_vertex_ids,
            access_region,
        )
        local_num_sims, global_num_sims, sims_per_rank = domain_decomposer.finalize(
            local_macro_coords
        )

        if local_num_sims == 0 and self._mpi.is_parallel() and self._lazy_init:
            raise Exception(
                "The macro mesh has no vertices in the specified access region, "
                "but lazy initialization is turned on. Lazy initialization cannot be used "
                "if there are no vertices in the access region, as there would be no data to compute "
                "the adaptivity and determine which simulations to initialize."
            )

        if self._is_adaptivity_on:
            for name in self._adaptivity_data_names:
                self._data_for_adaptivity[name] = [0] * local_num_sims

        # Create lists of global IDs
        local_gids = self._coupling.generate_gids(
            local_macro_coords,
            local_macro_ids,
            sims_per_rank,
        )

        self._sim_container.initialize(
            glob_num_sims=global_num_sims,
            local_num_sims=local_num_sims,
            local_gids=local_gids,
            local_coords=local_macro_coords,
        )

        # Setup for simulation crashes
        self._has_sim_crashed = [False] * self._sim_container.local_num_sims
        if self._interpolate_crashed_sims:
            self._interpolant = Interpolation(self._logger)

        # Setup remote workers
        base_dir = os.path.dirname(os.path.abspath(__file__))
        worker_exec = os.path.join(base_dir, "tasking", "worker_main.py")
        num_ranks = self._config.tasking_num_workers()
        self._conn = spawn_local_workers(
            worker_exec,
            num_ranks,
            self._config.tasking_backend(),
            self._config.enable_tasking_slurm(),
            self._config.mpi_impl(),
            self._config.tasking_hostfile(),
        )

        # load micro sim
        self._model_manager.load_models(self._config, num_ranks, self._conn)
        micro_problem_cls = self._model_manager.get_cls_by_idx(0)

        if self._is_model_adaptivity_on:
            self._model_adaptivity_controller: ModelAdaptivity = ModelAdaptivity(
                self._model_manager,
                self._sim_container,
                self._config,
                self._mpi,
                self._log_file,
            )

        # Create micro simulation objects
        if not self._lazy_init:
            for lid in self._sim_container.range_lid:
                self._sim_container[lid] = self._model_manager.get_instance(
                    self._sim_container.local_gids[lid], micro_problem_cls
                )

        if self._is_adaptivity_on:
            self._adaptivity_controller = create_adaptivity_calculator(
                self._config,
                self._sim_container,
                self._profiler,
                self._logger,
                self._mpi,
                micro_problem_cls,
                self._model_manager,
            )

            self._micro_sims_active_steps = np.zeros(
                self._sim_container.global_num_sims
            )  # DECLARATION

        self._micro_sims_init = False  # DECLARATION

        # Read initial data from preCICE, if it is available
        read_buffer = {} if self._is_adaptivity_on else None
        initial_macro_data = self._coupling.read_from_precice(0, read_buffer)
        # TODO: will be removed in adaptivity refactoring
        if self._is_adaptivity_on and read_buffer is not None:
            for name in read_buffer.keys():
                if name in self._adaptivity_macro_data_names:
                    self._data_for_adaptivity[name] = read_buffer[name]

        first_id = 0  # 0 if lazy initialization is off, otherwise the first active simulation ID
        micro_sims_to_init = range(
            1, self._sim_container.local_num_sims
        )  # All sims if lazy init is off, otherwise all active simulations

        # Additional bool to check if there are sims to init
        are_there_sims_to_init = True

        if not initial_macro_data:
            is_initial_data_available = False

            if self._lazy_init:
                raise Exception(
                    "Initial macro data is required for lazy initialization."
                )
        else:
            is_initial_data_available = True

            # For lazy initialization, compute adaptivity with the initial macro data
            if self._lazy_init:
                for i in self._sim_container.range_lid:
                    for name in self._adaptivity_macro_data_names:
                        self._data_for_adaptivity[name][i] = initial_macro_data[i][name]

                self._adaptivity_controller.compute_adaptivity(
                    self._coupling.micro_dt, self._data_for_adaptivity
                )

                active_sim_lids = self._adaptivity_controller.get_active_sim_local_ids()

                if active_sim_lids.size == 0:
                    self._logger.log_info(
                        "There are no active simulations on this rank."
                    )
                    micro_sims_to_init = []
                    are_there_sims_to_init = False
                else:
                    for i in active_sim_lids:
                        self._sim_container[i] = self._model_manager.get_instance(
                            self._sim_container.local_gids[i], micro_problem_cls
                        )

                    first_id = active_sim_lids[0]  # First active simulation ID
                    micro_sims_to_init = (
                        active_sim_lids  # Only active simulations will be initialized
                    )
                    are_there_sims_to_init = True

        test_instance = self._model_manager.get_instance(
            self._sim_container.global_num_sims + 1, micro_problem_cls
        )
        test_data = None
        if is_initial_data_available:
            test_data = initial_macro_data[0]
        (
            self._micro_sims_init,
            sim_requires_init_data,
        ) = micro_problem_cls.check_initialize(
            test_instance,
            test_data,
        )
        test_instance.destroy()
        del test_instance

        if sim_requires_init_data and not is_initial_data_available:
            raise Exception(
                "The initialize() method of the Micro simulation requires initial data, but no initial macro data has been provided."
            )

        initial_micro_data: dict[str, list] = dict()

        if are_there_sims_to_init and self._micro_sims_init:
            # Call initialize() method of the micro simulation to check if it returns any initial data
            if sim_requires_init_data:
                initial_micro_output = self._sim_container[first_id].initialize(
                    initial_macro_data[first_id]
                )
            else:
                initial_micro_output = self._sim_container[first_id].initialize()

            if initial_micro_output is None:
                self._logger.log_warning_rank_zero(
                    "The initialize() call of the Micro simulation has not returned any initial data."
                    " This means that the initialize() call has no effect on the adaptivity. The initialize method will nevertheless still be called."
                )
                self._micro_sims_init = False

                if sim_requires_init_data:
                    for i in micro_sims_to_init:
                        self._sim_container[i].initialize(initial_macro_data[i])
                else:
                    for i in micro_sims_to_init:
                        self._sim_container[i].initialize()
            else:  # Case where the initialize() method returns data
                if self._is_adaptivity_on:
                    # Check for missing data
                    expected, provided = set(self._adaptivity_micro_data_names), set(
                        initial_micro_output.keys()
                    )
                    if missing := expected - provided:
                        raise Exception(
                            "The initialize() method needs to return data which is required for the adaptivity calculation. "
                            f'Of the expected data {", ".join(expected)}, the following is missing: {", ".join(missing)}'
                        )
                    elif extra := provided - set(
                        self._adaptivity_macro_data_names
                        + self._adaptivity_micro_data_names
                    ):
                        self._logger.log_warning_rank_zero(
                            f'The initialize() method of the Micro simulation returns extra initial data which isn\'t used by the adaptivity: {", ".join(extra)}'
                        )

                    for name in initial_micro_output.keys():
                        initial_micro_data[name] = [
                            0
                        ] * self._sim_container.local_num_sims
                        # Save initial data from first micro simulation as we anyway have it
                        initial_micro_data[name][first_id] = initial_micro_output[name]

                    # Save initial data from first micro simulation as we anyway have it
                    for name in initial_micro_output.keys():
                        if name in self._data_for_adaptivity:
                            self._data_for_adaptivity[name][
                                first_id
                            ] = initial_micro_output[name]

                    # Gather initial data from the rest of the micro simulations
                    if sim_requires_init_data:
                        for i in micro_sims_to_init:
                            initial_micro_output = self._sim_container[i].initialize(
                                initial_macro_data[i]
                            )
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                                initial_micro_data[name][i] = initial_micro_output[name]
                    else:
                        for i in micro_sims_to_init:
                            initial_micro_output = self._sim_container[i].initialize()
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                                initial_micro_data[name][i] = initial_micro_output[name]
                else:  # If adaptivity is off, the returned initial data from the initialize() method will be ignored
                    self._logger.log_warning_rank_zero(
                        "The initialize() method of the Micro simulation returns initial data, but adaptivity is turned off. The returned data will be ignored. The initialize method will nevertheless still be called."
                    )
                    if sim_requires_init_data:
                        for i in range(1, self._sim_container.local_num_sims):
                            self._sim_container[i].initialize(initial_macro_data[i])
                    else:
                        for i in range(1, self._sim_container.local_num_sims):
                            self._sim_container[i].initialize()

        self._micro_sims_have_output = micro_problem_cls.check_output()

        self.load_balancing = create_load_balancer(
            self._profiler,
            self._model_manager,
            self._adaptivity_controller if self._is_adaptivity_on else None,
            self._sim_container,
            self._logger,
            self._config,
            self._mpi,
        )

        # If lazy initialization is on, initial states of inactive simulations need to be determined
        if self._lazy_init:
            # If there is initial micro data, and if this rank has sims to init, then the data is to be gathered
            if initial_micro_data and are_there_sims_to_init:
                initial_micro_data_list: list[dict] = [
                    dict(zip(initial_micro_data, t))
                    for t in zip(*initial_micro_data.values())
                ]
            else:
                # Ranks without active simulations provide empty dicts
                initial_micro_data_list: list[dict] = [
                    dict() for _ in range(self._sim_container.local_num_sims)
                ]

            initial_micro_data_list = (
                self._adaptivity_controller.get_full_field_micro_output(
                    initial_macro_data, initial_micro_data_list
                )
            )

            for i in range(self._sim_container.local_num_sims):
                for name in self._adaptivity_micro_data_names:
                    self._data_for_adaptivity[name][i] = initial_micro_data_list[i][
                        name
                    ]

        self._profiler.end()

    # ***************
    # Private methods
    # ***************

    def _solve_micro_simulations(
        self, micro_sims_input: list, dt: float, computed_outputs: dict = {}
    ) -> list:
        """
        Solve all micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        dt : float
            Time step size.
        computed_outputs : dict
            Dictionary of global ids to already computed outputs

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data which are required outputs of
        """
        micro_sims_output: list[dict] = [None] * self._sim_container.local_num_sims

        for lid in self._sim_container.range_lid:
            sim = self._sim_container[lid]
            # skip already computed outputs
            gid = self._sim_container.local_gids[lid]
            if gid in computed_outputs:
                micro_sims_output[lid] = computed_outputs[gid]
                continue

            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[lid]:
                # Attempt to solve the micro simulation
                try:
                    self.load_balancing.pre_sim_solve(gid)
                    micro_sims_output[lid] = sim.solve(micro_sims_input[lid], dt)
                    self.load_balancing.post_sim_solve(gid)

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.log_error(
                        "Micro simulation at macro coordinates {} with input {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._sim_container.local_coords[lid], micro_sims_input[lid]
                        )
                    )
                    self._logger.log_error(error_message)
                    self._has_sim_crashed[lid] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
            self._mpi.comm.Allgather(
                np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
            )
            if sum(crashed_sims_on_all_ranks) > 0:
                self._logger.log_info(
                    "Exiting simulation after micro simulation crash."
                )
                sys.exit()

        # Interpolate result for crashed simulation
        unset_sims = [
            lid for lid, value in enumerate(micro_sims_output) if value is None
        ]

        # Iterate over all crashed simulations to interpolate output
        if self._interpolate_crashed_sims:
            for lid in unset_sims:
                self._logger.log_info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._sim_container.local_coords[lid]
                    )
                )
                micro_sims_output[lid] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, lid
                )

        return micro_sims_output

    def _solve_micro_simulations_with_adaptivity(
        self, micro_sims_input: list, dt: float, computed_outputs: dict = {}
    ) -> list:
        """
        Adaptively solve micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        dt : float
            Time step size.
        computed_outputs : dict
            Dictionary of global ids to already computed outputs

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data which are required outputs of
        """
        active_sim_lids = self._adaptivity_controller.get_active_sim_local_ids()

        micro_sims_output = [0] * self._sim_container.local_num_sims

        # Solve all active micro simulations
        for lid in active_sim_lids:
            # skip already computed outputs
            gid = self._sim_container.local_gids[lid]
            if gid in computed_outputs:
                micro_sims_output[lid] = computed_outputs[gid]
                continue

            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[lid]:
                try:
                    self.load_balancing.pre_sim_solve(gid)
                    micro_sims_output[lid] = self._sim_container[lid].solve(
                        micro_sims_input[lid], dt
                    )
                    self.load_balancing.post_sim_solve(gid)

                    # Mark the micro sim as active for export
                    micro_sims_output[lid]["Active-State"] = 1
                    micro_sims_output[lid][
                        "Active-Steps"
                    ] = self._micro_sims_active_steps[gid]

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.log_error(
                        "Micro simulation at macro coordinates {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._sim_container.local_coords[lid]
                        )
                    )
                    self._logger.log_error(error_message)
                    self._has_sim_crashed[lid] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._mpi.size, dtype=np.int64)
            self._mpi.comm.Allgather(
                np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
            )
            if sum(crashed_sims_on_all_ranks) > 0:
                self._logger.log_error(
                    "Exiting simulation after micro simulation crash."
                )
                sys.exit()

        # Interpolate result for crashed simulation
        unset_sims = []
        for lid in active_sim_lids:
            if micro_sims_output[lid] == 0:
                unset_sims.append(lid)

        # Iterate over all crashed simulations to interpolate output
        if self._interpolate_crashed_sims:
            for lid in unset_sims:
                self._logger.log_info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._sim_container.local_coords[lid]
                    )
                )

                micro_sims_output[lid] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, lid, active_sim_lids
                )

        micro_sims_output = self._adaptivity_controller.get_full_field_micro_output(
            micro_sims_input, micro_sims_output
        )

        inactive_sim_lids = self._adaptivity_controller.get_inactive_sim_local_ids()

        # Resolve micro sim output data for inactive simulations
        for inactive_lid in inactive_sim_lids:
            self.load_balancing.pre_sim_solve(
                self._sim_container.local_gids[inactive_lid]
            )
            micro_sims_output[inactive_lid]["Active-State"] = 0
            gid = self._sim_container.local_gids[inactive_lid]
            micro_sims_output[inactive_lid][
                "Active-Steps"
            ] = self._micro_sims_active_steps[gid]
            self.load_balancing.post_sim_solve(
                self._sim_container.local_gids[inactive_lid]
            )

        # Collect micro sim output for adaptivity calculation
        for lid in self._sim_container.range_lid:
            for name in self._adaptivity_micro_data_names:
                self._data_for_adaptivity[name][lid] = micro_sims_output[lid][name]

        return micro_sims_output

    def _solve_micro_simulations_with_model_adaptivity(
        self, micro_sims_input: list, dt: float, solve_variant: Callable
    ) -> list:
        self._model_adaptivity_controller.initialise_solve()

        active_sim_ids = None
        if self._is_adaptivity_on:
            active_sim_ids = self._adaptivity_controller.get_active_sim_local_ids()
        output = None

        while self._model_adaptivity_controller.should_iterate():
            switched_lids = self._model_adaptivity_controller.switch_models(
                self._t,
                micro_sims_input,
                output,
                active_sim_ids,
            )
            computed_outputs = {}
            if output is not None:
                for lid, out in enumerate(output):
                    if lid in switched_lids:
                        continue
                    gid = self._sim_container.local_gids[int(lid)]
                    computed_outputs[gid] = out
            output = solve_variant(micro_sims_input, dt, computed_outputs)
            self._model_adaptivity_controller.check_convergence(
                self._t,
                micro_sims_input,
                output,
                active_sim_ids,
            )

        self._model_adaptivity_controller.finalise_solve()

        for lid in self._sim_container.range_lid:
            sim = self._sim_container[lid]
            res = -1
            if sim is not None:
                res = self._model_manager.get_idx_of_sim(sim)
            output[lid]["model_resolution"] = res
        return output

    def _get_solve_variant(self) -> Callable[[list, float], list]:
        """
        Get the solve variant function based on the adaptivity type.

        Returns
        -------
        solve_variant : Callable
            Solve variant function based on the adaptivity type.
        """
        if self._is_adaptivity_on:
            solve_variant = self._solve_micro_simulations_with_adaptivity
        else:
            solve_variant = self._solve_micro_simulations

        if self._is_model_adaptivity_on:
            return partial(
                self._solve_micro_simulations_with_model_adaptivity,
                solve_variant=solve_variant,
            )
        else:
            return solve_variant

    def _interpolate_output_for_crashed_sim(
        self,
        micro_sims_input: list,
        micro_sims_output: list,
        unset_sim_lid: int,
        active_sim_ids: np.ndarray = None,
    ) -> dict:
        """
        Using the output of neighboring simulations, interpolate the output for a crashed simulation.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        micro_sims_output : list
            List dicts containing output of local micro simulations.
        unset_sim_lid : int
            Index of the crashed simulation in the list of all local simulations currently interpolating.
        active_sim_ids : numpy.ndarray, optional
            Array of active simulation IDs.

        Returns
        -------
        output_interpol : dict
            Result of the interpolation in which keys are names of data and the values are the data.
        """
        # Find neighbors of the crashed simulation in active and non-crashed simulations
        # Set iteration length to only iterate over active simulations
        if self._is_adaptivity_on:
            iter_length = active_sim_ids
        else:
            iter_length = range(len(micro_sims_input))
        micro_sims_active_input_lists = []
        micro_sims_active_values = []
        # Turn crashed simulation macro parameters into list to use as coordinate for interpolation
        crashed_position = []
        for value in micro_sims_input[unset_sim_lid].values():
            if isinstance(value, np.ndarray) or isinstance(value, list):
                crashed_position.extend(value)
            else:
                crashed_position.append(value)
        # Turn active simulation macro parameters into lists to use as coordinates for interpolation based on parameters
        for i in iter_length:
            if not self._has_sim_crashed[i]:
                # Collect macro data at one macro vertex
                intermediate_list = []
                for value in micro_sims_input[i].values():
                    if isinstance(value, np.ndarray) or isinstance(value, list):
                        intermediate_list.extend(value)
                    else:
                        intermediate_list.append(value)
                # Create lists of macro data for interpolation
                micro_sims_active_input_lists.append(intermediate_list)
                micro_sims_active_values.append(micro_sims_output[i].copy())
        # Find nearest neighbors
        if len(micro_sims_active_input_lists) == 0:
            self._logger.log_error(
                "No active neighbors available for interpolation at macro vertex {}. Value cannot be interpolated".format(
                    self._sim_container.local_coords[unset_sim_lid]
                )
            )
            return None
        else:
            nearest_neighbors = self._interpolant.get_nearest_neighbor_indices(
                micro_sims_active_input_lists,
                crashed_position,
                self._number_of_nearest_neighbors,
            )
        # Interpolate
        interpol_space = []
        interpol_values = []
        # Collect neighbor vertices for interpolation
        for neighbor in nearest_neighbors:
            # Remove data not required for interpolation from values
            if self._is_adaptivity_on:
                interpol_space.append(micro_sims_active_input_lists[neighbor].copy())
                interpol_values.append(micro_sims_active_values[neighbor].copy())
                interpol_values[-1].pop("Active-State", None)
                interpol_values[-1].pop("Active-Steps", None)
            else:
                interpol_space.append(micro_sims_active_input_lists[neighbor].copy())
                interpol_values.append(micro_sims_active_values[neighbor].copy())

        # Interpolate for each parameter
        output_interpol = dict()
        for key in interpol_values[0].keys():
            key_values = []  # DECLARATION
            # Collect values of current parameter from neighboring simulations
            for elems in range(len(interpol_values)):
                key_values.append(interpol_values[elems][key])
            output_interpol[key] = self._interpolant.interpolate(
                interpol_space, crashed_position, key_values
            )
        # Reintroduce removed information
        if self._is_adaptivity_on:
            output_interpol["Active-State"] = 1
            output_interpol["Active-Steps"] = self._micro_sims_active_steps[
                unset_sim_lid
            ]
        return output_interpol
