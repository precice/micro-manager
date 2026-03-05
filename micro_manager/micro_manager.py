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

import importlib
import os
import sys
import inspect
from typing import Callable
import numpy as np
from psutil import Process
import csv
import subprocess
from functools import partial

import precice

from .model_manager import ModelManager
from .micro_manager_base import MicroManager

from .adaptivity.model_adaptivity import ModelAdaptivity
from .adaptivity.adaptivity_selection import create_adaptivity_calculator

from .domain_decomposition import DomainDecomposer
from .tasking.connection import spawn_local_workers
from .micro_simulation import create_simulation_class, load_backend_class
from .tools.logging_wrapper import Logger

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

        self._log_file = log_file
        self._logger = Logger(__name__, log_file, self._rank)

        self._config.set_logger(self._logger)
        self._config.read_json_micro_manager()

        self._memory_usage_output_type = self._config.get_memory_usage_output_type()

        self._memory_usage_output_n = self._config.get_memory_usage_output_n()

        self._output_dir = self._config.get_output_dir()

        if self._output_dir is not None:
            self._output_dir = os.path.abspath(self._output_dir) + "/"
            subprocess.run(["mkdir", "-p", self._output_dir])  # Create output directory
        else:
            self._output_dir = os.path.abspath(os.getcwd()) + "/"

        # Data names of data to output to the snapshot database
        self._write_data_names = self._config.get_write_data_names()

        # Data names of data to read as input parameter to the simulations
        self._read_data_names = self._config.get_read_data_names()

        self._micro_dt = self._config.get_micro_dt()

        self._macro_mesh_name = self._config.get_macro_mesh_name()

        self._macro_bounds = self._config.get_macro_domain_bounds()

        if self._is_parallel:  # Simulation is run in parallel
            self._ranks_per_axis = self._config.get_ranks_per_axis()

        # Parameter for interpolation in case of a simulation crash
        self._interpolate_crashed_sims = self._config.interpolate_crashed_micro_sim()
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

        self._micro_n_out = self._config.get_micro_output_n()

        self._lazy_init = self._config.initialize_sims_lazily()

        self._is_adaptivity_on = self._config.turn_on_adaptivity()

        self._is_adaptivity_with_load_balancing = (
            self._config.is_adaptivity_with_load_balancing()
        )

        if self._is_adaptivity_on:
            self._data_for_adaptivity: dict[str, list] = dict()

            self._adaptivity_data_names = self._config.get_data_for_adaptivity()

            # Names of macro data to be used for adaptivity computation
            self._adaptivity_macro_data_names: list = []

            # Names of micro data to be used for adaptivity computation
            self._adaptivity_micro_data_names: list = []
            for name in self._adaptivity_data_names:
                if name in self._read_data_names:
                    self._adaptivity_macro_data_names.append(name)
                if name in self._write_data_names:
                    self._adaptivity_micro_data_names.append(name)

            self._adaptivity_in_every_implicit_step = (
                self._config.is_adaptivity_required_in_every_implicit_iteration()
            )

            if self._is_adaptivity_with_load_balancing:
                self._load_balancing_n = self._config.get_load_balancing_n()

        self._adaptivity_n = self._config.get_adaptivity_n()

        self._adaptivity_output_type = self._config.get_adaptivity_output_type()

        self._adaptivity_output_n = self._config.get_adaptivity_output_n()

        self._is_model_adaptivity_on = self._config.turn_on_model_adaptivity()

        # Define the preCICE Participant
        self._participant = precice.Participant(
            "Micro-Manager",
            self._config.get_precice_config_file_name(),
            self._rank,
            self._size,
        )

        self._t = 0  # global time
        self._n = 0  # sim-step

        self._model_manager = ModelManager()
        self._conn = None

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
        sim_states_cp = [None] * self._local_number_of_sims
        mem_usage: list = []
        mem_usage_n = []

        process = Process()

        micro_sim_solve = self._get_solve_variant()
        state_loader = lambda sim: sim.get_state()
        state_setter = lambda sim, state: sim.set_state(state)
        if self._is_model_adaptivity_on:
            state_loader = lambda sim: sim.attachments
            state_setter = lambda sim, state: sim.attachments.update(state)
        # TODO adapt variant to also use special mada case, iterates in itself and will prob
        # call _solve_micro_simulations or _solve_micro_simulations_with_adaptivity internally
        # should use ModelAdaptivity methods to coordinate

        dt = min(self._participant.get_max_time_step_size(), self._micro_dt)

        first_iteration = True

        if self._is_adaptivity_on:
            # Log initial adaptivity metrics
            self._adaptivity_controller.log_metrics(self._n)

        while self._participant.is_coupling_ongoing():

            dt = min(self._participant.get_max_time_step_size(), self._micro_dt)

            if self._is_adaptivity_on:
                if (self._adaptivity_in_every_implicit_step or first_iteration) and (
                    self._n % self._adaptivity_n == 0
                ):
                    self._participant.start_profiling_section(
                        "micro_manager.solve.adaptivity_computation"
                    )

                    self._adaptivity_controller.compute_adaptivity(
                        dt,
                        self._micro_sims,
                        self._data_for_adaptivity,
                    )
                    active_sim_gids = (
                        self._adaptivity_controller.get_active_sim_global_ids()
                    )
                    for gid in active_sim_gids:
                        self._micro_sims_active_steps[gid] += 1

                    # Write a checkpoint if a simulation is just activated.
                    # This checkpoint will be asynchronous to the checkpoints written at the start of the time window.
                    if self._is_model_adaptivity_on:
                        self._model_adaptivity_controller.update_states(
                            self._micro_sims, active_sim_gids
                        )
                    for i in range(self._local_number_of_sims):
                        if sim_states_cp[i] is None and self._micro_sims[i]:
                            sim_states_cp[i] = state_loader(self._micro_sims[i])

                    self._participant.stop_last_profiling_section()

            if self._is_adaptivity_with_load_balancing:
                if self._n % self._load_balancing_n == 0 and first_iteration:
                    self._participant.start_profiling_section(
                        "micro_manager.solve.load_balancing"
                    )

                    self._adaptivity_controller.redistribute_sims(self._micro_sims)

                    self._local_number_of_sims = len(self._global_ids_of_local_sims)

                    # Reset simulation state checkpoints after load balancing
                    sim_states_cp = [None] * self._local_number_of_sims

                    for name in self._adaptivity_data_names:
                        self._data_for_adaptivity[name] = [
                            0
                        ] * self._local_number_of_sims

                    # Reset simulation crash state information after load balancing
                    self._has_sim_crashed = [False] * self._local_number_of_sims

                    self._participant.stop_last_profiling_section()

            # Write a checkpoint
            if self._participant.requires_writing_checkpoint():
                if self._is_model_adaptivity_on:
                    active_sim_gids = None
                    if self._is_adaptivity_on:
                        active_sim_gids = (
                            self._adaptivity_controller.get_active_sim_local_ids()
                        )
                    self._model_adaptivity_controller.update_states(
                        self._micro_sims, active_sim_gids
                    )
                for i in range(self._local_number_of_sims):
                    sim_states_cp[i] = (
                        state_loader(self._micro_sims[i])
                        if self._micro_sims[i]
                        else None
                    )

            micro_sims_input = self._read_data_from_precice(dt)

            self._participant.start_profiling_section(
                "micro_manager.solve.solve_micro_simulations"
            )

            micro_sims_output = micro_sim_solve(micro_sims_input, dt)

            self._participant.stop_last_profiling_section()

            if self._is_adaptivity_with_load_balancing:
                for i in range(self._local_number_of_sims):
                    micro_sims_output[i]["Rank"] = self._rank

            # Check if more than a certain percentage of the micro simulations have crashed and terminate if threshold is exceeded
            if self._interpolate_crashed_sims:
                crashed_sims_on_all_ranks = np.zeros(self._size, dtype=np.int64)
                self._comm.Allgather(
                    np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
                )

                if self._is_parallel:
                    crash_ratio = (
                        np.sum(crashed_sims_on_all_ranks) / self._global_number_of_sims
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

            self._write_data_to_precice(micro_sims_output)

            self._participant.advance(dt)

            # Revert micro simulations to their last checkpoints if required
            if self._participant.requires_reading_checkpoint():
                for i in range(self._local_number_of_sims):
                    if self._micro_sims[i]:
                        state_setter(self._micro_sims[i], sim_states_cp[i])

                if self._is_model_adaptivity_on:
                    active_sim_gids = None
                    if self._is_adaptivity_on:
                        active_sim_gids = (
                            self._adaptivity_controller.get_active_sim_local_ids()
                        )
                    self._model_adaptivity_controller.write_back_states(
                        self._micro_sims, active_sim_gids
                    )

                first_iteration = False

            # Time window has converged, now micro output can be generated
            if self._participant.is_time_window_complete():
                self._t += dt  # Update time to the end of the time window
                self._n += 1  # Update time step to the end of the time window

                if self._micro_sims_have_output:
                    if self._n % self._micro_n_out == 0:
                        for sim in self._micro_sims:
                            if sim:
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
                self._output_dir + "peak_mem_usage_" + str(self._rank) + ".csv"
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
            if self._rank == 0:
                global_mem_usage = np.empty(
                    [self._size, len(mem_usage)], dtype=np.float64
                )

            self._comm.Gather(mem_usage, global_mem_usage, root=0)

            if self._rank == 0:
                avg_mem_usage = np.zeros((len(mem_usage)))
                for t in range(len(mem_usage)):
                    rank_wise_mem_usage = 0
                    for r in range(self._size):
                        rank_wise_mem_usage += global_mem_usage[r][t]
                    avg_mem_usage[t] = rank_wise_mem_usage / self._size

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
        self._participant.finalize()

    def initialize(self) -> None:
        """
        Initialize the Micro Manager by performing the following tasks:
        - Decompose the domain if the Micro Manager is executed in parallel.
        - Initialize preCICE.
        - Gets the macro mesh information from preCICE.
        - Create all micro simulation objects and initialize them if an initialize() method is available.
        - If required, write initial data to preCICE.
        """
        self._participant.start_profiling_section(
            "micro_manager.initialize.direct_access"
        )

        # Decompose the macro-domain and set the mesh access region for each partition in preCICE
        if not len(self._macro_bounds) / 2 == self._participant.get_mesh_dimensions(
            self._macro_mesh_name
        ):
            raise Exception("Provided macro mesh bounds are of incorrect dimension")

        if self._is_parallel:
            if not len(self._ranks_per_axis) == self._participant.get_mesh_dimensions(
                self._macro_mesh_name
            ):
                raise Exception(
                    "Provided ranks combination is of incorrect dimension"
                    " and does not match the dimensions of the macro mesh."
                )

            domain_decomposer = DomainDecomposer(self._rank, self._size)

        if self._is_parallel and not self._is_adaptivity_with_load_balancing:
            coupling_mesh_bounds = domain_decomposer.get_local_mesh_bounds(
                self._macro_bounds, self._ranks_per_axis
            )
        else:  # When serial or load balancing, the whole macro domain is assigned to one/each rank
            coupling_mesh_bounds = self._macro_bounds

        self._participant.set_mesh_access_region(
            self._macro_mesh_name, coupling_mesh_bounds
        )

        self._participant.stop_last_profiling_section()

        # initialize preCICE
        self._participant.initialize()

        self._participant.start_profiling_section(
            "micro_manager.initialize.initialize_micro_sims"
        )

        (
            self._mesh_vertex_ids,
            self._mesh_vertex_coords,
        ) = self._participant.get_mesh_vertex_ids_and_coordinates(self._macro_mesh_name)

        if self._is_parallel:
            # Gather all vertex coords and IDs from all ranks onto all ranks,
            # filter out coords already claimed by lower-ranked ranks.
            all_coords = self._comm.allgather(self._mesh_vertex_coords)
            all_ids = self._comm.allgather(self._mesh_vertex_ids)

            (
                self._mesh_vertex_coords,
                self._mesh_vertex_ids,
            ) = domain_decomposer.filter_duplicate_coords(all_coords, all_ids)

        if self._mesh_vertex_coords.size == 0:
            raise Exception("Macro mesh has no vertices.")

        if self._is_adaptivity_with_load_balancing:
            (
                self._local_number_of_sims,
                local_macro_coords,
            ) = domain_decomposer.get_local_sims_and_macro_coords(
                self._macro_bounds, self._ranks_per_axis, self._mesh_vertex_coords
            )
        else:
            self._local_number_of_sims, _ = self._mesh_vertex_coords.shape

        if self._local_number_of_sims == 0:
            if self._is_parallel:
                self._logger.log_info(
                    "Rank {} has no micro simulations and hence will not do any computation.".format(
                        self._rank
                    )
                )
                self._is_rank_empty = True
            else:
                raise Exception("Micro Manager has no micro simulations.")

        nms_all_ranks = np.zeros(self._size, dtype=np.int64)
        # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
        # simulations have been created by previous ranks, so that it can set
        # the correct global IDs
        self._comm.Allgatherv(np.array(self._local_number_of_sims), nms_all_ranks)

        max_nms = np.max(nms_all_ranks)
        min_nms = np.min(nms_all_ranks)

        if (
            max_nms != min_nms
        ):  # if the number of maximum and minimum micro simulations per rank are different
            self._logger.log_info_rank_zero(
                "The following ranks have the maximum number of micro simulations ({}): {}".format(
                    max_nms, np.where(nms_all_ranks == max_nms)[0]
                )
            )
            self._logger.log_info_rank_zero(
                "The following ranks have the minimum number of micro simulations ({}): {}".format(
                    min_nms, np.where(nms_all_ranks == min_nms)[0]
                )
            )
        else:  # if the number of maximum and minimum micro simulations per rank are the same
            self._logger.log_info_rank_zero(
                "All ranks have the same number of micro simulations: {}".format(
                    max_nms
                )
            )

        # Get global number of micro simulations
        self._global_number_of_sims: int = np.sum(nms_all_ranks)

        self._logger.log_info_rank_zero(
            "Total number of micro simulations: {}".format(self._global_number_of_sims)
        )

        if self._is_adaptivity_on:
            for name in self._adaptivity_data_names:
                self._data_for_adaptivity[name] = [0] * self._local_number_of_sims

        # Create lists of global IDs
        self._global_ids_of_local_sims = []  # DECLARATION

        if self._is_adaptivity_with_load_balancing:
            # Create a set of global coordinate indices for faster lookup
            coord_to_index = {
                tuple(coord): i for i, coord in enumerate(self._mesh_vertex_coords)
            }

            # Set global IDs based on the coordinate ordering in preCICE to be consistent with the scenario without load balancing
            for coords in local_macro_coords:
                coord_tuple = tuple(coords)
                self._global_ids_of_local_sims.append(coord_to_index[coord_tuple])
        else:
            sim_id = np.sum(nms_all_ranks[: self._rank])
            for i in range(self._local_number_of_sims):
                self._global_ids_of_local_sims.append(sim_id)
                sim_id += 1

        # Setup for simulation crashes
        self._has_sim_crashed = [False] * self._local_number_of_sims
        if self._interpolate_crashed_sims:
            self._interpolant = Interpolation(self._logger)

        # Setup remote workers
        base_dir = os.path.dirname(os.path.abspath(__file__))
        worker_exec = os.path.join(base_dir, "tasking", "worker_main.py")
        num_ranks = self._config.get_tasking_num_workers()
        self._conn = spawn_local_workers(
            worker_exec,
            num_ranks,
            self._config.get_tasking_backend(),
            self._config.get_tasking_use_slurm(),
            self._config.get_mpi_impl(),
            self._config.get_tasking_hostfile(),
        )

        # load micro sim
        micro_problem_cls = None
        if self._is_model_adaptivity_on:
            self._model_adaptivity_controller: ModelAdaptivity = ModelAdaptivity(
                self._model_manager,
                self._config,
                self._rank,
                self._log_file,
                self._conn,
                num_ranks,
            )
            micro_problem_cls = (
                self._model_adaptivity_controller.get_resolution_sim_class(0)
            )
        else:
            micro_problem_base = load_backend_class(self._config.get_micro_file_name())
            micro_problem_cls = create_simulation_class(
                self._logger,
                micro_problem_base,
                self._config.get_micro_file_name(),
                self._config.get_tasking_num_workers(),
                self._conn,
                "MicroSimulationDefault",
            )
            self._model_manager.register(
                micro_problem_cls, self._config.turn_on_micro_stateless()
            )

        # Create micro simulation objects
        self._micro_sims = [0] * self._local_number_of_sims
        if not self._lazy_init:
            for i in range(self._local_number_of_sims):
                self._micro_sims[i] = self._model_manager.get_instance(
                    self._global_ids_of_local_sims[i], micro_problem_cls
                )

        if self._is_adaptivity_on:
            self._adaptivity_controller = create_adaptivity_calculator(
                self._config,
                self._local_number_of_sims,
                self._global_number_of_sims,
                self._global_ids_of_local_sims,
                self._participant,
                self._logger,
                self._rank,
                self._comm,
                micro_problem_cls,
                self._model_manager,
                self._is_adaptivity_with_load_balancing,
            )

            self._micro_sims_active_steps = np.zeros(
                self._global_number_of_sims
            )  # DECLARATION

        self._micro_sims_init = False  # DECLARATION

        # Read initial data from preCICE, if it is available
        initial_data = self._read_data_from_precice(dt=0)

        first_id = 0  # 0 if lazy initialization is off, otherwise the first active simulation ID
        micro_sims_to_init = range(
            1, self._local_number_of_sims
        )  # All sims if lazy init is off, otherwise all active simulations

        if not initial_data:
            is_initial_data_available = False
        else:
            is_initial_data_available = True
            # For lazy initialization, compute adaptivity with the initial macro data
            if self._lazy_init:
                for i in range(self._local_number_of_sims):
                    for name in self._adaptivity_macro_data_names:
                        self._data_for_adaptivity[name][i] = initial_data[i][name]

                self._adaptivity_controller.compute_adaptivity(
                    self._micro_dt, self._micro_sims, self._data_for_adaptivity
                )

                active_sim_lids = self._adaptivity_controller.get_active_sim_local_ids()

                if active_sim_lids.size == 0:
                    self._logger.log_info(
                        "There are no active simulations on this rank."
                    )
                    return

                for i in active_sim_lids:
                    self._micro_sims[i] = self._model_manager.get_instance(
                        self._global_ids_of_local_sims[i], micro_problem_cls
                    )

                first_id = active_sim_lids[0]  # First active simulation ID
                micro_sims_to_init = (
                    active_sim_lids  # Only active simulations will be initialized
                )

        # Boolean which states if the initialize() method of the micro simulation requires initial data
        (
            self._micro_sims_init,
            sim_requires_init_data,
        ) = micro_problem_cls.check_initialize(
            self._micro_sims[first_id],
            initial_data[first_id] if is_initial_data_available else None,
        )

        if sim_requires_init_data and not is_initial_data_available:
            raise Exception(
                "The initialize() method of the Micro simulation requires initial data, but no initial macro data has been provided."
            )

        # Get initial data from micro simulations if initialize() method exists
        if self._micro_sims_init:
            # Call initialize() method of the micro simulation to check if it returns any initial data
            if sim_requires_init_data:
                initial_micro_output = self._micro_sims[first_id].initialize(
                    initial_data[first_id]
                )
            else:
                initial_micro_output = self._micro_sims[first_id].initialize()

            # Check if the detected initialize() method returns any data
            if initial_micro_output is None:
                self._logger.log_warning_rank_zero(
                    "The initialize() call of the Micro simulation has not returned any initial data."
                    " This means that the initialize() call has no effect on the adaptivity. The initialize method will nevertheless still be called."
                )
                self._micro_sims_init = False

                if sim_requires_init_data:
                    for i in micro_sims_to_init:
                        self._micro_sims[i].initialize(initial_data[i])
                else:
                    for i in micro_sims_to_init:
                        self._micro_sims[i].initialize()
            else:  # Case where the initialize() method returns data
                if self._is_adaptivity_on:
                    initial_micro_data: dict[str, list] = dict()

                    for name in initial_micro_output.keys():
                        initial_micro_data[name] = [0] * self._local_number_of_sims
                        # Save initial data from first micro simulation as we anyway have it
                        initial_micro_data[name][first_id] = initial_micro_output[name]

                    # Save initial data from first micro simulation as we anyway have it
                    for name in initial_micro_output.keys():
                        if name in self._data_for_adaptivity:
                            self._data_for_adaptivity[name][
                                first_id
                            ] = initial_micro_output[name]
                        else:
                            raise Exception(
                                "The initialize() method needs to return data which is required for the adaptivity calculation."
                            )

                    # Gather initial data from the rest of the micro simulations
                    if sim_requires_init_data:
                        for i in micro_sims_to_init:
                            initial_micro_output = self._micro_sims[i].initialize(
                                initial_data[i]
                            )
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                                initial_micro_data[name][i] = initial_micro_output[name]
                    else:
                        for i in micro_sims_to_init:
                            initial_micro_output = self._micro_sims[i].initialize()
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                                initial_micro_data[name][i] = initial_micro_output[name]

                    # If lazy initialization is on, initial states of inactive simulations need to be determined
                    if self._lazy_init:
                        self._adaptivity_controller.get_full_field_micro_output(
                            initial_micro_data
                        )
                        for i in range(self._local_number_of_sims):
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][i] = initial_micro_data[
                                    name
                                ][i]
                        del initial_micro_data  # Once the initial data is fed into the adaptivity data, it is no longer required

                else:
                    self._logger.log_warning_rank_zero(
                        "The initialize() method of the Micro simulation returns initial data, but adaptivity is turned off. The returned data will be ignored. The initialize method will nevertheless still be called."
                    )
                    if sim_requires_init_data:
                        for i in range(1, self._local_number_of_sims):
                            self._micro_sims[i].initialize(initial_data[i])
                    else:
                        for i in range(1, self._local_number_of_sims):
                            self._micro_sims[i].initialize()

        self._micro_sims_have_output = micro_problem_cls.check_output()

        self._participant.stop_last_profiling_section()

    # ***************
    # Private methods
    # ***************

    def _read_data_from_precice(self, dt) -> list:
        """
        Read data from preCICE.

        Parameters
        ----------
        dt : float
            Time step size at which data is to be read from preCICE.

        Returns
        -------
        local_read_data : list
            List of dicts in which keys are names of data being read and the values are the data from preCICE.
        """
        read_data: dict[str, list] = dict()

        if self._is_adaptivity_with_load_balancing:
            read_vertex_ids = self._global_ids_of_local_sims
        else:
            read_vertex_ids = self._mesh_vertex_ids

        for name in self._read_data_names:
            read_data[name] = []

        for name in self._read_data_names:
            read_data.update(
                {
                    name: self._participant.read_data(
                        self._macro_mesh_name, name, read_vertex_ids, dt
                    )
                }
            )

            if self._is_adaptivity_on:
                if name in self._adaptivity_macro_data_names:
                    self._data_for_adaptivity[name] = read_data[name]

        return [dict(zip(read_data, t)) for t in zip(*read_data.values())]

    def _write_data_to_precice(self, data: list) -> None:
        """
        Write data to preCICE.

        Parameters
        ----------
        data : list
            List of dicts in which keys are names of data and the values are the data to be written to preCICE.
        """
        if self._is_adaptivity_with_load_balancing:
            write_vertex_ids = self._global_ids_of_local_sims
        else:
            write_vertex_ids = self._mesh_vertex_ids

        data_dict: dict[str, list] = dict()
        if not self._is_rank_empty:
            for name in data[0]:
                data_dict[name] = []

            for d in data:
                for name, values in d.items():
                    data_dict[name].append(values)

            for dname in self._write_data_names:
                self._participant.write_data(
                    self._macro_mesh_name,
                    dname,
                    write_vertex_ids,
                    data_dict[dname],
                )
        else:
            for dname in self._write_data_names:
                self._participant.write_data(
                    self._macro_mesh_name, dname, [], np.array([])
                )

    def _solve_micro_simulations(self, micro_sims_input: list, dt: float) -> list:
        """
        Solve all micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        dt : float
            Time step size.

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data which are required outputs of
        """
        micro_sims_output: list[dict] = [None] * self._local_number_of_sims

        for count, sim in enumerate(self._micro_sims):
            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[count]:
                # Attempt to solve the micro simulation
                try:
                    micro_sims_output[count] = sim.solve(micro_sims_input[count], dt)

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.log_error(
                        "Micro simulation at macro coordinates {} with input {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._mesh_vertex_coords[count], micro_sims_input[count]
                        )
                    )
                    self._logger.log_error(error_message)
                    self._has_sim_crashed[count] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._size, dtype=np.int64)
            self._comm.Allgather(
                np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
            )
            if sum(crashed_sims_on_all_ranks) > 0:
                self._logger.log_info(
                    "Exiting simulation after micro simulation crash."
                )
                sys.exit()

        # Interpolate result for crashed simulation
        unset_sims = [
            count for count, value in enumerate(micro_sims_output) if value is None
        ]

        # Iterate over all crashed simulations to interpolate output
        if self._interpolate_crashed_sims:
            for unset_sim in unset_sims:
                self._logger.log_info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._mesh_vertex_coords[unset_sim]
                    )
                )
                micro_sims_output[unset_sim] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, unset_sim
                )

        return micro_sims_output

    def _solve_micro_simulations_with_adaptivity(
        self, micro_sims_input: list, dt: float
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

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data which are required outputs of
        """
        active_sim_lids = self._adaptivity_controller.get_active_sim_local_ids()

        micro_sims_output = [0] * self._local_number_of_sims

        # Solve all active micro simulations
        for lid in active_sim_lids:
            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[lid]:
                try:
                    micro_sims_output[lid] = self._micro_sims[lid].solve(
                        micro_sims_input[lid], dt
                    )

                    # Mark the micro sim as active for export
                    micro_sims_output[lid]["Active-State"] = 1
                    gid = self._global_ids_of_local_sims[lid]
                    micro_sims_output[lid][
                        "Active-Steps"
                    ] = self._micro_sims_active_steps[gid]

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.log_error(
                        "Micro simulation at macro coordinates {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._mesh_vertex_coords[lid]
                        )
                    )
                    self._logger.log_error(error_message)
                    self._has_sim_crashed[lid] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._size, dtype=np.int64)
            self._comm.Allgather(
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
            for unset_sim in unset_sims:
                self._logger.log_info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._mesh_vertex_coords[unset_sim]
                    )
                )

                micro_sims_output[unset_sim] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, unset_sim, active_sim_lids
                )

        micro_sims_output = self._adaptivity_controller.get_full_field_micro_output(
            micro_sims_output
        )

        inactive_sim_lids = self._adaptivity_controller.get_inactive_sim_local_ids()

        # Resolve micro sim output data for inactive simulations
        for inactive_lid in inactive_sim_lids:
            micro_sims_output[inactive_lid]["Active-State"] = 0
            gid = self._global_ids_of_local_sims[inactive_lid]
            micro_sims_output[inactive_lid][
                "Active-Steps"
            ] = self._micro_sims_active_steps[gid]

        # Collect micro sim output for adaptivity calculation
        for i in range(self._local_number_of_sims):
            for name in self._adaptivity_micro_data_names:
                self._data_for_adaptivity[name][i] = micro_sims_output[i][name]

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
            self._model_adaptivity_controller.switch_models(
                self._mesh_vertex_coords,
                self._t,
                micro_sims_input,
                output,
                self._micro_sims,
                active_sim_ids,
            )
            output = solve_variant(micro_sims_input, dt)
            self._model_adaptivity_controller.check_convergence(
                self._mesh_vertex_coords,
                self._t,
                micro_sims_input,
                output,
                self._micro_sims,
                active_sim_ids,
            )

        self._model_adaptivity_controller.finalise_solve()
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
        unset_sim: int,
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
        unset_sim : int
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
        for value in micro_sims_input[unset_sim].values():
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
                    self._mesh_vertex_coords[unset_sim]
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
            output_interpol["Active-Steps"] = self._micro_sims_active_steps[unset_sim]
        return output_interpol
