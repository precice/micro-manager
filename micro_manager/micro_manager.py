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
import time
import inspect
from copy import deepcopy
from typing import Dict
from warnings import warn

import numpy as np
import precice

from .micro_manager_base import MicroManager
from .adaptivity.global_adaptivity import GlobalAdaptivityCalculator
from .adaptivity.local_adaptivity import LocalAdaptivityCalculator
from .domain_decomposition import DomainDecomposer
from .micro_simulation import create_simulation_class

try:
    from .interpolation import Interpolation
except ImportError:
    Interpolation = None

sys.path.append(os.getcwd())


class MicroManagerCoupling(MicroManager):
    def __init__(self, config_file: str) -> None:
        """
        Constructor.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (provided by the user).
        """
        super().__init__(config_file)
        self._config.read_json_micro_manager()
        # Define the preCICE Participant
        self._participant = precice.Participant(
            "Micro-Manager", self._config.get_config_file_name(), self._rank, self._size
        )

        self._macro_mesh_name = self._config.get_macro_mesh_name()

        self._macro_bounds = self._config.get_macro_domain_bounds()

        if self._is_parallel:  # Simulation is run in parallel
            self._ranks_per_axis = self._config.get_ranks_per_axis()

        # Parameter for interpolation in case of a simulation crash
        self._interpolate_crashed_sims = self._config.interpolate_crashed_micro_sim()
        if self._interpolate_crashed_sims:
            if Interpolation is None:
                self._logger.info(
                    "Interpolation is turned off as the required package is not installed."
                )
                self._interpolate_crashed_sims = False
            else:
                # The following parameters can potentially become configurable by the user in the future
                self._crash_threshold = 0.2
                self._number_of_nearest_neighbors = 4

        self._mesh_vertex_ids = None  # IDs of macro vertices as set by preCICE
        self._micro_n_out = self._config.get_micro_output_n()

        self._is_adaptivity_on = self._config.turn_on_adaptivity()

        if self._is_adaptivity_on:
            self._number_of_sims_for_adaptivity = 0

            self._data_for_adaptivity: Dict[str, np.ndarray] = dict()
            self._adaptivity_type = self._config.get_adaptivity_type()

            self._adaptivity_data_names = self._config.get_data_for_adaptivity()

            # Names of macro data to be used for adaptivity computation
            self._adaptivity_macro_data_names = dict()
            # Names of micro data to be used for adaptivity computation
            self._adaptivity_micro_data_names = dict()
            for name, is_data_vector in self._adaptivity_data_names.items():
                if name in self._read_data_names:
                    self._adaptivity_macro_data_names[name] = is_data_vector
                if name in self._write_data_names:
                    self._adaptivity_micro_data_names[name] = is_data_vector

            self._adaptivity_in_every_implicit_step = (
                self._config.is_adaptivity_required_in_every_implicit_iteration()
            )
            self._micro_sims_active_steps = None

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
        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0
        similarity_dists_cp = None
        is_sim_active_cp = None
        sim_is_associated_to_cp = None
        sim_states_cp = [None] * self._local_number_of_sims

        dt = min(self._participant.get_max_time_step_size(), self._micro_dt)

        if self._is_adaptivity_on:
            similarity_dists = np.zeros(
                (
                    self._number_of_sims_for_adaptivity,
                    self._number_of_sims_for_adaptivity,
                )
            )

            # Start adaptivity calculation with all sims active
            is_sim_active = np.array([True] * self._number_of_sims_for_adaptivity)

            # Active sims do not have an associated sim
            sim_is_associated_to = np.full(
                (self._number_of_sims_for_adaptivity), -2, dtype=np.intc
            )

            # If micro simulations have been initialized, compute adaptivity before starting the coupling
            if self._micro_sims_init:
                (
                    similarity_dists,
                    is_sim_active,
                    sim_is_associated_to,
                ) = self._adaptivity_controller.compute_adaptivity(
                    dt,
                    self._micro_sims,
                    similarity_dists,
                    is_sim_active,
                    sim_is_associated_to,
                    self._data_for_adaptivity,
                )

        while self._participant.is_coupling_ongoing():

            dt = min(self._participant.get_max_time_step_size(), self._micro_dt)

            # Write a checkpoint
            if self._participant.requires_writing_checkpoint():
                for i in range(self._local_number_of_sims):
                    sim_states_cp[i] = self._micro_sims[i].get_state()
                t_checkpoint = t
                n_checkpoint = n

                if self._is_adaptivity_on:
                    if not self._adaptivity_in_every_implicit_step:
                        (
                            similarity_dists,
                            is_sim_active,
                            sim_is_associated_to,
                        ) = self._adaptivity_controller.compute_adaptivity(
                            dt,
                            self._micro_sims,
                            similarity_dists,
                            is_sim_active,
                            sim_is_associated_to,
                            self._data_for_adaptivity,
                        )

                        # Only checkpoint the adaptivity configuration if adaptivity is computed
                        # once in every time window
                        similarity_dists_cp = np.copy(similarity_dists)
                        is_sim_active_cp = np.copy(is_sim_active)
                        sim_is_associated_to_cp = np.copy(sim_is_associated_to)

                    if self._adaptivity_type == "local":
                        active_sim_ids = np.where(is_sim_active)[0]
                    elif self._adaptivity_type == "global":
                        active_sim_ids = np.where(
                            is_sim_active[
                                self._global_ids_of_local_sims[
                                    0
                                ] : self._global_ids_of_local_sims[-1]
                                + 1
                            ]
                        )[0]

                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

            micro_sims_input = self._read_data_from_precice(dt)

            if self._is_adaptivity_on:
                if self._adaptivity_in_every_implicit_step:
                    (
                        similarity_dists,
                        is_sim_active,
                        sim_is_associated_to,
                    ) = self._adaptivity_controller.compute_adaptivity(
                        dt,
                        self._micro_sims,
                        similarity_dists,
                        is_sim_active,
                        sim_is_associated_to,
                        self._data_for_adaptivity,
                    )

                    if self._adaptivity_type == "local":
                        active_sim_ids = np.where(is_sim_active)[0]
                    elif self._adaptivity_type == "global":
                        active_sim_ids = np.where(
                            is_sim_active[
                                self._global_ids_of_local_sims[
                                    0
                                ] : self._global_ids_of_local_sims[-1]
                                + 1
                            ]
                        )[0]

                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

                micro_sims_output = self._solve_micro_simulations_with_adaptivity(
                    micro_sims_input, is_sim_active, sim_is_associated_to, dt
                )
            else:
                micro_sims_output = self._solve_micro_simulations(micro_sims_input, dt)

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
                    self._logger.info(
                        "{:.1%} of the micro simulations have crashed exceeding the threshold of {:.1%}. "
                        "Exiting simulation.".format(crash_ratio, self._crash_threshold)
                    )
                    sys.exit()

            self._write_data_to_precice(micro_sims_output)

            t += dt  # increase internal time when time step is done.
            n += 1  # increase counter
            self._participant.advance(
                dt
            )  # notify preCICE that time step of size dt is complete

            # Revert micro simulations to their last checkpoints if required
            if self._participant.requires_reading_checkpoint():
                for i in range(self._local_number_of_sims):
                    self._micro_sims[i].set_state(sim_states_cp[i])
                n = n_checkpoint
                t = t_checkpoint

                # If adaptivity is computed only once per time window, the states of sims need to be reset too
                if self._is_adaptivity_on:
                    if not self._adaptivity_in_every_implicit_step:
                        similarity_dists = np.copy(similarity_dists_cp)
                        is_sim_active = np.copy(is_sim_active_cp)
                        sim_is_associated_to = np.copy(sim_is_associated_to_cp)

            if (
                self._participant.is_time_window_complete()
            ):  # Time window has converged, now micro output can be generated
                self._logger.info(
                    "Micro simulations {} - {} have converged at t = {}".format(
                        self._micro_sims[0].get_global_id(),
                        self._micro_sims[-1].get_global_id(),
                        t,
                    )
                )

                if self._micro_sims_have_output:
                    if n % self._micro_n_out == 0:
                        for sim in self._micro_sims:
                            sim.output()

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
        # Decompose the macro-domain and set the mesh access region for each partition in preCICE
        assert len(self._macro_bounds) / 2 == self._participant.get_mesh_dimensions(
            self._macro_mesh_name
        ), "Provided macro mesh bounds are of incorrect dimension"
        if self._is_parallel:
            domain_decomposer = DomainDecomposer(
                self._logger,
                self._participant.get_mesh_dimensions(self._macro_mesh_name),
                self._rank,
                self._size,
            )
            coupling_mesh_bounds = domain_decomposer.decompose_macro_domain(
                self._macro_bounds, self._ranks_per_axis
            )
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._participant.set_mesh_access_region(
            self._macro_mesh_name, coupling_mesh_bounds
        )

        # initialize preCICE
        self._participant.initialize()

        (
            self._mesh_vertex_ids,
            self._mesh_vertex_coords,
        ) = self._participant.get_mesh_vertex_ids_and_coordinates(self._macro_mesh_name)
        assert self._mesh_vertex_coords.size != 0, "Macro mesh has no vertices."

        self._local_number_of_sims, _ = self._mesh_vertex_coords.shape
        self._logger.info(
            "Number of local micro simulations = {}".format(self._local_number_of_sims)
        )

        if self._local_number_of_sims == 0:
            if self._is_parallel:
                self._logger.info(
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

        # Get global number of micro simulations
        self._global_number_of_sims = np.sum(nms_all_ranks)

        if self._is_adaptivity_on:
            for name, is_data_vector in self._adaptivity_data_names.items():
                if is_data_vector:
                    self._data_for_adaptivity[name] = np.zeros(
                        (
                            self._local_number_of_sims,
                            self._participant.get_data_dimensions(
                                self._macro_mesh_name, name
                            ),
                        )
                    )
                else:
                    self._data_for_adaptivity[name] = np.zeros(
                        (self._local_number_of_sims)
                    )

        # Create lists of local and global IDs
        sim_id = np.sum(nms_all_ranks[: self._rank])
        self._global_ids_of_local_sims = []  # DECLARATION
        for i in range(self._local_number_of_sims):
            self._global_ids_of_local_sims.append(sim_id)
            sim_id += 1

        self._micro_sims = [None] * self._local_number_of_sims  # DECLARATION

        # Setup for simulation crashes
        self._has_sim_crashed = [False] * self._local_number_of_sims
        if self._interpolate_crashed_sims:
            self._interpolant = Interpolation(self._logger)

        micro_problem = getattr(
            importlib.import_module(
                self._config.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation",
        )

        # Create micro simulation objects
        for i in range(self._local_number_of_sims):
            self._micro_sims[i] = create_simulation_class(micro_problem)(
                self._global_ids_of_local_sims[i]
            )

        self._logger.info(
            "Micro simulations with global IDs {} - {} created.".format(
                self._global_ids_of_local_sims[0], self._global_ids_of_local_sims[-1]
            )
        )

        if self._is_adaptivity_on:
            if self._adaptivity_type == "local":
                self._adaptivity_controller = LocalAdaptivityCalculator(
                    self._config, self._logger
                )
                self._number_of_sims_for_adaptivity = self._local_number_of_sims
            elif self._adaptivity_type == "global":
                self._adaptivity_controller = GlobalAdaptivityCalculator(
                    self._config,
                    self._logger,
                    self._global_number_of_sims,
                    self._global_ids_of_local_sims,
                    self._rank,
                    self._comm,
                )
                self._number_of_sims_for_adaptivity = self._global_number_of_sims

            self._micro_sims_active_steps = np.zeros(self._local_number_of_sims)

        self._micro_sims_init = False  # DECLARATION

        # Read initial data from preCICE, if it is available
        initial_data = self._read_data_from_precice(dt=0)

        if not initial_data:
            is_initial_data_available = False
        else:
            is_initial_data_available = True

        # Boolean which states if the initialize() method of the micro simulation requires initial data
        is_initial_data_required = False

        # Check if provided micro simulation has an initialize() method
        if hasattr(micro_problem, "initialize") and callable(
            getattr(micro_problem, "initialize")
        ):
            self._micro_sims_init = True  # Starting value before setting

            try:  # Try to get the signature of the initialize() method, if it is written in Python
                argspec = inspect.getfullargspec(micro_problem.initialize)
                if (
                    len(argspec.args) == 1
                ):  # The first argument in the signature is self
                    is_initial_data_required = False
                elif len(argspec.args) == 2:
                    is_initial_data_required = True
                else:
                    raise Exception(
                        "The initialize() method of the Micro simulation has an incorrect number of arguments."
                    )
            except TypeError:
                self._logger.info(
                    "The signature of initialize() method of the micro simulation cannot be determined. Trying to determine the signature by calling the method."
                )
                # Try to get the signature of the initialize() method, if it is not written in Python
                try:  # Try to call the initialize() method without initial data
                    self._micro_sims[0].initialize()
                    is_initial_data_required = False
                except TypeError:
                    self._logger.info(
                        "The initialize() method of the micro simulation has arguments. Attempting to call it again with initial data."
                    )
                    try:  # Try to call the initialize() method with initial data
                        self._micro_sims[0].initialize(initial_data[0])
                        is_initial_data_required = True
                    except TypeError:
                        raise Exception(
                            "The initialize() method of the Micro simulation has an incorrect number of arguments."
                        )

        if is_initial_data_required and not is_initial_data_available:
            raise Exception(
                "The initialize() method of the Micro simulation requires initial data, but no initial data has been provided."
            )

        if not is_initial_data_required and is_initial_data_available:
            warn(
                "The initialize() method is only allowed to return data which is required for the adaptivity calculation."
            )

        # Get initial data from micro simulations if initialize() method exists
        if self._micro_sims_init:

            # Call initialize() method of the micro simulation to check if it returns any initial data
            if is_initial_data_required:
                initial_micro_output = self._micro_sims[0].initialize(initial_data[0])
            else:
                initial_micro_output = self._micro_sims[0].initialize()

            if (
                initial_micro_output is None
            ):  # Check if the detected initialize() method returns any data
                warn(
                    "The initialize() call of the Micro simulation has not returned any initial data."
                    " This means that the initialize() call has no effect on the adaptivity. The initialize method will nevertheless still be called."
                )
                self._micro_sims_init = False

                if is_initial_data_required:
                    for i in range(1, self._local_number_of_sims):
                        self._micro_sims[i].initialize(initial_data[i])
                else:
                    for i in range(1, self._local_number_of_sims):
                        self._micro_sims[i].initialize()
            else:  # Case where the initialize() method returns data
                if self._is_adaptivity_on:
                    # Save initial data from first micro simulation as we anyway have it
                    for name in initial_micro_output.keys():
                        if name in self._data_for_adaptivity:
                            self._data_for_adaptivity[name][0] = initial_micro_output[
                                name
                            ]
                        else:
                            raise Exception(
                                "The initialize() method needs to return data which is required for the adaptivity calculation."
                            )

                    # Gather initial data from the rest of the micro simulations
                    if is_initial_data_required:
                        for i in range(1, self._local_number_of_sims):
                            initial_micro_output = self._micro_sims[i].initialize(
                                initial_data[i]
                            )
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                    else:
                        for i in range(1, self._local_number_of_sims):
                            initial_micro_output = self._micro_sims[i].initialize()
                            for name in self._adaptivity_micro_data_names:
                                self._data_for_adaptivity[name][
                                    i
                                ] = initial_micro_output[name]
                else:
                    warn(
                        "The initialize() method of the Micro simulation returns initial data, but adaptivity is turned off. The returned data will be ignored. The initialize method will nevertheless still be called."
                    )
                    if is_initial_data_required:
                        for i in range(1, self._local_number_of_sims):
                            self._micro_sims[i].initialize(initial_data[i])
                    else:
                        for i in range(1, self._local_number_of_sims):
                            self._micro_sims[i].initialize()

        self._micro_sims_have_output = False
        if hasattr(micro_problem, "output") and callable(
            getattr(micro_problem, "output")
        ):
            self._micro_sims_have_output = True

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
        read_data: Dict[str, list] = dict()
        for name in self._read_data_names.keys():
            read_data[name] = []

        for name in self._read_data_names.keys():
            read_data.update(
                {
                    name: self._participant.read_data(
                        self._macro_mesh_name, name, self._mesh_vertex_ids, dt
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
        data_dict: Dict[str, list] = dict()
        if not self._is_rank_empty:
            for name in data[0]:
                data_dict[name] = []

            for d in data:
                for name, values in d.items():
                    data_dict[name].append(values)

            for dname in self._write_data_names.keys():
                self._participant.write_data(
                    self._macro_mesh_name,
                    dname,
                    self._mesh_vertex_ids,
                    data_dict[dname],
                )
        else:
            for dname in self._write_data_names.keys():
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
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        micro_sims_output = [None] * self._local_number_of_sims

        for count, sim in enumerate(self._micro_sims):
            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[count]:
                # Attempt to solve the micro simulation
                try:
                    start_time = time.time()
                    micro_sims_output[count] = sim.solve(micro_sims_input[count], dt)
                    end_time = time.time()
                    # Write solve time of the macro simulation if required and the simulation has not crashed
                    if self._is_micro_solve_time_required:
                        micro_sims_output[count]["micro_sim_time"] = (
                            end_time - start_time
                        )

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.error(
                        "Micro simulation at macro coordinates {} with input {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._mesh_vertex_coords[count], micro_sims_input[count]
                        )
                    )
                    self._logger.error(error_message)
                    self._has_sim_crashed[count] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._size, dtype=np.int64)
            self._comm.Allgather(
                np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
            )
            if sum(crashed_sims_on_all_ranks) > 0:
                self._logger.info("Exiting simulation after micro simulation crash.")
                sys.exit()

        # Interpolate result for crashed simulation
        unset_sims = [
            count for count, value in enumerate(micro_sims_output) if value is None
        ]

        # Iterate over all crashed simulations to interpolate output
        if self._interpolate_crashed_sims:
            for unset_sim in unset_sims:
                self._logger.info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._mesh_vertex_coords[unset_sim]
                    )
                )
                micro_sims_output[unset_sim] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, unset_sim
                )

        return micro_sims_output

    def _solve_micro_simulations_with_adaptivity(
        self,
        micro_sims_input: list,
        is_sim_active: np.ndarray,
        sim_is_associated_to: np.ndarray,
        dt: float,
    ) -> list:
        """
        Solve all micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        is_sim_active : numpy array
            1D array having state (active or inactive) of each micro simulation
        sim_is_associated_to : numpy array
            1D array with values of associated simulations of inactive simulations. Active simulations have None
        dt : float
            Time step size.

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        if self._adaptivity_type == "global":
            active_sim_ids = np.where(
                is_sim_active[
                    self._global_ids_of_local_sims[0] : self._global_ids_of_local_sims[
                        -1
                    ]
                    + 1
                ]
            )[0]
            inactive_sim_ids = np.where(
                is_sim_active[
                    self._global_ids_of_local_sims[0] : self._global_ids_of_local_sims[
                        -1
                    ]
                    + 1
                ]
                == False
            )[0]
        elif self._adaptivity_type == "local":
            active_sim_ids = np.where(is_sim_active)[0]
            inactive_sim_ids = np.where(is_sim_active == False)[0]

        micro_sims_output = [None] * self._local_number_of_sims

        # Solve all active micro simulations
        for active_id in active_sim_ids:
            # If micro simulation has not crashed in a previous iteration, attempt to solve it
            if not self._has_sim_crashed[active_id]:
                # Attempt to solve the micro simulation
                try:
                    start_time = time.time()
                    micro_sims_output[active_id] = self._micro_sims[active_id].solve(
                        micro_sims_input[active_id], dt
                    )
                    end_time = time.time()
                    # Write solve time of the macro simulation if required and the simulation has not crashed
                    if self._is_micro_solve_time_required:
                        micro_sims_output[active_id]["micro_sim_time"] = (
                            end_time - start_time
                        )

                    # Mark the micro sim as active for export
                    micro_sims_output[active_id]["active_state"] = 1
                    micro_sims_output[active_id][
                        "active_steps"
                    ] = self._micro_sims_active_steps[active_id]

                # If simulation crashes, log the error and keep the output constant at the previous iteration's output
                except Exception as error_message:
                    self._logger.error(
                        "Micro simulation at macro coordinates {} has experienced an error. "
                        "See next entry on this rank for error message.".format(
                            self._mesh_vertex_coords[active_id]
                        )
                    )
                    self._logger.error(error_message)
                    self._has_sim_crashed[active_id] = True

        # If interpolate is off, terminate after crash
        if not self._interpolate_crashed_sims:
            crashed_sims_on_all_ranks = np.zeros(self._size, dtype=np.int64)
            self._comm.Allgather(
                np.sum(self._has_sim_crashed), crashed_sims_on_all_ranks
            )
            if sum(crashed_sims_on_all_ranks) > 0:
                self._logger.info("Exiting simulation after micro simulation crash.")
                sys.exit()
        # Interpolate result for crashed simulation
        unset_sims = []
        for active_id in active_sim_ids:
            if micro_sims_output[active_id] is None:
                unset_sims.append(active_id)

        # Iterate over all crashed simulations to interpolate output
        if self._interpolate_crashed_sims:
            for unset_sim in unset_sims:
                self._logger.info(
                    "Interpolating output for crashed simulation at macro vertex {}.".format(
                        self._mesh_vertex_coords[unset_sim]
                    )
                )

                micro_sims_output[unset_sim] = self._interpolate_output_for_crashed_sim(
                    micro_sims_input, micro_sims_output, unset_sim, active_sim_ids
                )

        # For each inactive simulation, copy data from most similar active simulation
        if self._adaptivity_type == "global":
            self._adaptivity_controller.communicate_micro_output(
                is_sim_active, sim_is_associated_to, micro_sims_output
            )
        elif self._adaptivity_type == "local":
            for inactive_id in inactive_sim_ids:
                micro_sims_output[inactive_id] = deepcopy(
                    micro_sims_output[sim_is_associated_to[inactive_id]]
                )

        # Resolve micro sim output data for inactive simulations
        for inactive_id in inactive_sim_ids:
            micro_sims_output[inactive_id]["active_state"] = 0
            micro_sims_output[inactive_id][
                "active_steps"
            ] = self._micro_sims_active_steps[inactive_id]

            if self._is_micro_solve_time_required:
                micro_sims_output[inactive_id]["micro_sim_time"] = 0

        # Collect micro sim output for adaptivity calculation
        for i in range(self._local_number_of_sims):
            for name in self._adaptivity_micro_data_names:
                self._data_for_adaptivity[name][i] = micro_sims_output[i][name]

        return micro_sims_output

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
            self._logger.error(
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
                interpol_values[-1].pop("micro_sim_time", None)
                interpol_values[-1].pop("active_state", None)
                interpol_values[-1].pop("active_steps", None)
            else:
                interpol_space.append(micro_sims_active_input_lists[neighbor].copy())
                interpol_values.append(micro_sims_active_values[neighbor].copy())
                interpol_values[-1].pop("micro_sim_time", None)

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
        if self._is_micro_solve_time_required:
            output_interpol["micro_sim_time"] = 0
        if self._is_adaptivity_on:
            output_interpol["active_state"] = 1
            output_interpol["active_steps"] = self._micro_sims_active_steps[unset_sim]
        return output_interpol
