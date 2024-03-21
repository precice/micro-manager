#!/usr/bin/env python3
"""
Micro Manager is a tool to initialize and adaptively control micro simulations and couple them via preCICE to a macro simulation.
This files the class MicroManager which has the following callable public methods:

- solve

This file is directly executable as it consists of a main() function. Upon execution, an object of the class MicroManager is created using a given JSON file,
and the initialize and solve methods are called.

Detailed documentation: https://precice.org/tooling-micro-manager-overview.html
"""

import argparse
import os
import sys
import precice
from mpi4py import MPI
import numpy as np
import logging
import time
from copy import deepcopy
from typing import Dict
from warnings import warn

from .config import Config
from .micro_simulation import create_simulation_class
from .adaptivity.local_adaptivity import LocalAdaptivityCalculator
from .adaptivity.global_adaptivity import GlobalAdaptivityCalculator
from .domain_decomposition import DomainDecomposer

sys.path.append(os.getcwd())


class MicroManager:
    def __init__(self, config_file: str) -> None:
        """
        Constructor.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (provided by the user).
        """
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.INFO)

        # Create file handler which logs messages
        fh = logging.FileHandler('micro-manager.log')
        fh.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter('[' + str(self._rank) + '] %(name)s -  %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)  # add the handlers to the logger

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        self._logger.info("Provided configuration file: {}".format(config_file))
        self._config = Config(self._logger, config_file)

        # Define the preCICE Participant
        self._participant = precice.Participant(
            "Micro-Manager",
            self._config.get_config_file_name(),
            self._rank,
            self._size)

        micro_file_name = self._config.get_micro_file_name()

        self._macro_mesh_name = self._config.get_macro_mesh_name()

        # Data names of data written to preCICE
        self._write_data_names = self._config.get_write_data_names()

        # Data names of data read from preCICE
        self._read_data_names = self._config.get_read_data_names()

        self._macro_bounds = self._config.get_macro_domain_bounds()

        if self._is_parallel:  # Simulation is run in parallel
            self._ranks_per_axis = self._config.get_ranks_per_axis()

        self._is_micro_solve_time_required = self._config.write_micro_solve_time()

        self._local_number_of_sims = 0
        self._global_number_of_sims = 0
        self._is_rank_empty = False
        self._dt = 0
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

            self._adaptivity_in_every_implicit_step = self._config.is_adaptivity_required_in_every_implicit_iteration()
            self._micro_sims_active_steps = None

        self._initialize()

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

        if self._is_adaptivity_on:
            similarity_dists = np.zeros(
                (self._number_of_sims_for_adaptivity,
                 self._number_of_sims_for_adaptivity))

            # Start adaptivity calculation with all sims active
            is_sim_active = np.array([True] * self._number_of_sims_for_adaptivity)

            # Active sims do not have an associated sim
            sim_is_associated_to = np.full((self._number_of_sims_for_adaptivity), -2, dtype=np.intc)

            # If micro simulations have been initialized, compute adaptivity based on initial data
            if self._micro_sims_init:
                # Compute adaptivity based on initial data of micro sims
                similarity_dists, is_sim_active, sim_is_associated_to = self._adaptivity_controller.compute_adaptivity(
                    self._dt, self._micro_sims, similarity_dists, is_sim_active, sim_is_associated_to, self._data_for_adaptivity)

        while self._participant.is_coupling_ongoing():
            # Write a checkpoint
            if self._participant.requires_writing_checkpoint():
                for i in range(self._local_number_of_sims):
                    sim_states_cp[i] = self._micro_sims[i].get_state()
                t_checkpoint = t
                n_checkpoint = n

                if self._is_adaptivity_on:
                    if not self._adaptivity_in_every_implicit_step:
                        similarity_dists, is_sim_active, sim_is_associated_to = self._adaptivity_controller.compute_adaptivity(
                            self._dt, self._micro_sims, similarity_dists, is_sim_active, sim_is_associated_to, self._data_for_adaptivity)

                        # Only checkpoint the adaptivity configuration if adaptivity is computed
                        # once in every time window
                        similarity_dists_cp = np.copy(similarity_dists)
                        is_sim_active_cp = np.copy(is_sim_active)
                        sim_is_associated_to_cp = np.copy(sim_is_associated_to)

                    if self._adaptivity_type == "local":
                        active_sim_ids = np.where(is_sim_active)[0]
                    elif self._adaptivity_type == "global":
                        active_sim_ids = np.where(
                            is_sim_active[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1])[0]

                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

            micro_sims_input = self._read_data_from_precice()

            if self._is_adaptivity_on:
                if self._adaptivity_in_every_implicit_step:
                    similarity_dists, is_sim_active, sim_is_associated_to = self._adaptivity_controller.compute_adaptivity(
                        self._dt, self._micro_sims, similarity_dists, is_sim_active, sim_is_associated_to, self._data_for_adaptivity)

                    if self._adaptivity_type == "local":
                        active_sim_ids = np.where(is_sim_active)[0]
                    elif self._adaptivity_type == "global":
                        active_sim_ids = np.where(
                            is_sim_active[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1])[0]

                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

                micro_sims_output = self._solve_micro_simulations_with_adaptivity(
                    micro_sims_input, is_sim_active, sim_is_associated_to)
            else:
                micro_sims_output = self._solve_micro_simulations(micro_sims_input)

            self._write_data_to_precice(micro_sims_output)

            self._participant.advance(self._dt)
            self._dt = self._participant.get_max_time_step_size()

            t += self._dt
            n += 1

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

            else:  # Time window has converged, now micro output can be generated
                self._logger.info("Micro simulations {} - {} have converged at t = {}".format(
                    self._micro_sims[0].get_global_id(), self._micro_sims[-1].get_global_id(), t))

                if self._micro_sims_have_output:
                    if n % self._micro_n_out == 0:
                        for sim in self._micro_sims:
                            sim.output()

        self._participant.finalize()

    # ***************
    # Private methods
    # ***************

    def _initialize(self) -> None:
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
            self._macro_mesh_name), "Provided macro mesh bounds are of incorrect dimension"
        if self._is_parallel:
            domain_decomposer = DomainDecomposer(
                self._logger, self._participant.get_mesh_dimensions(self._macro_mesh_name), self._rank, self._size)
            coupling_mesh_bounds = domain_decomposer.decompose_macro_domain(self._macro_bounds, self._ranks_per_axis)
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._participant.set_mesh_access_region(self._macro_mesh_name, coupling_mesh_bounds)

        # initialize preCICE
        self._participant.initialize()

        self._mesh_vertex_ids, mesh_vertex_coords = self._participant.get_mesh_vertex_ids_and_coordinates(
            self._macro_mesh_name)
        assert (mesh_vertex_coords.size != 0), "Macro mesh has no vertices."

        self._local_number_of_sims, _ = mesh_vertex_coords.shape
        self._logger.info("Number of local micro simulations = {}".format(self._local_number_of_sims))

        if self._local_number_of_sims == 0:
            if self._is_parallel:
                self._logger.info(
                    "Rank {} has no micro simulations and hence will not do any computation.".format(
                        self._rank))
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
                        (self._local_number_of_sims, self._participant.get_data_dimensions(
                            self._macro_mesh_name, name)))
                else:
                    self._data_for_adaptivity[name] = np.zeros((self._local_number_of_sims))

        # Create lists of local and global IDs
        sim_id = np.sum(nms_all_ranks[:self._rank])
        self._global_ids_of_local_sims = []  # DECLARATION
        for i in range(self._local_number_of_sims):
            self._global_ids_of_local_sims.append(sim_id)
            sim_id += 1

        self._micro_sims = [None] * self._local_number_of_sims  # DECLARATION

        micro_problem = getattr(
            __import__(
                self._config.get_micro_file_name(),
                fromlist=["MicroSimulation"]),
            "MicroSimulation")

        # Create micro simulation objects
        for i in range(self._local_number_of_sims):
            self._micro_sims[i] = create_simulation_class(
                micro_problem)(self._global_ids_of_local_sims[i])

        self._logger.info("Micro simulations with global IDs {} - {} created.".format(
            self._global_ids_of_local_sims[0], self._global_ids_of_local_sims[-1]))

        if self._is_adaptivity_on:
            if self._adaptivity_type == "local":
                self._adaptivity_controller = LocalAdaptivityCalculator(
                    self._config, self._logger)
                self._number_of_sims_for_adaptivity = self._local_number_of_sims
            elif self._adaptivity_type == "global":
                self._adaptivity_controller = GlobalAdaptivityCalculator(
                    self._config,
                    self._logger,
                    self._global_number_of_sims,
                    self._global_ids_of_local_sims,
                    self._rank,
                    self._comm)
                self._number_of_sims_for_adaptivity = self._global_number_of_sims

            self._micro_sims_active_steps = np.zeros(self._local_number_of_sims)

        self._micro_sims_init = False  # DECLARATION

        # Get initial data from micro simulations if initialize() method exists
        if hasattr(micro_problem, 'initialize') and callable(getattr(micro_problem, 'initialize')):
            if self._is_adaptivity_on:
                self._micro_sims_init = True
                initial_micro_output = self._micro_sims[0].initialize()  # Call initialize() of the first simulation
                if initial_micro_output is None:  # Check if the detected initialize() method returns any data
                    warn("The initialize() call of the Micro simulation has not returned any initial data."
                         " The initialize call is stopped.")
                    self._micro_sims_init = False
                else:
                    # Save initial data from first micro simulation as we anyway have it
                    for name in initial_micro_output.keys():
                        self._data_for_adaptivity[name][0] = initial_micro_output[name]

                    # Gather initial data from the rest of the micro simulations
                    for i in range(1, self._local_number_of_sims):
                        initial_micro_output = self._micro_sims[i].initialize()
                        for name in self._adaptivity_micro_data_names:
                            self._data_for_adaptivity[name][i] = initial_micro_output[name]
            else:
                self._logger.info(
                    "Micro simulation has the method initialize(), but it is not called, because adaptivity is off.")

        self._micro_sims_have_output = False
        if hasattr(micro_problem, 'output') and callable(getattr(micro_problem, 'output')):
            self._micro_sims_have_output = True

        self._dt = self._participant.get_max_time_step_size()

    def _read_data_from_precice(self) -> list:
        """
        Read data from preCICE.

        Returns
        -------
        local_read_data : list
            List of dicts in which keys are names of data being read and the values are the data from preCICE.
        """
        read_data: Dict[str, list] = dict()
        for name in self._read_data_names.keys():
            read_data[name] = []

        for name in self._read_data_names.keys():
            read_data.update({name: self._participant.read_data(
                self._macro_mesh_name, name, self._mesh_vertex_ids, self._dt)})

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
                    self._macro_mesh_name, dname, self._mesh_vertex_ids, data_dict[dname])
        else:
            for dname in self._write_data_names.keys():
                self._participant.write_data(self._macro_mesh_name, dname, [], np.array([]))

    def _solve_micro_simulations(self, micro_sims_input: list) -> list:
        """
        Solve all micro simulations and assemble the micro simulations outputs in a list of dicts format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        micro_sims_output = [None] * self._local_number_of_sims

        for count, sim in enumerate(self._micro_sims):
            start_time = time.time()
            micro_sims_output[count] = sim.solve(micro_sims_input[count], self._dt)
            end_time = time.time()

            if self._is_micro_solve_time_required:
                micro_sims_output[count]["micro_sim_time"] = end_time - start_time

        return micro_sims_output

    def _solve_micro_simulations_with_adaptivity(
            self,
            micro_sims_input: list,
            is_sim_active: np.ndarray,
            sim_is_associated_to: np.ndarray) -> list:
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

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        if self._adaptivity_type == "global":
            active_sim_ids = np.where(
                is_sim_active[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1])[0]
            inactive_sim_ids = np.where(
                is_sim_active[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1] == False)[0]
        elif self._adaptivity_type == "local":
            active_sim_ids = np.where(is_sim_active)[0]
            inactive_sim_ids = np.where(is_sim_active == False)[0]

        micro_sims_output = [None] * self._local_number_of_sims

        # Solve all active micro simulations
        for active_id in active_sim_ids:
            start_time = time.time()
            micro_sims_output[active_id] = self._micro_sims[active_id].solve(micro_sims_input[active_id], self._dt)
            end_time = time.time()

            # Mark the micro sim as active for export
            micro_sims_output[active_id]["active_state"] = 1
            micro_sims_output[active_id]["active_steps"] = self._micro_sims_active_steps[active_id]

            if self._is_micro_solve_time_required:
                micro_sims_output[active_id]["micro_sim_time"] = end_time - start_time

        # For each inactive simulation, copy data from most similar active simulation
        if self._adaptivity_type == "global":
            self._adaptivity_controller.communicate_micro_output(is_sim_active, sim_is_associated_to, micro_sims_output)
        elif self._adaptivity_type == "local":
            for inactive_id in inactive_sim_ids:
                micro_sims_output[inactive_id] = deepcopy(
                    micro_sims_output[sim_is_associated_to[inactive_id]])

        # Resolve micro sim output data for inactive simulations
        for inactive_id in inactive_sim_ids:
            micro_sims_output[inactive_id]["active_state"] = 0
            micro_sims_output[inactive_id]["active_steps"] = self._micro_sims_active_steps[inactive_id]

            if self._is_micro_solve_time_required:
                micro_sims_output[inactive_id]["micro_sim_time"] = 0

        # Collect micro sim output for adaptivity calculation
        for i in range(self._local_number_of_sims):
            for name in self._adaptivity_micro_data_names:
                self._data_for_adaptivity[name][i] = micro_sims_output[i][name]

        return micro_sims_output


def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON config file of the manager.')

    args = parser.parse_args()
    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    manager = MicroManager(config_file_path)

    manager.solve()


if __name__ == "__main__":
    main()
