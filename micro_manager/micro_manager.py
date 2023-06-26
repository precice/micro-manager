#!/usr/bin/env python3
"""
Micro Manager: a tool to initialize and adaptively control micro simulations and couple them via preCICE to a macro simulation
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

from .config import Config
from .micro_simulation import create_micro_problem_class
from .adaptivity.local_adaptivity import LocalAdaptivityCalculator
from .adaptivity.global_adaptivity import GlobalAdaptivityCalculator
from .domain_decomposition import DomainDecomposer

sys.path.append(os.getcwd())


class MicroManager:
    def __init__(self, config_file: str) -> None:
        """
        Constructor of MicroManager class.

        Parameters
        ----------
        config_file : string
            Name of the JSON configuration file (to be provided by the user)
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
        self._config = Config(config_file)

        # Define the preCICE interface
        self._interface = precice.Interface(
            "Micro-Manager",
            self._config.get_config_file_name(),
            self._rank,
            self._size)

        micro_file_name = self._config.get_micro_file_name()
        self._micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

        self._macro_mesh_id = self._interface.get_mesh_id(self._config.get_macro_mesh_name())

        # Data names and ids of data written to preCICE
        self._write_data_names = self._config.get_write_data_names()
        self._write_data_ids = dict()
        for name in self._write_data_names.keys():
            self._write_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        # Data names and ids of data read from preCICE
        self._read_data_names = self._config.get_read_data_names()
        self._read_data_ids = dict()
        for name in self._read_data_names.keys():
            self._read_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        self._macro_bounds = self._config.get_macro_domain_bounds()

        if self._is_parallel:  # Simulation is run in parallel
            self._ranks_per_axis = self._config.get_ranks_per_axis()

        self._is_micro_solve_time_required = self._config.write_micro_solve_time()

        self._local_number_of_micro_sims = None
        self._global_number_of_micro_sims = None
        self._is_rank_empty = False
        self._dt = None
        self._mesh_vertex_ids = None  # IDs of macro vertices as set by preCICE
        self._micro_n_out = self._config.get_micro_output_n()

        self._is_adaptivity_on = self._config.turn_on_adaptivity()

        if self._is_adaptivity_on:
            self._number_of_micro_sims_for_adaptivity = 0

            self._data_for_adaptivity = dict()
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

            self._is_adaptivity_required_in_every_implicit_iteration = self._config.is_adaptivity_required_in_every_implicit_iteration()
            self._micro_sims_active_steps = None

    def initialize(self) -> None:
        """
        This function does the following things:
        - Decomposes the domain if the Micro Manager is executed in parallel
        - Initializes preCICE
        - Gets the macro mesh information from preCICE
        - Creates all micro simulation objects and initializes them if an initialization procedure is available
        - Writes initial data to preCICE
        """
        # Decompose the macro-domain and set the mesh access region for each
        # partition in preCICE
        assert len(self._macro_bounds) / \
            2 == self._interface.get_dimensions(), "Provided macro mesh bounds are of incorrect dimension"
        if self._is_parallel:
            domain_decomposer = DomainDecomposer(self._logger, self._interface.get_dimensions(), self._rank, self._size)
            coupling_mesh_bounds = domain_decomposer.decompose_macro_domain(self._macro_bounds, self._ranks_per_axis)
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._interface.set_mesh_access_region(self._macro_mesh_id, coupling_mesh_bounds)

        # initialize preCICE
        self._dt = self._interface.initialize()

        self._mesh_vertex_ids, mesh_vertex_coords = self._interface.get_mesh_vertices_and_ids(self._macro_mesh_id)
        self._local_number_of_micro_sims, _ = mesh_vertex_coords.shape
        self._logger.info("Number of local micro simulations = {}".format(self._local_number_of_micro_sims))

        if self._local_number_of_micro_sims == 0:
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
        self._comm.Allgather(np.array(self._local_number_of_micro_sims), nms_all_ranks)

        # Get global number of micro simulations
        self._global_number_of_micro_sims = np.sum(nms_all_ranks)

        if self._is_adaptivity_on:
            for name, is_data_vector in self._adaptivity_data_names.items():
                if is_data_vector:
                    self._data_for_adaptivity[name] = np.zeros(
                        (self._local_number_of_micro_sims, self._interface.get_dimensions()))
                else:
                    self._data_for_adaptivity[name] = np.zeros((self._local_number_of_micro_sims))

        # Create lists of local and global IDs
        sim_id = np.sum(nms_all_ranks[:self._rank])
        self._global_ids_of_local_sims = []  # DECLARATION
        for i in range(self._local_number_of_micro_sims):
            self._global_ids_of_local_sims.append(sim_id)
            sim_id += 1

        self._micro_sims = [None] * self._local_number_of_micro_sims  # DECLARATION

        if self._is_adaptivity_on:
            # Create micro simulation objects
            for i in range(self._local_number_of_micro_sims):
                self._micro_sims[i] = create_micro_problem_class(
                    self._micro_problem)(self._global_ids_of_local_sims[i])

            # Create a map of micro simulation global IDs and the ranks on which they are
            micro_sims_on_this_rank = np.zeros(self._local_number_of_micro_sims)
            for i in range(self._local_number_of_micro_sims):
                micro_sims_on_this_rank[i] = self._rank

            self._rank_of_sim = np.zeros(self._global_number_of_micro_sims)  # DECLARATION
            self._comm.Allgather(micro_sims_on_this_rank, self._rank_of_sim)

            self._is_sim_on_this_rank = [False] * self._global_number_of_micro_sims  # DECLARATION
            for i in range(self._global_number_of_micro_sims):
                if self._rank_of_sim[i] == self._rank:
                    self._is_sim_on_this_rank[i] = True

            if self._adaptivity_type == "local":
                self._adaptivity_controller = LocalAdaptivityCalculator(
                    self._config, self._logger)
                self._number_of_micro_sims_for_adaptivity = self._local_number_of_micro_sims
            elif self._adaptivity_type == "global":
                self._adaptivity_controller = GlobalAdaptivityCalculator(
                    self._config,
                    self._logger,
                    self._is_sim_on_this_rank,
                    self._rank_of_sim,
                    self._global_ids_of_local_sims,
                    self._comm,
                    self._rank)
                self._number_of_micro_sims_for_adaptivity = self._global_number_of_micro_sims

            self._micro_sims_active_steps = np.zeros(self._local_number_of_micro_sims)
        else:
            for i in range(self._local_number_of_micro_sims):
                self._micro_sims[i] = (
                    create_micro_problem_class(
                        self._micro_problem)(self._global_ids_of_local_sims[i]))

        micro_sims_output = list(range(self._local_number_of_micro_sims))

        # Initialize micro simulations if initialize() method exists
        if hasattr(self._micro_problem, 'initialize') and callable(getattr(self._micro_problem, 'initialize')):
            for i in range(self._local_number_of_micro_sims):
                micro_sims_output[i] = self._micro_sims[i].initialize()
                if micro_sims_output[i] is not None:
                    if self._is_micro_solve_time_required:
                        micro_sims_output[i]["micro_sim_time"] = 0.0
                    if self._is_adaptivity_on:
                        micro_sims_output[i]["active_state"] = 0
                        micro_sims_output[i]["active_steps"] = 0
                else:
                    micro_sims_output[i] = dict()
                    for name, is_data_vector in self._write_data_names.items():
                        if is_data_vector:
                            micro_sims_output[i][name] = np.zeros(self._interface.get_dimensions())
                        else:
                            micro_sims_output[i][name] = 0.0

        self._logger.info("Micro simulations with global IDs {} - {} initialized.".format(
            self._global_ids_of_local_sims[0], self._global_ids_of_local_sims[-1]))

        self._micro_sims_have_output = False
        if hasattr(self._micro_problem, 'output') and callable(getattr(self._micro_problem, 'output')):
            self._micro_sims_have_output = True

        # Write initial data if required
        if self._interface.is_action_required(precice.action_write_initial_data()):
            self.write_data_to_precice(micro_sims_output)
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

    def read_data_from_precice(self) -> list:
        """
        Read data from preCICE. Depending on initial definition of whether a data is scalar or vector, the appropriate
        preCICE API command is called.

        Returns
        -------
        local_read_data : list
            List of dicts in which keys are names of data being read and the values are the data from preCICE.
        """
        read_data = dict()
        for name in self._read_data_names.keys():
            read_data[name] = []

        for name, is_data_vector in self._read_data_names.items():
            if is_data_vector:
                read_data.update({name: self._interface.read_block_vector_data(
                    self._read_data_ids[name], self._mesh_vertex_ids)})
            else:
                read_data.update({name: self._interface.read_block_scalar_data(
                    self._read_data_ids[name], self._mesh_vertex_ids)})

            if self._is_adaptivity_on:
                if name in self._adaptivity_macro_data_names:
                    self._data_for_adaptivity[name] = read_data[name]

        read_data = [dict(zip(read_data, t)) for t in zip(*read_data.values())]

        return read_data

    def write_data_to_precice(self, micro_sims_output: list) -> None:
        """
        Write output of micro simulations to preCICE.

        Parameters
        ----------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data to be written to preCICE.
        """
        write_data = dict()
        if not self._is_rank_empty:
            for name in micro_sims_output[0]:
                write_data[name] = []

            for output_dict in micro_sims_output:
                for name, values in output_dict.items():
                    write_data[name].append(values)

            for dname, is_data_vector in self._write_data_names.items():
                if is_data_vector:
                    self._interface.write_block_vector_data(
                        self._write_data_ids[dname], self._mesh_vertex_ids, write_data[dname])
                else:
                    self._interface.write_block_scalar_data(
                        self._write_data_ids[dname], self._mesh_vertex_ids, write_data[dname])
        else:
            for dname, is_data_vector in self._write_data_names.items():
                if is_data_vector:
                    self._interface.write_block_vector_data(
                        self._write_data_ids[dname], [], np.array([]))
                else:
                    self._interface.write_block_scalar_data(
                        self._write_data_ids[dname], [], np.array([]))

    def solve_micro_simulations(self, micro_sims_input: list) -> list:
        """
        Solve all micro simulations using the data read from preCICE and assemble the micro simulations outputs in a list of dicts
        format.

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
        micro_sims_output = [None] * self._local_number_of_micro_sims

        for count, sim in enumerate(self._micro_sims):
            start_time = time.time()
            micro_sims_output[count] = sim.solve(micro_sims_input[count], self._dt)
            end_time = time.time()

            if self._is_micro_solve_time_required:
                micro_sims_output[count]["micro_sim_time"] = end_time - start_time

        return micro_sims_output

    def solve_micro_simulations_with_adaptivity(self, micro_sims_input: list, micro_sim_states: np.ndarray) -> list:
        """
        Solve all micro simulations using the data read from preCICE and assemble the micro simulations outputs in a list of dicts
        format.

        Parameters
        ----------
        micro_sims_input : list
            List of dicts in which keys are names of data and the values are the data which are required inputs to
            solve a micro simulation.
        micro_sim_states : numpy array
            1D array having state (active or inactive) of each micro simulation

        Returns
        -------
        micro_sims_output : list
            List of dicts in which keys are names of data and the values are the data of the output of the micro
            simulations.
        """
        if self._adaptivity_type == "global":
            active_sim_ids = np.where(
                micro_sim_states[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1] == 1)[0]
            inactive_sim_ids = np.where(
                micro_sim_states[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1] == 0)[0]
        elif self._adaptivity_type == "local":
            active_sim_ids = np.where(micro_sim_states == 1)[0]
            inactive_sim_ids = np.where(micro_sim_states == 0)[0]

        micro_sims_output = [None] * self._local_number_of_micro_sims

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
            self._adaptivity_controller.communicate_micro_output(self._micro_sims, micro_sim_states, micro_sims_output)
        elif self._adaptivity_type == "local":
            for inactive_id in inactive_sim_ids:
                micro_sims_output[inactive_id] = deepcopy(
                    micro_sims_output[self._micro_sims[inactive_id].get_associated_active_id()])

        # Resolve micro sim output data for inactive simulations
        for inactive_id in inactive_sim_ids:
            micro_sims_output[inactive_id]["active_state"] = 0
            micro_sims_output[inactive_id]["active_steps"] = self._micro_sims_active_steps[inactive_id]

            if self._is_micro_solve_time_required:
                micro_sims_output[inactive_id]["micro_sim_time"] = 0

        # Collect micro sim output for adaptivity
        for i in micro_sim_states[self._global_ids_of_local_sims[0]:self._global_ids_of_local_sims[-1] + 1]:
            for name in self._adaptivity_micro_data_names:
                self._data_for_adaptivity[name][i] = micro_sims_output[i][name]

        return micro_sims_output

    def solve(self):
        """
        This function handles the coupling time loop, including checkpointing and output.
        """
        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0

        if self._is_adaptivity_on:
            similarity_dists = np.zeros(
                (self._number_of_micro_sims_for_adaptivity,
                 self._number_of_micro_sims_for_adaptivity))

            # Start adaptivity calculation with all sims inactive
            micro_sim_states = np.zeros((self._number_of_micro_sims_for_adaptivity), dtype=np.intc)

            # If all sims are inactive, activate the first one (a random choice)
            micro_sim_states[0] = 1

            # All inactive sims are associated to the one active sim
            if self._adaptivity_type == "local":
                self._micro_sims[0].activate()
                for sim in self._micro_sims:
                    if not sim.is_active():
                        sim.is_associated_to_active_sim(self._global_ids_of_local_sims[0])
            elif self._adaptivity_type == "global":
                if self._rank == 0:
                    # Only activate one micro simulation globally
                    self._micro_sims[0].activate()
                for sim in self._micro_sims:
                    if not sim.is_active():
                        sim.is_associated_to_active_sim(0)

        similarity_dists_cp = None
        micro_sim_states_cp = None
        micro_sims_cp = None

        while self._interface.is_coupling_ongoing():
            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                for micro_sim in self._micro_sims:
                    micro_sim.save_checkpoint()
                t_checkpoint = t
                n_checkpoint = n

                if self._is_adaptivity_on:
                    if not self._is_adaptivity_required_in_every_implicit_iteration:
                        similarity_dists, micro_sim_states = self._adaptivity_controller.compute_adaptivity(
                            self._dt, self._micro_sims, similarity_dists, micro_sim_states, self._data_for_adaptivity)

                        # Only do checkpointing if adaptivity is computed once in every time window
                        similarity_dists_cp = np.copy(similarity_dists)
                        micro_sim_states_cp = np.copy(micro_sim_states)
                        micro_sims_cp = self._micro_sims.copy()

                    active_sim_ids = np.where(micro_sim_states == 1)[0]
                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

                self._interface.mark_action_fulfilled(
                    precice.action_write_iteration_checkpoint())

            micro_sims_input = self.read_data_from_precice()

            if self._is_adaptivity_on:
                if self._is_adaptivity_required_in_every_implicit_iteration:
                    similarity_dists, micro_sim_states = self._adaptivity_controller.compute_adaptivity(
                        self._dt, self._micro_sims, similarity_dists, micro_sim_states, self._data_for_adaptivity)

                    active_sim_ids = np.where(micro_sim_states == 1)[0]
                    for active_id in active_sim_ids:
                        self._micro_sims_active_steps[active_id] += 1

                micro_sims_output = self.solve_micro_simulations_with_adaptivity(micro_sims_input, micro_sim_states)
            else:
                micro_sims_output = self.solve_micro_simulations(micro_sims_input)

            self.write_data_to_precice(micro_sims_output)

            self._dt = self._interface.advance(self._dt)

            t += self._dt
            n += 1

            # Revert all micro simulations to checkpoints if required
            if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
                for micro_sim in self._micro_sims:
                    micro_sim.reload_checkpoint()
                n = n_checkpoint
                t = t_checkpoint

                if self._is_adaptivity_on:
                    if not self._is_adaptivity_required_in_every_implicit_iteration:
                        similarity_dists = np.copy(similarity_dists_cp)
                        micro_sim_states = np.copy(micro_sim_states_cp)
                        self._micro_sims = micro_sims_cp.copy()

                self._interface.mark_action_fulfilled(
                    precice.action_read_iteration_checkpoint())
            else:  # Time window has converged, now micro output can be generated
                self._logger.info("Micro simulations {} - {} have converged at t = {}".format(
                    self._micro_sims[0].get_global_id(), self._micro_sims[-1].get_global_id(), t))

                if self._micro_sims_have_output:
                    if n % self._micro_n_out == 0:
                        for micro_sim in self._micro_sims:
                            micro_sim.output()

        self._interface.finalize()


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

    manager.initialize()

    manager.solve()


if __name__ == "__main__":
    main()
