#!/usr/bin/env python3
"""
Snapshot Computation is a tool to initialize and adaptively control micro simulations
and create snapshot databases by simulating a macro simulation via prescribes parameters.
This files the class SnapshotComputation which has the following callable public methods:

- solve

This file is directly executable as it consists of a main() function. Upon execution, an object of the class SnapshotComputation is created using a given JSON file,
and the initialize and solve methods are called.

Detailed documentation: https://precice.org/tooling-micro-manager-overview.html
"""

import argparse
import importlib
import os
import sys
from mpi4py import MPI
import numpy as np
import logging
import time
from copy import deepcopy
from typing import Dict
from warnings import warn

from .config import Config
from .micro_simulation import create_simulation_class
from .domain_decomposition import DomainDecomposer

sys.path.append(os.getcwd())


class SnapshotComputation:
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
        fh = logging.FileHandler('snapshot-computation.log')
        fh.setLevel(logging.INFO)

        # Create formatter and add it to handlers
        formatter = logging.Formatter('[' + str(self._rank) + '] %(name)s -  %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)  # add the handlers to the logger

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        self._logger.info("Provided configuration file: {}".format(config_file))
        self._config = Config(self._logger, config_file)

        self._macro_mesh_name = self._config.get_macro_mesh_name()

        # Data names of data to output - TODO: rename but reuse
        self._write_data_names = self._config.get_write_data_names()

        # Data names of data read as macro input parameters - TODO: rename but reuse
        self._read_data_names = self._config.get_read_data_names()

        self._macro_bounds = self._config.get_macro_domain_bounds()

        if self._is_parallel:  # Simulation is run in parallel
            self._ranks_per_axis = self._config.get_ranks_per_axis()

        self._is_micro_solve_time_required = self._config.write_micro_solve_time()

        self._local_number_of_sims = 0
        self._global_number_of_sims = 0
        self._is_rank_empty = False
        self._dt = 0
        self._mesh_vertex_ids = None  # IDs of macro vertices as set by preCICE TODO: rewrite so it works without precise
        self._micro_n_out = self._config.get_micro_output_n()


        self._initialize()

    # **************
    # Public methods
    # **************

    def solve(self) -> None:
        """
        Solve the problem using given macro parameters.
        - Read macro parameters from the config file, solve micro simulations, and write data to a file.
        - If adaptivity is on, compute micro simulations adaptively.
        """

        # Loop over parameter step sizes

        micro_sims_input = self._read_data_from_precice()  # TODO: replace with own read_data function or just a list of dicts
        
        micro_sims_output = self._solve_micro_simulations(micro_sims_input)

        self._write_data_to_precice(micro_sims_output) # TODO: Replace with write data to file

        self._logger.info("Snapshots for micro simulations {} - {} have been created".format(
            self._micro_sims[0].get_global_id(), self._micro_sims[-1].get_global_id()))


    # ***************
    # Private methods
    # ***************

    def _initialize(self) -> None:
        """
        Initialize the Micro Manager by performing the following tasks:
        - Decompose the domain if the snapshot creation is executed in parallel.
        - Simulate macro information.
        - Create all micro simulation objects and initialize them if an initialize() method is available.
        """
        # Decompose the macro-domain and set the mesh access region for each partition
        assert len(self._macro_bounds) / 2 == self._participant.get_mesh_dimensions(
            self._macro_mesh_name), "Provided macro mesh bounds are of incorrect dimension"  # TODO: Replace with own function
        if self._is_parallel:
            domain_decomposer = DomainDecomposer(
                self._logger, self._participant.get_mesh_dimensions(self._macro_mesh_name), self._rank, self._size)  # TODO: Replace _participant with own function
            coupling_mesh_bounds = domain_decomposer.decompose_macro_domain(self._macro_bounds, self._ranks_per_axis)
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._participant.set_mesh_access_region(self._macro_mesh_name, coupling_mesh_bounds)  # TODO: write own set_mesh_access_region

        self._mesh_vertex_ids, mesh_vertex_coords = self._participant.get_mesh_vertex_ids_and_coordinates(
            self._macro_mesh_name)  # TODO: write own get_mesh_vertex_ids_and_coordinates
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

        # Create lists of local and global IDs
        sim_id = np.sum(nms_all_ranks[:self._rank])
        self._global_ids_of_local_sims = []  # DECLARATION
        for i in range(self._local_number_of_sims):
            self._global_ids_of_local_sims.append(sim_id)
            sim_id += 1

        self._micro_sims = [None] * self._local_number_of_sims  # DECLARATION

        micro_problem = getattr(
            importlib.import_module(
                self._config.get_micro_file_name(), "MicroSimulation"
            ),
            "MicroSimulation")

        # Create micro simulation objects
        for i in range(self._local_number_of_sims):
            self._micro_sims[i] = create_simulation_class(
                micro_problem)(self._global_ids_of_local_sims[i])

        self._logger.info("Micro simulations with global IDs {} - {} created.".format(
            self._global_ids_of_local_sims[0], self._global_ids_of_local_sims[-1]))
        
        self._micro_sims_init = False  # DECLARATION

        # Get initial data from micro simulations if initialize() method exists
        if hasattr(micro_problem, 'initialize') and callable(getattr(micro_problem, 'initialize')):
                self._logger.info(
                    "Micro simulation has the method initialize(), but it is not called, because adaptivity is not used for snapshot computation.")

        self._micro_sims_have_output = False
        if hasattr(micro_problem, 'output') and callable(getattr(micro_problem, 'output')):
            self._micro_sims_have_output = True
            

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

    manager = SnapshotComputation(config_file_path)

    manager.solve()


if __name__ == "__main__":
    main()
