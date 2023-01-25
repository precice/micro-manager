#!/usr/bin/env python3
"""
Micro Manager: a tool to initialize and adaptively control micro simulations and couple them via preCICE to a macro simulation
"""

import argparse
import os
import sys
import precice
from mpi4py import MPI
from math import sqrt, exp
import numpy as np
import logging
import time

from .config import Config
from .adaptivity import AdaptiveController

sys.path.append(os.getcwd())


def create_micro_problem_class(base_micro_simulation):
    """
    Creates a class MicroProblem which inherits from the class of the micro simulation.

    Parameters
    ----------
    base_micro_simulation : class
        The base class from the micro simulation script.

    Returns
    -------
    MicroProblem : class
        Definition of class MicroProblem defined in this function.
    """
    class MicroProblem(base_micro_simulation):
        def __init__(self, local_id, global_id):
            base_micro_simulation.__init__(self, local_id)
            self._local_id = local_id
            self._global_id = global_id
            self._is_active = False
            self._most_similar_active_local_id = 0

        def get_local_id(self):
            return self._local_id

        def get_global_id(self):
            return self._global_id

        def activate(self):
            self._is_active = True

        def deactivate(self):
            self._is_active = False

        def is_most_similar_to(self, similar_active_local_id):
            assert self._is_active is False, "Micro simulation {} is active and hence cannot be most similar to another active simulation".format(
                self._global_id)
            self._most_similar_active_id = similar_active_local_id

        def get_most_similar_active_id(self):
            assert self._is_active is False, "Micro simulation {} is active and hence cannot have a most similar active id".format(
                self._global_id)
            return self._most_similar_active_local_id

        def is_active(self):
            return self._is_active

    return MicroProblem


class MicroManager:
    def __init__(self, config_file) -> None:
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
        config = Config(config_file)

        # Define the preCICE interface
        self._interface = precice.Interface("Micro-Manager", config.get_config_file_name(), self._rank, self._size)

        micro_file_name = config.get_micro_file_name()
        self._micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

        self._macro_mesh_id = self._interface.get_mesh_id(config.get_macro_mesh_name())

        # Data names and ids of data written to preCICE
        self._write_data_names = config.get_write_data_names()
        self._write_data_ids = dict()
        for name in self._write_data_names.keys():
            self._write_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        # Data names and ids of data read from preCICE
        self._read_data_names = config.get_read_data_names()
        self._read_data_ids = dict()
        for name in self._read_data_names.keys():
            self._read_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        self._data_used_for_adaptivity = dict()

        self._macro_bounds = config.get_macro_domain_bounds()
        self._is_micro_solve_time_required = config.write_micro_solve_time()

        self._local_number_of_micro_sims = None
        self._global_number_of_micro_sims = None
        self._is_rank_empty = False
        self._micro_sims = None  # Array carrying micro simulation objects
        self._dt = None
        self._mesh_vertex_ids = None  # IDs of macro vertices as set by preCICE
        self._micro_n_out = config.get_micro_output_n()

        self._is_adaptivity_on = config.turn_on_adaptivity()

        if self._is_adaptivity_on:
            self._adaptivity_controller = AdaptiveController(config)
            self._hist_param = config.get_adaptivity_hist_param()
            self._adaptivity_data_names = config.get_data_for_adaptivity()

            # Names of macro data to be used for adaptivity computation
            self._adaptivity_macro_data_names = dict()
            # Names of micro data to be used for adaptivity computation
            self._adaptivity_micro_data_names = dict()
            for name, is_data_vector in self._adaptivity_data_names.items():
                if name in self._read_data_names:
                    self._adaptivity_macro_data_names[name] = is_data_vector
                if name in self._write_data_names:
                    self._adaptivity_micro_data_names[name] = is_data_vector

    def decompose_macro_domain(self, macro_bounds) -> list:
        """
        Decompose the macro domain equally among all ranks, if the Micro Manager is run in parallel.

        Parameters
        ----------
        macro_bounds : list
            List containing upper and lower bounds of the macro domain.
            Format in 2D is [x_min, x_max, y_min, y_max]
            Format in 2D is [x_min, x_max, y_min, y_max, z_min, z_max]

        Returns
        -------
        mesh_bounds : list
            List containing the upper and lower bounds of the domain pertaining to this rank.
            Format is same as input parameter macro_bounds.
        """
        size_x = int(sqrt(self._size))
        while self._size % size_x != 0:
            size_x -= 1

        size_y = int(self._size / size_x)

        dx = abs(macro_bounds[1] - macro_bounds[0]) / size_x
        dy = abs(macro_bounds[3] - macro_bounds[2]) / size_y

        local_xmin = macro_bounds[0] + dx * (self._rank % size_x)
        local_ymin = macro_bounds[2] + dy * int(self._rank / size_x)

        mesh_bounds = []
        if self._interface.get_dimensions() == 2:
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
        elif self._interface.get_dimensions() == 3:
            # TODO: Domain needs to be decomposed optimally in the Z direction
            # too
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy, macro_bounds[4], macro_bounds[5]]

        self._logger.info("Bounding box limits are {}".format(mesh_bounds))

        return mesh_bounds

    def initialize(self) -> None:
        """
        This function does the following things:
        - If the Micro Manager has been executed in parallel, it decomposes the domain as uniformly as possible.
        - Initializes preCICE.
        - Gets the macro mesh information from preCICE.
        - Creates all micro simulation objects and initializes them if an initialization procedure is available.
        - Writes initial data to preCICE.
        """
        # Decompose the macro-domain and set the mesh access region for each
        # partition in preCICE
        assert len(self._macro_bounds) / \
            2 == self._interface.get_dimensions(), "Provided macro mesh bounds are of incorrect dimension"
        if self._is_parallel:
            coupling_mesh_bounds = self.decompose_macro_domain(self._macro_bounds)
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._interface.set_mesh_access_region(self._macro_mesh_id, coupling_mesh_bounds)

        # initialize preCICE
        self._dt = self._interface.initialize()

        self._mesh_vertex_ids, mesh_vertex_coords = self._interface.get_mesh_vertices_and_ids(self._macro_mesh_id)
        self._local_number_of_micro_sims, _ = mesh_vertex_coords.shape
        self._logger.info("Number of local micro simulations = {}".format(self._local_number_of_micro_sims))

        for name, is_data_vector in self._adaptivity_data_names.items():
            if is_data_vector:
                self._data_used_for_adaptivity[name] = np.zeros(
                    (self._local_number_of_micro_sims, self._interface.get_dimensions()))
            else:
                self._data_used_for_adaptivity[name] = np.zeros((self._local_number_of_micro_sims))

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

        self._adaptivity_controller.set_number_of_sims(self._local_number_of_micro_sims)

        # Create all micro simulations
        sim_id = 0
        if self._rank != 0:
            for i in range(self._rank - 1, -1, -1):
                sim_id += nms_all_ranks[i]

        self._micro_sims = []
        self._micro_sim_global_ids = []
        for i in range(self._local_number_of_micro_sims):
            self._micro_sims.append(create_micro_problem_class(self._micro_problem)(i, sim_id))
            self._micro_sim_global_ids.append(sim_id)
            sim_id += 1

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
                else:
                    micro_sims_output[i] = dict()
                    for name, is_data_vector in self._write_data_names.items():
                        if is_data_vector:
                            micro_sims_output[i][name] = np.zeros(self._interface.get_dimensions())
                        else:
                            micro_sims_output[i][name] = 0.0

        self._logger.info("Micro simulations with global IDs {} - {} initialized.".format(
            self._micro_sim_global_ids[0], self._micro_sim_global_ids[-1]))

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
                if name in self._adaptivity_macro_data_names:
                    self._data_used_for_adaptivity[name] = read_data[name]
            else:
                read_data.update({name: self._interface.read_block_scalar_data(
                    self._read_data_ids[name], self._mesh_vertex_ids)})
                if name in self._adaptivity_macro_data_names:
                    self._data_used_for_adaptivity[name] = read_data[name]

        local_read_data = [dict(zip(read_data, t)) for t in zip(*read_data.values())]

        return local_read_data

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

            for i in range(self._local_number_of_micro_sims):
                for name, values in micro_sims_output[i].items():
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

    def solve_micro_simulations(self, micro_sims_input: dict, similarity_dists_nm1: np.ndarray,
                                micro_sim_states_nm1: np.ndarray):
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
        if self._is_adaptivity_on:
            # Multiply old similarity distance by history term to get current distances
            similarity_dists_n = exp(-self._hist_param * self._dt) * similarity_dists_nm1

            for name, _ in self._adaptivity_data_names.items():
                similarity_dists_n = self._adaptivity_controller.get_similarity_dists(
                    self._dt, similarity_dists_n, self._data_used_for_adaptivity[name])

            micro_sim_states_n = self._adaptivity_controller.update_active_micro_sims(
                similarity_dists_n, micro_sim_states_nm1, self._micro_sims)

            micro_sim_states_n = self._adaptivity_controller.update_inactive_micro_sims(
                similarity_dists_n, micro_sim_states_n, self._micro_sims)

            self._adaptivity_controller.associate_inactive_to_active(
                similarity_dists_n, micro_sim_states_n, self._micro_sims)

            active_sim_ids = np.where(micro_sim_states_n == 1)[0]
            inactive_sim_ids = np.where(micro_sim_states_n == 0)[0]

        else:
            # If adaptivity is off, all micro simulations are active
            active_sim_ids = np.where(micro_sim_states_nm1 == 1)[0]
            inactive_sim_ids = np.where(micro_sim_states_nm1 == 0)[0]

        micro_sims_output = list(range(self._local_number_of_micro_sims))

        # Solve all active micro simulations
        for i in active_sim_ids:
            self._logger.info("Solving active micro sim [{}]".format(self._micro_sims[i].get_global_id()))

            start_time = time.time()
            micro_sims_output[i] = self._micro_sims[i].solve(micro_sims_input[i], self._dt)
            end_time = time.time()

            if self._is_adaptivity_on:
                # Mark the micro sim as active for export
                micro_sims_output[i]["active_state"] = 1

            for name in self._adaptivity_micro_data_names:
                # Collect micro sim output for adaptivity
                self._data_used_for_adaptivity[name][i] = micro_sims_output[i][name]

            if self._is_micro_solve_time_required:
                micro_sims_output[i]["micro_sim_time"] = end_time - start_time

        # For each inactive simulation, copy data from most similar active simulation
        for i in inactive_sim_ids:
            self._logger.info("Micro sim [{}] is inactive. Copying data from most similar active micro " "sim [{}]".format(
                self._micro_sims[i].get_global_id(), self._micro_sim_global_ids[self._micro_sims[i].get_most_similar_active_id()]))

            micro_sims_output[i] = dict()
            for dname, values in micro_sims_output[self._micro_sims[i].get_most_similar_active_id()].items():
                micro_sims_output[i][dname] = values

            start_time = end_time = 0
            micro_sims_output[i]["active_state"] = 0

            for name in self._adaptivity_micro_data_names:
                # Collect micro sim output for adaptivity
                self._data_used_for_adaptivity[name][i] = micro_sims_output[i][name]

            if self._is_micro_solve_time_required:
                micro_sims_output[i]["micro_sim_time"] = end_time - start_time

        return micro_sims_output, similarity_dists_n, micro_sim_states_n

    def solve(self):
        """
        This function handles the coupling time loop, including checkpointing and output.
        """
        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0
        similarity_dists = np.zeros((self._local_number_of_micro_sims, self._local_number_of_micro_sims))
        micro_sim_states = np.zeros((self._local_number_of_micro_sims))

        similarity_dists_cp = None
        micro_sim_states_cp = None

        while self._interface.is_coupling_ongoing():
            # Write checkpoints for all micro simulations
            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                for micro_sim in self._micro_sims:
                    micro_sim.save_checkpoint()
                t_checkpoint = t
                n_checkpoint = n

                if self._is_adaptivity_on:
                    similarity_dists_cp = similarity_dists
                    micro_sim_states_cp = micro_sim_states

                self._interface.mark_action_fulfilled(
                    precice.action_write_iteration_checkpoint())

            micro_sims_input = self.read_data_from_precice()

            micro_sims_output, similarity_dists, micro_sim_states = self.solve_micro_simulations(
                micro_sims_input, similarity_dists, micro_sim_states)

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
                    similarity_dists = similarity_dists_cp
                    micro_sim_states = micro_sim_states_cp

                self._interface.mark_action_fulfilled(
                    precice.action_read_iteration_checkpoint())
            else:  # Time window has converged, now micro output can be generated
                self._logger.info("Micro simulations {} - {}: time window t = {} has converged".format(
                    self._micro_sims[0].get_global_id(), self._micro_sims[-1].get_global_id(), t))

                if self._micro_sims_have_output:
                    if n % self._micro_n_out == 0:
                        for micro_sim in self._micro_sims:
                            micro_sim.output(n)

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
