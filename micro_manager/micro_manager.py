#!/usr/bin/env python3
"""
Micro manager to organize many micro simulations and couple them via preCICE to a macro simulation
"""

import argparse
import os
import sys
import precice
from .config import Config
from mpi4py import MPI
from math import sqrt
import numpy as np
from functools import reduce
from operator import iconcat
import hashlib
import logging
import time

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
        def __init__(self, micro_sim_id):
            base_micro_simulation.__init__(self, micro_sim_id)
            self._id = micro_sim_id
            self._is_active = False

        def get_id(self):
            return self._id

        def activate(self):
            self._is_active = True

        def deactivate(self):
            self._is_active = False

    return MicroProblem


class MicroManager:
    def __init__(self, config_file):
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
        fh = logging.FileHandler('micro-manager.log')  # Create file handler which logs messages
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

        self._macro_bounds = config.get_macro_domain_bounds()
        self._is_micro_solve_time_required = config.write_micro_solve_time()

        self._number_of_micro_simulations = None
        self._is_rank_empty = False
        self._micro_sims = None
        self._dt = None
        self._mesh_vertex_ids = None
        self._micro_n_out = config.get_micro_output_n()

    def decompose_macro_domain(self, macro_bounds):
        """
        Decompose the macro domain equally among all ranks, if the Micro Manager is run in paralle.

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
            # TODO: Domain needs to be decomposed optimally in the Z direction too
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy, macro_bounds[4],
                           macro_bounds[5]]

        self._logger.info("Bounding box limits are {}".format(mesh_bounds))

        return mesh_bounds

    def initialize(self):
        """
        This function does the following things:
        - If the Micro Manager has been executed in parallel, it decomposes the domain as equally as possible.
        - Initializes preCICE.
        - Get the macro mesh information.
        - Creates all micro simulation objects and initializes them if the an initialization procedure is available.
        - Writes initial data to preCICE.
        """
        # Decompose the macro-domain and set the mesh access region for each partition in preCICE
        assert len(
            self._macro_bounds) / 2 == self._interface.get_dimensions(), "Provided macro mesh bounds are of " \
                                                                         "incorrect dimension"
        if self._is_parallel:
            coupling_mesh_bounds = self.decompose_macro_domain(self._macro_bounds)
        else:
            coupling_mesh_bounds = self._macro_bounds

        self._interface.set_mesh_access_region(self._macro_mesh_id, coupling_mesh_bounds)

        # initialize preCICE
        self._dt = self._interface.initialize()

        self._mesh_vertex_ids, mesh_vertex_coords = self._interface.get_mesh_vertices_and_ids(self._macro_mesh_id)
        self._number_of_micro_simulations, _ = mesh_vertex_coords.shape
        self._logger.info("Number of micro simulations = {}".format(self._number_of_micro_simulations))

        if self._number_of_micro_simulations == 0:
            if self._is_parallel:
                self._logger.info("Rank {} has no micro simulations and hence will not do any computation.".format(
                    self._rank))
                self._is_rank_empty = True
            else:
                raise Exception("Micro Manager has no micro simulations.")

        nms_all_ranks = np.zeros(self._size, dtype=np.int64)
        # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
        # simulations have been created by previous ranks, so that it can set the correct IDs
        self._comm.Allgather(np.array(self._number_of_micro_simulations), nms_all_ranks)

        # Create all micro simulations
        sim_id = 0
        if self._rank != 0:
            for i in range(self._rank - 1, -1, -1):
                sim_id += nms_all_ranks[i]

        self._micro_sims = []
        for _ in range(self._number_of_micro_simulations):
            self._micro_sims.append(create_micro_problem_class(self._micro_problem)(sim_id))
            sim_id += 1

        write_data = dict()
        for name in self._write_data_names.keys():
            write_data[name] = []

        # Initialize all micro simulations
        if hasattr(self._micro_problem, 'initialize') and callable(getattr(self._micro_problem, 'initialize')):
            for micro_sim in self._micro_sims:
                micro_sims_output = micro_sim.initialize()
                if micro_sims_output is not None:
                    if self._is_micro_solve_time_required:
                        micro_sims_output["micro_sim_time"] = 0.0

                    for data_name, data in micro_sims_output.items():
                        write_data[data_name].append(data)
                else:
                    for name, is_data_vector in self._write_data_names.items():
                        if is_data_vector:
                            write_data[name].append(np.zeros(self._interface.get_dimensions()))
                        else:
                            write_data[name].append(0.0)

        self._logger.info("Micro simulations {} - {} initialized.".format(self._micro_sims[0].get_id(),
                                                                          self._micro_sims[-1].get_id()))

        self._micro_sims_have_output = False
        if hasattr(self._micro_problem, 'output') and callable(getattr(self._micro_problem, 'output')):
            self._micro_sims_have_output = True

        # Initialize coupling data
        if self._interface.is_action_required(precice.action_write_initial_data()):
            for dname, dim in self._write_data_names.items():
                if dim == 1:
                    self._interface.write_block_vector_data(self._write_data_ids[dname], self._mesh_vertex_ids,
                                                            write_data[dname])
                elif dim == 0:
                    self._interface.write_block_scalar_data(self._write_data_ids[dname], self._mesh_vertex_ids,
                                                            write_data[dname])
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

    def read_data_from_precice(self):
        """
        Read data from preCICE. Depending on initial definition of whether a data is scalar or vector, the appropriate
        preCICE API command is called.

        Returns
        -------
        list : list
            List of dicts in which keys are names of data being read and the values are the data from preCICE.
        """
        read_data = dict()
        for name in self._read_data_names.keys():
            read_data[name] = []

        for name, is_data_vector in self._read_data_names.items():
            if is_data_vector:
                read_data.update({name: self._interface.read_block_vector_data(self._read_data_ids[name],
                                                                               self._mesh_vertex_ids)})
            else:
                read_data.update({name: self._interface.read_block_scalar_data(self._read_data_ids[name],
                                                                               self._mesh_vertex_ids)})

        return [dict(zip(read_data, t)) for t in zip(*read_data.values())]

    def write_data_to_precice(self, micro_sims_output):
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

            for dic in micro_sims_output:
                for name, values in dic.items():
                    write_data[name].append(values)

            for dname, is_data_vector in self._write_data_names.items():
                if is_data_vector:
                    self._interface.write_block_vector_data(self._write_data_ids[dname], self._mesh_vertex_ids,
                                                            write_data[dname])
                else:
                    self._interface.write_block_scalar_data(self._write_data_ids[dname], self._mesh_vertex_ids,
                                                            write_data[dname])
        else:
            for dname, is_data_vector in self._write_data_names.items():
                if is_data_vector:
                    self._interface.write_block_vector_data(self._write_data_ids[dname], [], np.array([]))
                else:
                    self._interface.write_block_scalar_data(self._write_data_ids[dname], [], np.array([]))

    def solve_micro_simulations(self, micro_sims_input):
        """
        Solve all micro simulations using the input data and assemble the micro simulations outputs in a list of dicts
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
        micro_sims_output = []
        for i in range(self._number_of_micro_simulations):
            self._logger.info("Solving micro simulation ({})".format(self._micro_sims[i].get_id()))
            start_time = time.time()
            micro_sims_output.append(self._micro_sims[i].solve(micro_sims_input[i], self._dt))
            end_time = time.time()
            if self._is_micro_solve_time_required:
                micro_sims_output[i]["micro_sim_time"] = end_time - start_time

        return micro_sims_output

    def solve(self):
        """
        This function handles the coupling time loop, including checkpointing and output.
        """
        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0

        while self._interface.is_coupling_ongoing():
            # Write checkpoints for all micro simulations
            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                for micro_sim in self._micro_sims:
                    micro_sim.save_checkpoint()
                t_checkpoint = t
                n_checkpoint = n
                self._interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

            micro_sims_input = self.read_data_from_precice()

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
                self._interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
            else:  # Time window has converged, now micro output can be generated
                self._logger.info("Micro simulations {} - {}: time window t = {} has converged".format(
                    self._micro_sims[0].get_id(), self._micro_sims[-1].get_id(), t))

                if self._micro_sims_have_output:
                    if n % self._micro_n_out == 0:
                        for micro_sim in self._micro_sims:
                            micro_sim.output(n)

        self._interface.finalize()


def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('config_file', type=str, help='Path to the JSON config file of the manager.')

    args = parser.parse_args()
    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    manager = MicroManager(config_file_path)

    manager.initialize()

    manager.solve()


if __name__ == "__main__":
    main()
