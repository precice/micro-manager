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

        def get_id(self):
            return self._id

    return MicroProblem


class MicroManager:
    def __init__(self, config_filename="micro-manager-config.json"):
        """
        Constructor of MicroManager class.

        Parameters
        ----------
        config_filename : string
            Name of the JSON configuration file (to be provided by the user)
        """
        # MPI related variables
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.INFO)
        # Create file handler which logs messages
        fh = logging.FileHandler('micro-manager.log')
        fh.setLevel(logging.INFO)
        # Create formater and add it to handlers
        formatter = logging.Formatter('[' + str(self._rank) + '] %(name)s -  %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add the handlers to the logger
        self._logger.addHandler(fh)

        self._is_parallel = self._size > 1
        self._micro_sims_have_output = False

        print("Provided configuration file: {}".format(config_filename))
        self._config = Config(config_filename)

        micro_file_name = self._config.get_micro_file_name()
        self._micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

        self._interface = precice.Interface("Micro-Manager", self._config.get_config_file_name(),
                                            self._rank, self._size)

    def _decompose_macro_domain(self, macro_bounds):
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

    def run(self):
        """
        This function is called to start running the Micro Manager. This function has all the preCICE API calls.
        """
        macro_mesh_id = self._interface.get_mesh_id(self._config.get_macro_mesh_name())

        # Decompose the macro-domain and set the mesh access region for each partition in preCICE
        macro_bounds = self._config.get_macro_domain_bounds()
        assert len(macro_bounds) / 2 == self._interface.get_dimensions(), "Provided macro mesh bounds are of " \
                                                                          "incorrect dimension"
        coupling_mesh_bounds = self._decompose_macro_domain(macro_bounds)
        self._interface.set_mesh_access_region(macro_mesh_id, coupling_mesh_bounds)

        # Data names and ids of data written to preCICE
        write_data_names = self._config.get_write_data_names()
        write_data_ids = dict()
        for name in write_data_names.keys():
            write_data_ids[name] = self._interface.get_data_id(name, macro_mesh_id)

        is_micro_solve_time_required = self._config.write_micro_sim_solve_time()

        # Data names and ids of data read from preCICE
        read_data_names = self._config.get_read_data_names()
        read_data_ids = dict()
        for name in read_data_names.keys():
            read_data_ids[name] = self._interface.get_data_id(name, macro_mesh_id)

        write_data = dict()
        for name in write_data_names.keys():
            write_data[name] = []

        read_data = dict()
        for name in read_data_names.keys():
            read_data[name] = []

        # initialize preCICE
        dt = self._interface.initialize()

        mesh_vertex_ids, mesh_vertex_coords = self._interface.get_mesh_vertices_and_ids(macro_mesh_id)
        number_of_micro_simulations, _ = mesh_vertex_coords.shape
        assert(number_of_micro_simulations != 0, "Micro manager does not see any macro vertices. This is most likely "
                                                 "because of an irregular number of processors provided for the "
                                                 "parallel run, which leads in an irregular domain decomposition. "
                                                 "Please try to run the micro manager again with a different number "
                                                 "of processors")
        self._logger.info("Number of micro simulations = {}".format(number_of_micro_simulations))

        nms_all_ranks = np.zeros(self._size, dtype=np.int)
        # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
        # simulations have been created by previous ranks, so that it can set the correct IDs
        self._comm.Allgather(np.array(number_of_micro_simulations), nms_all_ranks)

        # Create all micro simulations
        sim_id = 0
        if self._rank != 0:
            for i in range(self._rank - 1, -1, -1):
                sim_id += nms_all_ranks[i]

        micro_sims = []
        for n in range(number_of_micro_simulations):
            micro_sims.append(create_micro_problem_class(self._micro_problem)(sim_id))
            sim_id += 1

        # Initialize all micro simulations
        if hasattr(self._micro_problem, 'initialize') and callable(getattr(self._micro_problem, 'initialize')):
            for micro_sim in micro_sims:
                micro_sims_output = micro_sim.initialize()
                if is_micro_solve_time_required:
                    micro_sims_output["micro_sim_time"] = 0.0
                if micro_sims_output is not None:
                    for data_name, data in micro_sims_output.items():
                        write_data[data_name].append(data)
                else:
                    for name, is_data_vector in write_data_names.items():
                        if is_data_vector:
                            write_data[name].append(np.zeros(self._interface.get_dimensions()))
                        else:
                            write_data[name].append(0.0)

        self._logger.info("Micro simulations {} - {} initialized.".format(micro_sims[0].get_id(),
                                                                          micro_sims[-1].get_id()))

        micro_sims_have_output = False
        if hasattr(self._micro_problem, 'output') and callable(getattr(self._micro_problem, 'output')):
            micro_sims_have_output = True

        # Initialize coupling data
        if self._interface.is_action_required(precice.action_write_initial_data()):
            for dname, dim in write_data_names.items():
                if dim == 1:
                    self._interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
                elif dim == 0:
                    self._interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0
        n_out = self._config.get_micro_output_n()

        while self._interface.is_coupling_ongoing():
            # Write checkpoints for all micro simulations
            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                for micro_sim in micro_sims:
                    micro_sim.save_checkpoint()
                t_checkpoint = t
                n_checkpoint = n
                self._interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

            for name, is_data_vector in read_data_names.items():
                if is_data_vector:
                    read_data.update({name: self._interface.read_block_vector_data(read_data_ids[name],
                                                                                   mesh_vertex_ids)})
                else:
                    read_data.update({name: self._interface.read_block_scalar_data(read_data_ids[name],
                                                                                   mesh_vertex_ids)})

            micro_sims_input = [dict(zip(read_data, t)) for t in zip(*read_data.values())]
            micro_sims_output = []
            for i in range(number_of_micro_simulations):
                start_time = time.time()
                micro_sims_output.append(micro_sims[i].solve(micro_sims_input[i], dt))
                end_time = time.time()
                if is_micro_solve_time_required:
                    micro_sims_output[i]["micro_sim_time"] = end_time - start_time

            write_data = dict()
            for name in micro_sims_output[0]:
                write_data[name] = []

            for dic in micro_sims_output:
                for name, values in dic.items():
                    write_data[name].append(values)

            for dname, is_data_vector in write_data_names.items():
                if is_data_vector:
                    self._interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
                else:
                    self._interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])

            dt = self._interface.advance(dt)

            t += dt
            n += 1

            # Revert all micro simulations to checkpoints if required
            if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
                for micro_sim in micro_sims:
                    micro_sim.reload_checkpoint()
                n = n_checkpoint
                t = t_checkpoint
                self._interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
            else:  # Time window has converged, now micro output can be generated
                self._logger.info("Micro simulations {} - {}: time window t = {} "
                                  "has converged".format(micro_sims[0].get_id(), micro_sims[-1].get_id(), t))

                if micro_sims_have_output:
                    if n % n_out == 0:
                        for micro_sim in micro_sims:
                            micro_sim.output(n)

        self._interface.finalize()


def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('config_file', type=str,
                        help='Path to the config file that should be used.')

    args = parser.parse_args()
    path = args.config_file
    if not os.path.isabs(path):
        path = os.getcwd() + "/" + path
    manager = MicroManager(path)

    manager.run()


if __name__ == "__main__":
    main()
