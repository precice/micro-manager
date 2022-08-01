#!/usr/bin/env python3
"""
Micro manager to organize many micro simulations and couple them via preCICE to a macro simulation
"""

import argparse
import os
import sys
sys.path.append(os.getcwd())
import precice
from .config import Config
from mpi4py import MPI
from math import sqrt
import numpy as np
from functools import reduce
from operator import iconcat
import hashlib
import logging


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
            base_micro_simulation.__init__(self)
            self._id = micro_sim_id

        def get_id(self):
            return self._id

    return MicroProblem


class MicroManager:
    def __init__(self, rank, size, logger):
        """
        Constructor of MicroManager class.

        Parameters
        ----------
        config_filename : string
            Name of the JSON configuration file (to be provided by the user)
        """
        self._size = size
        self._rank = rank
        self._logger = logger

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

        return mesh_bounds

    def initialize(self, interface, micro_problem):
        # Initialize all micro simulations
        if hasattr(micro_problem, 'initialize') and callable(getattr(micro_problem, 'initialize')):
            for micro_sim in micro_sims:
                micro_sims_output = micro_sim.initialize()
                self._logger.info("Micro simulation ({}) initialized.".format(micro_sim.get_id()))
                if micro_sims_output is not None:
                    for data_name, data in micro_sims_output.items():
                        write_data[data_name].append(data)
                else:
                    for name, is_data_vector in write_data_names.items():
                        if is_data_vector:
                            write_data[name].append(np.zeros(interface.get_dimensions()))
                        else:
                            write_data[name].append(0.0)

        # initialize preCICE
        dt = interface.initialize()

        # Initialize coupling data
        if interface.is_action_required(precice.action_write_initial_data()):
            for dname, dim in write_data_names.items():
                if dim == 1:
                    interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
                elif dim == 0:
                    interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
            interface.mark_action_fulfilled(precice.action_write_initial_data())

        interface.initialize_data()

        return dt


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler('micro-manager.log')  # Create file handler which logs messages
    fh.setLevel(logging.INFO)
    # Create formater and add it to handlers
    formatter = logging.Formatter('[' + str(rank) + '] %(name)s -  %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # add the handlers to the logger

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('config_file', type=str, help='Path to the JSON config file of the manager.')

    args = parser.parse_args()
    path = args.config_file
    if not os.path.isabs(path):
        path = os.getcwd() + "/" + path

    logger.info("Provided configuration file: {}".format(config_filename))
    config = Config(path)

    is_parallel = size > 1
    is_rank_empty = False

    # Define the preCICE interface
    interface = precice.Interface("Micro-Manager", config.get_config_file_name(), rank, size)

    manager = MicroManager(rank, size, logger)

    micro_file_name = config.get_micro_file_name()
    micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

    macro_mesh_id = interface.get_mesh_id(config.get_macro_mesh_name())

    # Data names and ids of data written to preCICE
    write_data_names = config.get_write_data_names()
    write_data_ids = dict()
    for name in write_data_names.keys():
        write_data_ids[name] = interface.get_data_id(name, macro_mesh_id)

    # Data names and ids of data read from preCICE
    read_data_names = config.get_read_data_names()
    read_data_ids = dict()
    for name in read_data_names.keys():
        read_data_ids[name] = interface.get_data_id(name, macro_mesh_id)

    write_data = dict()
    for name in write_data_names.keys():
        write_data[name] = []

    read_data = dict()
    for name in read_data_names.keys():
        read_data[name] = []

    # Decompose the macro-domain and set the mesh access region for each partition in preCICE
    macro_bounds = config.get_macro_domain_bounds()
    assert len(macro_bounds) / 2 == interface.get_dimensions(), "Provided macro mesh bounds are of incorrect dimension"
    if is_parallel:
        coupling_mesh_bounds = manager.decompose_macro_domain(macro_bounds)
    else:
        coupling_mesh_bounds = macro_bounds

    interface.set_mesh_access_region(macro_mesh_id, coupling_mesh_bounds)

    mesh_vertex_ids, mesh_vertex_coords = interface.get_mesh_vertices_and_ids(macro_mesh_id)
    number_of_micro_simulations, _ = mesh_vertex_coords.shape
    logger.info("Number of micro simulations = {}".format(number_of_micro_simulations))

    if number_of_micro_simulations == 0:
        if is_parallel:
            logger.info("Rank {} has no micro simulations and hence will not do any computation.".format(rank))
            is_rank_empty = True
        else:
            raise Exception("Micro Manager has no micro simulations.")

    if not is_rank_empty:
        dt = manager.initialize(interface, micro_problem)

    nms_all_ranks = np.zeros(size, dtype=np.int)
    # Gather number of micro simulations that each rank has, because this rank needs to know how many micro
    # simulations have been created by previous ranks, so that it can set the correct IDs
    comm.Allgather(np.array(number_of_micro_simulations), nms_all_ranks)

    # Create all micro simulations
    sim_id = 0
    if rank != 0:
        for i in range(rank - 1, -1, -1):
            sim_id += nms_all_ranks[i]

    micro_sims = []
    for _ in range(number_of_micro_simulations):
        micro_sims.append(create_micro_problem_class(micro_problem)(sim_id))
        sim_id += 1

    t, n = 0, 0
    t_checkpoint, n_checkpoint = 0, 0

    while interface.is_coupling_ongoing():
        # Write checkpoint
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            for micro_sim in micro_sims:
                micro_sim.save_checkpoint()
            t_checkpoint = t
            n_checkpoint = n
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

        for name, is_data_vector in read_data_names.items():
            if is_data_vector:
                read_data.update({name: interface.read_block_vector_data(read_data_ids[name], mesh_vertex_ids)})
            else:
                read_data.update({name: interface.read_block_scalar_data(read_data_ids[name], mesh_vertex_ids)})

        micro_sims_input = [dict(zip(read_data, t)) for t in zip(*read_data.values())]
        micro_sims_output = []
        for i in range(number_of_micro_simulations):
            logger.info("Solving micro simulation ({})".format(micro_sims[i].get_id()))
            micro_sims_output.append(micro_sims[i].solve(micro_sims_input[i], dt))

        write_data = dict()
        if not is_rank_empty:
            for name in micro_sims_output[0]:
                write_data[name] = []

            for dic in micro_sims_output:
                for name, values in dic.items():
                    write_data[name].append(values)

            for dname, is_data_vector in write_data_names.items():
                if is_data_vector:
                    interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
                else:
                    interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
        else:
            for dname, is_data_vector in write_data_names.items():
                if is_data_vector:
                    interface.write_block_vector_data(write_data_ids[dname], [], np.array([]))
                else:
                    interface.write_block_scalar_data(write_data_ids[dname], [], np.array([]))

        dt = self._interface.advance(dt)

        t += dt
        n += 1

        # Revert to checkpoint if required
        if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
            for micro_sim in micro_sims:
                micro_sim.reload_checkpoint()
            n = n_checkpoint
            t = t_checkpoint
            self._interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

    self._interface.finalize()


if __name__ == "__main__":
    main()
