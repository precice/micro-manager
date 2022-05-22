#!/usr/bin/env python3
"""
Micro manager to organize many micro simulations and couple them via preCICE to a macro simulation
"""

import precice
from .config import Config
from mpi4py import MPI
from math import sqrt
import numpy as np
from functools import reduce
from operator import iconcat


class MicroManager:
    def __init__(self, config_filename="micro-manager-config.json"):
        """
        Constructor of MicroManager class.
        """
        # MPI related variables
        comm = MPI.COMM_WORLD
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

        print("Provided configuration file: {}".format(config_filename))
        self._config = Config(config_filename)

        micro_file_name = self._config.get_micro_file_name()
        self._micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

        self._interface = precice.Interface("Micro-Manager", self._config.get_config_file_name(),
                                            self._rank, self._size)

    def _decompose_macro_domain(self, macro_bounds):
        size_x = int(sqrt(self._size))
        while self._size % size_x != 0:
            size_x -= 1

        size_y = int(self._size / size_x)

        dx = abs(macro_bounds[0] - macro_bounds[1]) / size_x
        dy = abs(macro_bounds[2] - macro_bounds[3]) / size_y

        local_xmin = dx * (self._rank % size_x)
        local_ymin = dy * int(self._rank / size_x)

        mesh_bounds = []
        if self._interface.get_dimensions() == 2:
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
        elif self._interface.get_dimensions() == 3:
            # TODO: Domain needs to be decomposed optimally in the Z direction too
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy, macro_bounds[4],
                           macro_bounds[5]]

        return mesh_bounds

    def run(self):

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

        # Create all micro simulations
        micro_sims = []
        for _ in range(number_of_micro_simulations):
            micro_sims.append(self._micro_problem())

        # Initialize all micro simulations
        if hasattr(self._micro_problem, 'initialize') and callable(getattr(self._micro_problem, 'initialize')):
            for micro_sim in micro_sims:
                micro_sims_output = micro_sim.initialize()
                if micro_sims_output is not None:
                    for data_name, data in micro_sims_output.items():
                        write_data[data_name].append(data)
                else:
                    for name, is_data_vector in write_data_names.items():
                        if is_data_vector:
                            write_data[name].append(np.zeros(self._interface.get_dimensions()))
                        else:
                            write_data[name].append(0.0)

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

        while self._interface.is_coupling_ongoing():
            # Write checkpoint
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
                micro_sims_output.append(micro_sims[i].solve(micro_sims_input[i], dt))

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

            # Read checkpoint if required
            if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
                for micro_sim in micro_sims:
                    micro_sim.reload_checkpoint()
                n = n_checkpoint
                t = t_checkpoint
                self._interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

        self._interface.finalize()
