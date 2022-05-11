"""
Micro manager to couple a macro code to multiple micro codes
"""

import precice
from .config import Config
from mpi4py import MPI
from math import sqrt
import numpy as np
from functools import reduce
from operator import iconcat


class MicroManager:
    def __init__(self, config_filename='micro-manager-config.json'):
        """
        Constructor of MicroManager class.
        """
        # MPI related variables
        comm = MPI.COMM_WORLD
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

        print("Provided configuration file: {}".format(config_filename))
        config = Config(config_filename)

        micro_file_name = config.get_micro_file_name()
        self._micro_problem = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

        self._dt = config.get_dt()
        self._t_out = config.get_t_output()

        self._interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(),
                                            self._rank, self._size)

        # coupling mesh names and ids
        self._macro_mesh_id = self._interface.get_mesh_id(config.get_macro_mesh_name())

        # Data names and ids of data written to preCICE
        self._write_data_ids = dict()
        self._write_data_names = config.get_write_data_name()
        assert isinstance(self._write_data_names, dict)
        for name in self._write_data_names.keys():
            self._write_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        # Data names and ids of data read from preCICE
        self._read_data_ids = dict()
        self._read_data_names = config.get_read_data_name()
        assert isinstance(self._read_data_names, dict)
        for name in self._read_data_names.keys():
            self._read_data_ids[name] = self._interface.get_data_id(name, self._macro_mesh_id)

        self._macro_bounds = config.get_macro_domain_bounds()

        assert len(self._macro_bounds) / 2 == self._interface.get_dimensions(), "Provided macro mesh bounds are of " \
                                                                                "incorrect dimension"

    def run(self):
        # Domain decomposition
        size_x = int(sqrt(self._size))
        while self._size % size_x != 0:
            size_x -= 1

        size_y = int(self._size / size_x)

        dx = abs(self._macro_bounds[0] - self._macro_bounds[1]) / size_x
        dy = abs(self._macro_bounds[2] - self._macro_bounds[3]) / size_y

        local_xmin = dx * (self._rank % size_x)
        local_ymin = dy * int(self._rank / size_x)

        mesh_bounds = []
        if self._interface.get_dimensions() == 2:
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
        elif self._interface.get_dimensions() == 3:
            # TODO: Domain needs to be decomposed optimally in the Z direction too
            mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy, macro_bounds[4], macro_bounds[5]]

        self._interface.set_mesh_access_region(self._macro_mesh_id, mesh_bounds)

        # Configure data written to preCICE
        write_data = dict()
        for name in self._write_data_names.keys():
            write_data[name] = []

        # Configure data read from preCICE
        read_data = dict()
        for name in self._read_data_names.keys():
            read_data[name] = []

        # initialize preCICE
        precice_dt = self._interface.initialize()
        self._dt = min(precice_dt, self._dt)

        # Get macro mesh from preCICE (API function is experimental)
        mesh_vertex_ids, mesh_vertex_coords = self._interface.get_mesh_vertices_and_ids(self._macro_mesh_id)
        nms, _ = mesh_vertex_coords.shape

        # Create all micro simulations
        micro_sims = []
        for v in range(nms):
            micro_sims.append(self._micro_problem())

        # Initialize all micro simulations
        if hasattr(self._micro_problem, 'initialize') and callable(getattr(self._micro_problem, 'initialize')):
            for v in range(nms):
                micro_sims_output = micro_sims[v].initialize()
                if micro_sims_output is not None:
                    for data_name, data in micro_sims_output.items():
                        write_data[data_name].append(data)
                else:
                    for name, dim in self._write_data_names.items():
                        if dim == 0:
                            write_data[name].append(0.0)
                        elif dim == 1:
                            write_data[name].append(np.zeros(self._interface.get_dimensions()))

        # Initialize coupling data
        if self._interface.is_action_required(precice.action_write_initial_data()):
            for dname, dim in self._write_data_names.items():
                if dim == 1:
                    self._interface.write_block_vector_data(self._write_data_ids[dname], mesh_vertex_ids,
                                                            write_data[dname])
                elif dim == 0:
                    self._interface.write_block_scalar_data(self._write_data_ids[dname], mesh_vertex_ids,
                                                            write_data[dname])
            self._interface.mark_action_fulfilled(precice.action_write_initial_data())

        self._interface.initialize_data()

        t, n = 0, 0
        t_checkpoint, n_checkpoint = 0, 0

        while self._interface.is_coupling_ongoing():
            # Write checkpoint
            if self._interface.is_action_required(precice.action_write_iteration_checkpoint()):
                for v in range(nms):
                    micro_sims[v].save_checkpoint()
                t_checkpoint = t
                n_checkpoint = n
                self._interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

            for name, dims in self._read_data_names.items():
                if dims == 0:
                    read_data.update({name: self._interface.read_block_scalar_data(self._read_data_ids[name],
                                                                                   mesh_vertex_ids)})
                elif dims == 1:
                    read_data.update({name: self._interface.read_block_vector_data(self._read_data_ids[name],
                                                                                   mesh_vertex_ids)})

            micro_sims_input = [dict(zip(read_data, t)) for t in zip(*read_data.values())]
            micro_sims_output = []
            for i in range(nms):
                micro_sims_output.append(micro_sims[i].solve(micro_sims_input[i], self._dt))

            # write_data = {k: reduce(iconcat, [dic[k] for dic in micro_sims_output], []) for k in micro_sims_output[0]}

            write_data = dict()
            for name in micro_sims_output[0]:
                write_data[name] = []

            for dic in micro_sims_output:
                for name, values in dic.items():
                    write_data[name].append(values)

            for dname, dim in self._write_data_names.items():
                if dim == 0:
                    self._interface.write_block_scalar_data(self._write_data_ids[dname], mesh_vertex_ids,
                                                            write_data[dname])
                elif dim == 1:
                    self._interface.write_block_vector_data(self._write_data_ids[dname], mesh_vertex_ids,
                                                            write_data[dname])

            precice_dt = self._interface.advance(self._dt)
            self._dt = min(precice_dt, self._dt)

            t += self._dt
            n += 1

            # Read checkpoint if required
            if self._interface.is_action_required(precice.action_read_iteration_checkpoint()):
                for v in range(nms):
                    micro_sims[v].reload_checkpoint()
                n = n_checkpoint
                t = t_checkpoint
                self._interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

        self._interface.finalize()
