#! /usr/bin/env python3
#

import numpy as np
import precice
from config import Config


def main():
    """
    Dummy macro simulation which is coupled to a set of micro simulations via preCICE and the Micro Manager
    """
    config = Config("macro-dummy-config.json")

    nv = 25

    n = n_checkpoint = 0
    t = t_checkpoint = 0
    dt = config.get_dt()

    # preCICE setup
    interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

    # define coupling meshes
    read_mesh_name = config.get_read_mesh_name()
    read_mesh_id = interface.get_mesh_id(read_mesh_name)
    read_data_names = config.get_read_data_name()

    write_mesh_name = config.get_write_mesh_name()
    write_mesh_id = interface.get_mesh_id(write_mesh_name)
    write_data_names = config.get_write_data_name()

    # Coupling mesh
    coords = np.zeros((nv, interface.get_dimensions()))
    for x in range(nv):
        for d in range(interface.get_dimensions()):
            coords[x, d] = x

    # Define Gauss points on entire domain as coupling mesh
    vertex_ids = interface.set_mesh_vertices(read_mesh_id, coords)

    read_data_ids = dict()
    # coupling data
    for name, dim in read_data_names.items():
        read_data_ids[name] = interface.get_data_id(name, read_mesh_id)

    write_data_ids = dict()
    for name, dim in write_data_names.items():
        write_data_ids[name] = interface.get_data_id(name, write_mesh_id)

    # initialize preCICE
    precice_dt = interface.initialize()
    dt = min(precice_dt, dt)

    write_scalar_data = np.zeros(nv)
    write_vector_data = np.zeros((nv, interface.get_dimensions()))

    for i in range(nv):
        write_scalar_data[i] = i
        for d in range(interface.get_dimensions()):
            write_vector_data[i, d] = i

    if interface.is_action_required(precice.action_write_initial_data()):
        for name, dim in write_data_names.items():
            if dim == 0:
                interface.write_block_scalar_data(write_data_ids[name], vertex_ids, write_scalar_data)
            elif dim == 1:
                interface.write_block_vector_data(write_data_ids[name], vertex_ids, write_vector_data)
        interface.mark_action_fulfilled(precice.action_write_initial_data())

    interface.initialize_data()

    # time loop
    while interface.is_coupling_ongoing():
        # write checkpoint
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            print("Saving macro state")
            t_checkpoint = t
            n_checkpoint = n
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

        # Read porosity and apply
        for name, dim in read_data_names.items():
            if dim == 0:
                read_scalar_data = interface.read_block_scalar_data(read_data_ids[name], vertex_ids)
            elif dim == 1:
                read_vector_data = interface.read_block_vector_data(read_data_ids[name], vertex_ids)

        write_scalar_data[:] = read_scalar_data[:]
        for i in range(nv):
            for d in range(interface.get_dimensions()):
                write_vector_data[i, d] = read_vector_data[i, d]

        for name, dim in write_data_names.items():
            if dim == 0:
                interface.write_block_scalar_data(write_data_ids[name], vertex_ids, write_scalar_data)
            elif dim == 1:
                interface.write_block_vector_data(write_data_ids[name], vertex_ids, write_vector_data)

        # do the coupling
        precice_dt = interface.advance(dt)
        dt = min(precice_dt, dt)

        # advance variables
        n += 1
        t += dt

        if interface.is_action_required(precice.action_read_iteration_checkpoint()):
            print("Reverting to old macro state")
            t = t_checkpoint
            n = n_checkpoint
            interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

    interface.finalize()


if __name__ == '__main__':
    main()
