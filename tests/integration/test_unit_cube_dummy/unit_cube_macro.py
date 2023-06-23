#! /usr/bin/env python3
#

import numpy as np
import precice


def main():
    """
    Dummy macro simulation which is coupled to a set of micro simulations via preCICE and the Micro Manager
    """
    n = n_checkpoint = 0
    t = t_checkpoint = 0

    # preCICE setup
    interface = precice.Interface("macro-cube", "precice-config.xml", 0, 1)

    # define coupling meshes
    read_mesh_name = write_mesh_name = "macro-cube-mesh"
    read_mesh_id = interface.get_mesh_id(read_mesh_name)
    read_data_names = {"micro-scalar-data": 0, "micro-vector-data": 1}

    write_mesh_id = interface.get_mesh_id(write_mesh_name)
    write_data_names = {"macro-scalar-data": 0, "macro-vector-data": 1}

    # Coupling mesh - unit cube with 5 points in each direction
    np_axis = 2
    x_coords, y_coords, z_coords = np.meshgrid(
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis)
    )

    nv = np_axis ** interface.get_dimensions()
    coords = np.zeros((nv, interface.get_dimensions()))

    write_scalar_data = np.zeros(nv)
    write_vector_data = np.zeros((nv, interface.get_dimensions()))

    scalar_value = 1.0
    vector_value = [2.0, 3.0, 4.0]
    for z in range(np_axis):
        for y in range(np_axis):
            for x in range(np_axis):
                n = x + y * np_axis + z * np_axis * np_axis
                coords[n, 0] = x_coords[x, y, z]
                coords[n, 1] = y_coords[x, y, z]
                coords[n, 2] = z_coords[x, y, z]
                write_scalar_data[n] = scalar_value
                write_vector_data[n, 0] = vector_value[0]
                write_vector_data[n, 1] = vector_value[1]
                write_vector_data[n, 2] = vector_value[2]
        scalar_value += 1
        vector_value = [x + 1 for x in vector_value]

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
    dt = interface.initialize()

    # Set initial data to write to preCICE
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

        # Read data from preCICE
        for name, dim in read_data_names.items():
            if dim == 0:
                read_scalar_data = interface.read_block_scalar_data(read_data_ids[name], vertex_ids)
            elif dim == 1:
                read_vector_data = interface.read_block_vector_data(read_data_ids[name], vertex_ids)

        # Set the read data as the write data with an increment
        write_scalar_data = read_scalar_data + 1
        write_vector_data = read_vector_data + 1

        # Write data to preCICE
        for name, dim in write_data_names.items():
            if dim == 0:
                interface.write_block_scalar_data(write_data_ids[name], vertex_ids, write_scalar_data)
            elif dim == 1:
                interface.write_block_vector_data(write_data_ids[name], vertex_ids, write_vector_data)

        # do the coupling
        dt = interface.advance(dt)

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
