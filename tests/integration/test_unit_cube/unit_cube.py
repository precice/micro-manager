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

    t_end = 10

    # preCICE setup
    participant = precice.Participant("macro-cube", "precice-config.xml", 0, 1)
    mesh_name = "macro-cube-mesh"
    read_data_names = {"micro-scalar-data": 0, "micro-vector-data": 1}
    write_data_names = {"macro-scalar-data": 0, "macro-vector-data": 1}

    # Coupling mesh - unit cube with 5 points in each direction
    np_axis = 2
    x_coords, y_coords, z_coords = np.meshgrid(
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis)
    )

    nv = np_axis ** participant.get_mesh_dimensions(mesh_name)
    coords = np.zeros((nv, participant.get_mesh_dimensions(mesh_name)))

    # Define unit cube coordinates
    for z in range(np_axis):
        for y in range(np_axis):
            for x in range(np_axis):
                n = x + y * np_axis + z * np_axis * np_axis
                coords[n, 0] = x_coords[x, y, z]
                coords[n, 1] = y_coords[x, y, z]
                coords[n, 2] = z_coords[x, y, z]

    # Define points on entire domain as coupling mesh
    vertex_ids = participant.set_mesh_vertices(mesh_name, coords)

    write_data = []
    write_data.append(np.zeros(nv))
    write_data.append(np.zeros((nv, participant.get_mesh_dimensions(mesh_name))))

    # Define initial data to write to preCICE
    scalar_value = 1.0
    vector_value = [2.0, 3.0, 4.0]
    for z in range(np_axis):
        for y in range(np_axis):
            for x in range(np_axis):
                n = x + y * np_axis + z * np_axis * np_axis
                write_data[0][n] = scalar_value
                write_data[1][n, 0] = vector_value[0]
                write_data[1][n, 1] = vector_value[1]
                write_data[1][n, 2] = vector_value[2]
        scalar_value += 1
        vector_value = [x + 1 for x in vector_value]

    # Write initial data to preCICE
    if participant.requires_initial_data():
        for count, data_name in enumerate(write_data_names.keys()):
            participant.write_data(mesh_name, data_name, vertex_ids, write_data[count])

    participant.initialize()

    read_data = [None, None]
    dt = participant.get_max_time_step_size()

    # time loop
    while participant.is_coupling_ongoing():
        # write checkpoint
        if participant.requires_writing_checkpoint():
            print("Saving macro state")
            t_checkpoint = t
            n_checkpoint = n

        # Read data from preCICE
        for count, data_name in enumerate(read_data_names.keys()):
            read_data[count] = participant.read_data(mesh_name, data_name, vertex_ids, 1.)

        # Set the read data as the write data with an increment
        write_data[0] = read_data[0] + 1
        write_data[1] = read_data[1] + 1

        # Define new data to write to preCICE midway through the simulation
        if t == t_end / 2:
            scalar_value = 1.0
            vector_value = [2.0, 3.0, 4.0]
            for z in range(np_axis):
                for y in range(np_axis):
                    for x in range(np_axis):
                        n = x + y * np_axis + z * np_axis * np_axis
                        write_data[0][n] = scalar_value
                        write_data[1][n, 0] = vector_value[0]
                        write_data[1][n, 1] = vector_value[1]
                        write_data[1][n, 2] = vector_value[2]

        # Write data to preCICE
        for count, data_name in enumerate(write_data_names.keys()):
            participant.write_data(mesh_name, data_name, vertex_ids, write_data[count])

        participant.advance(dt)
        dt = participant.get_max_time_step_size()

        # advance variables
        n += 1
        t += dt

        if participant.requires_reading_checkpoint():
            print("Reverting to old macro state")
            t = t_checkpoint
            n = n_checkpoint

    participant.finalize()


if __name__ == '__main__':
    main()
