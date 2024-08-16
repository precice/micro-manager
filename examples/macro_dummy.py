#! /usr/bin/env python3
#

import numpy as np
import precice


def main():
    """
    Dummy macro simulation which is coupled to a set of micro simulations via preCICE and the Micro Manager
    """
    nv = 25  # number of vertices

    n = n_checkpoint = 0
    t = t_checkpoint = 0

    # preCICE setup
    interface = precice.Participant("Macro-dummy", "precice-config.xml", 0, 1)

    # define coupling meshes
    read_mesh_name = write_mesh_name = "macro-mesh"
    read_data_names = {"micro-scalar-data": 0, "micro-vector-data": 1}

    write_data_names = {"macro-scalar-data": 0, "macro-vector-data": 1}

    # Coupling mesh
    coords = np.zeros((nv, interface.get_mesh_dimensions(write_mesh_name)))
    for x in range(nv):
        for d in range(interface.get_mesh_dimensions(write_mesh_name)):
            coords[x, d] = x

    # Define Gauss points on entire domain as coupling mesh
    vertex_ids = interface.set_mesh_vertices(read_mesh_name, coords)

    write_scalar_data = np.zeros(nv)
    write_vector_data = np.zeros(
        (nv, interface.get_data_dimensions(write_mesh_name, "macro-vector-data"))
    )

    for i in range(nv):
        write_scalar_data[i] = i
        for d in range(
            interface.get_data_dimensions(write_mesh_name, "macro-vector-data")
        ):
            write_vector_data[i, d] = i

    if interface.requires_initial_data():
        for name, dim in write_data_names.items():
            if dim == 0:
                interface.write_data(
                    write_mesh_name, name, vertex_ids, write_scalar_data
                )
            elif dim == 1:
                interface.write_data(
                    write_mesh_name, name, vertex_ids, write_vector_data
                )

    # initialize preCICE
    interface.initialize()
    dt = interface.get_max_time_step_size()

    # time loop
    while interface.is_coupling_ongoing():
        # write checkpoint
        if interface.requires_writing_checkpoint():
            print("Saving macro state")
            t_checkpoint = t
            n_checkpoint = n

        for name, dim in read_data_names.items():
            if dim == 0:
                read_scalar_data = interface.read_data(
                    read_mesh_name, name, vertex_ids, 1
                )
            elif dim == 1:
                read_vector_data = interface.read_data(
                    read_mesh_name, name, vertex_ids, 1
                )

        write_scalar_data[:] = read_scalar_data[:]
        for i in range(nv):
            for d in range(
                interface.get_data_dimensions(read_mesh_name, "micro-vector-data")
            ):
                write_vector_data[i, d] = read_vector_data[i, d]
                if t > 1:  # to trigger adaptivity after some time
                    # ensure that the data is different from the previous time step
                    # previously inactive microsimulations will be activated
                    write_vector_data[i, d] += np.random.randint(0, 10)

        for name, dim in write_data_names.items():
            if dim == 0:
                interface.write_data(
                    write_mesh_name, name, vertex_ids, write_scalar_data
                )
            elif dim == 1:
                interface.write_data(
                    write_mesh_name, name, vertex_ids, write_vector_data
                )

        # do the coupling
        interface.advance(dt)
        dt = interface.get_max_time_step_size()

        # advance variables
        n += 1
        t += dt

        if interface.requires_reading_checkpoint():
            print("Reverting to old macro state")
            t = t_checkpoint
            n = n_checkpoint

    interface.finalize()


if __name__ == "__main__":
    main()
