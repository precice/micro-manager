#! /usr/bin/env python3
#

import argparse
import numpy as np
import precice


def main():
    """
    Dummy macro simulation which is coupled to a set of micro simulations via preCICE and the Micro Manager
    """

    parser = argparse.ArgumentParser(description="Macro simulation")
    parser.add_argument("np_axis", type=int, help="Number of points in each axis")
    args = parser.parse_args()

    t = 0

    # preCICE setup
    participant = precice.Participant("macro-cube", "precice-config.xml", 0, 1)
    mesh_name = "macro-cube-mesh"
    read_data_names = {"micro-data-1": 0, "micro-data-2": 1}

    # Coupling mesh - unit cube with 5 points in each direction
    np_axis = args.np_axis
    x_coords, y_coords, z_coords = np.meshgrid(
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis),
        np.linspace(0, 1, np_axis),
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

    participant.initialize()

    read_data = [None, None]
    dt = participant.get_max_time_step_size()

    # time loop
    while participant.is_coupling_ongoing():
        # Read data from preCICE
        for count, data_name in enumerate(read_data_names.keys()):
            read_data[count] = participant.read_data(
                mesh_name, data_name, vertex_ids, 1.0
            )

        participant.write_data(mesh_name, "macro-data-1", vertex_ids, np.ones(nv))

        participant.advance(dt)
        dt = participant.get_max_time_step_size()

        t += dt

    participant.finalize()


if __name__ == "__main__":
    main()
