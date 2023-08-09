# This file mocks pyprecice, the Python bindings for preCICE, and is used _only_ for unit testing the Micro Manager.
from typing import Any
import numpy as np


class Participant:
    def __init__(self, solver_name, config_file_name, solver_process_index, solver_process_size):
        self.read_write_vector_buffer = []
        self.read_write_scalar_buffer = []

    def initialize(self):
        return 0.1  # dt

    def advance(self, dt):
        pass

    def finalize(self):
        pass

    def get_mesh_dimensions(self, mesh_name):
        return 3

    def get_data_dimensions(self, mesh_name, data_name):
        return 3

    def is_coupling_ongoing(self):
        yield True
        yield False

    def is_time_window_complete(self):
        return True

    def get_max_time_step_size(self):
        return 0.1  # dt

    def requires_initial_data(self):
        return True

    def requires_writing_checkpoint(self):
        return True

    def requires_reading_checkpoint(self):
        return True

    def set_mesh_access_region(self, mesh_name, bounds):
        pass

    def get_mesh_vertex_ids_and_coordinates(self, mesh_name):
        return np.array([0, 1, 2, 3]), np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def write_data(self, data_name, vertex_ids, data):
        if data_name == "micro-scalar-data":
            self.read_write_scalar_buffer = data

    def read_data(self, data_name, vertex_ids):
        return self.read_write_scalar_buffer
