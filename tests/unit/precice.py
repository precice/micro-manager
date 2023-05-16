# This file mocks  pyprecice, the Python bindings for preCICE and is used _only_ for unit testing the Micro Manager.
import numpy as np


def action_write_initial_data():
    return "ActionWriteInitialData"


def action_write_iteration_checkpoint():
    return "ActionWriteIterationCheckpoint"


class Interface:
    def __init__(self, solver_name, config_file_name, solver_process_index, solver_process_size):
        self.read_write_vector_buffer = []
        self.read_write_scalar_buffer = []

    def get_mesh_id(self, mesh_name):
        return 0

    def get_data_id(self, data_name, mesh_id):
        return int(data_name == "micro-scalar-data")

    def get_dimensions(self):
        return 3

    def set_mesh_access_region(self, mesh_id, bounds):
        pass

    def initialize(self):
        return 0.1  # dt

    def get_mesh_vertices_and_ids(self, mesh_id):
        return np.array([0, 1, 2, 3]), np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def is_action_required(self, action):
        return True

    def mark_action_fulfilled(self, action):
        pass

    def initialize_data(self):
        pass

    def write_block_scalar_data(self, data_id, vertex_ids, data):
        if data_id == 1:  # micro-scalar-data not micro_sim_time
            self.read_write_scalar_buffer = data
        print("write_block_scalar_data", data)

    def write_block_vector_data(self, data_id, vertex_ids, data):
        self.read_write_vector_buffer = data
        print("write_block_vector_data", data)

    def write_scalar_data(self, data_id, vertex_id, data):
        pass

    def write_vector_data(self, data_id, vertex_id, data):
        pass

    def read_block_scalar_data(self, data_id, vertex_ids):
        return self.read_write_scalar_buffer

    def read_block_vector_data(self, data_id, vertex_ids):
        return self.read_write_vector_buffer

    def read_scalar_data(self, data_id, vertex_id):
        return 0

    def read_vector_data(self, data_id, vertex_id):
        return [0, 0]

    def finalize(self):
        pass

    def is_coupling_ongoing(self):
        yield True
        yield False

    def advance(self, dt):
        pass
