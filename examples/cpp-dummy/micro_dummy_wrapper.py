import ctypes
import numpy as np

class MicroSimulation:
    def __init__(self, sim_id:int):
        self.MicroSimulation_lib = ctypes.CDLL('./micro_dummy.so')        
        self.MicroSimulation_lib.MicroSimulation_solve.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double]
        # return type is a std::pair of float and numpy array
        self.obj = self.MicroSimulation_lib.MicroSimulation_new(sim_id)

    def initialize(self):
        self.MicroSimulation_lib.MicroSimulation_initialize(self.obj)

    def solve(self, macro_data, dt:float):
        macro_scalar_data_ptr = ctypes.c_double(macro_data["macro-scalar-data"])
        macro_vector_data_ptr = macro_data["macro-vector-data"].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_dt = ctypes.c_double(dt)
        self.MicroSimulation_lib.MicroSimulation_solve(self.obj, macro_scalar_data_ptr, macro_vector_data_ptr, c_dt)
        # convert back to double and numpy array
        micro_data = {}
        micro_data["micro-scalar-data"] = macro_scalar_data_ptr.value
        micro_data["micro-vector-data"] = np.ctypeslib.as_array(macro_vector_data_ptr, shape=(3,))
        return micro_data
            
    def save_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_save_checkpoint(self.obj)
    
    def reload_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_reload_checkpoint(self.obj)
