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
        # macro_data["macro-scalar-data"]=1.0
        # macro_data["macro-vector-data"]=np.array([1.,2.,3.])
        macro_scalar_data_ptr = ctypes.c_double(macro_data["macro-scalar-data"])
        macro_vector_data_ptr = macro_data["macro-vector-data"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) # now in c++: macro_vector_data_ptr[0] = 1.0, macro_vector_data_ptr[1] = 2.0, macro_vector_data_ptr[2] = 3.0
        c_dt = ctypes.c_double(dt)
        res = self.MicroSimulation_lib.MicroSimulation_solve(self.obj, macro_scalar_data_ptr, macro_vector_data_ptr, c_dt)
        return {"micro-scalar-data": res, "micro-vector-data": [res]*3}

    def save_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_save_checkpoint(self.obj)
    
    def reload_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_reload_checkpoint(self.obj)
