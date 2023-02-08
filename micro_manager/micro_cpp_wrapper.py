import ctypes
import numpy as np

class MicroSimulation:
    def __init__(self, lib_path:str, micro_read_data:dict, micro_write_data:dict, sim_id:int):
        print("Loading MicroSimulation library from", lib_path)
        if not lib_path.endswith(".so"):
            lib_path += ".so"
        self.MicroSimulation_lib = ctypes.CDLL(lib_path)
        self.MicroSimulation_lib.MicroSimulation_solve.argtypes = [ctypes.c_void_p, *([ctypes.POINTER(ctypes.c_double)]*len(micro_write_data)), ctypes.c_double, *([ctypes.POINTER(ctypes.c_double)]*len(micro_read_data)), ] # object, write_data, dt, read_data
        self._micro_read_data = micro_read_data
        self._micro_write_data = micro_write_data
        
        self.obj = self.MicroSimulation_lib.MicroSimulation_new(sim_id)

    def initialize(self):
        self.MicroSimulation_lib.MicroSimulation_initialize(self.obj)

    def solve(self, macro_data:dict, dt:float):
        # convert to ctypes
        write_data_pointers = []
        for data in macro_data.values():
            if isinstance(data, np.ndarray): # vector
                write_data_pointers.append(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            elif isinstance(data, float): # scalar
                write_data_pointers.append(ctypes.c_double(data))
            else:
                raise Exception("Unknown data type")
        read_data_pointers = [ctypes.POINTER(ctypes.c_double) for _ in self._micro_read_data]


        c_dt = ctypes.c_double(dt)
        self.MicroSimulation_lib.MicroSimulation_solve(self.obj, *write_data_pointers, c_dt, *read_data_pointers)
        # convert back to double and numpy array
        micro_data = {}
        for i, key in enumerate(self._micro_read_data.keys()):
            if self._micro_read_data[key] == "scalar":
                micro_data[key] = read_data_pointers[i].contents.value
            elif self._micro_read_data[key] == "vector":
                micro_data[key] = np.ctypeslib.as_array(read_data_pointers[i], shape=(3,))
        
        return micro_data
            
    def save_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_save_checkpoint(self.obj)
    
    def reload_checkpoint(self):
        self.MicroSimulation_lib.MicroSimulation_reload_checkpoint(self.obj)
