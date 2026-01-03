"""
This file provides a function which creates a class Simulation. This class inherits from the user-provided
class MicroSimulation. A global ID member variable is defined for the class Simulation, which ensures that each
created object is uniquely identifiable in a global setting.
"""


def create_simulation_class(micro_simulation_class, sim_class_name=None):
    """
    Creates a class Simulation which inherits from the class of the micro simulation.

    Parameters
    ----------
    micro_simulation_class : class
        The base class from the micro simulation script.

    sim_class_name : [string, None]
        The name of the class to be created. If None, a unique name will be generated.

    Returns
    -------
    Simulation : class
        Definition of class Simulation defined in this function.
    """
    if sim_class_name is None:
        if not hasattr(create_simulation_class, "sim_id"):
            create_simulation_class.sim_id = 0
        else:
            create_simulation_class.sim_id += 1
        sim_class_name = f"MicroSimulation{create_simulation_class.sim_id}"

    sim_class_body = """
def __init__(self, global_id):
    micro_simulation_class.__init__(self, global_id)
    self._global_id = global_id

def get_global_id(self) -> int:
    return self._global_id
    
def get_state(self):
    if hasattr(micro_simulation_class, "get_state"):
        return super().get_state()
    else:
        return None
        
def set_state(self, state):
    if hasattr(micro_simulation_class, "set_state"):
        super().set_state(state)
    else:
        return
    """
    sim_class_dict = {}
    local_globals = {
        "__builtins__": __builtins__,
        "micro_simulation_class": micro_simulation_class,
    }
    exec(sim_class_body, local_globals, sim_class_dict)
    # print(sim_class_dict)
    sim_class = type(sim_class_name, (micro_simulation_class,), sim_class_dict)

    return sim_class
