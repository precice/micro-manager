"""
This file provides a function which creates a class Simulation. This class inherits from the user-provided
class MicroSimulation. A global ID member variable is defined for the class Simulation, which ensures that each
created object is uniquely identifiable in a global setting.
"""


def create_simulation_class(micro_simulation_class):
    """
    Creates a class Simulation which inherits from the class of the micro simulation.

    Parameters
    ----------
    base_micro_simulation : class
        The base class from the micro simulation script.

    Returns
    -------
    Simulation : class
        Definition of class Simulation defined in this function.
    """
    class Simulation(micro_simulation_class):
        def __init__(self, global_id):
            micro_simulation_class.__init__(self)
            self._global_id = global_id

        def get_global_id(self) -> int:
            return self._global_id

        def set_global_id(self, global_id) -> None:
            self._global_id = global_id

    return Simulation
