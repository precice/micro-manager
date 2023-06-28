"""
Functionality to create MicroSimulation class objects which inherit from user provided base_micro_simulation class.
"""


def create_micro_problem_class(base_micro_simulation):
    """
    Creates a class MicroSimulation which inherits from the class of the micro simulation.

    Parameters
    ----------
    base_micro_simulation : class
        The base class from the micro simulation script.

    Returns
    -------
    MicroSimulation : class
        Definition of class MicroSimulation defined in this function.
    """
    class MicroSimulation(base_micro_simulation):
        def __init__(self, global_id):
            base_micro_simulation.__init__(self)
            self._global_id = global_id

        def get_global_id(self) -> int:
            return self._global_id

        def set_global_id(self, global_id) -> None:
            self._global_id = global_id

    return MicroSimulation
