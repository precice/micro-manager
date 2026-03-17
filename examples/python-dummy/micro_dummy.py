"""
Micro simulation
In this script we solve a dummy micro problem to just show the working of the macro-micro coupling.
This example shows how to inherit from MicroSimulationInterface provided by the Micro Manager.
"""

from micro_manager import MicroSimulationInterface


class MicroSimulation(MicroSimulationInterface):
    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id
        self._dims = 3
        self._micro_scalar_data = None
        self._micro_vector_data = None
        self._state = None

    def initialize(self, initial_data=None):
        pass

    def solve(self, macro_data, dt):
        assert dt != 0
        self._micro_vector_data = []
        self._micro_scalar_data = macro_data["Macro-Scalar"] + 1
        for d in range(self._dims):
            self._micro_vector_data.append(macro_data["Macro-Vector"][d] + 1)

        return {
            "Micro-Scalar": self._micro_scalar_data.copy(),
            "Micro-Vector": self._micro_vector_data.copy(),
        }

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def get_global_id(self):
        return self._sim_id

    def set_global_id(self, global_id):
        self._sim_id = global_id

    def output(self):
        pass
