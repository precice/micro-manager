"""
Micro simulation
In this script we solve a dummy micro problem to just show the working of the micro simulation snapshot computation.
Set 'crash == True' to make micro simulation 13 crash to see how the parameter is skipped
"""


class MicroSimulation:
    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id
        self._dims = 3
        self._micro_scalar_data = None
        self._micro_vector_data = None
        self._state = None
        self._crash = False

    def solve(self, macro_data, dt):
        self._micro_vector_data = []

        if self._crash == True and macro_data["macro-scalar-data"] == 13:
            raise ValueError("Macro scalar data is unlucky number 13!")
        self._micro_scalar_data = macro_data["macro-scalar-data"] + 1
        for d in range(self._dims):
            self._micro_vector_data.append(macro_data["macro-vector-data"][d] + 1)

        return {
            "micro-scalar-data": self._micro_scalar_data.copy(),
            "micro-vector-data": self._micro_vector_data.copy(),
        }

    def postprocessing(self, micro_data):
        micro_data["micro-scalar-data"] = micro_data["micro-scalar-data"] + 10
        return micro_data

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state
