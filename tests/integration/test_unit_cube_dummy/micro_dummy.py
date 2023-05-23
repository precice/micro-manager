"""
Micro simulation
In this script we solve a dummy micro problem to just show the working of the macro-micro coupling
"""


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        self._micro_scalar_data = None
        self._micro_vector_data = None
        self._checkpoint = None

    def initialize(self):
        self._micro_scalar_data = 0
        self._micro_vector_data = []
        self._checkpoint = 0

    def solve(self, macro_data, dt):
        assert dt != 0
        self._micro_vector_data = macro_data["macro-vector-data"] + 1
        self._micro_scalar_data = macro_data["macro-scalar-data"] + 1

        return {"micro-scalar-data": self._micro_scalar_data,
                "micro-vector-data": self._micro_vector_data}

    def save_checkpoint(self):
        self._checkpoint = self._micro_scalar_data

    def reload_checkpoint(self):
        self._micro_scalar_data = self._checkpoint
