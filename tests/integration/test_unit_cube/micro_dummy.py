"""
Micro simulation
In this script we solve a dummy micro problem to just show the working of the macro-micro coupling
"""
import copy
import random
import time


class MicroSimulation:
    def __init__(self, sim_id):
        """
        Constructor of MicroSimulation class.
        """
        self._sim_id = sim_id

        self._n = 0

        sim_types = [4, 88, 37, 12, 1, 23, 134]

        self._this_sim_type = random.choice(sim_types)

        # Artificial state of 100 floats
        self._state = [x * 0.1 for x in range(100)]

    def initialize(self):
        return {
            "micro-data-1": self._this_sim_type * 0.5,
            "micro-data-2": [
                self._this_sim_type * 2,
                self._this_sim_type * 3,
                self._this_sim_type * 4,
            ],
        }

    def solve(self, macro_data, dt):
        time.sleep(self._this_sim_type * 0.001)

        return {
            "micro-data-1": self._this_sim_type * 0.5,
            "micro-data-2": [
                self._this_sim_type * 2,
                self._this_sim_type * 3,
                self._this_sim_type * 4,
            ],
        }

    def get_state(self):
        return copy.deepcopy(self._state)

    def set_state(self, state):
        self._state = copy.deepcopy(state)
