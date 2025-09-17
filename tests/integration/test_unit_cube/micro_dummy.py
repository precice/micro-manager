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

        if sim_id == 0 or sim_id == 4:
            self._this_sim_type = 1
        elif sim_id == 1 or sim_id == 5:
            self._this_sim_type = 3
        elif sim_id == 2 or sim_id == 6:
            self._this_sim_type = 6
        elif sim_id == 3 or sim_id == 7:
            self._this_sim_type = 9

        # Artificial state
        self._state = [x * 0.1 for x in range(1000)]

    def initialize(self):
        return {
            "micro-data-1": self._this_sim_type,
            "micro-data-2": [
                self._this_sim_type,
                self._this_sim_type,
                self._this_sim_type,
            ],
        }

    def solve(self, macro_data, dt):
        time.sleep(self._this_sim_type * 0.001)

        return {
            "micro-data-1": self._this_sim_type,
            "micro-data-2": [
                self._this_sim_type,
                self._this_sim_type,
                self._this_sim_type,
            ],
        }

    def get_state(self):
        return copy.deepcopy(self._state)

    def set_state(self, state):
        self._state = copy.deepcopy(state)
