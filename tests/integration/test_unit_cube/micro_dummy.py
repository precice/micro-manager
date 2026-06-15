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

        match sim_id:
            case 0, 4:
                self._this_sim_type = 1
            case 1, 5:
                self._this_sim_type = 3
            case 2, 6:
                self._this_sim_type = 6
            case 3, 7:
                self._this_sim_type = 9
            case _:
                self._this_sim_type = -1

        # Artificial state
        self._state = [x * 0.1 for x in range(1000)]

    def initialize(self):
        return {
            "Micro-1": self._this_sim_type,
            "Micro-2": [
                self._this_sim_type,
                self._this_sim_type,
                self._this_sim_type,
            ],
        }

    def solve(self, macro_data, dt):
        time.sleep(self._this_sim_type * 0.001)

        return {
            "Micro-1": self._this_sim_type,
            "Micro-2": [
                self._this_sim_type,
                self._this_sim_type,
                self._this_sim_type,
            ],
        }

    def get_state(self):
        return copy.deepcopy(self._state)

    def set_state(self, state):
        self._state = copy.deepcopy(state)

    def get_global_id(self):
        return self._sim_id

    def set_global_id(self, sim_id):
        self._sim_id = sim_id
