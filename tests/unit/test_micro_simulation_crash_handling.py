import numpy as np
from unittest import TestCase
import micro_manager


class MicroSimulation:
    def __init__(self, sim_id):
        self.very_important_value = 0
        self.sim_id = sim_id
        self.current_time = 0

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        if self.sim_id == 0:
            self.current_time += dt
            if self.current_time > dt:
                raise Exception("Crash")

        return {"micro-scalar-data": macro_data["macro-scalar-data"] + 1,
                "micro-vector-data": macro_data["macro-vector-data"] + 1}


class TestSimulationCrashHandling(TestCase):
    def setUp(self):
        self.fake_read_data_names = {
            "macro-scalar-data": False, "macro-vector-data": True}
        self.fake_read_data = [{"macro-scalar-data": 1,
                                "macro-vector-data": np.array([0, 1, 2])}] * 10
        self.fake_write_data = [{"micro-scalar-data": 1,
                                 "micro-vector-data": np.array([0, 1, 2]),
                                 "micro_sim_time": 0,
                                 "active_state": 0,
                                 "active_steps": 0}] * 10

    def test_crash_handling(self):
        """
        Test if the micro manager catches a simulation crash and handles it adequately.
        """
        manager = micro_manager.MicroManager('micro-manager-config_crash.json')

        manager._local_number_of_sims = 10
        manager._crashed_sims = [False] * 10
        manager._micro_sims = [MicroSimulation(i) for i in range(10)]
        manager._micro_sims_active_steps = np.zeros(10, dtype=np.int32)
        # Crash during first time step has to be handled differently

        micro_sims_output = manager._solve_micro_simulations(
            self.fake_read_data)
        for i, data in enumerate(micro_sims_output):
            self.fake_read_data[i]["macro-scalar-data"] = data["micro-scalar-data"]
            self.fake_read_data[i]["macro-vector-data"] = data["micro-vector-data"]
        micro_sims_output = manager._solve_micro_simulations(
            self.fake_read_data)
        # The crashed simulation should have the same data as the previous step
        data_crashed = micro_sims_output[0]
        self.assertEqual(data_crashed["micro-scalar-data"], 2)
        self.assertListEqual(data_crashed["micro-vector-data"].tolist(),
                             (self.fake_write_data[0]["micro-vector-data"] + 1).tolist())
        # Non-crashed simulations should have updated data
        data_normal = micro_sims_output[1]
        self.assertEqual(data_normal["micro-scalar-data"], 3)
        self.assertListEqual(data_normal["micro-vector-data"].tolist(),
                             (self.fake_write_data[1]["micro-vector-data"] + 2).tolist())

    def test_crash_handling_with_adaptivity(self):
        """
        Test if the micro manager catches a simulation crash and handles it adequately with adaptivity.
        """
        manager = micro_manager.MicroManager('micro-manager-config_crash.json')

        manager._local_number_of_sims = 10
        manager._crashed_sims = [False] * 10
        manager._micro_sims = [MicroSimulation(i) for i in range(10)]
        manager._micro_sims_active_steps = np.zeros(10, dtype=np.int32)
        is_sim_active = np.array(
            [True, True, False, True, False, False, False, True, True, False,])
        sim_is_associated_to = np.array([-2, -2, 1, -2, 3, 3, 0, -2, -2, 8])
        # Crash in the first time step is handled differently

        micro_sims_output = manager._solve_micro_simulations_with_adaptivity(
            self.fake_read_data, is_sim_active, sim_is_associated_to)
        for i, data in enumerate(micro_sims_output):
            self.fake_read_data[i]["macro-scalar-data"] = data["micro-scalar-data"]
            self.fake_read_data[i]["macro-vector-data"] = data["micro-vector-data"]
        micro_sims_output = manager._solve_micro_simulations_with_adaptivity(
            self.fake_read_data, is_sim_active, sim_is_associated_to)
        # The crashed simulation should have the same data as the previous step
        data_crashed = micro_sims_output[0]
        self.assertEqual(data_crashed["micro-scalar-data"], 2)
        self.assertListEqual(data_crashed["micro-vector-data"].tolist(),
                             (self.fake_write_data[0]["micro-vector-data"] + 1).tolist())
        # Non-crashed simulations should have updated data
        data_normal = micro_sims_output[1]
        self.assertEqual(data_normal["micro-scalar-data"], 3)
        self.assertListEqual(data_normal["micro-vector-data"].tolist(),
                             (self.fake_write_data[1]["micro-vector-data"] + 2).tolist())


if __name__ == '__main__':
    import unittest
    unittest.main()
