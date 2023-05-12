import numpy as np
from unittest import TestCase
import micro_manager


class MicroSimulation:
    def __init__(self):
        self.very_important_value = 0

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        assert macro_data["macro-scalar-data"] == 1
        assert macro_data["macro-vector-data"].tolist() == [0, 1, 2]
        return {"micro-scalar-data": macro_data["macro-scalar-data"] + 1,
                "micro-vector-data": macro_data["macro-vector-data"] + 1}


class TestFunctioncalls(TestCase):

    def test_micromanager_initialize(self):
        manager = micro_manager.MicroManager('test_unit.json')
        self.assertTrue(True)

    def test_initialize(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        self.assertEquals(manager._dt, 0.1)  # from Interface.initialize
        self.assertEquals(manager._global_number_of_micro_sims, 4)
        self.assertListEqual(manager._macro_bounds, [0.0, 25.0, 0.0, 25.0, 0.0, 25.0])
        self.assertListEqual(manager._mesh_vertex_ids.tolist(), [0, 1, 2, 3])
        self.assertEquals(len(manager._micro_sims), 4)
        self.assertEquals(manager._micro_sims[0].very_important_value, 0)  # test inheritance
        self.assertDictEqual(manager._read_data_names, {"macro-scalar-data": False, "macro-vector-data": True})
        self.assertDictEqual(
            manager._write_data_names, {
                "micro-scalar-data": False, "micro-vector-data": True, "micro_sim_time": False})

    def test_read_write_data_from_precice(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        fake_write_data = [{"micro-scalar-data": 1, "micro-vector-data": np.array([0, 1, 2]), "micro_sim_time": 0}] * 3
        manager.write_data_to_precice(fake_write_data)
        read_data = manager.read_data_from_precice()
        for i in range(3):
            self.assertEqual(read_data[i]["macro-scalar-data"], 1)
            self.assertListEqual(read_data[i]["macro-vector-data"].tolist(), [0, 1, 2])

    def test_solve_mico_sims(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        micro_sims_output = manager.solve_micro_simulations(
            [{"macro-scalar-data": 1, "macro-vector-data": np.array([0, 1, 2]), "micro_sim_time": 0}] * 4, np.array([True, True, True, True]))
        for i in range(4):
            self.assertEqual(micro_sims_output[i]["micro-scalar-data"], 2)
            self.assertListEqual(micro_sims_output[i]["micro-vector-data"].tolist(), [1, 2, 3])

    def test_config(self):
        config = micro_manager.Config('test_unit.json')
        self.assertEqual(config._config_file_name.split("/")[-1], "precice-config.xml")
        self.assertEqual(config._micro_file_name, "test_functioncalls")
        self.assertEqual(config._macro_mesh_name, "macro-mesh")
        self.assertEqual(config._micro_output_n, 10)
        self.assertDictEqual(config._read_data_names, {"macro-scalar-data": False, "macro-vector-data": True})
        self.assertDictEqual(
            config._write_data_names, {
                "micro-scalar-data": False, "micro-vector-data": True, "micro_sim_time": False})


if __name__ == '__main__':
    import unittest
    unittest.main()
