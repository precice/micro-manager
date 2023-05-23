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
    def setUp(self):
        self.fake_read_data_names = {"macro-scalar-data": False, "macro-vector-data": True}
        self.fake_read_data = [{"macro-scalar-data": 1, "macro-vector-data": np.array([0, 1, 2])}] * 4
        self.fake_write_data_names = {
            "micro-scalar-data": False,
            "micro-vector-data": True,
            'micro_sim_time': False,
            'active_state': False,
            'active_steps': False}
        self.fake_write_data = [{"micro-scalar-data": 1,
                                 "micro-vector-data": np.array([0, 1, 2]),
                                 "micro_sim_time": 0,
                                 "active_state": 0,
                                 "active_steps": 0}] * 4
        self.macro_bounds = [0.0, 25.0, 0.0, 25.0, 0.0, 25.0]

    def test_micromanager_constructor(self):
        manager = micro_manager.MicroManager('test_unit.json')
        self.assertListEqual(manager._macro_bounds, self.macro_bounds)
        self.assertDictEqual(manager._read_data_names, self.fake_read_data_names)
        self.assertDictEqual(self.fake_write_data_names, manager._write_data_names)
        self.assertEqual(manager._micro_n_out, 10)

    def test_initialize(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        self.assertEqual(manager._dt, 0.1)  # from Interface.initialize
        self.assertEqual(manager._global_number_of_micro_sims, 4)
        self.assertListEqual(manager._macro_bounds, self.macro_bounds)
        self.assertListEqual(manager._mesh_vertex_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(len(manager._micro_sims), 4)
        self.assertEqual(manager._micro_sims[0].very_important_value, 0)  # test inheritance
        self.assertDictEqual(manager._read_data_names, self.fake_read_data_names)
        self.assertDictEqual(self.fake_write_data_names, manager._write_data_names)

    def test_read_write_data_from_precice(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.write_data_to_precice(self.fake_write_data)
        read_data = manager.read_data_from_precice()
        for data, fake_data in zip(read_data, self.fake_write_data):
            self.assertEqual(data["macro-scalar-data"], 1)
            self.assertListEqual(data["macro-vector-data"].tolist(),
                                 fake_data["micro-vector-data"].tolist())

    def test_solve_mico_sims(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager._local_number_of_micro_sims = 4
        manager._micro_sims = [MicroSimulation() for _ in range(4)]
        manager._micro_sims_active_steps = np.zeros(4, dtype=np.int32)
        micro_sims_output = manager.solve_micro_simulations(self.fake_read_data, np.array([True, True, True, True]))
        for data, fake_data in zip(micro_sims_output, self.fake_write_data):
            self.assertEqual(data["micro-scalar-data"], 2)
            self.assertListEqual(data["micro-vector-data"].tolist(),
                                 (fake_data["micro-vector-data"] + 1).tolist())

    def test_config(self):
        config = micro_manager.Config('test_unit.json')

        self.assertEqual(config._config_file_name.split("/")[-1], "precice-config.xml")
        self.assertEqual(config._micro_file_name, "test_functioncalls")
        self.assertEqual(config._macro_mesh_name, "macro-mesh")
        self.assertEqual(config._micro_output_n, 10)
        self.assertDictEqual(config._read_data_names, self.fake_read_data_names)
        self.assertDictEqual(self.fake_write_data_names, config._write_data_names)

        # test adaptivity
        self.assertEqual(config._adaptivity, True)
        self.assertDictEqual(config._data_for_adaptivity, self.fake_read_data_names)
        self.assertEqual(config._adaptivity_type, "local")
        self.assertEqual(config._adaptivity_history_param, 0.5)
        self.assertEqual(config._adaptivity_coarsening_constant, 0.3)
        self.assertEqual(config._adaptivity_refining_constant, 0.4)
        self.assertEqual(config._adaptivity_every_implicit_iteration, False)


if __name__ == '__main__':
    import unittest
    unittest.main()
