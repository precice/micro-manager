import sys
sys.path.append('/home/erik/software/python-bindings')  # to use the dummy python bindings
from unittest import TestCase
import micro_manager
import numpy as np


class MicroSimulation:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def solve(self, macro_data, dt):
        return {"micro-scalar-data": 0, "micro-vector-data": [0, 0, 0]}


class TestFunctioncalls(TestCase):
    def test_micromanager_initialize(self):
        print(sys.path)
        manager = micro_manager.MicroManager('test_unit.json')
        self.assertTrue(True)

    def test_initialize(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        self.assertTrue(True)

    def test_decompose_macro_domain(self):
        manager = micro_manager.MicroManager('test_unit.json')
        mesh_bounds = manager.decompose_macro_domain(macro_bounds=[0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(mesh_bounds, [0, 0, 0, 1, 1, 1])

    def test_read_write_data_from_precice(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        fake_write_data = [{"micro-scalar-data": 1, "micro-vector-data": [0, 1, 2], "micro_sim_time": 0}] * 3
        manager.write_data_to_precice(fake_write_data)
        read_data = manager.read_data_from_precice()
        for i in range(3):
            self.assertEqual(read_data[i]["macro-scalar-data"], 1)
            self.assertEqual(read_data[i]["macro-vector-data"].tolist(), [0, 1, 2])

    def test_solve(self):
        manager = micro_manager.MicroManager('test_unit.json')
        manager.initialize()
        manager.solve()
        self.assertTrue(True)

    # def test_compute_adaptivity(self):
    #     manager = micro_manager.MicroManager('test_unit.json')

    #     manager.compute_adaptivity(np.zeros((4,4)),np.ones(4))
    #     self.assertTrue(True)


if __name__ == '__main__':
    import unittest
    unittest.main()
