from unittest import TestCase
from unittest.mock import MagicMock

import os
import sys
from pathlib import Path

# no clue why this TestCase does not see the local precice.py
# So we add path here and load it
sys.path.append(str(Path(__file__).resolve().parent))
import precice

import numpy as np
from mpi4py import MPI

from micro_manager.micro_simulation import create_simulation_class
from micro_manager.tasking.connection import spawn_local_workers
from micro_manager.tasking.task import (
    ConstructTask,
    ConstructLateTask,
    InitializeTask,
    OutputTask,
    SolveTask,
    SetStateTask,
    GetStateTask,
)

data_size = 32
num_workers = 2


class MicroSimulation:
    def __init__(self, sim_id):
        self._rank = MPI.COMM_WORLD.Get_rank()
        self._state = "Some State" if sim_id >= 0 else "No State"
        self._sim_id = sim_id

    def initialize(self):
        return "initialized"

    def output(self):
        return "output"

    def solve(self, macro_data, dt):
        data = macro_data["task-data"]

        data_per_rank = data_size // num_workers

        data_local = data[data_per_rank * self._rank : data_per_rank * (self._rank + 1)]
        sum_local = np.asarray(data_local).sum()
        sum_global = MPI.COMM_WORLD.allreduce(sum_local, op=MPI.SUM)
        result = {
            "task-result": sum_global,
        }
        return result

    def get_global_id(self):
        return self._sim_id

    def get_state(self):
        return {
            "gid": self._sim_id,
            "state": self._state,
        }

    def set_state(self, state):
        assert "gid" in state
        assert "state" in state

        self._sim_id = state["gid"]
        self._state = state["state"]


class TestTasking(TestCase):
    """
    Can only test for general functionality, not pinning.
    """

    def setUp(self):
        # cannot use mpi, would need to set certain env flags
        mm_dir = os.path.abspath(str(Path(__file__).resolve().parent.parent.parent))
        worker_exec = os.path.join(mm_dir, "micro_manager", "tasking", "worker_main.py")

        self.conn = spawn_local_workers(
            worker_exec, num_workers, "socket", False, "open", ""
        )
        self.input_data = {"task-data": np.arange(data_size)}
        self.expected_output = (data_size - 1) * data_size / 2
        self.cls_path = os.path.abspath(str(Path(__file__).resolve())).replace(
            ".py", ""
        )
        self.sim_cls = create_simulation_class(
            MagicMock(),
            MicroSimulation,
            self.cls_path,
            num_workers,
            self.conn,
        )

    def tearDown(self):
        super().tearDown()
        if self.conn is not None:
            self.conn.close()

    def test_construct(self):
        gid = 0

        self.send(ConstructTask.send_args(gid, self.cls_path))
        self.recv()

    def test_construct_late(self):
        gid = 0

        self.send(ConstructLateTask.send_args(gid, self.cls_path))
        self.recv()

        base_state = {
            "gid": gid,
            "state": "Important State Information",
        }
        states = [base_state, base_state]
        for i in range(num_workers):
            self.conn.send(i, SetStateTask.send_args(gid, states[i]))
        self.recv()

    def test_initialize(self):
        gid = 0
        sim = self.sim_cls(gid)

        self.send(InitializeTask.send_args(gid))
        result = self.recv()
        for i in range(num_workers):
            self.assertEqual(result[i], "initialized")

    def test_output(self):
        gid = 0
        sim = self.sim_cls(gid)

        self.send(OutputTask.send_args(gid))
        result = self.recv()
        for i in range(num_workers):
            self.assertEqual(result[i], "output")

    def test_solve(self):
        gid = 0
        sim = self.sim_cls(gid)

        result_interface = sim.solve(self.input_data, 0.0)
        self.assertEqual(result_interface["task-result"], self.expected_output)

        self.send(SolveTask.send_args(gid, self.input_data, 0.0))
        result_manual = self.recv()
        self.assertTrue(result_manual[0] is not None)
        self.assertDictEqual(result_interface, result_manual[0])

    def test_get_state(self):
        gid = 0
        sim = self.sim_cls(gid)

        result_interface = list(sim.get_state().values())
        for i in range(num_workers):
            self.assertEqual(result_interface[i]["gid"], gid)
            self.assertEqual(result_interface[i]["state"], "Some State")

        self.send(GetStateTask.send_args(gid))
        result_manual = self.recv()
        self.assertListEqual(result_manual, result_interface)

    def test_set_state(self):
        gid = -1
        gid_new = 0
        sim = self.sim_cls(gid)
        base_state = {
            "gid": gid_new,
            "state": "Important State Information",
        }
        states = [base_state, base_state]

        sim.set_state(states)
        result_interface = sim.get_state()
        self.assertEqual(result_interface[0]["gid"], gid_new)
        self.assertEqual(result_interface[0]["state"], "Important State Information")

        del sim

        sim = self.sim_cls(gid)
        for i in range(num_workers):
            self.conn.send(i, SetStateTask.send_args(gid, states[i]))
        self.recv()

        self.send(GetStateTask.send_args(gid_new))
        result_manual = self.recv()
        self.assertEqual(result_manual[0]["gid"], gid_new)
        self.assertEqual(result_manual[0]["state"], "Important State Information")

    def send(self, args):
        for i in range(num_workers):
            self.conn.send(i, args)

    def recv(self):
        return [self.conn.recv(i) for i in range(num_workers)]


if __name__ == "__main__":
    import unittest

    unittest.main()
