"""
Tests for micro_simulation.py covering MicroSimulationInterface,
MicroSimulationLocal, MicroSimulationClass, and create_simulation_class.
"""

import unittest
import warnings
from unittest.mock import MagicMock

from micro_manager.micro_simulation import (
    MicroSimulationInterface,
    MicroSimulationLocal,
    create_simulation_class,
)


class MinimalSim(MicroSimulationInterface):
    """Minimal implementation of MicroSimulationInterface."""

    def __init__(self, gid):
        self._gid = gid

    def solve(self, micro_sim_input, dt):
        return {"out": 1}

    def get_state(self):
        return self._gid

    def set_state(self, state):
        self._gid = state

    def get_global_id(self):
        return self._gid

    def set_global_id(self, gid):
        self._gid = gid


class SimWithInitialize(MinimalSim):
    def initialize(self, data=None):
        return {"init": True}


class SimWithOutput(MinimalSim):
    def output(self):
        pass


class TestMicroSimulationInterface(unittest.TestCase):
    def test_requires_initialize_false_when_not_overridden(self):
        sim = MinimalSim(0)
        self.assertFalse(sim.requires_initialize())

    def test_requires_initialize_true_when_overridden(self):
        sim = SimWithInitialize(0)
        self.assertTrue(sim.requires_initialize())

    def test_requires_output_false_when_not_overridden(self):
        sim = MinimalSim(0)
        self.assertFalse(sim.requires_output())

    def test_requires_output_true_when_overridden(self):
        sim = SimWithOutput(0)
        self.assertTrue(sim.requires_output())

    def test_default_initialize_returns_none(self):
        sim = MinimalSim(0)
        self.assertIsNone(sim.initialize())

    def test_default_output_returns_none(self):
        sim = MinimalSim(0)
        self.assertIsNone(sim.output())


class TestMicroSimulationLocal(unittest.TestCase):
    def test_solve(self):
        local = MicroSimulationLocal(0, False, MinimalSim)
        result = local.solve({"in": 1}, 0.1)
        self.assertEqual(result, {"out": 1})

    def test_get_set_state(self):
        local = MicroSimulationLocal(0, False, MinimalSim)
        local.set_state(42)
        self.assertEqual(local.get_state(), 42)

    def test_get_set_global_id(self):
        local = MicroSimulationLocal(5, False, MinimalSim)
        self.assertEqual(local.get_global_id(), 5)
        local.set_global_id(99)
        self.assertEqual(local.get_global_id(), 99)

    def test_late_init_sets_instance_gid_to_minus_one(self):
        """When late_init=True, the wrapped instance should be constructed with gid=-1."""
        local = MicroSimulationLocal(3, True, MinimalSim)
        # The outer local gid should remain 3
        self.assertEqual(local.get_global_id(), 3)
        # The inner instance should have been constructed with -1
        self.assertEqual(local._instance.get_global_id(), -1)

    def test_initialize(self):
        local = MicroSimulationLocal(0, False, SimWithInitialize)
        result = local.initialize({"data": 1})
        self.assertEqual(result, {"init": True})

    def test_output(self):
        local = MicroSimulationLocal(0, False, SimWithOutput)
        self.assertIsNone(local.output())

    def test_requires_initialize(self):
        local = MicroSimulationLocal(0, False, SimWithInitialize)
        self.assertTrue(local.requires_initialize())

    def test_requires_output(self):
        local = MicroSimulationLocal(0, False, SimWithOutput)
        self.assertTrue(local.requires_output())

    def test_getattr_delegates_to_instance(self):
        """__getattr__ should delegate unknown attributes to the wrapped instance."""

        class SimWithExtra(MinimalSim):
            extra_attr = "hello"

        local = MicroSimulationLocal(7, False, SimWithExtra)
        # extra_attr is not defined on MicroSimulationLocal — must come via __getattr__
        self.assertEqual(local.extra_attr, "hello")



class TestMicroSimulationRemote(unittest.TestCase):
    def _make_remote(self, late_init=False):
        from micro_manager.micro_simulation import MicroSimulationRemote

        conn = MagicMock()
        conn.recv.return_value = None
        return (
            MicroSimulationRemote(
                gid=0,
                late_init=late_init,
                num_ranks=1,
                conn=conn,
                cls_path="dummy_path",
                sim_cls=MinimalSim,
            ),
            conn,
        )

    def test_get_set_global_id(self):
        remote, _ = self._make_remote()
        self.assertEqual(remote.get_global_id(), 0)
        remote.set_global_id(42)
        self.assertEqual(remote.get_global_id(), 42)

    def test_solve_returns_worker0_result(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = {"out": 99}
        result = remote.solve({"in": 1}, 0.1)
        self.assertEqual(result, {"out": 99})

    def test_get_state_returns_dict_keyed_by_worker(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = {"gid": 0, "state": "s"}
        state = remote.get_state()
        self.assertIn(0, state)
        self.assertEqual(state[0], {"gid": 0, "state": "s"})

    def test_set_state_updates_gid_from_worker0(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = 7
        remote.set_state({0: {"gid": 7, "state": "s"}})
        self.assertEqual(remote.get_global_id(), 7)

    def test_initialize_returns_worker0_result(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = {"init": True}
        result = remote.initialize()
        self.assertEqual(result, {"init": True})

    def test_output_returns_worker0_result(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = None
        result = remote.output()
        self.assertIsNone(result)

    def test_requires_initialize_false_for_minimal_sim(self):
        remote, _ = self._make_remote()
        self.assertFalse(remote.requires_initialize())

    def test_requires_initialize_true_for_sim_with_initialize(self):
        from micro_manager.micro_simulation import MicroSimulationRemote

        conn = MagicMock()
        conn.recv.return_value = None
        remote = MicroSimulationRemote(
            gid=0,
            late_init=False,
            num_ranks=1,
            conn=conn,
            cls_path="dummy_path",
            sim_cls=SimWithInitialize,
        )
        self.assertTrue(remote.requires_initialize())

    def test_requires_output_false_for_minimal_sim(self):
        remote, _ = self._make_remote()
        self.assertFalse(remote.requires_output())

    def test_requires_output_true_for_sim_with_output(self):
        from micro_manager.micro_simulation import MicroSimulationRemote

        conn = MagicMock()
        conn.recv.return_value = None
        remote = MicroSimulationRemote(
            gid=0,
            late_init=False,
            num_ranks=1,
            conn=conn,
            cls_path="dummy_path",
            sim_cls=SimWithOutput,
        )
        self.assertTrue(remote.requires_output())

    def test_destroy_sends_delete_task_to_all_workers(self):
        remote, conn = self._make_remote()
        conn.recv.return_value = None
        remote.destroy()
        conn.send.assert_called()
        conn.recv.assert_called()

    def test_late_init_uses_construct_late_task(self):
        from micro_manager.micro_simulation import MicroSimulationRemote
        from micro_manager.tasking.task import ConstructLateTask

        conn = MagicMock()
        conn.recv.return_value = None
        remote = MicroSimulationRemote(
            gid=5,
            late_init=True,
            num_ranks=1,
            conn=conn,
            cls_path="dummy_path",
            sim_cls=MinimalSim,
        )
        sent_task = conn.send.call_args_list[0][0][1]
        self.assertEqual(sent_task[0], "ConstructLateTask")



class TestCreateSimulationClass(unittest.TestCase):
    def test_valid_class(self):
        log = MagicMock()
        sim_cls = create_simulation_class(log, MinimalSim, "dummy_path", 1)
        self.assertIsNotNone(sim_cls)

    def test_missing_get_global_id_raises(self):
        class BadSim:
            def solve(self, i, dt):
                pass

            def get_state(self):
                pass

            def set_state(self, s):
                pass

        log = MagicMock()
        with self.assertRaises(ValueError):
            create_simulation_class(log, BadSim, "dummy_path", 1)

    def test_missing_solve_raises(self):
        class BadSim:
            def get_global_id(self):
                pass

            def get_state(self):
                pass

            def set_state(self, s):
                pass

        log = MagicMock()
        with self.assertRaises(ValueError):
            create_simulation_class(log, BadSim, "dummy_path", 1)

    def test_missing_get_state_raises(self):
        class BadSim:
            def get_global_id(self):
                pass

            def set_state(self, s):
                pass

            def solve(self, i, dt):
                pass

        log = MagicMock()
        with self.assertRaises(ValueError):
            create_simulation_class(log, BadSim, "dummy_path", 1)

    def test_missing_set_state_raises(self):
        class BadSim:
            def get_global_id(self):
                pass

            def get_state(self):
                pass

            def solve(self, i, dt):
                pass

        log = MagicMock()
        with self.assertRaises(ValueError):
            create_simulation_class(log, BadSim, "dummy_path", 1)

    def test_custom_sim_class_name(self):
        log = MagicMock()
        sim_cls = create_simulation_class(
            log, MinimalSim, "dummy_path", 1, sim_class_name="MyTestSim"
        )
        self.assertEqual(sim_cls.name, "MyTestSim")

    def test_interface_subclass_accepted_without_wrapping(self):
        """A class that already inherits MicroSimulationInterface is accepted as-is."""
        log = MagicMock()
        sim_cls = create_simulation_class(log, MinimalSim, "dummy_path", 1)
        # backend_cls should be exactly MinimalSim — no wrapping applied
        self.assertIs(sim_cls.backend_cls, MinimalSim)


class TestMicroSimulationClassMethods(unittest.TestCase):
    def setUp(self):
        self.log = MagicMock()
        self.sim_cls = create_simulation_class(self.log, MinimalSim, "dummy_path", 1)
        self.sim_cls_with_init = create_simulation_class(
            self.log, SimWithInitialize, "dummy_path", 1
        )
        self.sim_cls_with_output = create_simulation_class(
            self.log, SimWithOutput, "dummy_path", 1
        )

    def test_check_output_false(self):
        self.assertFalse(self.sim_cls.check_output())

    def test_check_output_true(self):
        self.assertTrue(self.sim_cls_with_output.check_output())

    def test_check_initialize_false(self):
        instance = MinimalSim(0)
        has_init, has_args = self.sim_cls.check_initialize(instance, {})
        self.assertFalse(has_init)
        self.assertFalse(has_args)

    def test_check_initialize_true_no_args(self):
        class SimInitNoArgs(MinimalSim):
            def initialize(self):
                return None

        log = MagicMock()
        sim_cls = create_simulation_class(log, SimInitNoArgs, "dummy", 1)
        instance = SimInitNoArgs(0)
        has_init, has_args = sim_cls.check_initialize(instance, {})
        self.assertTrue(has_init)
        self.assertFalse(has_args)

    def test_check_initialize_true_with_args(self):
        instance = SimWithInitialize(0)
        has_init, has_args = self.sim_cls_with_init.check_initialize(
            instance, {"data": 1}
        )
        self.assertTrue(has_init)
        self.assertTrue(has_args)

    def test_call_creates_wrapper(self):
        wrapper = self.sim_cls(0)
        self.assertIsNotNone(wrapper)

if __name__ == "__main__":
    unittest.main()
