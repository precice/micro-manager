"""
Tests for micro_simulation.py covering MicroSimulationInterface,
MicroSimulationLocal, MicroSimulationClass, and create_simulation_class.
"""
import unittest
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

    def test_late_init(self):
        local = MicroSimulationLocal(3, True, MinimalSim)
        self.assertEqual(local.get_global_id(), 3)

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

    def test_getattr_delegates(self):
        local = MicroSimulationLocal(7, False, MinimalSim)
        self.assertEqual(local._gid, 7)


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

    def test_non_interface_class_wrapped(self):
        """Non-interface class should be wrapped and callable."""

        class LegacySim:
            def __init__(self, gid):
                self._gid = gid

            def solve(self, i, dt):
                return {}

            def get_state(self):
                return None

            def set_state(self, s):
                pass

            def get_global_id(self):
                return self._gid

            def set_global_id(self, gid):
                self._gid = gid

        import warnings

        log = MagicMock()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim_cls = create_simulation_class(log, LegacySim, "legacy_path", 1)
            self.assertIsNotNone(sim_cls)


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
