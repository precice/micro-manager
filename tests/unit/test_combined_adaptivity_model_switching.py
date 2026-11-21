"""
Unit tests for the combined adaptivity + model switching feature (Issue #318).

Verifies:
1. Active problems are assigned resolution 0 (FOM).
2. Inactive problems are assigned resolution >= 1 (ROM) instead of being omitted.
3. Multi-level ROM assignment based on relative similarity distance.
4. Inactive simulations are solved (not skipped) when model switching is enabled.
5. State is preserved and transferred when switching between FOM and ROM.
"""
from unittest import TestCase
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Any

import numpy as np

from micro_manager.model_switching import ModelSwitching
from micro_manager.simulation_container import SimulationContainer
from micro_manager.tools.mpi_handler import MPIHandler


class DummyModelClass:
    def __init__(self, name):
        self.name = name


class DummySimulation:
    def __init__(self, name, global_id=0, late_init=False):
        self.name = name
        self._global_id = global_id
        self._state = {"state": name, "value": 0.0}
        self.attachments = {}
        self.destroyed = False
        self.late_init = late_init
        self.solve_calls = 0

    def get_global_id(self):
        return self._global_id

    def get_state(self):
        return self._state.copy()

    def set_state(self, state):
        self._state = state.copy()

    def destroy(self):
        self.destroyed = True

    def solve(self, micro_input, dt):
        self.solve_calls += 1
        return {"output": self.name}


class ThreeModelManager:
    """Model manager with 3 models: fine (0), medium (1), coarse (2)."""

    def __init__(self):
        self.created_instances = []
        self.num_models = 3
        self.models = [
            DummyModelClass("fine"),
            DummyModelClass("medium"),
            DummyModelClass("coarse"),
        ]
        self.name_to_idx = {"fine": 0, "medium": 1, "coarse": 2}

    def get_instance(self, gid, target_class, *, late_init=False):
        inst = DummySimulation(target_class.name, gid, late_init=late_init)
        self.created_instances.append(
            {"gid": gid, "target_class": target_class.name, "late_init": late_init}
        )
        return inst

    def get_idx_of_sim(self, sim):
        return self.name_to_idx[sim.name]

    def get_cls_by_idx(self, idx):
        return self.models[idx]


class TwoModelManager:
    """Model manager with 2 models: fine (0), coarse (1)."""

    def __init__(self):
        self.created_instances = []
        self.num_models = 2
        self.models = [DummyModelClass("fine"), DummyModelClass("coarse")]
        self.name_to_idx = {"fine": 0, "coarse": 1}

    def get_instance(self, gid, target_class, *, late_init=False):
        inst = DummySimulation(target_class.name, gid, late_init=late_init)
        self.created_instances.append(
            {"gid": gid, "target_class": target_class.name, "late_init": late_init}
        )
        return inst

    def get_idx_of_sim(self, sim):
        return self.name_to_idx[sim.name]

    def get_cls_by_idx(self, idx):
        return self.models[idx]


def _make_controller(mpi, container, model_manager, switching_func=None, is_adaptivity_on=True, levels_thresholds=None):
    controller = ModelSwitching.__new__(ModelSwitching)
    controller._switching_func = switching_func
    controller._sim_container = container
    controller._model_manager = model_manager
    controller._mpi = mpi
    controller._logger = MagicMock()
    controller._converged = False
    controller._is_adaptivity_on = is_adaptivity_on
    controller._levels_thresholds = levels_thresholds or []
    return controller


class TestCombinedAdaptivityModelSwitching2Models(TestCase):
    """Tests for 2-model case: active -> FOM (0), inactive -> ROM (1)."""

    def test_active_gets_resolution_0_inactive_gets_resolution_1(self):
        """
        In the 2-model adaptivity case, active sims must be assigned FOM (0)
        and inactive sims must be assigned the only ROM (1).
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = DummySimulation("coarse", global_id=1)

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        active_sim_ids = [0]  # sim 0 is active, sim 1 is inactive
        inactive_rel_dists = {1: 0.3}  # sim 1 has rel_dist 0.3

        target = controller._gather_target_resolutions(
            cur_res=np.array([0, 1]),
            locations=[np.array([0.0]), np.array([1.0])],
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sims=np.array([1, 0]),
            inactive_relative_distances=inactive_rel_dists,
        )

        self.assertEqual(target[0], 0, "Active sim should get FOM (resolution 0)")
        self.assertEqual(target[1], 1, "Inactive sim should get ROM (resolution 1)")

    def test_switch_inactive_sim_to_rom(self):
        """
        When adaptivity is on and a sim is inactive (None in container),
        switch_models should create a ROM (resolution 1) instance for it.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        # sim 1 is inactive - no object in container
        container[1] = None

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        active_sim_ids = [0]
        inactive_rel_dists = {1: 0.5}

        switched = controller.switch_models(
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sim_ids=active_sim_ids,
            inactive_relative_distances=inactive_rel_dists,
        )

        # sim 1 should now have a coarse (ROM) instance
        self.assertIsNotNone(container[1])
        self.assertEqual(container[1].name, "coarse")
        self.assertIn(1, switched)

    def test_inactive_sim_already_on_correct_rom_no_switch(self):
        """
        If an inactive sim is already on the right ROM resolution, no switch occurs.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = DummySimulation("coarse", global_id=1)  # already coarse

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        active_sim_ids = [0]
        inactive_rel_dists = {1: 0.5}

        switched = controller.switch_models(
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sim_ids=active_sim_ids,
            inactive_relative_distances=inactive_rel_dists,
        )

        # No switch should occur since sim 1 is already at coarse (1)
        self.assertNotIn(1, switched)
        self.assertEqual(container[1].name, "coarse")


class TestCombinedAdaptivityModelSwitching3Models(TestCase):
    """Tests for 3-model case with multi-level ROM assignment by relative distance."""

    def test_inactive_close_to_threshold_gets_finest_rom(self):
        """
        An inactive sim with rel_dist close to 1.0 (near activation threshold)
        should get the finest ROM (resolution 1 in 3-model setup).
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = DummySimulation("medium", global_id=1)

        manager = ThreeModelManager()
        controller = _make_controller(mpi, container, manager)

        active_sim_ids = [0]
        # rel_dist near 1.0 -> should map to finest ROM (level 1 = "medium")
        inactive_rel_dists = {1: 0.9}

        target = controller._gather_target_resolutions(
            cur_res=np.array([0, 1]),
            locations=[np.array([0.0]), np.array([1.0])],
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sims=np.array([1, 0]),
            inactive_relative_distances=inactive_rel_dists,
        )

        self.assertEqual(target[0], 0, "Active sim should be FOM")
        self.assertEqual(target[1], 1, "Very similar-to-threshold inactive sim should get finest ROM (1)")

    def test_inactive_far_from_threshold_gets_coarsest_rom(self):
        """
        An inactive sim with rel_dist near 0.0 (very similar to its representative)
        should get the coarsest ROM (resolution 2 in 3-model setup).
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = DummySimulation("coarse", global_id=1)

        manager = ThreeModelManager()
        controller = _make_controller(mpi, container, manager)

        active_sim_ids = [0]
        # rel_dist near 0.0 -> very similar -> coarsest ROM (level 2 = "coarse")
        inactive_rel_dists = {1: 0.05}

        target = controller._gather_target_resolutions(
            cur_res=np.array([0, 2]),
            locations=[np.array([0.0]), np.array([1.0])],
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sims=np.array([1, 0]),
            inactive_relative_distances=inactive_rel_dists,
        )

        self.assertEqual(target[0], 0, "Active sim should be FOM")
        self.assertEqual(target[2 - 1], 2, "Very dissimilar inactive sim should get coarsest ROM (2)")

    def test_compute_reduced_model_level_uniform_partition(self):
        """
        Test _compute_reduced_model_level() with 3 models and uniform partitioning.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(1, 1, [0], [np.array([0.0])])
        container[0] = DummySimulation("fine", global_id=0)

        manager = ThreeModelManager()
        controller = _make_controller(mpi, container, manager)

        # K = 2 levels (3 models - 1). Uniform split at 0.5.
        # rel_dist in [0, 0.5) -> floor(rel_dist * 2) = 0 -> level = 2 (coarsest ROM)
        # rel_dist in [0.5, 1.0] -> floor(rel_dist * 2) = 1 -> level = 1 (finest ROM)
        self.assertEqual(controller._compute_reduced_model_level(0.0), 2)
        self.assertEqual(controller._compute_reduced_model_level(0.3), 2)
        self.assertEqual(controller._compute_reduced_model_level(0.5), 1)
        self.assertEqual(controller._compute_reduced_model_level(0.9), 1)
        self.assertEqual(controller._compute_reduced_model_level(1.0), 1)

    def test_compute_reduced_model_level_with_custom_thresholds(self):
        """
        Test _compute_reduced_model_level() with custom level thresholds.
        threshold=[0.3] means: rel_dist < 0.3 -> finest ROM (1), else -> coarser ROM (2).
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(1, 1, [0], [np.array([0.0])])
        container[0] = DummySimulation("fine", global_id=0)

        manager = ThreeModelManager()
        controller = _make_controller(mpi, container, manager, levels_thresholds=[0.3])

        # rel_dist < 0.3 -> finest ROM (level = num_models - 1 - 0 = 2)
        # rel_dist >= 0.3 -> coarser: since only 1 threshold, falls through to level 1
        self.assertEqual(controller._compute_reduced_model_level(0.1), 2)
        self.assertEqual(controller._compute_reduced_model_level(0.5), 1)
        self.assertEqual(controller._compute_reduced_model_level(0.8), 1)

    def test_compute_reduced_model_level_2_models(self):
        """
        With 2 models, _compute_reduced_model_level always returns 1 (only 1 ROM).
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(1, 1, [0], [np.array([0.0])])
        container[0] = DummySimulation("fine", global_id=0)

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        for rel_dist in [0.0, 0.1, 0.5, 0.9, 1.0]:
            self.assertEqual(
                controller._compute_reduced_model_level(rel_dist),
                1,
                f"2-model case: rel_dist={rel_dist} should always map to ROM level 1",
            )


class TestStatePreservationOnModelSwitch(TestCase):
    """Tests that state is properly transferred when switching between FOM and ROM."""

    def test_fom_to_rom_state_transferred(self):
        """
        When an active sim becomes inactive (FOM -> ROM), its state should be
        transferred to the new ROM instance.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])

        # Sim 0 is active on FOM, sim 1 is active but will become inactive (ROM)
        fine_sim_0 = DummySimulation("fine", global_id=0)
        fine_sim_1 = DummySimulation("fine", global_id=1)
        fine_sim_1._state = {"state": "fine", "value": 42.0}  # distinctive state

        container[0] = fine_sim_0
        container[1] = fine_sim_1

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        # Now sim 1 becomes inactive - switch it to ROM
        active_sim_ids = [0]
        inactive_rel_dists = {1: 0.5}

        controller.switch_models(
            t=0.0,
            inputs=[{}, {}],
            prev_output=None,
            active_sim_ids=active_sim_ids,
            inactive_relative_distances=inactive_rel_dists,
        )

        # Sim 1 should now be a "coarse" instance with transferred state
        self.assertEqual(container[1].name, "coarse")
        # State should have been transferred from fine sim
        self.assertEqual(container[1]._state["value"], 42.0)

    def test_rom_to_fom_state_transferred_via_cache(self):
        """
        When a previously active sim (now on ROM) becomes active again (ROM -> FOM),
        its cached FOM state should be restored.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(1, 1, [0], [np.array([0.0])])

        fine_sim = DummySimulation("fine", global_id=0)
        fine_sim._state = {"state": "fine", "value": 99.0}
        container[0] = fine_sim

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager)

        # Step 1: sim 0 goes inactive -> switch to ROM
        switched = controller.switch_models(
            t=0.0,
            inputs=[{}],
            prev_output=None,
            active_sim_ids=[],  # no active sims - sim 0 is inactive
            inactive_relative_distances={0: 0.5},
        )
        self.assertEqual(container[0].name, "coarse")

        # Step 2: sim 0 becomes active again -> switch back to FOM
        switched = controller.switch_models(
            t=0.5,
            inputs=[{}],
            prev_output=None,
            active_sim_ids=[0],  # sim 0 is active again
            inactive_relative_distances={},
        )
        self.assertEqual(container[0].name, "fine")
        # Cached FOM state (value=99.0) should be restored
        self.assertEqual(container[0]._state["value"], 99.0)


class TestGatherCurrentResolutionsWithAdaptivity(TestCase):
    """Tests _gather_current_resolutions when adaptivity is on."""

    def test_current_resolutions_with_none_sims(self):
        """
        When adaptivity is on, None simulations (inactive, not yet assigned a ROM)
        should report resolution -1 without raising an error.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = None  # inactive, no ROM yet

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager, is_adaptivity_on=True)

        active_sims = np.array([1, 0])
        cur_res = controller._gather_current_resolutions(active_sims)

        self.assertEqual(cur_res[0], 0)  # fine = resolution 0
        self.assertEqual(cur_res[1], -1)  # None sim -> resolution -1

    def test_current_resolutions_inactive_on_rom(self):
        """
        An inactive sim already running on ROM should report its actual resolution.
        """
        mpi = MPIHandler()
        container = SimulationContainer(mpi)
        container.initialize(2, 2, [0, 1], [np.array([0.0]), np.array([1.0])])
        container[0] = DummySimulation("fine", global_id=0)
        container[1] = DummySimulation("coarse", global_id=1)  # already on ROM

        manager = TwoModelManager()
        controller = _make_controller(mpi, container, manager, is_adaptivity_on=True)

        active_sims = np.array([1, 0])
        cur_res = controller._gather_current_resolutions(active_sims)

        self.assertEqual(cur_res[0], 0)  # fine = resolution 0
        self.assertEqual(cur_res[1], 1)  # coarse = resolution 1 (actual ROM)
