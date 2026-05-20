from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from mpi4py import MPI

from micro_manager.adaptivity.model_adaptivity import ModelAdaptivity
from micro_manager.micro_manager import MicroManagerCoupling


class DummyModelClass:
    def __init__(self, name):
        self.name = name


class DummySimulation:
    def __init__(self, name, global_id=0, late_init=False):
        self.name = name
        self._global_id = global_id
        self._state = {"state": name}
        self.attachments = {}
        self.destroyed = False
        self.late_init = late_init

    def get_global_id(self):
        return self._global_id

    def get_state(self):
        return self._state.copy()

    def set_state(self, state):
        self._state = state.copy()

    def destroy(self):
        self.destroyed = True


class DummyModelManager:
    def __init__(self):
        self.created_instances = []

    def get_instance(self, gid, target_class, *, late_init=False):
        self.created_instances.append(
            {
                "gid": gid,
                "target_class": target_class.name,
                "late_init": late_init,
            }
        )
        return DummySimulation(target_class.name, gid, late_init=late_init)


class TestModelAdaptivity(TestCase):
    def _make_controller(self, switching_func):
        controller = ModelAdaptivity.__new__(ModelAdaptivity)
        controller._switching_func = switching_func
        controller._model_classes = [
            DummyModelClass("fine"),
            DummyModelClass("coarse"),
        ]
        controller._model_manager = DummyModelManager()
        controller._comm = MPI.COMM_SELF
        controller._logger = MagicMock()
        controller._converged = False
        return controller

    def test_check_convergence_ignores_invalid_switch_at_finest_resolution(self):
        """
        Check that convergence is reached when the switching function requests a
        finer model while the simulation is already using the finest available
        resolution. Such an out-of-range request should be clamped to the
        current resolution and treated as no model change.
        """
        controller = self._make_controller(lambda resolution, *_: -1)

        controller.check_convergence(
            np.array([[0.0, 0.0, 0.0]]),
            1.0,
            [{}],
            None,
            [DummySimulation("fine")],
        )

        self.assertTrue(controller._converged)

    def test_check_convergence_ignores_invalid_switch_at_coarsest_resolution(self):
        """
        Check that convergence is reached when the switching function requests a
        coarser model while the simulation is already using the coarsest
        available resolution. This guards against endless iterations caused by
        repeated out-of-range coarsening requests.
        """
        controller = self._make_controller(lambda resolution, *_: 1)

        controller.check_convergence(
            np.array([[0.0, 0.0, 0.0]]),
            1.0,
            [{}],
            None,
            [DummySimulation("coarse")],
        )

        self.assertTrue(controller._converged)

    def test_check_convergence_detects_valid_switch(self):
        """
        Check that convergence is not reported when the switching function
        requests a valid change to another available model resolution. The
        adaptivity loop must continue in this case so the requested switch can
        be applied.
        """
        controller = self._make_controller(lambda resolution, *_: 1)

        controller.check_convergence(
            np.array([[0.0, 0.0, 0.0]]),
            1.0,
            [{}],
            None,
            [DummySimulation("fine")],
        )

        self.assertFalse(controller._converged)

    def test_manager_loop_switches_once_then_exits_on_invalid_boundary_request(self):
        """
        Reproduce the regression scenario where a model is switched once and
        the switching function then keeps requesting another change beyond the
        available resolution range. The manager should solve with the new model,
        avoid reusing output from the previous model, and stop once the repeated
        boundary request is clamped to a no-op.
        """

        def switching_function(resolution, location, t, input, prev_output):
            if prev_output is None:
                return 0
            return 1

        controller = self._make_controller(switching_function)
        manager = MicroManagerCoupling.__new__(MicroManagerCoupling)
        manager._model_adaptivity_controller = controller
        manager._is_adaptivity_on = False
        manager._mesh_vertex_coords = np.array([[0.0, 0.0, 0.0]])
        manager._global_ids_of_local_sims = [0]
        manager._t = 1.0
        manager._micro_sims = [DummySimulation("fine", global_id=0)]

        solve_calls = []

        def solve_variant(micro_sims_input, dt, computed_outputs):
            solve_calls.append(
                {
                    "sim_name": manager._micro_sims[0].name,
                    "computed_outputs": computed_outputs.copy(),
                }
            )
            return [{"result": len(solve_calls)}]

        result = MicroManagerCoupling._solve_micro_simulations_with_model_adaptivity(
            manager,
            [{"input": 1.0}],
            0.1,
            solve_variant,
        )

        self.assertEqual(len(solve_calls), 2)
        self.assertEqual(solve_calls[0]["sim_name"], "fine")
        self.assertEqual(solve_calls[0]["computed_outputs"], {})

        self.assertEqual(solve_calls[1]["sim_name"], "coarse")
        self.assertEqual(
            solve_calls[1]["computed_outputs"],
            {},
            "Output from the previous resolution must not be reused after a model switch.",
        )

        self.assertEqual(manager._micro_sims[0].name, "coarse")
        self.assertEqual(result, [{"result": 2, "model_resolution": 1}])
        self.assertTrue(controller._converged)

    def test_manager_loop_exits_on_invalid_switch_request(self):
        """
        Check the manager loop for the simpler boundary case where the
        simulation starts at the coarsest model and the switching function keeps
        requesting an even coarser model. The loop should perform one solve,
        recognize that no valid model change remains, and return normally.
        """
        controller = self._make_controller(lambda resolution, *_: 1)
        manager = MicroManagerCoupling.__new__(MicroManagerCoupling)
        manager._model_adaptivity_controller = controller
        manager._is_adaptivity_on = False
        manager._mesh_vertex_coords = np.array([[0.0, 0.0, 0.0]])
        manager._global_ids_of_local_sims = [0]
        manager._t = 1.0
        manager._micro_sims = [DummySimulation("coarse")]

        solve_calls = []

        def solve_variant(micro_sims_input, dt, computed_outputs):
            solve_calls.append(
                {
                    "micro_sims_input": micro_sims_input,
                    "dt": dt,
                    "computed_outputs": computed_outputs,
                }
            )
            return [{"result": 1.0}]

        result = MicroManagerCoupling._solve_micro_simulations_with_model_adaptivity(
            manager,
            [{"input": 1.0}],
            0.1,
            solve_variant,
        )

        self.assertEqual(len(solve_calls), 1)
        self.assertEqual(solve_calls[0]["computed_outputs"], {})
        self.assertEqual(result, [{"result": 1.0, "model_resolution": 1}])
