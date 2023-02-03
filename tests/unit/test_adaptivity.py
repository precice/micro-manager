from unittest import TestCase
from micro_manager.adaptivity import AdaptiveController
from micro_manager.config import Config

class TestAdaptivity(TestCase):
    
    def setUp(self):
        self._refine_const = 0.5
        self._coarse_const = 0.5
        self._adaptivity_controller = AdaptiveController(Config("./tests/unit/test_adaptivity_config.json"))
        self._number_of_sims = 5
        self._coarse_tol = 0.2

    def test_set_number_of_sims(self):
        self._adaptivity_controller.set_number_of_sims(self._number_of_sims)
        self.assertEqual(self._number_of_sims, self._adaptivity_controller._number_of_sims)

