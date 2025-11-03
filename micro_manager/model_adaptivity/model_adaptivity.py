"""
Class ModelAdaptivity provides methods to change micro simulation resolution on the fly.
"""

from ..config import Config
from ..micro_simulation import create_simulation_class
from micro_manager.tools.logging_wrapper import Logger

import numpy as np
import importlib

class ModelAdaptivity:
    def __init__(self, configurator : Config):
        self._model_files = configurator.get_model_adaptivity_file_names()
        self._model_thresholds = configurator.get_model_adaptivity_thresholds()
        self._model_adaptivity_names = configurator.get_model_adaptivity_data()

        self._model_classes = []
        CLASS_NAME = 'MicroSimulation'
        for model_file in self._model_files:
            try:
                model = getattr(importlib.import_module(model_file, CLASS_NAME), CLASS_NAME,)
                self._model_classes.append(create_simulation_class(model))
            except Exception as e:
                print("Failed to load model class with error: ", e) # TODO replace with loggers
        if len(self._model_classes) != len(self._model_files) or len(self._model_classes) == 0:
            raise RuntimeError("Not all models were loaded. Stopping!")

        self._converged = False

    def initialise_solve(self):
        self._converged = False

    def finalise_solve(self):
        pass

    def should_iterate(self):
        return not self._converged

    def switch_models(self, sims, active_sim_ids=None):
        size = len(sims)
        active_sims = self._create_active_mask(active_sim_ids, size)
        cur_cls = self._gather_current_resolutions(sims, active_sims)
        tgt_cls = self._gather_target_resolutions(sims, active_sims)

        for idx in range(size):
            if cur_cls[idx] == tgt_cls[idx]: continue

            sim_state = sims[idx].get_state()
            sim_id = sims[idx].get_global_id()
            sims[idx] = tgt_cls[idx](sim_id)
            sims[idx].set_state(sim_state)

        # TODO change this later
        self._converged = True

    def get_resolution_sim_class(self, resolution):
        return self._model_classes[self._clamp_in_range(resolution)]

    def get_sim_class_resolution(self, sim):
        return next((idx for idx, cls in enumerate(self._model_classes) if cls == type(sim)))

    def _clamp_in_range(self, value):
        return max(0, min(value, len(self._model_classes)-1))

    def _gather_current_resolutions(self, sims, active_sims):
        return [self.get_sim_class_resolution(sim) if active_sims[idx] == 1 else type(sim) for idx, sim in
                enumerate(sims)]

    def _gather_target_resolutions(self, sims, active_sims):
        # TODO this needs to be based on input params or sim metric
        # for now we just set one half to smth else
        return [self.get_resolution_sim_class(1) if active_sims[idx] == 1 and idx >= (len(sims) / 2) else type(sim) for
                idx, sim in enumerate(sims)]

    def _create_active_mask(self, active_sim_ids, size):
        if active_sim_ids is None: active_sims = np.ones(size)
        else:
            mask = np.zeros(size)
            mask[active_sim_ids] = 1
            active_sims = mask
        return active_sims
