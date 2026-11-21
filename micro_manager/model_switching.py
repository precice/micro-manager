"""
Class ModelSwitching provides methods to change micro simulation resolution on the fly.
"""
from typing import Union, Optional, List, Dict, Any

from micro_manager.config import Config
from micro_manager.micro_simulation import (
    create_simulation_class,
    load_backend_class,
    MicroSimulationClass,
)
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.misc import clamp_in_range
from micro_manager.tools.mpi_handler import MPIHandler, MPI
from micro_manager.model_manager import ModelManager
from micro_manager.tasking.connection import Connection
from micro_manager.simulation_container import SimulationContainer

import numpy as np
import importlib
import math


class ModelSwitching:
    def __init__(
        self,
        model_manager: ModelManager,
        sim_container: SimulationContainer,
        config: Config,
        mpi: MPIHandler,
        log_file: str,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        model_manager: ModelManager
            ModelManager instance
        sim_container: SimulationContainer
            SimulationContainer instance
        config : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        mpi: MPIHandler
            MPIHandler object
        log_file : str
            Path to the log file to write to.
        """
        self._logger = Logger(__name__, log_file, mpi.rank)

        self._mpi = mpi
        self._model_manager = model_manager
        self._sim_container = sim_container
        self._switching_func_name = config.model_switching_function()
        self._levels_thresholds = config.model_switching_levels_thresholds()
        self._is_adaptivity_on = config.enable_adaptivity()

        FUNC_NAME = "switching_function"
        self._switching_func = None
        if self._switching_func_name is not None:
            try:
                self._switching_func = getattr(
                    importlib.import_module(self._switching_func_name, FUNC_NAME), FUNC_NAME
                )
            except Exception as e:
                self._logger.log_info_rank_zero(
                    f"Failed to load switching function with error: {e}"
                )
                self._switching_func = ModelSwitching.switching_interface

        self._converged = False

    @staticmethod
    def switching_interface(
        resolution: int,
        location: np.ndarray,
        t: float,
        input: Dict[str, Any],
        prev_output: Optional[Dict[str, Any]],
    ) -> int:
        """
        Switching interface function, use as reference

        Parameters
        ----------
        resolution : int
            resolution information as get_sim_class_resolution would return for a sim obj.
        location : np.array - shape(D)
            Array with gaussian point for respective sim. D is the mesh dimension.
        t : float
            Current time in simulation.
        input : Dict[str, Any]
            input object.
        prev_output : Optional[Dict[str, Any]]
            Contains the output of the previous model evaluation.

        """
        return 0

    def initialise_solve(self) -> None:
        """
        Initialise the model switching. Currently only resets convergence flag.
        """
        self._converged = False

    def finalise_solve(self) -> None:
        """
        Perform final clean up. Currently NOOP.
        """
        pass

    def should_iterate(self) -> bool:
        """
        Returns whether or not to further iterate and switch models.
        """
        return not self._converged

    def _compute_reduced_model_level(self, rel_dist: float) -> int:
        """
        Determines the reduced model resolution index (>= 1) for an inactive simulation.
        Resolution 0 is reserved for high-fidelity active simulations.
        """
        num_models = self.get_num_resolutions()
        if num_models <= 2:
            return 1

        k_levels = num_models - 1
        levels_thresholds = getattr(self, "_levels_thresholds", None)
        if levels_thresholds is not None and len(levels_thresholds) > 0:
            thresholds = sorted(levels_thresholds)
            for idx, thresh in enumerate(thresholds):
                if rel_dist < thresh:
                    return clamp_in_range(num_models - 1 - idx, 1, num_models - 1)
            return 1
        else:
            # Uniform partition:
            # rel_dist in [0, 1]. rel_dist close to 1 -> level 1 (finest ROM), close to 0 -> level K (coarsest ROM)
            idx = int(math.floor(rel_dist * k_levels))
            idx = clamp_in_range(idx, 0, k_levels - 1)
            level = k_levels - idx
            return clamp_in_range(level, 1, num_models - 1)

    def switch_models(
        self,
        t: float,
        inputs: List[Dict[str, Any]],
        prev_output: Optional[List[Dict[str, Any]]],
        active_sim_ids: Optional[List[int]] = None,
        inactive_relative_distances: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Switches models within sims list. If active_sim_ids is None, all sims are considered as active.

        Parameters
        ----------
        t : float
            Current time in simulation.
        inputs : List[Dict[str, Any]]
            List of input objects.
        prev_output : Optional[List[Dict[str, Any]]]
            Contains the outputs of the previous model evaluation.
        active_sim_ids : Optional[List[int]]
            List of all active simulation ids.
        inactive_relative_distances : Optional[Dict[int, float]]
            Dictionary mapping inactive simulation lids to their relative similarity distance (d / eps_r).

        Returns
        -------
        switched_lids : list[int]
            List of lids of simulations that were switched.
        """
        locations = self._sim_container.local_coords
        active_sims = self._create_active_mask(
            active_sim_ids, self._sim_container.local_num_sims
        )
        current_res = self._gather_current_resolutions(active_sims)
        target_res = self._gather_target_resolutions(
            current_res,
            locations,
            t,
            inputs,
            prev_output,
            active_sims,
            inactive_relative_distances,
        )

        self._logger.log_info(f"New resolutions for t={t}: {target_res}")

        for lid in self._sim_container.range_lid:
            if current_res[lid] == target_res[lid]:
                continue

            sim = self._sim_container[lid]
            gid = self._sim_container.local_gids[lid]
            target_class = self._model_manager.get_cls_by_idx(
                clamp_in_range(target_res[lid], 0, self.get_num_resolutions() - 1)
            )

            # we store state for each resolution separately
            # keys are sim names of respective resolution
            key_new = f"{target_class.name}-state"

            # check if a state of the target resolution exists
            # then update state buffer with current state
            if sim is not None:
                state_dict = self._sim_container.get_sim_state(lid)
            else:
                state_dict = self._sim_container._sim_checkpoints[lid]
            new_state_exists = key_new in state_dict

            # construct new sim and delay initialization if possible
            sim_new = self._model_manager.get_instance(
                gid, target_class, late_init=new_state_exists
            )

            # if state of target resolution exists
            # use it to initialize
            if new_state_exists:
                sim_new_state = state_dict[key_new]
                sim_new.set_state(sim_new_state)
            else:
                # If target resolution has no saved state yet, attempt to transfer state from the current model
                if sim is not None and hasattr(sim, "get_state") and hasattr(sim_new, "set_state"):
                    try:
                        sim_new.set_state(sim.get_state())
                    except Exception:
                        pass
                state_dict[key_new] = sim_new.get_state()

            # release resources of previous sim and set to new sim
            if sim is not None:
                sim.destroy()
            self._sim_container[lid] = sim_new

        return np.flatnonzero(current_res != target_res).tolist()

    def check_convergence(
        self,
        t: float,
        inputs: List[Dict[str, Any]],
        prev_output: Optional[List[Dict[str, Any]]],
        active_sim_ids: Optional[List[int]] = None,
        inactive_relative_distances: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Similarly to switch_models, checks whether models would be switched in next step.
        If no further changes in model resolution are detected, convergence flag is set to True.

        Parameters
        ----------
        t : float
            Current time in simulation.
        inputs : List[Dict[str, Any]]
            List of all input objects.
        prev_output : Optional[List[Dict[str, Any]]]
            Contains the outputs of the previous model evaluation.
        active_sim_ids : Optional[List[int]]
            List of all active simulation ids.
        inactive_relative_distances : Optional[Dict[int, float]]
            Dictionary mapping inactive simulation lids to their relative similarity distance.
        """
        locations = self._sim_container.local_coords
        active_sims = self._create_active_mask(
            active_sim_ids, self._sim_container.local_num_sims
        )
        resolutions = self._gather_current_resolutions(active_sims)
        target_resolutions = self._gather_target_resolutions(
            resolutions,
            locations,
            t,
            inputs,
            prev_output,
            active_sims,
            inactive_relative_distances,
        )
        local_num_changes = np.sum(target_resolutions != resolutions)
        global_num_changes = self._mpi.comm.allreduce(local_num_changes, op=MPI.SUM)
        self._converged = global_num_changes == 0

    def get_num_resolutions(self) -> int:
        """
        Gets the number of loaded resolutions.

        Returns
        -------
        num_resolutions : int
            Number of loaded resolutions.
        """
        return self._model_manager.num_models

    def _gather_current_resolutions(self, active_sims: np.ndarray) -> np.ndarray:
        """
        Gathers current resolutions. When adaptivity is off, inactive sims have resolution -1.
        When adaptivity is on, each sim's resolution is gathered from the container.

        Parameters
        ----------
        active_sims : np.array
            Boolean array indicating whether the model is active or not.

        Returns
        -------
        resolutions : np.array
            Current resolutions.
        """
        if not getattr(self, "_is_adaptivity_on", False):
            return np.array(
                [
                    self._model_manager.get_idx_of_sim(self._sim_container[lid])
                    if active_sims[lid] == 1
                    else -1
                    for lid in self._sim_container.range_lid
                ]
            )
        else:
            return np.array(
                [
                    self._model_manager.get_idx_of_sim(self._sim_container[lid])
                    if self._sim_container[lid] is not None
                    else -1
                    for lid in self._sim_container.range_lid
                ]
            )

    def _gather_target_resolutions(
        self,
        cur_res: np.ndarray,
        locations: List[np.ndarray],
        t: float,
        inputs: List[Dict[str, Any]],
        prev_output: Optional[List[Dict[str, Any]]],
        active_sims: np.ndarray,
        inactive_relative_distances: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """
        Gathers target resolutions.

        Parameters
        ----------
        cur_res : np.ndarray
            Current resolutions, from _gather_current_resolutions.
        locations : List[np.ndarray]
            Coordinates of all macro-scale integration points in a 2D array of shape [N x D] on this rank, where N is the number of local simulations and D is the mesh dimension.
        t : float
            Current time in simulation.
        inputs : List[Dict[str, Any]]
            List of all input objects.
        prev_output : Optional[List[Dict[str, Any]]]
            Contains the outputs of the previous model evaluation.
        active_sims : np.array
            Boolean array indicating whether the model is active or not.
        inactive_relative_distances : Optional[Dict[int, float]]
            Relative similarity distances for inactive simulations.

        Returns
        -------
        resolutions : np.ndarray
            Target resolutions.
        """
        is_adaptivity_on = getattr(self, "_is_adaptivity_on", False)
        if not is_adaptivity_on:
            switch_tgt = np.zeros_like(cur_res)
            for idx in range(active_sims.shape[0]):
                if active_sims[idx] != 1:
                    continue
                prev_out = prev_output[idx] if prev_output is not None else None
                if self._switching_func is not None:
                    switch_tgt[idx] = self._switching_func(
                        cur_res[idx], locations[idx], t, inputs[idx], prev_out
                    )
            res_tgt = cur_res.copy()
            res_tgt[active_sims == 1] = clamp_in_range(
                switch_tgt[active_sims == 1] + cur_res[active_sims == 1],
                0,
                self.get_num_resolutions() - 1,
            )
            return res_tgt
        else:
            res_tgt = np.zeros_like(cur_res)
            for idx in range(active_sims.shape[0]):
                if active_sims[idx] == 1:
                    # Active problems are solved with high-fidelity model (Resolution 0, FOM)
                    res_tgt[idx] = 0
                else:
                    # Inactive problems are solved with reduced-order models (Resolution >= 1, ROMs)
                    prev_out = prev_output[idx] if prev_output is not None else None
                    if (
                        self._switching_func is not None
                        and self._switching_func is not ModelSwitching.switching_interface
                        and (self._levels_thresholds is None or len(self._levels_thresholds) == 0)
                    ):
                        switch = self._switching_func(
                            cur_res[idx], locations[idx], t, inputs[idx], prev_out
                        )
                        base_res = cur_res[idx] if cur_res[idx] >= 1 else 1
                        res_tgt[idx] = clamp_in_range(
                            base_res + switch,
                            1,
                            self.get_num_resolutions() - 1,
                        )
                    else:
                        rel_dist = 0.0
                        if (
                            inactive_relative_distances is not None
                            and idx in inactive_relative_distances
                        ):
                            rel_dist = inactive_relative_distances[idx]
                        res_tgt[idx] = self._compute_reduced_model_level(rel_dist)
            return res_tgt

    def _create_active_mask(
        self, active_sim_ids: Optional[List[int]], size: int
    ) -> np.ndarray:
        """
        Converts list of active simulation ids to np boolean mask.

        Parameters
        ----------
        active_sim_ids : Optional[List[int]]
            List of all active simulation ids.
        size : int
            size of active_sim_ids

        Returns
        -------
        active_mask : np.ndarray
            Boolean mask of active simulation ids.
        """
        if active_sim_ids is None:
            active_sims = np.ones(size)
        else:
            mask = np.zeros(size)
            if len(active_sim_ids) > 0:
                mask[active_sim_ids] = 1
            active_sims = mask
        return active_sims.astype(bool)
