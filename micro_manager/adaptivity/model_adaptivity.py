"""
Class ModelAdaptivity provides methods to change micro simulation resolution on the fly.
"""
from typing import Union, Optional

from micro_manager.config import Config
from micro_manager.micro_simulation import (
    create_simulation_class,
    load_backend_class,
    MicroSimulationClass,
)
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.misc import clamp_in_range
from micro_manager.model_manager import ModelManager
from micro_manager.tasking.connection import Connection

import numpy as np
import importlib


class ModelAdaptivity:
    def __init__(
        self,
        model_manager: ModelManager,
        configurator: Config,
        rank: int,
        log_file: str,
        conn: Connection,
        num_ranks: int,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        configurator : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        rank : int
            Rank of the MPI communicator.
        log_file : str
            Path to the log file to write to.
        """
        self._logger = Logger(__name__, log_file, rank)

        self._model_manager = model_manager
        self._model_files = configurator.get_model_adaptivity_file_names()
        self._switching_func_name = (
            configurator.get_model_adaptivity_switching_function()
        )

        stateless_flags = configurator.get_model_adaptivity_micro_stateless()
        self._model_classes = []
        pos = 0
        for model_file in self._model_files:
            try:
                model = load_backend_class(model_file)
                self._model_classes.append(
                    create_simulation_class(
                        self._logger, model, model_file, num_ranks, conn
                    )
                )
                self._model_manager.register(
                    self._model_classes[pos], stateless_flags[pos]
                )
                pos += 1
            except Exception as e:
                self._logger.log_info_rank_zero(
                    f"Failed to load model class with error: {e}"
                )
        if (
            len(self._model_classes) != len(self._model_files)
            or len(self._model_classes) == 0
        ):
            raise RuntimeError("Not all models were loaded. Stopping!")

        FUNC_NAME = "switching_function"
        self._switching_func = ModelAdaptivity.switching_interface
        try:
            self._switching_func = getattr(
                importlib.import_module(self._switching_func_name, FUNC_NAME), FUNC_NAME
            )
        except Exception as e:
            self._logger.log_info_rank_zero(
                f"Failed to load switching function with error: {e}"
            )

        self._converged = False

    @staticmethod
    def switching_interface(
        resolution: int,
        location: np.ndarray,
        t: float,
        input: dict,
        prev_output: Optional[dict],
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
        input : dict
            input object.
        prev_output : [None, dict-like]
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

    def switch_models(
        self,
        locations: np.ndarray,
        t: float,
        inputs: list[dict],
        prev_output: Optional[list[dict]],
        sims: list,
        active_sim_ids: Optional[list] = None,
    ) -> None:
        """
        Switches models within sims list. If active_sim_ids is None, all sims are considered as active.

        Parameters
        ----------
        locations : np.array - shape(N,D)
            Array with gaussian points for all sims. D is the mesh dimension.
        t : float
            Current time in simulation.
        inputs : list[dict]
            List of input objects.
        prev_output : [None, list[dict]]
            Contains the outputs of the previous model evaluation.
        sims : list
            List of all simulation objects.
        active_sim_ids : [list, None]
            List of all active simulation ids.
        """
        size = len(sims)
        active_sims = self._create_active_mask(active_sim_ids, size)
        current_res = self._gather_current_resolutions(sims, active_sims)
        target_res = self._gather_target_resolutions(
            current_res, locations, t, inputs, prev_output, active_sims
        )

        self._logger.log_info_rank_zero(f"New resolutions for t={t}: {target_res}")

        for idx in range(size):
            if current_res[idx] == target_res[idx]:
                continue

            sim = sims[idx]
            gid = sim.get_global_id()
            target_class = self.get_resolution_sim_class(target_res[idx])

            # we store state for each resolution separately
            # keys are sim names of respective resolution
            key = f"{sim.name}-state"
            key_new = f"{target_class.name}-state"

            # check if a state of the target resolution exists
            # then update state buffer with current state
            new_state_exists = key_new in sim.attachments
            sim.attachments[key] = sim.get_state()

            # construct new sim and delay initialization if possible
            sim_new = self._model_manager.get_instance(
                gid, target_class, late_init=new_state_exists
            )
            # need to copy over the multi-state buffer to new sim object
            sim_new.attachments = sim.attachments
            sim_new.attachments[key_new] = sim_new.get_state()

            # if state of target resolution exists
            # use it to initialize
            if new_state_exists:
                sim_new_state = sim.attachments[key_new]
                sim_new.set_state(sim_new_state)

            sims[idx] = sim_new

    def update_states(
        self,
        sims: list,
        active_sim_ids: Optional[list] = None,
    ):
        """
        Updates the current state of the current model in the local buffers.

        Parameters
        ----------
        sims : list
            List of all simulation objects.
        active_sim_ids : [list, None]
            List of all active simulation ids.
        """
        size = len(sims)
        active_sims = self._create_active_mask(active_sim_ids, size)

        for idx in range(size):
            if not active_sims[idx]:
                continue

            sim = sims[idx]
            key = f"{sim.name}-state"
            sim.attachments[key] = sim.get_state()

    def write_back_states(
        self,
        sims: list,
        active_sim_ids: Optional[list] = None,
    ):
        """
        Loads the current state of the current model into local buffers.

        Parameters
        ----------
        sims : list
            List of all simulation objects.
        active_sim_ids : [list, None]
            List of all active simulation ids.
        """
        size = len(sims)
        active_sims = self._create_active_mask(active_sim_ids, size)

        for idx in range(size):
            if not active_sims[idx]:
                continue

            sim = sims[idx]
            key = f"{sim.name}-state"
            sim.set_state(sim.attachments[key])

    def check_convergence(
        self,
        locations: np.ndarray,
        t: float,
        inputs: list,
        prev_output: Optional[list[dict]],
        sims: list,
        active_sim_ids: Optional[list] = None,
    ) -> None:
        """
        Similarly to switch_models, checks whether models would be switched in next step.
        If no further changes in model resolution are detected, convergence flag is set to True.

        Parameters
        ----------
        locations : np.array - shape(N,D)
            Array with gaussian points for all sims. D is the mesh dimension.
        t : float
            Current time in simulation.
        inputs : list[dict]
            List of all input objects.
        prev_output : [None, list[dict]]
            Contains the outputs of the previous model evaluation.
        sims : list
            List of all simulation objects.
        active_sim_ids : [list, None]
            List of all active simulation ids.
        """
        size = len(sims)
        active_sims = self._create_active_mask(active_sim_ids, size)
        resolutions = self._gather_current_resolutions(sims, active_sims)
        next_switch = np.zeros_like(resolutions)
        for idx in range(active_sims.shape[0]):
            if active_sims[idx] != 1:
                continue
            prev_out = prev_output[idx] if prev_output is not None else None
            next_switch[idx] = self._switching_func(
                resolutions[idx], locations[idx], t, inputs[idx], prev_out
            )
        self._converged = np.all(next_switch == 0)

    def get_num_resolutions(self) -> int:
        """
        Gets the number of loaded resolutions.

        Returns
        -------
        num_resolutions : int
            Number of loaded resolutions.
        """
        return len(self._model_classes)

    def get_resolution_sim_class(
        self, resolution: Union[int, np.ndarray]
    ) -> Union[MicroSimulationClass, list[MicroSimulationClass]]:
        """
        Looks up the class associated with the provided resolution.

        Parameters
        ----------
        resolution : [int, np.array]
            target resolution

        Returns
        -------
        sim_class : class
            associated class
        """
        return self._model_classes[
            clamp_in_range(resolution, 0, len(self._model_classes) - 1)
        ]

    def get_sim_class_resolution(self, sim: MicroSimulationClass) -> int:
        """
        Looks up the resolution associated with the provided simulation object.

        Parameters
        ----------
        sim : Simulation
            Simulation object

        Returns
        -------
        resolution : int
            target resolution
        """
        return next(
            (idx for idx, cls in enumerate(self._model_classes) if cls.name == sim.name)
        )

    def _gather_current_resolutions(
        self, sims: list, active_sims: np.ndarray
    ) -> np.ndarray:
        """
        Gathers current resolutions. Inactive sims have resolution -1.

        Parameters
        ----------
        sims : list
            List of all simulation objects.
        active_sims : np.array
            Boolean array indicating whether the model is active or not.

        Returns
        -------
        resolutions : np.array
            Current resolutions.
        """
        return np.array(
            [
                self.get_sim_class_resolution(sim) if active_sims[idx] == 1 else -1
                for idx, sim in enumerate(sims)
            ]
        )

    def _gather_target_resolutions(
        self,
        cur_res: np.ndarray,
        locations: np.ndarray,
        t: float,
        inputs: list[dict],
        prev_output: Optional[list[dict]],
        active_sims: np.ndarray,
    ) -> np.ndarray:
        """
        Gathers target resolutions. Inactive sims have resolution -1.

        Parameters
        ----------
        cur_res : np.array
            Current resolutions, from _gather_current_resolutions.
        locations : np.array
            Array with gaussian points for all sims. D is the mesh dimension.
        t : float
            Current time in simulation.
        inputs : list[dict]
            List of all input objects.
        prev_output : [None, list[dict]]
            Contains the outputs of the previous model evaluation.
        active_sims : np.array
            Boolean array indicating whether the model is active or not.

        Returns
        -------
        resolutions : np.array
            Target resolutions.
        """
        switch_tgt = np.zeros_like(cur_res)
        for idx in range(active_sims.shape[0]):
            if active_sims[idx] != 1:
                continue
            prev_out = prev_output[idx] if prev_output is not None else None
            switch_tgt[idx] = self._switching_func(
                cur_res[idx], locations[idx], t, inputs[idx], prev_out
            )
        res_tgt = cur_res.copy()
        res_tgt[active_sims] = clamp_in_range(
            switch_tgt[active_sims] + cur_res[active_sims],
            0,
            len(self._model_classes) - 1,
        )
        return res_tgt

    def _create_active_mask(self, active_sim_ids: list, size: int) -> np.ndarray:
        """
        Converts list of active simulation ids to np boolean mask.

        Parameters
        ----------
        active_sim_ids : np.array
            List of all active simulation ids.
        size : int
            size of active_sim_ids

        Returns
        -------
        active_mask : np.array
            Boolean mask of active simulation ids.
        """
        if active_sim_ids is None:
            active_sims = np.ones(size)
        else:
            mask = np.zeros(size)
            mask[active_sim_ids] = 1
            active_sims = mask
        return active_sims.astype(bool)
