"""
Functionality for adaptive initialization and control of micro simulations
"""
from collections import defaultdict
from math import exp
from typing import Callable, Dict, Optional, List, Any, Union, Tuple, Hashable
from abc import ABC

from micro_manager.config import Config
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager
from micro_manager.simulation_container import SimulationContainer
from micro_manager.adaptivity.adaptivity_interface import AdaptivityInterface
from micro_manager.interpolation import Interpolator
from micro_manager.tools.profiling import Profiler
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler

import numpy as np


class NoOpAdaptivity(AdaptivityInterface):
    """
    Adaptivity implementation that does not perform any adaptivity.
    All micro problems are active in all time windows.
    """

    def __init__(self, container: SimulationContainer) -> None:
        self._container: SimulationContainer = container

    def get_active_steps(self) -> Dict[int, int]:
        return defaultdict(lambda: 0)

    def get_active_lids(self) -> List[int]:
        return list(self._container.range_lid)

    def get_inactive_lids(self) -> List[int]:
        return []

    def get_active_gids(self) -> List[int]:
        return self._container.local_gids

    def get_inactive_gids(self) -> List[int]:
        return []

    def get_full_field_micro_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_output: List[Optional[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if any([entry is None for entry in micro_output]):
            raise RuntimeError(
                "Some micro outputs are None. NoOpAdaptivity requires fully populated micro output."
            )
        return micro_output

    def get_adaptivity_buffer(self) -> Dict[str, List[Any]]:
        return {}

    def get_associated_map(self) -> Dict[int, int]:
        return {}


class AdaptivityCalculator(AdaptivityInterface, ABC):
    def __init__(
        self,
        config: Config,
        nsims: int,
        sim_container: SimulationContainer,
        profiler: Profiler,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
        base_logger: Logger,
        mpi: MPIHandler,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        config : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        nsims : int
            Initial number of micro simulations.
        sim_container : SimulationContainer
            Simulation container object.
        profiler: Profiler
            Profiler object
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object
            Handles instantiation of micro simulation.
        base_logger : object of class Logger
            Logger object to log messages.
        mpi : MPIHandler
            mpi handler object
        """
        # ===============================
        # Other Modules and Configuration
        # ===============================
        self._micro_problem_cls: MicroSimulationClass = micro_problem_cls
        self._model_manager: ModelManager = model_manager
        self._sim_container: SimulationContainer = sim_container
        self._profiler: Profiler = profiler
        self._mpi: MPIHandler = mpi
        self._base_logger: Logger = base_logger
        self._logger_global_metrics: Optional[Logger] = None
        self._logger_local_metrics: Optional[Logger] = None
        self._interp_min: Optional[int] = None
        self._interp_ids: List[Hashable] = []
        self._mappings: List[Tuple[List[str], List[str]]] = []

        self._refine_const: float = config.adaptivity_refining_constant()
        self._coarse_const: float = config.adaptivity_coarsening_constant()
        self._hist_param: float = config.adaptivity_history_param()

        self._coarse_tol: float = 0.0
        self._ref_tol: float = 0.0
        self._max_similarity_dist: float = 0.0

        self._similarity_measure: Callable[
            [np.ndarray], np.ndarray
        ] = self._get_similarity_measure(config.adaptivity_similarity_measure())

        self._data_names: List[str] = config.data_for_adaptivity()
        # Names of macro data to be used for adaptivity computation
        self._macro_data_names: List[str] = []
        # Names of micro data to be used for adaptivity computation
        self._micro_data_names: List[str] = []
        self._run_every_step: bool = config.enable_adaptivity_each_implicit_iteration()
        self._n: int = config.adaptivity_n()
        self._output_n: int = config.adaptivity_output_n()

        # ===============================
        #         Data Buffers
        # ===============================
        self._data_for_adaptivity: Dict[str, List[Any]] = dict()
        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        # This array is modified in place via the function update_active_sims and update_inactive_sims
        self._is_sim_active: np.ndarray = np.array([True] * nsims, dtype=np.bool_)
        self._sim_active_steps: Dict[int, int] = {
            gid: 0 for gid in self._sim_container.local_gids
        }
        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        # This array is modified in place via the function associate_inactive_to_active
        self._sim_is_associated_to: Dict[int, int] = {}
        self._just_deactivated: List[int] = []
        self._similarity_dists: np.ndarray = np.empty(shape=0)

        # ===============================
        #      Member Initialization
        # ===============================
        self.update_buffers(alloc=True)

        # initialize mappings
        mappings = config.adaptivity_mapping_configs()
        self._load_mappings(mappings)

        # initialize read/write names
        coupling_read_names = config.read_data_names()
        coupling_write_names = config.write_data_names()
        for name in self._data_names:
            if name in coupling_read_names:
                self._macro_data_names.append(name)
            if name in coupling_write_names:
                self._micro_data_names.append(name)

        # initialize output
        output_dir = config.output_dir()
        metrics_output_dir = "adaptivity-metrics"
        if output_dir is not None:
            metrics_output_dir = f"{output_dir}/{metrics_output_dir}"
        output_type = config.adaptivity_output_type()

        # initialize global logging
        if output_type in ["global", "all"]:
            self._logger_global_metrics = Logger(
                "global-metrics-logger",
                metrics_output_dir + "-global.csv",
                self._mpi.rank,
                csv_logger=True,
            )
            self._logger_global_metrics.log_info_rank_zero(
                "n|n active|n inactive|avg active|avg inactive|max active|rank of max active|max inactive|rank of max inactive"
            )

        # initialize local logging
        if output_type in ["local", "all"]:
            self._logger_local_metrics = Logger(
                "metrics-logger",
                metrics_output_dir + "-" + str(self._mpi.rank) + ".csv",
                self._mpi.rank,
                csv_logger=True,
            )
            self._logger_local_metrics.log_info("n|n active|n inactive|assoc ranks")

    def compute_step(self, n: int, first_iteration: bool, dt: float):
        # See AdaptivityInterface for docstring
        if n % self._n != 0:
            return
        if not (self._run_every_step or first_iteration):
            return

        with self._profiler.measure("micro_manager.solve.adaptivity_computation"):
            # compute adaptivity
            self.compute(dt)

            # update counters
            active_gids = self.get_active_gids()
            for gid in active_gids:
                self._sim_active_steps[gid] += 1
            # Write a checkpoint if a simulation is just activated.
            # This checkpoint will be asynchronous to the checkpoints written at the start of the time window.
            self._sim_container.write_checkpoints(only_none=True)

    def get_active_steps(self) -> Dict[int, int]:
        # See AdaptivityInterface for docstring
        return self._sim_active_steps

    def get_macro_data_names(self) -> Optional[List[str]]:
        # See AdaptivityInterface for docstring
        return self._macro_data_names

    def get_micro_data_names(self) -> Optional[List[str]]:
        # See AdaptivityInterface for docstring
        return self._micro_data_names

    def postprocess_active_output(self, micro_output: Dict[str, Any], gid: int) -> None:
        # See AdaptivityInterface for docstring
        micro_output["Active-State"] = 1
        micro_output["Active-Steps"] = self._sim_active_steps[gid]

    def postprocess_inactive_output(
        self, micro_output: Dict[str, Any], gid: int
    ) -> None:
        # See AdaptivityInterface for docstring
        micro_output["Active-State"] = 0
        micro_output["Active-Steps"] = self._sim_active_steps[gid]

    def postprocess_remove(self, micro_output: Dict[str, Any]) -> None:
        # See AdaptivityInterface for docstring
        del micro_output["Active-State"]
        del micro_output["Active-Steps"]

    def get_associated_map(self) -> Dict[int, int]:
        # See AdaptivityInterface for docstring
        return self._sim_is_associated_to

    def get_adaptivity_buffer(self) -> Dict[str, List[Any]]:
        # See AdaptivityInterface for docstring
        return self._data_for_adaptivity

    def get_read_buffer(self) -> Optional[Dict[str, List[Any]]]:
        # See AdaptivityInterface for docstring
        return {}

    def update_buffers(
        self,
        micro_data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None,
        macro_data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None,
        invert: bool = False,
        alloc: bool = False,
    ) -> None:
        # See AdaptivityInterface for docstring
        if alloc:
            for name in self._data_names:
                self._data_for_adaptivity[name] = [
                    0
                ] * self._sim_container.local_num_sims

        self._update_data_buffers_impl(self._micro_data_names, micro_data, invert)
        self._update_data_buffers_impl(self._macro_data_names, macro_data, invert)

    def _update_data_buffers_impl(
        self,
        names: List[str],
        data: Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]] = None,
        invert: bool = False,
    ) -> None:
        """
        Copies data from the provided data buffer into the adaptivity buffer with the selected names.

        Parameters
        ----------
        names : List[str]
            Name selection.
        data : Optional[Union[List[Dict[str, Any]], Dict[str, List[Any]]]]
            Data to be copied.
        invert : bool
            If True, then the expected data format is a dictionary of lists.
        """
        if data is None:
            return

        # writing loops explicitly to avoid branching within
        if invert:
            if len(data.keys()) == 0:
                return
            for name in names:
                for lid in self._sim_container.range_lid:
                    self._data_for_adaptivity[name][lid] = data[name][lid]
        else:
            if len(data) == 0:
                return
            for lid in self._sim_container.range_lid:
                for name in names:
                    self._data_for_adaptivity[name][lid] = data[lid][name]

    def _load_mappings(self, mappings: List[Dict[str, Any]]) -> None:
        """
        Translates the mapping information provided from the configuration file into a
        interpolation method parseable structure.

        This will populate the self._mappings and self._mapping_configs buffers.
        Called once during __init__.

        Parameters
        ----------
        mappings : List[Dict[str, Any]]
            List of mappings as provided by the configuration file.
        """
        for mapping in mappings:
            src_fields = mapping["src_fields"]
            dst_fields = mapping["dst_fields"]
            self._mappings.append((src_fields, dst_fields))

            interp_id = mapping["interp_id"]
            self._interp_ids.append(interp_id)
            if not Interpolator.is_id_valid(interp_id):
                raise ValueError(f"Unknown Interpolation config ID: {interp_id}")
            interp = Interpolator.get_instance(interp_id)
            interp_min = interp.get_min_support_size()

            self._interp_min = self._interp_min or interp_min
            self._interp_min = min(interp_min, self._interp_min)

    def _update_similarity_dists(self, dt: float, data: dict) -> None:
        """
        Calculate metric which determines if two micro simulations are similar enough to have one of them deactivated.

        Parameters
        ----------
        dt : float
            Current time step
        data : dict
            Data to be used in similarity distance calculation
        """
        # Update similarity distances without copying
        self._similarity_dists *= exp(-self._hist_param * dt)

        for name in data.keys():
            data_vals = np.asarray(data[name])
            if data_vals.ndim == 1:
                # If the adaptivity data is a scalar for each simulation,
                # expand the dimension to make it a 2D array to unify the calculation.
                # The axis is later reduced with a norm.
                data_vals = np.expand_dims(data_vals, axis=1)

            self._similarity_dists += dt * self._similarity_measure(data_vals)

    def _associate_inactive_to_active(self) -> None:
        """
        Associate inactive micro simulations to most similar active micro simulation.
        """
        active_ids = np.where(self._is_sim_active)[0]
        inactive_ids = np.where(self._is_sim_active == False)[0]

        # Start with a large distance to trigger the search for the most similar active sim
        # Add the +1 for the case when the similarity distance matrix is zeros
        dist_min_start_value = self._max_similarity_dist + 1

        # Associate inactive micro sims to active micro sims
        for inactive_id in inactive_ids:
            # Begin with a large distance to trigger the search for the most similar active sim
            dist_min = dist_min_start_value
            for active_id in active_ids:
                # Find most similar active sim for every inactive sim
                if self._similarity_dists[inactive_id, active_id] < dist_min:
                    associated_active_id = active_id
                    dist_min = self._similarity_dists[inactive_id, active_id]

            self._sim_is_associated_to[inactive_id] = associated_active_id

    def _check_for_activation(self, inactive_id: int, active_ids: List[int]) -> bool:
        """
        Check if an inactive simulation needs to be activated.

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation.
        active_ids : List[int]
            List containing IDs of active micro simulations.
        Return
        ------
        tag : bool
            True if the inactive simulation needs to be activated, False otherwise.
        """
        dists = self._similarity_dists[inactive_id, active_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(self, active_id: int, active_ids: List[int]) -> bool:
        """
        Check if an active simulation needs to be deactivated.

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation.
        active_ids : List[int]
            List having IDs of active micro simulations.

        Return
        ------
        tag : bool
            True if the active simulation needs to be deactivated, False otherwise.
        """
        for active_id_2 in active_ids:
            if active_id != active_id_2:  # don't compare active sim to itself
                # If active sim is similar to another active sim, deactivate it
                if self._similarity_dists[active_id, active_id_2] <= self._coarse_tol:
                    return True
        return False

    def _interpolate_output(
        self,
        micro_input: List[Dict[str, Any]],
        micro_sims_output: List[Dict[str, Any]],
    ) -> None:
        """
        Interpolates the micro output based on the available inputs and outputs using the selected
        interpolation method and desired mappings.
        Will compute functions f1 ... fN described in the config.
        fi: X -> Y, X and Y must be subsets of the coupled fields.
        Every output field may only be used once as interpolation target, meaning there may not be
        a function fi and fj with shared Yi and Yj.

        This method will edit the output buffer, instead of returning a new buffer.

        Parameters
        ----------
        micro_input : List[Dict[str, Any]]
            List of all local micro simulation inputs.

        micro_sims_output : List[Dict[str, Any]]
            List of all local micro simulation outputs. (current state)
        """
        targets = []
        for _, target_args in self._mappings:
            targets.extend(target_args)
        assert len(targets) == len(set(targets))

        # precompute arg sizes
        active_lids = self.get_active_lids()
        inactive_lids = self.get_inactive_lids()
        arg_sizes = {}
        for name, value in micro_input[-1].items():
            arg_sizes[name] = (
                1 if type(value) != np.ndarray and type(value) != list else len(value)
            )
        for name, value in micro_sims_output[-1].items():
            arg_sizes[name] = (
                1 if type(value) != np.ndarray and type(value) != list else len(value)
            )

        # create interpolation data structures
        n_points = len(active_lids)
        n_points_inactive = len(inactive_lids)
        for m_idx, fun in enumerate(self._mappings):
            src_args, dst_args = fun
            src_size = np.array([arg_sizes[name] for name in src_args]).sum()
            dst_size = np.array([arg_sizes[name] for name in dst_args]).sum()
            input_data = np.zeros((n_points, src_size))
            output_data = np.zeros((n_points, dst_size))
            for idx, lid in enumerate(active_lids):
                offset = 0
                for src_arg in src_args:
                    input_data[idx, offset : offset + arg_sizes[src_arg]] = micro_input[
                        lid
                    ][src_arg]
                    offset += arg_sizes[src_arg]
                offset = 0
                for dst_arg in dst_args:
                    output_data[
                        idx, offset : offset + arg_sizes[dst_arg]
                    ] = micro_sims_output[lid][dst_arg]
                    offset += arg_sizes[dst_arg]
            input_data_inactive = np.zeros((n_points_inactive, src_size))
            for idx, lid in enumerate(inactive_lids):
                offset = 0
                for src_arg in src_args:
                    input_data_inactive[
                        idx, offset : offset + arg_sizes[src_arg]
                    ] = micro_input[lid][src_arg]
                    offset += arg_sizes[src_arg]

            # use interpolant
            interp = Interpolator.get_instance(self._interp_ids[m_idx])
            interp.set_local_data(input_data, input_data_inactive, output_data)
            output_data_inactive = interp.interpolate()

            for idx, lid in enumerate(inactive_lids):
                offset = 0
                for dst_arg in dst_args:
                    micro_sims_output[lid][dst_arg] = output_data_inactive[
                        idx, offset : offset + arg_sizes[dst_arg]
                    ]
                    # unwrap array of size 1
                    if arg_sizes[dst_arg] == 1:
                        micro_sims_output[lid][dst_arg] = micro_sims_output[lid][dst_arg][0]
                    offset += arg_sizes[dst_arg]

    @staticmethod
    def _get_similarity_measure(
        similarity_measure: str,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get similarity measure to be used for similarity calculation

        Parameters
        ----------
        similarity_measure : str
            String specifying the similarity measure to be used

        Returns
        -------
        similarity_measure : function
            Function to be used for similarity calculation. Takes data as input and returns similarity measure
        """
        if similarity_measure == "L1":
            return AdaptivityCalculator._l1
        elif similarity_measure == "L2":
            return AdaptivityCalculator._l2
        elif similarity_measure == "L1rel":
            return AdaptivityCalculator._l1rel
        elif similarity_measure == "L2rel":
            return AdaptivityCalculator._l2rel
        else:
            raise ValueError(
                'Similarity measure not supported. Currently supported similarity measures are "L1", "L2", "L1rel", "L2rel".'
            )

    @staticmethod
    def _l1(data: np.ndarray) -> np.ndarray:
        """
        Calculate L1 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        return np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=1, axis=-1)

    @staticmethod
    def _l2(data: np.ndarray) -> np.ndarray:
        """
        Calculate L2 norm of data

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        return np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], ord=2, axis=-1)

    @staticmethod
    def _l1rel(data: np.ndarray) -> np.ndarray:
        """
        Calculate L1 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        eps = np.finfo(np.float64).eps
        data_bc = data[np.newaxis, :]
        data_abs = np.absolute(data_bc)
        denom = np.maximum(data_abs, np.swapaxes(data_abs, 0, 1))
        return np.linalg.norm(
            (data_bc - np.swapaxes(data_bc, 0, 1)) / np.maximum(denom, eps),
            ord=1,
            axis=-1,
        )

    @staticmethod
    def _l2rel(data: np.ndarray) -> np.ndarray:
        """
        Calculate L2 norm of relative difference of data.
        The relative difference is calculated by dividing the difference of two data points by the maximum of the absolute value of the two data points.

        Parameters
        ----------
        data : numpy array
            Data to be used in similarity distance calculation

        Returns
        -------
        similarity_dists : numpy array
            Updated 2D array having similarity distances between each micro simulation pair
        """
        eps = np.finfo(np.float64).eps
        data_bc = data[np.newaxis, :]
        data_abs = np.absolute(data_bc)
        denom = np.maximum(data_abs, np.swapaxes(data_abs, 0, 1))
        return np.linalg.norm(
            (data_bc - np.swapaxes(data_bc, 0, 1)) / np.maximum(denom, eps),
            ord=2,
            axis=-1,
        )
