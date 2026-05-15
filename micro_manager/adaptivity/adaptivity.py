"""
Functionality for adaptive initialization and control of micro simulations
"""

from math import exp
from typing import Callable
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.config import Config
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager

import numpy as np


class AdaptivityCalculator:
    def __init__(
        self,
        config: Config,
        nsims: int,
        micro_problem_cls: MicroSimulationClass,
        model_manager: ModelManager,
        base_logger: Logger,
        rank: int,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        config : object of class Config
            Object which has getter functions to get parameters defined in the configuration file.
        nsims : int
            Number of micro simulations.
        micro_problem_cls : callable
            Class of micro problem.
        model_manager : object
            Handles instantiation of micro simulation.
        base_logger : object of class Logger
            Logger object to log messages.
        rank : int
            Rank of the MPI communicator.
        """
        self._refine_const = config.adaptivity_refining_constant()
        self._coarse_const = config.adaptivity_coarsening_constant()
        self._hist_param = config.adaptivity_history_param()
        self._adaptivity_data_names = config.data_for_adaptivity()
        self._adaptivity_type = config.adaptivity_type()
        self._adaptivity_output_type = config.adaptivity_output_type()

        self._micro_problem_cls = micro_problem_cls
        self._model_manager = model_manager

        self._coarse_tol = 0.0
        self._ref_tol = 0.0

        self._rank = rank
        self._base_logger = base_logger

        self._max_similarity_dist = 0.0

        self._interpolation = None
        self._interp_min = -1
        self._mappings = []
        self._mapping_configs = []
        mappings = configurator.get_adaptivity_mapping_configs()
        self._load_mappings(mappings)

        # is_sim_active: 1D array having state (active or inactive) of each micro simulation
        # Start adaptivity calculation with all sims active
        # This array is modified in place via the function update_active_sims and update_inactive_sims
        self._is_sim_active = np.array([True] * nsims, dtype=np.bool_)

        # sim_is_associated_to: 1D array with values of associated simulations of inactive simulations. Active simulations have None
        # Active sims do not have an associated sim
        # This array is modified in place via the function associate_inactive_to_active
        self._sim_is_associated_to = np.full((nsims), -2, dtype=np.intc)

        self._just_deactivated: list[int] = []

        self._similarity_measure = self._get_similarity_measure(
            config.adaptivity_similarity_measure()
        )

        output_dir = config.output_dir()

        if output_dir is not None:
            metrics_output_dir = output_dir + "/adaptivity-metrics"
        else:
            metrics_output_dir = "adaptivity-metrics"

        if self._rank == 0 and (
            self._adaptivity_output_type == "global"
            or self._adaptivity_output_type == "all"
        ):
            self._global_metrics_logger = Logger(
                "global-metrics-logger",
                metrics_output_dir + "-global.csv",
                rank,
                csv_logger=True,
            )

            self._global_metrics_logger.log_info(
                "n|n active|n inactive|avg active|avg inactive|max active|rank of max active|max inactive|rank of max inactive"
            )

        if (
            self._adaptivity_output_type == "local"
            or self._adaptivity_output_type == "all"
        ):
            self._metrics_logger = Logger(
                "metrics-logger",
                metrics_output_dir + "-" + str(rank) + ".csv",
                rank,
                csv_logger=True,
            )

            self._metrics_logger.log_info("n|n active|n inactive|assoc ranks")

    def _load_mappings(self, mappings: list) -> None:
        """
        Translates the mapping information provided from the configuration file into a
        interpolation method parseable structure.

        This will populate the self._mappings and self._mapping_configs buffers.
        Called once during __init__.

        Parameters
        ----------
        mappings : list
            List of mappings as provided by the configuration file.
        """
        for mapping in mappings:
            src_fields = mapping["src_fields"]
            dst_fields = mapping["dst_fields"]
            n_neighbors = mapping["n_neighbors"]
            if self._interp_min == -1:
                self._interp_min = n_neighbors
            else:
                self._interp_min = min(n_neighbors, self._interp_min)

            self._mappings.append((src_fields, dst_fields))
            config = {}
            if "use_pu" in mapping["rbf_config"]:
                config["use_pu"] = mapping["rbf_config"]["use_pu"]
            if "pu_overlap" in mapping["rbf_config"]:
                config["pu_overlap"] = mapping["rbf_config"]["pu_overlap"]
            config["pu_cluster_size"] = n_neighbors
            if "basis" in mapping["rbf_config"]:
                if "type" in mapping["rbf_config"]["basis"]:
                    config["basis"] = mapping["rbf_config"]["basis"]["type"]
                if (
                    config["basis"] == "gauss"
                    and "eps" in mapping["rbf_config"]["basis"]
                ):
                    config["gauss_eps"] = mapping["rbf_config"]["basis"]["eps"]

            dom_config = {}
            dom_config["n_neighbors"] = n_neighbors
            if "max_filling" in mapping["domain_config"]:
                dom_config["max_filling"] = mapping["domain_config"]["max_filling"]
            if "coarsening_factor" in mapping["domain_config"]:
                dom_config["coarsening_factor"] = mapping["domain_config"][
                    "coarsening_factor"
                ]
            if "projection" in mapping["domain_config"]:
                if "type" in mapping["domain_config"]["projection"]:
                    dom_config["projection_type"] = mapping["domain_config"][
                        "projection"
                    ]["type"]
                if "target_dims" in mapping["domain_config"]["projection"]:
                    dom_config["projection_std_dims"] = mapping["domain_config"][
                        "projection"
                    ]["target_dims"]

            config["domain_config"] = dom_config
            self._mapping_configs.append(config)

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

    def _check_for_activation(self, inactive_id: int, active_ids: np.ndarray) -> bool:
        """
        Check if an inactive simulation needs to be activated.

        Parameters
        ----------
        inactive_id : int
            ID of inactive simulation which is checked for activation.
        active_ids : numpy array
            1D array having IDs of active micro simulations.
        Return
        ------
        tag : bool
            True if the inactive simulation needs to be activated, False otherwise.
        """
        dists = self._similarity_dists[inactive_id, active_ids]

        # If inactive sim is not similar to any active sim, activate it
        return min(dists) > self._ref_tol

    def _check_for_deactivation(self, active_id: int, active_ids: list) -> bool:
        """
        Check if an active simulation needs to be deactivated.

        Parameters
        ----------
        active_id : int
            ID of active simulation which is checked for deactivation.
        active_ids : list
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

    def _interpolate_output(self, micro_input, micro_sims_output) -> None:
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
        micro_input : list
            List of all local micro simulation inputs.

        micro_sims_output : list
            List of all local micro simulation outputs. (current state)
        """
        targets = []
        for _, target_args in self._mappings:
            targets.extend(target_args)
        assert len(targets) == len(set(targets))

        # precompute arg sizes
        active_lids = self.get_active_sim_local_ids()
        inactive_lids = self.get_inactive_sim_local_ids()
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
            self._interpolation.configure(self._mappings[m_idx])
            self._interpolation.set_local_data(
                input_data, input_data_inactive, output_data
            )
            output_data_inactive = self._interpolation.interpolate()

            for idx, lid in enumerate(inactive_lids):
                offset = 0
                for dst_arg in dst_args:
                    micro_sims_output[lid][dst_arg] = output_data_inactive[
                        idx, offset : offset + arg_sizes[dst_arg]
                    ]
                    offset += arg_sizes[dst_arg]

    def _get_similarity_measure(
        self, similarity_measure: str
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
            return self._l1
        elif similarity_measure == "L2":
            return self._l2
        elif similarity_measure == "L1rel":
            return self._l1rel
        elif similarity_measure == "L2rel":
            return self._l2rel
        else:
            raise ValueError(
                'Similarity measure not supported. Currently supported similarity measures are "L1", "L2", "L1rel", "L2rel".'
            )

    def _l1(self, data: np.ndarray) -> np.ndarray:
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

    def _l2(self, data: np.ndarray) -> np.ndarray:
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

    def _l1rel(self, data: np.ndarray) -> np.ndarray:
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

    def _l2rel(self, data: np.ndarray) -> np.ndarray:
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
