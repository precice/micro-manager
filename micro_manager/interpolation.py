from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Dict, Any, Optional, Hashable, Callable, List

import numpy as np

# handle compatibility issue between np version 1 and 2
if int(np.version.version.split(".")[0]) > 1:
    np.alltrue = np.all

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:

    class Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, item):
            return self

    NearestNeighbors = Dummy

from micro_manager.config import Config
from micro_manager.tools.mpi_handler import MPIHandler, MPIHandlerRankLocal
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.spatial_methods import InterleavedDomain


class Interpolator(ABC):
    # Associates Interpolator configuration IDs with the according config
    _registered_configs: Dict[Hashable, Tuple[str, Dict[str, Any]]] = dict()
    # Associates the respective class names of the used Interpolator implementations with their instances
    _instances: Dict[str, "Interpolator"] = dict()
    # Associates the respective class names of the Interpolator implementations with their classes
    _implementations: Dict[str, Callable[[Logger, MPIHandler], "Interpolator"]] = dict()

    def __init__(self, logger: Logger, mpi: MPIHandler):
        """
        Constructs the base interpolator.

        Parameters
        ----------
        logger : Logger
            Logger object.
        mpi : MPIHandler
            MPI handler object.
        """
        self._logger: Logger = logger
        self._mpi: MPIHandler = mpi

    def configure(self, interp_config: Dict[str, Any]) -> None:
        """
        Configures the current state of the interpolator to the specified settings.

        Parameters
        ----------
        interp_config : Dict[str, Any]
            Dict containing settings for interpolation.
        """
        pass

    def set_local_data(self, x: np.ndarray, x_: np.ndarray, f: np.ndarray) -> None:
        """
        Sets the rank local data for the next interpolation.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        x_ : np.ndarray
            Query points.
        f : np.ndarray
            Support point function values.
        """
        pass

    def interpolate(self) -> np.ndarray:
        """
        Interpolates the function values at the set query points.

        Returns
        -------
        interp_result : np.ndarray
            Interpolated function values.
        """
        pass

    @abstractmethod
    def get_min_support_size(self) -> int:
        """
        Gets the minimum number of support points for interpolation to be possible.

        Returns
        -------
        min_support_size : int
            Minimum number of support points.
        """
        pass

    @abstractmethod
    def is_local(self) -> bool:
        """
        Checks if the interpolation scheme is local or global.

        Returns
        -------
        Returns True if interpolation operates locally, else False.
        """
        pass

    @classmethod
    @abstractmethod
    def load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Loads and optionally transforms the configuration dict as provided in the input file.

        Returns
        -------
        config_dict : Dict[str, Any]
            Configuration dict for this interpolation class.
        """
        pass

    @staticmethod
    def initialize(config: Config, logger: Logger, mpi: MPIHandler) -> None:
        """
        Initializes all interpolation schemes.
        Loads the required configurations and constructs instances for all specified interpolation types.

        Parameters
        ----------
        config : Config
            Configuration object.
        logger : Logger
            Logger object.
        mpi : MPIHandler
            MPIHandler object.
        """
        interp_configs: List[Dict[str, Any]] = config.interpolation_configs()

        for interp_config in interp_configs:
            if "type" not in interp_config:
                logger.log_error("Interpolation type is missing.")
                raise ValueError("Provide interpolation type in configuration.")

            cls_name = interp_config["type"]
            if cls_name not in Interpolator._implementations.keys():
                logger.log_error(
                    f"Unknown Implementation type {cls_name}.\n"
                    f"Valid options are: {list(Interpolator._implementations.keys())}"
                )
                raise ValueError("Invalid Implementation type.")

            if "id" not in interp_config:
                logger.log_error("Interpolation config id is missing.")
                raise ValueError("Provide interpolation config id in configuration.")

            cls = Interpolator._implementations[cls_name]
            config_id = interp_config["id"]

            if config_id in Interpolator._registered_configs:
                logger.log_error(f"Interpolation config id:{config_id} already exists.")
                raise ValueError(
                    "Provide distinct interpolation config ids in configuration."
                )

            Interpolator._registered_configs[config_id] = cls.__name__, cls.load_config(
                interp_config
            )

            if cls_name in Interpolator._instances:
                continue

            Interpolator._instances[cls_name] = cls(logger, mpi)

    @staticmethod
    def is_id_valid(config_id: Hashable) -> bool:
        """
        Checks if the provided interpolation config ID is valid.

        Parameters
        ----------
        config_id : Hashable
            ID to be checked.

        Returns
        -------
        is_valid : bool
            True if the ID is known, else False.
        """
        return config_id in Interpolator._registered_configs

    @staticmethod
    def get_instance(id: Hashable) -> "Interpolator":
        """
        Gets a configured interpolation instance.
        The configuration is determined by the provided interpolation configuration ID.

        Parameters
        ----------
        id : Hashable
            Interpolation Configuration ID.

        Returns
        -------
        interp : Interpolator
            Configured interpolation instance.
        """
        cls_name, config = Interpolator._registered_configs[id]
        inst = Interpolator._instances[cls_name]
        inst.configure(config)
        return inst

    @staticmethod
    def get_config(id: Hashable) -> Dict[str, Any]:
        """
        Gets an interpolation configuration by its ID.

        Parameters
        ----------
        id : Hashable
            Interpolation Configuration ID.

        Returns
        -------
        config : Dict[str, Any]
            Interpolation configuration dict.
        """
        _, config = Interpolator._registered_configs[id]
        return config

    @staticmethod
    def register_impl(cls):
        """
        Registers an implementation of the Interpolator interface.
        Note: should be used as a class annotation.
        """
        Interpolator._implementations[cls.__name__] = cls
        return cls


@Interpolator.register_impl
class KNN(Interpolator):
    """
    Implements a k-Nearest-Neighbors interpolation scheme using sklearn.
    """

    def __init__(self, logger: Logger, mpi: MPIHandler):
        super().__init__(logger, mpi)

        self._k = 1
        self._x: Optional[np.ndarray] = None
        self._x_query: Optional[np.ndarray] = None
        self._f: Optional[np.ndarray] = None

    def configure(self, interp_config: Dict[str, Any]) -> None:
        self._k = 1 if "k" not in interp_config else interp_config["k"]

    def get_min_support_size(self) -> int:
        return 1

    def is_local(self) -> bool:
        return True

    def set_local_data(self, x: np.ndarray, x_: np.ndarray, f: np.ndarray) -> None:
        assert x.shape[1:] == x_.shape[1:]
        assert x.shape[0] == f.shape[0]

        self._x = x
        self._x_query = x_
        self._f = f

    def interpolate(self) -> np.ndarray:
        assert self._x is not None
        assert self._x_query is not None
        assert self._f is not None

        f_query = np.zeros(shape=(self._x_query.shape[0], self._f.shape[-1]))
        for idx_query in range(self._x_query.shape[0]):
            inter_point = self._x_query[idx_query]
            nearest_neighbors = self._get_nearest_neighbor_indices(
                self._x,
                inter_point,
                self._k,
            )
            f_query[idx_query, :] = self._interpolate_impl(
                self._x[nearest_neighbors, :],
                self._x_query[idx_query, :],
                self._f[nearest_neighbors, :],
            )
        return f_query

    @classmethod
    def load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        conf = config.copy()
        if "type" in conf:
            del conf["type"]
        if "id" in conf:
            del conf["id"]
        return conf

    def _get_nearest_neighbor_indices(
        self,
        coords: np.ndarray,
        inter_point: np.ndarray,
    ) -> np.ndarray:
        """
        Get local indices of the k nearest neighbors of a point.

        Parameters
        ----------
        coords : list
            List of coordinates of all points.
        inter_point : list | np.ndarray
            Coordinates of the point for which the neighbors are to be found.

        Returns
        ------
        neighbor_indices : np.ndarray
            Local indices of the k nearest neighbors in all local points.
        """
        k = self._k
        if len(coords) < k:
            self._logger.log_info(
                "Number of desired neighbors k = {} is larger than the number of available neighbors {}. Resetting k = {}.".format(
                    k, len(coords), len(coords)
                )
            )
            k = len(coords)
        if NearestNeighbors.__name__ != "NearestNeighbors":
            raise RuntimeError("scipy was not imported.")
        neighbors = NearestNeighbors(n_neighbors=k).fit(coords)

        neighbor_indices = neighbors.kneighbors(
            [inter_point], return_distance=False
        ).flatten()

        return neighbor_indices

    @staticmethod
    def _interpolate_impl(neighbors: np.ndarray, point: np.ndarray, values):
        r"""
            Interpolate a value at a point using inverse distance weighting. (https://en.wikipedia.org/wiki/Inverse_distance_weighting)
            .. math::
                f(x) = (\sum_{i=1}^{n} \frac{f_i}{\Vert x_i - x \Vert^2}) / (\sum_{j=1}^{n} \frac{1}{\Vert x_j - x \Vert^2})

        Parameters
        ----------
        neighbors : np.ndarray
            Coordinates at which the values are known.
        point : np.ndarray
            Coordinates at which the value is to be interpolated.
        values :
            Values at the known coordinates.

        Returns
        -------
        interpol_val / summed_weights :
            Value at interpolation point.
        """
        interpol_val = 0
        summed_weights = 0
        # Iterate over all neighbors
        for inx in range(len(neighbors)):
            # Compute the squared norm of the difference between interpolation point and neighbor
            norm = np.linalg.norm(np.array(neighbors[inx]) - np.array(point)) ** 2
            # If interpolation point is already part of the data it is returned as the interpolation result
            # This avoids division by zero
            if norm < 1e-16:
                return values[inx]
            # Update interpolation value
            interpol_val += values[inx] / norm
            # Extend normalization factor
            summed_weights += 1 / norm

        return interpol_val / summed_weights


@Interpolator.register_impl
class RBF_PU(Interpolator):
    """
    Interpolates f(x) for f: R^n -> R^m using partition of unity RBF interpolant.

    The approach here does not require a support radius as data is normalized.
    """

    def __init__(self, logger: Logger, mpi: MPIHandler):
        """
        Constructs the RBF_PU interpolator.
        For rank local interpolation provide MPI.COMM_SELF as comm with according rank and size.

        Parameters
        ----------
        logger : Logger
            Logger object.
        mpi : MPIHandler
            mpi handler object
        """
        super().__init__(logger, mpi)

        self._domain = InterleavedDomain(mpi)
        self._use_pu = False
        self._pu_overlap = 0.1
        self._pu_cluster_size = 50

        # RBF data
        self._phi = RBF_PU.basis_c6
        self._x = None
        self._x_query = None
        self._f = None

    def configure(self, interp_config: dict) -> None:
        """
        Configures the interpolator to the provided parameters.

        Parameters
        ----------
        interp_config : dict
            Interpolator configuration.
        """
        if "mpi" in interp_config:
            self._mpi = interp_config["mpi"]

        self._domain.configure(
            {}
            if "domain_config" not in interp_config
            else interp_config["domain_config"],
            self._mpi,
        )
        self._use_pu = (
            False if "use_pu" not in interp_config else interp_config["use_pu"]
        )
        if self._use_pu:
            self._pu_overlap = (
                0.1
                if "pu_overlap" not in interp_config
                else interp_config["pu_overlap"]
            )
            self._pu_cluster_size = (
                50
                if "pu_cluster_size" not in interp_config
                else interp_config["pu_cluster_size"]
            )
        if "basis" not in interp_config:
            return
        match interp_config["basis"]:
            case "c0":
                self._phi = RBF_PU.basis_c0
            case "c2":
                self._phi = RBF_PU.basis_c2
            case "c4":
                self._phi = RBF_PU.basis_c4
            case "c6":
                self._phi = RBF_PU.basis_c6
            case "gauss":
                eps = (
                    1.0
                    if "gauss_eps" not in interp_config
                    else interp_config["gauss_eps"]
                )
                self._phi = partial(RBF_PU.basis_gauss, eps=eps)

    def get_min_support_size(self) -> int:
        return self._domain.get_group_size()

    def is_local(self) -> bool:
        return self._mpi.size == 1

    def set_local_data(self, x: np.ndarray, x_: np.ndarray, f: np.ndarray) -> None:
        """
        Sets local data for interleaved domain.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        x_ : np.ndarray
            Query points.
        f : np.ndarray
            Support point function values.
        """
        self._domain.set_local_data(x, x_, f)

    def interpolate(self) -> np.ndarray:
        """
        Interpolates the function values at the set query points.

        Returns
        -------
        interp_result : np.ndarray
            Interpolated function values.
        """
        self._x, self._x_query, self._f = self._domain.decompose()

        interp = self.compute_interpolant(self._x, self._f)
        xq, fq = self.evaluate_interpolant(interp, self._x_query)

        fq_local = self._domain.reassemble(xq, fq)

        return fq_local

    @classmethod
    def load_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        conf: Dict[str, Any] = dict()
        if "use_pu" in config["rbf_config"]:
            conf["use_pu"] = config["rbf_config"]["use_pu"]
        if "pu_overlap" in config["rbf_config"]:
            conf["pu_overlap"] = config["rbf_config"]["pu_overlap"]
        conf["pu_cluster_size"] = config["rbf_config"]["n_neighbors"]
        if "basis" in config["rbf_config"]:
            if "type" in config["rbf_config"]["basis"]:
                conf["basis"] = config["rbf_config"]["basis"]["type"]
            if config["basis"] == "gauss" and "eps" in config["rbf_config"]["basis"]:
                conf["gauss_eps"] = config["rbf_config"]["basis"]["eps"]

        dom_config = {}
        dom_config["n_neighbors"] = config["rbf_config"]["n_neighbors"]
        if "local" == config["domain_config"]:
            conf["mpi"] = MPIHandlerRankLocal
        else:
            if "max_filling" in config["domain_config"]:
                dom_config["max_filling"] = config["domain_config"]["max_filling"]
            if "coarsening_factor" in config["domain_config"]:
                dom_config["coarsening_factor"] = config["domain_config"][
                    "coarsening_factor"
                ]
            if "projection" in config["domain_config"]:
                if "type" in config["domain_config"]["projection"]:
                    dom_config["projection_type"] = config["domain_config"][
                        "projection"
                    ]["type"]
                if "target_dims" in config["domain_config"]["projection"]:
                    dom_config["projection_std_dims"] = config["domain_config"][
                        "projection"
                    ]["target_dims"]

        conf["domain_config"] = dom_config
        return conf

    # ================================
    #              RBF
    # ================================
    @property
    def compute_interpolant(self):
        if self._use_pu:
            return self.compute_rbf_pu_interpolant
        else:
            return self.compute_rbf_interpolant

    @property
    def evaluate_interpolant(self):
        if self._use_pu:
            return self.evaluate_rbf_pu_interpolant
        else:
            return self.evaluate_rbf_interpolant

    def _compute_cluster_centers(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates cluster centers based on the provided support points.
        2 cluster centers per dimension and one in the middle.

        Parameters
        ----------
        x : np.ndarray
            Support points.

        Returns
        -------
        cluster_centers : np.ndarray
            Cluster centers.
        local_min : np.ndarray
            Minimum of local points.
        local_max : np.ndarray
            Maximum of local points.
        """
        assert self._use_pu
        local_min, local_max = np.min(x, axis=0), np.max(x, axis=0)
        d4 = (local_max - local_min) / 4

        center = local_min + 2.0 * d4
        centers = np.zeros((2 * x.shape[-1] + 1, x.shape[-1]))
        centers[-1, :] = center
        for d in range(x.shape[-1]):
            mask = np.zeros_like(d4)
            mask[d] = 1

            centers[2 * d + 0, :] = center - mask * d4
            centers[2 * d + 1, :] = center + mask * d4

        return centers, local_min, local_max

    def compute_rbf_pu_interpolant(self, x, f):
        # compute r_m
        c_centers, local_min, local_max = self._compute_cluster_centers(x)
        # index_tree = NDtree(
        #    NDtree.Mode.INDEX, local_min, local_max, *self._domain.get_depth_filling()
        # )
        # TODO later
        # determine clusters
        # ignore empty clusters
        # compute local RBF interpolant for remaining clusters
        pass

    def compute_rbf_interpolant(
        self, x: np.ndarray, f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs an interpolant based on the provided support points and function values.

        Parameters
        ----------
        x : np.ndarray
            Support points.
        f : np.ndarray
            Support point function values.

        Returns
        -------
        interp_weights_high: np.ndarray
            Interpolant weights, higher order.
        interp_weights_low: np.ndarray
            Interpolant weights, lower order.
        x : np.ndarray
            Support points.
        f : np.ndarray
            Support point function values.
        """
        n_points = x.shape[0]
        src_size = x.shape[-1]
        dst_size = f.shape[-1]

        r = np.linalg.norm(x[None, :, :] - x[:, None, :], ord=2, axis=-1)
        # compute linear and constant terms
        b = np.zeros((src_size + 1, dst_size))
        p = np.zeros((n_points, src_size + 1))
        p[:, 0] = 1
        p[:, 1:] = x
        for k in range(dst_size):
            b[:, k] = np.linalg.lstsq(p, f[:, k], rcond=None)[0]

        a = self._phi(r)
        # compute basis weights
        w = np.zeros((dst_size, n_points))
        for k in range(dst_size):
            w[k, :] = np.linalg.solve(a, f[:, k] - np.matmul(p, b[:, k]))

        return w, b, x, f

    def evaluate_rbf_pu_interpolant(self, interp, xq):
        # eval xq for all cluster interpolants
        # compute weights
        # sum contributions
        pass

    def evaluate_rbf_interpolant(
        self,
        interp: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        xq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates the function values at the set query points.

        Parameters
        ----------
        interp : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Interpolation model as computed by compute_rbf_interpolant
        xq : np.ndarray
            Query points.

        Returns
        -------
        xq : np.ndarray
            Query points.
        fq : np.ndarray
            Query point function values.
        """
        w, b, x, f = interp

        r = np.linalg.norm(xq[None, :, :] - x[:, None, :], ord=2, axis=-1)
        contrib_basis = np.matmul(w[:, :], self._phi(r))  # f_k x eval_p
        contrib_const = b[0, :]  # f_k
        # b: p_size+1 x f_k
        # xq: eval_p x p_size
        contrib_lin = np.matmul(xq[:, :], b[1:, :]).T  # f_k x eval_p

        fq = (contrib_basis + contrib_const[:, None] + contrib_lin).T
        return xq, fq

    # ================================
    #        BASIS FUNCTIONS
    # ================================
    @staticmethod
    def basis_c0(r):
        return np.maximum(0.0, np.power(1.0 - r, 2))

    @staticmethod
    def basis_c2(r):
        return np.maximum(0.0, np.power(1.0 - r, 4)) * (4.0 * r + 1)

    @staticmethod
    def basis_c4(r):
        return (
            np.maximum(0.0, np.power(1.0 - r, 6))
            * (35.0 * np.power(r, 2) + 18.0 * r + 3.0)
            / 3.0
        )

    @staticmethod
    def basis_c6(r):
        return np.maximum(0.0, np.power(1.0 - r, 8)) * (
            32.0 * np.power(r, 3) + 25.0 * np.power(r, 2) + 8.0 * r + 1.0
        )

    @staticmethod
    def basis_gauss(r, eps):
        return np.exp(-np.power(eps * r, 2.0))
