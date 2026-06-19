from functools import partial
from typing import Tuple

import numpy as np

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

from micro_manager.tools.mpi_handler import MPIHandler
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.spatial_methods import InterleavedDomain


# handle compat issue between np version 1 and 2
if int(np.version.version.split(".")[0]) > 1:
    np.alltrue = np.all


class Interpolation:
    def __init__(self, logger):

        self._logger = logger

    def get_nearest_neighbor_indices(
        self,
        coords: np.ndarray,
        inter_point: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Get local indices of the k nearest neighbors of a point.

        Parameters
        ----------
        coords : list
            List of coordinates of all points.
        inter_point : list | np.ndarray
            Coordinates of the point for which the neighbors are to be found.
        k : int
            Number of neighbors to consider.

        Returns
        ------
        neighbor_indices : np.ndarray
            Local indices of the k nearest neighbors in all local points.
        """
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

    def interpolate(self, neighbors: np.ndarray, point: np.ndarray, values):
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


class RBF_PU:
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
        self._logger = logger
        self._mpi = mpi

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
        self._domain.configure(
            {}
            if "domain_config" not in interp_config
            else interp_config["domain_config"]
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
