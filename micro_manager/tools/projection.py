from abc import ABC, abstractmethod
import numpy as np
from mpi4py import MPI


class Projector(ABC):
    """
    Interface to project high-dimensional data into low-dimensional space.
    """

    @abstractmethod
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        pass

    @abstractmethod
    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        pass


class STDProjector(Projector):
    """
    Projects high-dimensional data into low-dimensional space using the fields with the highest standard deviation.
    """

    def __init__(self, target_dims: int, comm: MPI.Comm):
        """
        Constructs STD projection.

        Parameters
        ----------
        target_dims : int
            Number of target dimensions.
        comm : MPI.Comm
            MPI communicator.
        """
        self.num_target_dims = target_dims
        self.target_dims = np.zeros(target_dims, dtype=np.int32)
        self.comm = comm

    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data.
        Computes target dimensions using provided data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        assert data.ndim > 1
        std = np.zeros(data.shape[-1])
        if data.shape[0] > 0:
            std = np.std(data, axis=0)
        stds = np.array(self.comm.allgather(std))
        stds = np.mean(stds, axis=0)
        self.target_dims[:] = np.sort(
            np.argsort(stds)[::-1][0 : self.num_target_dims]
        ).astype(np.int32)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data.

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        d = data
        if data.ndim == 1:
            d = d[np.newaxis, :]
        return d[:, self.target_dims]


class IdentityProjector(Projector):
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Performs projection on high-dimensional data. (does nothing)

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.

        Returns
        -------
        proj_data : np.ndarray
            Projected data.
        """
        return data

    def initialize(self, data: np.ndarray) -> None:
        """
        Initializes projection parameters based on data. (does nothing)

        Parameters
        ----------
        data : np.ndarray
            High-dimensional data.
        """
        pass
