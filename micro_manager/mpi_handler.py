from typing import Optional, Tuple

from mpi4py import MPI
import hashlib

class MPIHandler:
    """
    Encapsulates all MPI related aspects and provides Micro Manager specific functionality.
    All methods can override the rank and communicator if desired.
    """

    def __init__(self):
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    @property
    def size(self) -> int:
        return self._size

    @property
    def rank(self) -> int:
        return self._rank

    def is_parallel(self) -> bool:
        return self.size > 1

    # TODO: add all MPI IPC aspects here (OPs etc)

    @staticmethod
    def create_tag(gid: int, src_rank: int, dest_rank: int) -> int:
        """
        Creates a unique communication tag for a given GID, source rank, and destination rank.

        Parameters
        ----------
        gid : int
            Global ID of a simulation.
        src_rank : int
            Rank on which the simulation lives
        dest_rank : int
            Rank to which data of a simulation is to be sent to.

        Returns
        -------
        tag : int
            Unique tag.
        """
        send_hashtag = hashlib.sha256()
        send_hashtag.update((str(src_rank) + str(gid) + str(dest_rank)).encode("utf-8"))
        tag = int(send_hashtag.hexdigest()[:6], base=16)
        return tag

    def _gather_comm_rank(self, rank: Optional[int], comm: Optional[MPI.Comm]) -> Tuple[int, MPI.Comm]:
        rnk : int = self._rank
        if rank is not None:
            rnk = rank
        cmm : MPI.Comm = self._comm
        if comm is not None:
            cmm = comm

        return rnk, cmm