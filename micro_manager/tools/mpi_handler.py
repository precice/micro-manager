from typing import Optional, Tuple, List, Any, Dict, Union

from mpi4py import MPI
import numpy as np
import hashlib

class MPIHandler:
    """
    Encapsulates all MPI related aspects and provides Micro Manager specific functionality.
    All methods can override the rank and communicator if desired.
    """

    def __init__(self, comm: MPI.Comm=MPI.COMM_WORLD):
        self._comm = comm
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

    def create_node_buffer(self, dtype: MPI.Datatype, count: int):
        """
        Constructs a numpy array buffer of the given dimensions for the ranks located on the same node.
        The buffer will be owned by rank with ID 0 in the node local communicator.

        Parameters
        ----------
        dtype : MPI.Datatype
            data type of the array buffer.
        count : int
            number of elements in the array buffer.

        Returns
        -------
        buffer : np.ndarray
            buffer array
        node_handler : MPIHandler
            handles the node local MPI communicator
        """
        node_handler = MPIHandler(self.comm.Split_type(MPI.COMM_TYPE_SHARED))

        item_size = dtype.Get_size()
        n_bytes = 0
        if node_handler.rank == 0:
            n_bytes = count * count * item_size

        win = MPI.Win.Allocate_shared(n_bytes, item_size, comm=node_handler.comm)
        # Get the buffer on the local rank 0
        buffer, item_size = win.Shared_query(0)
        if item_size != dtype.Get_size():
            raise RuntimeError("Item size mismatch in shared memory.")

        # Create a numpy array from the buffer
        array_buffer = np.array(buffer, dtype="B", copy=False)

        return array_buffer, node_handler

    def get_ranks_of_objects(self, objects: List[Any], /, comm: Optional[MPI.Comm]=None, rank: Optional[int]=None) -> Dict[Any, int]:
        """
        Get the ranks of all objects.

        Parameters
        ----------
        objects : List[Any]
            List of objects to get ranks of.
        comm : Optional[MPI.Comm]
            MPI communicator. Defaults to MPIHandler.comm
        rank : Optional[int]
            MPI rank within provided communicator. Defaults to MPIHandler.rank

        Returns
        -------
        ranks_of_objects : Dict[Any, int]
            Mapping of all objects of all ranks to their respective rank IDs.
        """
        _rank, _comm = self._gather_comm_rank(rank, comm)
        glob_maps : List[Dict[Any, int]] = _comm.allgather({obj: _rank for obj in objects})

        obj_to_rank : Dict[Any, int] = dict()
        for map in glob_maps:
            for obj, rnk in map.items():
                obj_to_rank[obj] = rnk

        return obj_to_rank

    def create_empty_exchange_map(self) -> Dict[int, List[Any]]:
        return { r:[] for r in range(self.size)}

    def exchange(self, send_map: Dict[int, List[Any]], /, return_inverse: bool=False, comm: Optional[MPI.Comm]=None, rank: Optional[int]=None) -> Union[List[Any], Tuple[List[Any], Dict[int, List[Any]]]]:
        """
        Transfers data between ranks as p2p communication according to the provided send_map.

        Parameters
        ----------
        send_map : Dict[int, List[Any]]
            Mapping from destination rank to list of data to be transferred.
        return_inverse : bool
            Return inverse transfer or not.
        comm : Optional[MPI.Comm]
            MPI communicator. Defaults to MPIHandler.comm
        rank : Optional[int]
            MPI rank within provided communicator. Defaults to MPIHandler.rank

        Returns
        -------
        comm_result : Union[List[Any], Tuple[List[Any], Dict[int, List[Any]]]]
            If return_inverse is True, returns a list of all received data as well as a mapping from which rank
            which data was sent. Otherwise, returns only the received data list.
        """
        _rank, _comm = self._gather_comm_rank(rank, comm)
        _size = self.size if _comm == self.comm else _comm.Get_size()

        send_counts = [len(send_map[r]) for r in range(_size)]
        send_counts[_rank] = 0  # ignore local count
        glob_send_counts = _comm.allgather(send_counts)

        send_reqs = []
        for recv_rank, data in send_map.items():
            if recv_rank == _rank:
                continue
            if len(data) == 0:
                continue
            for d_idx, entry in enumerate(data):
                req = _comm.isend(
                    entry, dest=recv_rank, tag=self.create_tag(d_idx, _rank, recv_rank)
                )
                send_reqs.append(req)

        recv_reqs = []
        for send_rank in range(_size):
            if send_rank == _rank:
                continue
            if glob_send_counts[send_rank][_rank] == 0:
                continue
            for d_idx in range(glob_send_counts[send_rank][_rank]):
                req = _comm.irecv(
                    None, source=send_rank, tag=self.create_tag(d_idx, send_rank, _rank)
                )
                recv_reqs.append(tuple([send_rank, req]))

        MPI.Request.Waitall(send_reqs)

        result = []
        result.extend(send_map[_rank])
        inv_map = {r: [] for r in range(_size)}

        for source_rank, req in recv_reqs:
            data = req.wait()
            result.append(data)
            if return_inverse:
                inv_map[source_rank].append(data)

        if return_inverse:
            inv_map[_rank].extend(send_map[_rank])
            return result, inv_map
        else:
            return result

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
        rnk : int = self.rank
        if rank is not None:
            rnk = rank
        cmm : MPI.Comm = self.comm
        if comm is not None:
            cmm = comm

        return rnk, cmm


MPIHandlerRankLocal = MPIHandler(MPI.COMM_SELF)
