import pickle
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from mpi4py import MPI

class Connection(ABC):
    @abstractmethod
    def send(self, dst_id: int, obj: Any) -> None: pass
    @abstractmethod
    def recv(self, src_id: int) -> Any: pass
    @abstractmethod
    def close(self) -> None: pass


class MPIConnection(Connection):
    def __init__(self):
        self.inter_comm = None

    @classmethod
    def create_workers(cls, worker_exec: str, mpi_args: Optional, n_workers: int) -> "MPIConnection":
        comm = MPI.COMM_SELF
        conn = cls()
        conn.inter_comm = comm.Spawn(
            worker_exec,
            args=mpi_args or [],
            maxprocs=n_workers,
        )
        return conn

    @classmethod
    def connect_to_micromanager(cls, parent_comm) -> "MPIConnection":
        conn = cls()
        conn.inter_comm = parent_comm
        return conn

    def send(self, dst_id: int, obj: Any) -> None:
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.inter_comm.send(data, dest=dst_id, tag=0)

    def recv(self, src_id: int) -> Any:
        data = self.inter_comm.recv(source=src_id, tag=1)
        return pickle.loads(data)

    def close(self) -> None:
        self.inter_comm.Disconnect()


class SocketConnection(Connection):
    def __init__(self):
        self.sockets: Dict[int, socket.socket] = {}

    @classmethod
    def accept_workers(cls, host: str, port: int, n_workers: int) -> "SocketConnection":
        conn = cls()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen()

        for wid in range(n_workers):
            sock, _ = server.accept()
            conn.sockets[wid] = sock

        server.close()
        return conn

    @classmethod
    def connect_to_micromanager(
        cls, worker_id: int, host: str, port: int
    ) -> "SocketConnection":
        conn = cls()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        conn.sockets[worker_id] = sock
        return conn

    def send(self, dst_id: int, obj: Any) -> None:
        sock = self.sockets[dst_id]
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack("!Q", len(data))
        sock.sendall(header + data)

    def recv(self, src_id: int) -> Any:
        sock = self.sockets[src_id]
        header = sock.recv(8)
        if not header:
            raise EOFError
        (size,) = struct.unpack("!Q", header)
        payload = b""
        while len(payload) < size:
            chunk = sock.recv(size - len(payload))
            if not chunk:
                raise EOFError
            payload += chunk
        return pickle.loads(payload)

    def close(self) -> None:
        for sock in self.sockets.values(): sock.close()
        self.sockets.clear()