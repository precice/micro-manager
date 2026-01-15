import pickle
import psutil
import socket
import struct
import subprocess
import os
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
            f"python {worker_exec}",
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
    def create_workers(cls, worker_exec: str, launcher: list, host: str, n_workers: int) -> "SocketConnection":
        # create listening socket with ephemeral port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, 0))  # kernel picks free port
        server.listen()
        port = server.getsockname()[1]

        executable = [
            "python",
            worker_exec,
            "--backend", "socket",
            "--host", host,
            "--port", str(port),
        ]
        cmd = []
        cmd.extend(launcher)
        cmd.extend(executable)
        subprocess.Popen(cmd, env=os.environ.copy())

        conn = cls()
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


def get_local_ip(preferred_ifaces=None) -> str:
    """
    Returns a non-loopback IPv4 address without accessing external networks.

    Parameters
    ----------
    preferred_ifaces : list[str], optional
        If provided, try interfaces in this order first (e.g., ["ib0", "eno1"])

    Returns
    -------
    str
        The selected IPv4 address
    """
    addrs = psutil.net_if_addrs()

    candidates = []

    # Iterate over preferred interfaces first
    if preferred_ifaces:
        for name in preferred_ifaces:
            if name not in addrs:
                continue
            for a in addrs[name]:
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    return a.address

    # Fallback: iterate all interfaces
    for name, iface_addrs in addrs.items():
        for a in iface_addrs:
            if a.family == socket.AF_INET:
                ip = a.address
                if not ip.startswith("127.") and not ip.startswith("169.254."):
                    candidates.append(ip)

    if candidates:
        return candidates[0]

    raise RuntimeError("No non-loopback IPv4 address found")


def spawn_local_workers(
    worker_exec: str,
    n_workers: int,
    backend: str,
    is_slurm: bool,
):
    """
    Spawn worker processes. On Slurm systems: MPI spawn now supported, socket backend enforced.
    Ephemeral port auto-selected.

    Parameters
    ----------
    worker_exec : str
        path to worker executable
    n_workers : int
        number of worker processes, must be > 1 otherwise returns None
    backend : str
        mpi or socket
    is_slurm : bool
        is our system slurm based?

    Returns
    -------
    conn : Connection
        Established connection on generator side
    """
    from .task import RegisterAllTask

    if n_workers <= 1: return None
    conn = None

    # MPI BACKEND (non-Slurm only)
    if backend == "mpi":
        if is_slurm: raise RuntimeError(
            "MPI backend is not supported under Slurm. "
            "Use socket backend instead."
        )
        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        conn = MPIConnection.create_workers(
            worker_exec=worker_exec,
            mpi_args=[
                "--backend", "mpi",
                "--parentrank", str(local_rank),
            ],
            n_workers=n_workers,
        )

    # SOCKET BACKEND
    if backend == "socket":
        host = get_local_ip()

        # launch workers
        launcher = None
        if is_slurm:
            launcher = [
                "srun",
                #"--exclusive",
                "--ntasks", str(n_workers),
                "--kill-on-bad-exit=1",
            ]
        else:
            launcher = [
                "mpiexec",
                "-n", str(n_workers),
            ]

        conn = SocketConnection.create_workers(
            worker_exec=worker_exec,
            launcher=launcher,
            host=host,
            n_workers=n_workers
        )

    from ..micro_simulation import load_backend_class

    for worker_id in range(n_workers):
        conn.send(worker_id, RegisterAllTask(load_backend_class))
        conn.recv(worker_id)

    return conn
