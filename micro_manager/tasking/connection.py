import pickle
import psutil
import socket
import struct
import subprocess
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from mpi4py import MPI

# worker_main gets spawned not part of the micro_manager package.
# Therefore, imports do not work properly. As worker_main requires connection.py,
# this here is a workaround to still load it.
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from task import RegisterAllTask, ShutdownTask


class Connection(ABC):
    @abstractmethod
    def send(self, dst_id: int, obj: Any) -> None:
        pass

    @abstractmethod
    def recv(self, src_id: int) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class MPIConnection(Connection):
    def __init__(self):
        self.inter_comm = None
        self._num_workers = 0

    @classmethod
    def create_workers(
        cls, worker_exec: str, mpi_args: Optional, n_workers: int
    ) -> "MPIConnection":
        args = [worker_exec]
        args.extend(mpi_args or [])

        env = os.environ.copy()
        check_btl_base = (
            "OMPI_MCA_btl" not in env or env["OMPI_MCA_btl"] != "self,vader,tcp"
        )
        check_btl_tcp = (
            "OMPI_MCA_btl_tcp_if_include" not in env
            or env["OMPI_MCA_btl_tcp_if_include"] != "lo"
        )
        check_mca_oob = (
            "OMPI_MCA_oob_tcp_if_include" not in env
            or env["OMPI_MCA_oob_tcp_if_include"] != "lo"
        )
        if check_btl_base or check_btl_tcp or check_mca_oob:
            msg = (
                "Cannot launch MPI workers. Please set the following environment variables:\n"
                "\tOMPI_MCA_btl=self,vader,tcp\n"
                "\tOMPI_MCA_btl_tcp_if_include=lo\n"
                "\tOMPI_MCA_oob_tcp_if_include=lo\n"
            )
            raise RuntimeError(msg)

        comm = MPI.COMM_SELF
        conn = cls()
        conn._num_workers = n_workers
        conn.inter_comm = comm.Spawn(
            "python",
            args=args,
            maxprocs=n_workers,
        )
        return conn

    @classmethod
    def connect_to_micro_manager(cls, parent_comm) -> "MPIConnection":
        conn = cls()
        conn.inter_comm = parent_comm
        return conn

    def send(self, dst_id: int, obj: Any) -> None:
        """
        Sends any data (obj) to the worker with id (dst_id). dst_id corresponds to the
        worker MPI process rank. Data is encoded by pickling it. Thus, one form of pickling must be implemented for
        the data type of (obj). Data is transferred using an MPI inter-process communicator.

        Parameters
        ----------
        dst_id : int
            Worker MPI process rank.
        obj : Any
            Data to send. (needs implemented pickling interface)
        """
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.inter_comm.send(data, dest=dst_id, tag=0)

    def recv(self, src_id: int) -> Any:
        """
        Receives data from the worker with id (src_id). src_id corresponds to the
        worker MPI process rank. Data is transferred using an MPI inter-process communicator.

        Parameters
        ----------
        src_id : int
            Worker MPI process rank.
        """
        data = self.inter_comm.recv(source=src_id, tag=0)
        return pickle.loads(data)

    def close(self) -> None:
        if self._num_workers > 0:
            for i in range(self._num_workers):
                self.send(i, ShutdownTask.send_args())

        self.inter_comm.Disconnect()


class SocketConnection(Connection):
    def __init__(self):
        self.sockets: Dict[int, socket.socket] = {}

    @classmethod
    def create_workers(
        cls, worker_exec: str, launcher: list, host: str, n_workers: int, env_opts: dict
    ) -> "SocketConnection":
        # create listening socket with ephemeral port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, 0))  # kernel picks free port
        server.listen()
        port = server.getsockname()[1]

        executable = [
            "python",
            worker_exec,
            "--backend",
            "socket",
            "--host",
            host,
            "--port",
            str(port),
        ]
        cmd = []
        cmd.extend(launcher)
        cmd.extend(executable)

        env = os.environ.copy()
        env.update(env_opts)
        subprocess.Popen(cmd, env=env)

        conn = cls()
        for wid in range(n_workers):
            sock, _ = server.accept()
            conn.sockets[wid] = sock

        server.close()
        return conn

    @classmethod
    def connect_to_micro_manager(
        cls, worker_id: int, host: str, port: int
    ) -> "SocketConnection":
        conn = cls()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        conn.sockets[worker_id] = sock
        return conn

    def send(self, dst_id: int, obj: Any) -> None:
        """
        Sends any data (obj) to the worker with id (dst_id). dst_id corresponds to the
        worker MPI process rank. Data is encoded by pickling it. Thus, one form of pickling must be implemented for
        the data type of (obj). Data is transferred using a socket by first encoding a header
        (containing the data size), followed by the actual data.

        Parameters
        ----------
        dst_id : int
            Worker MPI process rank.
        obj : Any
            Data to send. (needs implemented pickling interface)
        """
        sock = self.sockets[dst_id]
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack("!Q", len(data))
        sock.sendall(header + data)

    def recv(self, src_id: int) -> Any:
        """
        Receives data from the worker with id (src_id). src_id corresponds to the
        worker MPI process rank. Data is transferred using a socket. First reads the header to determine the data size.
        Then reads the incoming data.

        Parameters
        ----------
        src_id : int
            Worker MPI process rank.
        """
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
        for sock in self.sockets.values():
            sock.close()
        self.sockets.clear()


def get_mpi_pinning(mpi_impl: str, num_workers: int, hostfile: str):
    """
    Returns a list containing args to determine MPI process pinning depending on the MPI implementation.

    Parameters
    ----------
    mpi_impl : string
        MPI implementation

    Returns
    -------
    list
        pinning args
    """
    args = []
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    options = {}
    if mpi_impl == "intel":
        if os.path.exists(hostfile):
            with open(hostfile, "r") as f:
                hosts = f.readlines()
        else:
            raise RuntimeError("Cannot determine target nodes")

        if size % len(hosts) != 0:
            raise RuntimeError("Number of ranks must be divisible by number of hosts")

        mm_ppn = size // len(hosts)
        node_idx = rank // mm_ppn
        node = hosts[node_idx].replace("\n", "")

        locations_int = list(os.sched_getaffinity(0))
        locations = ",".join([str(i) for i in locations_int])

        # See: https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-pinning-simulator.html
        # for more details and a nice visualization
        options.update(
            {
                "I_MPI_DEBUG": "5",
                "I_MPI_PIN": "1",
                "I_MPI_PIN_CELL": "core",
                "I_MPI_PIN_DOMAIN": "1",
                "I_MPI_PIN_PROCESSOR_LIST": locations,
            }
        )

        for key, value in options.items():
            args.append("-genv")
            args.append(f"{key}={value}")
        args.append("-host")
        args.append(f"{node}")

    if mpi_impl == "open":
        args.extend(["--bind-to", "core"])
        locations = ",".join([str(i + rank * num_workers) for i in range(num_workers)])
        args.extend(["--map-by", f"PE-LIST={locations}:ORDERED"])
        args.extend(["--report-bindings"])

    return args, options


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
                # trying to find an interface that is not loopback (127.xx)
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    return a.address

    # Fallback: iterate all interfaces
    for name, iface_addrs in addrs.items():
        for a in iface_addrs:
            if a.family == socket.AF_INET:
                ip = a.address
                # trying to find an interface that is not loopback (127.xx) or link-local (169.254.xx)
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
    mpi_impl: str,
    hostfile: str,
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
    mpi_impl : string
        MPI implementation [intel or open]
    hostfile : str
        Path to Hostfile containing hosts for workers

    Returns
    -------
    conn : Connection
        Established connection on generator side
    """
    if n_workers <= 1:
        return None
    conn = None

    # MPI BACKEND (non-Slurm only)
    if backend == "mpi":
        if is_slurm:
            raise RuntimeError(
                "MPI backend is not supported under Slurm. "
                "Use socket backend instead."
            )
        comm = MPI.COMM_WORLD
        local_rank = comm.Get_rank()
        conn = MPIConnection.create_workers(
            worker_exec=worker_exec,
            mpi_args=[
                "--backend",
                "mpi",
            ],
            n_workers=n_workers,
        )

    # SOCKET BACKEND
    if backend == "socket":
        host = get_local_ip()
        pin_args, pin_options = get_mpi_pinning(mpi_impl, n_workers, hostfile)
        # launch workers
        launcher = None
        if is_slurm:
            launcher = [
                "srun",
                f"--ntasks={n_workers}",
                "--cpus-per-task=1",
                "--cpu-bind=cores",
                "--exclusive",
            ]
        else:
            launcher = ["mpiexec"]
            if mpi_impl == "intel":
                launcher.extend(["-ppn", str(n_workers)])
            launcher.extend(["-n", str(n_workers)])
            launcher.extend(pin_args)

        conn = SocketConnection.create_workers(
            worker_exec=worker_exec,
            launcher=launcher,
            host=host,
            n_workers=n_workers,
            env_opts=pin_options,
        )

    from ..micro_simulation import load_backend_class

    # Send RegisterAllTask to all workers, which sets up the local worker state
    # and registers all potentially used tasks on the workers side.
    # By doing so, less pickling needs to be done during operation. Only the task name and data need to be transferred.
    # Workers can then locally re-construct the task based on the registered tasks.
    for worker_id in range(n_workers):
        conn.send(worker_id, RegisterAllTask(load_backend_class))
        conn.recv(worker_id)

    return conn
