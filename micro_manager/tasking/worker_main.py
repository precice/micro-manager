import argparse
import os
from mpi4py import MPI

# only used for tests that require precice
# pyprecice does not exist in CI, thus dummy is provided in test pipeline
# but for that cwd is needed in module PATH
import sys

sys.path.append(os.getcwd())

from task import handle_task

from connection import Connection, MPIConnection, SocketConnection
from micro_manager.tools.logging_wrapper import Logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mpi", "socket"])
    parser.add_argument("--host", help="IP or localhost")
    parser.add_argument("--port", type=int, help="Port to open port in micro manager")
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    log = Logger("Worker", rank=rank)
    worker_id = rank

    conn, dst_id, src_id = None, 0, 0
    if args.backend == "mpi":
        log.log_info(f"Launched Worker with MPI rank: {rank}")
        conn = MPIConnection.connect_to_micro_manager(MPI.Comm.Get_parent())
    else:
        log.log_info(
            f"Launched Worker with Socket rank: {rank}, IP: {args.host} Port: {args.port}"
        )
        conn = SocketConnection.connect_to_micro_manager(
            worker_id, args.host, args.port
        )
        dst_id = src_id = worker_id
    log.log_info(f"Worker rank {rank} connected to parent")
    state_data = {}

    # register possible tasks
    register_task = None
    try:
        register_task = conn.recv(src_id)
    except Exception:
        raise RuntimeError("Failed to recv register tasks")
    output = register_task(state_data)
    try:
        conn.send(dst_id, output)
    except Exception:
        raise RuntimeError("Failed to send register tasks output")

    while True:
        task_descriptor = None
        try:
            task_descriptor = conn.recv(src_id)
        except Exception:
            break

        output = None
        try:
            output = handle_task(state_data, task_descriptor)
        except Exception:
            break

        try:
            conn.send(dst_id, output)
        except Exception:
            break

    conn.close()
