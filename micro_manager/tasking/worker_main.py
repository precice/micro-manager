import argparse
import os
from mpi4py import MPI

from connection import Connection, MPIConnection, SocketConnection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mpi", "socket"])
    parser.add_argument("--host", help="IP or localhost")
    parser.add_argument("--port", type=int, help="Port to open port in micro manager")
    parser.add_argument("--parentrank", type=int, help="Parent rank of spawning micro manager mpi instance")
    args = parser.parse_args()

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    worker_id = rank

    conn, dst_id, src_id = None, 0, 0
    if args.backend == "mpi":
        conn = MPIConnection.connect_to_micromanager(MPI.Comm.Get_parent())
        dst_id = src_id = args.parentrank
    else:
        conn = SocketConnection.connect_to_micromanager(worker_id, args.host, args.port)
        dst_id = src_id = worker_id

    state_data = {}

    while True:
        task = None
        try: task = conn.recv(src_id)
        except Exception: break

        output = None
        try: output = task(state_data)
        except Exception: break

        try: conn.send(dst_id, output)
        except Exception: break

    conn.close()
