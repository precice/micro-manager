import hashlib
import numpy as np
from mpi4py import MPI


def get_ranks_of_sims(global_ids, rank, comm, global_number_of_sims) -> np.ndarray:
    """
    Get the ranks of all simulations.

    Parameters
    ----------
    global_ids : list
        Global ids on local rank.
    rank : int
        Rank of simulation.
    comm : MPI.Comm
        MPI communicator.
    global_number_of_sims : int
        Global number of sims.

    Returns
    -------
    ranks_of_sims : np.ndarray
        Array of ranks on which simulations exist.
    """
    gids_to_rank = dict()
    for gid in global_ids:
        gids_to_rank[gid] = rank

    ranks_maps_as_list = comm.allgather(gids_to_rank)

    ranks_of_sims = np.zeros(global_number_of_sims, dtype=np.intc)
    for ranks_map in ranks_maps_as_list:
        for gid, rank in ranks_map.items():
            ranks_of_sims[gid] = rank

    return ranks_of_sims


def create_tag(sim_id: int, src_rank: int, dest_rank: int) -> int:
    """
    For a given simulations ID, source rank, and destination rank, a unique tag is created.

    Parameters
    ----------
    sim_id : int
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
    send_hashtag.update((str(src_rank) + str(sim_id) + str(dest_rank)).encode("utf-8"))
    tag = int(send_hashtag.hexdigest()[:6], base=16)
    return tag


def p2p_comm(
    global_ids,
    rank,
    comm,
    global_number_of_sims,
    is_sim_on_this_rank,
    assoc_active_ids: list,
    data: list,
) -> list:
    """
    Handle process to process communication for a given set of associated active IDs and data.

    Parameters
    ----------
    global_ids : list
        Global ids on local rank.
    rank : int
        Rank of simulation.
    comm : MPI.Comm
        MPI communicator.
    global_number_of_sims : int
        Global number of sims.
    is_sim_on_this_rank: list
        TODO what?
    assoc_active_ids : list
        Global IDs of active simulations which are not on this rank and are associated to
        the inactive simulations on this rank.
    data : list
        Complete data from which parts are to be sent and received.

    Returns
    -------
    recv_reqs : list
        List of MPI requests of receive operations.
    """
    rank_of_sim = get_ranks_of_sims(global_ids, rank, comm, global_number_of_sims)

    send_map_local = dict()  # keys are global IDs, values are rank to send to
    send_map = (
        dict()
    )  # keys are global IDs of sims to send, values are ranks to send the sims to
    recv_map = (
        dict()
    )  # keys are global IDs to receive, values are ranks to receive from

    for i in assoc_active_ids:
        # Add simulation and its rank to receive map
        recv_map[i] = rank_of_sim[i]
        # Add simulation and this rank to local sending map
        send_map_local[i] = rank

    # Gather information about which sims to send where, from the sending perspective
    send_map_list = comm.allgather(send_map_local)

    for d in send_map_list:
        for i, rank in d.items():
            if is_sim_on_this_rank[i]:
                if i in send_map:
                    send_map[i].append(rank)
                else:
                    send_map[i] = [rank]

    # Asynchronous send operations
    send_reqs = []
    for gid, send_ranks in send_map.items():
        lid = global_ids.index(gid)
        for send_rank in send_ranks:
            tag = create_tag(gid, rank, send_rank)
            req = comm.isend(data[lid], dest=send_rank, tag=tag)
            send_reqs.append(req)

    # Asynchronous receive operations
    recv_reqs = []
    for gid, recv_rank in recv_map.items():
        tag = create_tag(gid, recv_rank, rank)
        bufsize = (
            1 << 30
        )  # allocate and use a temporary 1 MiB buffer size https://github.com/mpi4py/mpi4py/issues/389
        req = comm.irecv(bufsize, source=recv_rank, tag=tag)
        recv_reqs.append(req)

    # Wait for all non-blocking communication to complete
    MPI.Request.Waitall(send_reqs)

    return recv_reqs
