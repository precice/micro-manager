---
title: Running with multi-level parallelization
permalink: tooling-micro-manager-nested-parallelization.html
keywords: tooling, macro-micro, parallelization, workers, tasking
summary: Micro Manager supports MPI parallelized micro solvers with nested parallelization.
---

## Overview

With default configurations, running the Micro Manager in parallel with MPI results in $N$ ranks within the same MPI communicator.
In this approach, each operates under the assumption, that Micro Simulations are either solved in serial or use shared memory parallelization.
If one attempts to solve the Micro Simulation with MPI, this will result in incorrect behavior, as different Micro Simulation instances communicate with each other. Further, this will likely lead to race conditions or deadlocks.

To support MPI parallelization for Micro Simulations, multi-level parallelization is introduced. In this execution mode, each Micro Manager rank $n\in [N]$ creates its own sub-process group of $M$ ranks with MPI.
This results in a total of $N\cdot M$ processes. The sub-process groups are launched to run a "worker" process, and will be henceforth called workers. The worker executable is a generic task handler. Its sole task is to receive incoming tasks, decode them, solve the request and send back the results. Workers have a persistent storage, to which all incoming tasks have access.

The Micro Manager makes use of this, by first initializing the persistent storage with Micro Manager specific task-classes and loading the relevant Micro Simulation classes.
When the Micro Manager attempts to solve a Micro Simulation, this results in a "Solve-Request" being emitted and transferred to its $M$ workers.
Each of those, receive the request and solve the specified simulation in parallel. Upon completion, the results are transferred back to the Micro Manager rank. Other operations within the Micro Simulation Interface are handled similarly.

## Tasks

In accordance with the Micro Simulation Interface, each interface method has its corresponding Task implementation.
During operation no task objects are pickled (serialized) and transferred between the Micro Manager and its workers. Instead, only a string representation of the respective task class and the required data is transferred.
This is done for performance reasons. The task classes are only transferred once during worker initialization and are stored under the key "tasks" in each workers persistent storage.

```python
class Task:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, state_data: dict):
        return self.fn(*self.args, state_data=state_data, **self.kwargs)

    @classmethod
    def send_args(cls, *args, **kwargs):
        return cls.__name__, args, kwargs

class SolveTask(Task):
    def __init__(self, gid, sim_input, dt):
        super().__init__(SolveTask.solve, gid=gid, sim_input=sim_input, dt=dt)

    @staticmethod
    def solve(gid, sim_input, dt, state_data):
        sim_output = state_data["sim_instances"][gid].solve(sim_input, dt)
        return sim_output
```

The code snippet above showcases the "Solve-Task" alongside the "Task" interface.
The Micro Manager only calls the `send_args` method to transfer the required data.
Workers call the constructor, which registers the to-be-called method with the required parameters.
When the task is called via `__call__`, the registered method receives the previous parameters in addition to the persistent storage of the worker (`state_data`).

## Workers

Workers first establish a connection to their respective Micro Manager parent (either via MPI or Sockets) and allocate their persistent storage (`state_data`).
The Micro Manager initializes the `state_data` in the following form:

```python
task_dict = dict()
task_dict[SolveTask.__name__] = SolveTask
task_dict[...] = ...
state_data["tasks"] = task_dict
state_data["sim_classes"] = dict()
state_data["sim_instances"] = dict()
state_data["load_function"] = load_function
```

The key `tasks` contains all possible task classes, that may be encountered during simulation.
`sim_classes` is a dictionary mapping class names to loaded Micro Simulation classes.
`sim_instances` is a dictionary mapping Micro Simulation GIDs to their respective Micro Simulation instance.
`load_function` contains a function capable of loading a Micro Simulation class.

Afterwards, the main execution loop is entered:
```python
def handle_task(state_data, task_descriptor):
    name, args, kwargs = task_descriptor
    task = state_data["tasks"][name](*args, **kwargs)
    return task(state_data)

# Worker Main
conn, src_id, dst_id, state_data = ...
while True:
    try:
        task_descriptor = conn.recv(src_id)
        output = handle_task(state_data, task_descriptor)
        conn.send(dst_id, output)
    except Exception:
        break
conn.close()
```
After receiving a `task_descriptor`, that was created by `Task.send_args`, `handle_task` creates an instance of the task and attempts to handle the request.
The resulting output is then sent back to the Micro Manager parent rank.

## Connection

Connections between workers and Micro Manager ranks are realized as either MPI or socket based communication.
The MPI version (`MPIConnection`) calls internally `MPI_Comm_spawn`, which launches a separate process group and yields a MPI communicator between the Micro Manager parent and the newly create process group.
This approach is known to be more error-prone and required the definition of the following environment variables:

```bash
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_btl_tcp_if_include=lo
export OMPI_MCA_oob_tcp_if_include=lo
```

The socket based version (`SocketConnection`) manually creates a sub-process group either using `mpiexec` or `srun` and establishes socket connections from the Micro Manager rank to all of its workers.

### Pinning

During the creation of the worker processes, pinning is specified.
When each Micro Manager (MM) rank $n \in N$ creates its $M$ workers, it is assumed that each MM rank $n$ is pinned to a contiguous domain of size $M$ CPU cores.
Subsequently, the created workers (W) of rank $n$ are pinned to distinct cores within this region. In the following example $N=8$, $M=2$ and 2 nodes are used.

```
Node 0:       ╭─core─╮
┌──────┬──────┬──────┬──────┐  ┌──────┬──────┬──────┬──────┐
│MM0 W0│MM0 W1│MM1 W0│MM1 W1│  │MM2 W0│MM2 W1│MM3 W0│MM3 W1│
└──────┴──────┴──────┴──────┘  └──────┴──────┴──────┴──────┘
╰──────────socket 0─────────╯  ╰──────────socket 1─────────╯

Node 1:
┌──────┬──────┬──────┬──────┐  ┌──────┬──────┬──────┬──────┐
│MM4 W0│MM4 W1│MM5 W0│MM5 W1│  │MM6 W0│MM6 W1│MM7 W0│MM7 W1│
└──────┴──────┴──────┴──────┘  └──────┴──────┴──────┴──────┘
╰──────────socket 0─────────╯  ╰──────────socket 1─────────╯
```

__IMPORTANT__: For this to function correctly, the user must pin the Micro Manager ranks as described.
Additionally, when multiple nodes are used, then it is assumed that the Micro Manager ranks are distributed in an incrementing fashion per node first (as shown in the sketch).
The used nodes must be provided in a hostfile, otherwise all worker processes are created on the same node.
