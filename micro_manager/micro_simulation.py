"""
This file provides a function which creates a class Simulation. This class inherits from the user-provided
class MicroSimulation. A global ID member variable is defined for the class Simulation, which ensures that each
created object is uniquely identifiable in a global setting.
"""
class MicroSimulationWrapper:
    # TODO support optional initialize
    # TODO support optional output

    def __init__(self, sim_cls, name, global_id, num_ranks, executor, late_init):
        self._sim_cls = sim_cls # backend impl class
        self._name = name # needs to be unique
        self._gid = global_id
        self._num_ranks = num_ranks
        self._executor = executor

        self._states = [None] * num_ranks # list of sims
        self._instance = None

        if late_init: return

        if self._num_ranks <= 1:
            self._instance = sim_cls(self._gid)
        else:
            f_gen_instances = self._executor.submit(
                MicroSimulationWrapper.gen_instances,
                # args
                gid=self._gid,
                num_ranks=self._num_ranks,
                sim_cls=self._sim_cls,
                # execution params
                resource_dict={"cores": self._num_ranks},
            )

            for rank, sim_state in f_gen_instances.result():
                self._states[rank] = sim_state

    def solve(self, micro_sim_input, dt):
        if self._num_ranks <= 1:
            return self._instance.solve(micro_sim_input, dt)
        else:
            f_solve = self._executor.submit(
                MicroSimulationWrapper.solve_local,
                # args
                sim_cls=self._sim_cls,
                states=self._states,
                input=micro_sim_input,
                dt=dt,
                # execution params
                resource_dict={"cores": self._num_ranks},
            )

            results = f_solve.result()
            result = None
            for rank, output, state in results:
                if rank == 0: result = output
                self._states[rank] = state

            return result

    def get_state(self):
        if self._num_ranks <= 1: return self._instance.get_state()
        else: return self._states

    def set_state(self, states):
        if self._num_ranks <= 1: self._instance.set_state(states)
        else: self._states = states

    def get_global_id(self): return self._gid
    def get_name(self): return self._name

    @staticmethod
    def gen_instances(gid, num_ranks, sim_cls):
        from mpi4py import MPI

        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        assert size == num_ranks

        return rank, sim_cls(gid).get_state()

    @staticmethod
    def solve_local(sim_cls, states, input, dt):
        from mpi4py import MPI

        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        assert size == len(states)
        sim = sim_cls(-1)
        sim.set_state(states[rank])

        if rank == 0:
            output = sim.solve(input, dt)
            return rank, output, sim.get_state()
        else:
            sim.solve(input, dt)
            return rank, None, sim.get_state()


def create_simulation_class(micro_simulation_class, num_ranks, executor, sim_class_name=None):
    """
    Creates a class Simulation which inherits from the class of the micro simulation.

    Parameters
    ----------
    micro_simulation_class : class
        The base class from the micro simulation script.

    sim_class_name : [string, None]
        The name of the class to be created. If None, a unique name will be generated.

    Returns
    -------
    Simulation : class
        Definition of class Simulation defined in this function.
    """
    if not hasattr(micro_simulation_class, "get_global_id"): raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "get_state"):     raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "set_state"):     raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "solve"):        raise ValueError("Invalid micro simulation class")

    if sim_class_name is None:
        if not hasattr(create_simulation_class, "sim_id"): create_simulation_class.sim_id = 0
        else: create_simulation_class.sim_id += 1
        sim_class_name = f"MicroSimulation{create_simulation_class.sim_id}"

    cls_body = """
backend_cls = sim_cls
def __init__(self, global_id, late_init=False):
    wrapper_cls.__init__(self, sim_cls, name, global_id, num_ranks, executor, late_init)
    """
    cls_dict = {}
    local_globals = {
        "__builtins__": __builtins__,
        "wrapper_cls": MicroSimulationWrapper,
        "sim_cls": micro_simulation_class,
        "name": sim_class_name,
        "num_ranks": num_ranks,
        "executor": executor,
    }

    exec(cls_body, local_globals, cls_dict)
    result_cls = type(sim_class_name, (MicroSimulationWrapper,), cls_dict)
    return result_cls
