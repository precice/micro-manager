"""
This file provides a function which creates a class Simulation. This class inherits from the user-provided
class MicroSimulation. A global ID member variable is defined for the class Simulation, which ensures that each
created object is uniquely identifiable in a global setting.
"""

from abc import ABC, abstractmethod
import inspect
import importlib as ipl

from .tasking.task import (
    ConstructTask,
    ConstructLateTask,
    DeleteTask,
    InitializeTask,
    OutputTask,
    SolveTask,
    SetStateTask,
    GetStateTask,
)


class MicroSimulationInterface(ABC):
    @abstractmethod
    def solve(self, micro_sim_input, dt):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def get_global_id(self):
        pass

    @abstractmethod
    def set_global_id(self, global_id):
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def output(self):
        pass


class MicroSimulationLocal(MicroSimulationInterface):
    def __init__(self, gid, late_init, sim_cls):
        self._gid = gid
        self._instance = sim_cls(-1 if late_init else gid)

    def solve(self, micro_sim_input, dt):
        return self._instance.solve(micro_sim_input, dt)

    def get_state(self):
        return self._instance.get_state()

    def set_state(self, state):
        return self._instance.set_state(state)

    def get_global_id(self):
        return self._gid

    def set_global_id(self, global_id):
        self._gid = global_id

    def __getattr__(self, name):
        return getattr(self._instance, name)

    def initialize(self, *args, **kwargs):
        return self._instance.initialize(*args, **kwargs)

    def output(self):
        return self._instance.output()


class MicroSimulationRemote(MicroSimulationInterface):
    def __init__(self, gid, late_init, num_ranks, conn, cls_path):
        self._cls_path = cls_path
        self._gid = gid
        self._num_ranks = num_ranks
        self._conn = conn

        construct_cls = ConstructLateTask if late_init else ConstructTask
        for worker_id in range(self._num_ranks):
            task = construct_cls.send_args(self._gid, self._cls_path)
            self._conn.send(worker_id, task)

        for worker_id in range(self._num_ranks):
            self._conn.recv(worker_id)

    def __del__(self):
        for worker_id in range(self._num_ranks):
            task = DeleteTask.send_args(self._gid)
            self._conn.send(worker_id, task)

    def solve(self, micro_sim_input, dt):
        for worker_id in range(self._num_ranks):
            task = SolveTask.send_args(self._gid, micro_sim_input, dt)
            self._conn.send(worker_id, task)

        result = None
        for worker_id in range(self._num_ranks):
            output = self._conn.recv(worker_id)
            if worker_id == 0:
                result = output

        return result

    def get_state(self):
        for worker_id in range(self._num_ranks):
            task = GetStateTask.send_args(self._gid)
            self._conn.send(worker_id, task)

        result = {}
        for worker_id in range(self._num_ranks):
            result[worker_id] = self._conn.recv(worker_id)

        return result

    def set_state(self, state):
        for worker_id in range(self._num_ranks):
            task = SetStateTask.send_args(self._gid, state[worker_id])
            self._conn.send(worker_id, task)

        result = {}
        for worker_id in range(self._num_ranks):
            result[worker_id] = self._conn.recv(worker_id)
        self._gid = result[0]

    def get_global_id(self):
        return self._gid

    def set_global_id(self, global_id):
        self._gid = global_id

    def initialize(self, *args, **kwargs):
        for worker_id in range(self._num_ranks):
            task = InitializeTask.send_args(self._gid, *args, **kwargs)
            self._conn.send(worker_id, task)

        result = None
        for worker_id in range(self._num_ranks):
            output = self._conn.recv(worker_id)
            if worker_id == 0:
                result = output

        return result

    def output(self):
        for worker_id in range(self._num_ranks):
            task = OutputTask.send_args(self._gid)
            self._conn.send(worker_id, task)

        result = None
        for worker_id in range(self._num_ranks):
            output = self._conn.recv(worker_id)
            if worker_id == 0:
                result = output

        return result


class MicroSimulationWrapper(MicroSimulationInterface):
    """
    If only a single rank is in use: will contain the micro sim instance.
    Otherwise, it will delegate method calls to workers and not contain state.
    """

    def __init__(self, name, sim_cls, cls_path, global_id, late_init, num_ranks, conn):
        self._impl = None

        if num_ranks > 1 and conn is not None:
            self._impl = MicroSimulationRemote(
                global_id, late_init, num_ranks, conn, cls_path
            )
        else:
            self._impl = MicroSimulationLocal(global_id, late_init, sim_cls)

        self._external_data = dict()
        self._name = name

    def solve(self, micro_sim_input, dt):
        return self._impl.solve(micro_sim_input, dt)

    def get_state(self):
        return self._impl.get_state()

    def set_state(self, state):
        return self._impl.set_state(state)

    def get_global_id(self):
        return self._impl.get_global_id()

    def set_global_id(self, global_id):
        return self._impl.set_global_id(global_id)

    def initialize(self, *args, **kwargs):
        return self._impl.initialize(*args, **kwargs)

    def output(self):
        return self._impl.output()

    def __getattr__(self, name):
        return getattr(self._impl, name)

    @property
    def attachments(self):
        return self._external_data

    @attachments.setter
    def attachments(self, value):
        self._external_data = value

    @property
    def name(self):
        return self._name


class MicroSimulationClass:
    def __init__(self, sim_cls, cls_path, name, num_ranks, conn, log):
        self._sim_cls = sim_cls
        self._cls_path = cls_path
        self._name = name
        self._num_ranks = num_ranks
        self._conn = conn
        self._log = log

    @property
    def name(self):
        return self._name

    def __call__(self, gid, *, late_init=False):
        return MicroSimulationWrapper(
            self._name,
            self._sim_cls,
            self._cls_path,
            gid,
            late_init,
            self._num_ranks,
            self._conn,
        )

    @property
    def backend_cls(self):
        return self._sim_cls

    def check_initialize(self, test_instance, test_input):
        has_init = hasattr(self._sim_cls, "initialize")
        if not has_init:
            return False, False
        callable_init = callable(getattr(self._sim_cls, "initialize"))
        if not callable_init:
            return False, False

        has_args = False

        # Try to get the signature of the initialize() method, if it is written in Python
        try:
            argspec = inspect.getfullargspec(self._sim_cls.initialize)
            # The first argument in the signature is self
            if len(argspec.args) == 1:
                has_args = False
            elif len(argspec.args) == 2:
                has_args = True
            else:
                raise Exception(
                    "The initialize() method of the Micro simulation has an incorrect number of arguments."
                )
        except TypeError:
            self._log.log_info_rank_zero(
                "The signature of initialize() method of the micro simulation cannot be determined. "
                + "Trying to determine the signature by calling the method."
            )
            # Try to call the initialize() method without initial data
            try:
                test_instance.initialize()
                has_args = False
            except TypeError:
                self._log.log_info_rank_zero(
                    "The initialize() method of the micro simulation has arguments. "
                    + "Attempting to call it again with initial data."
                )
                try:
                    test_instance.initialize(test_input)
                    has_args = True
                except TypeError:
                    raise Exception(
                        "The initialize() method of the Micro simulation has an incorrect number of arguments."
                    )

        return has_init and callable_init, has_args

    def check_output(self):
        has_init = hasattr(self._sim_cls, "output")
        if not has_init:
            return False
        callable_init = callable(getattr(self._sim_cls, "output"))

        return has_init and callable_init


def load_backend_class(path_to_micro_file):
    def try_load(name):
        try:
            return getattr(ipl.import_module(path_to_micro_file, name), name)
        except ImportError as ie:
            return None
        except AttributeError as ae:
            return None

    CLS_NAME = "MicroSimulation"
    # attempt to load with base name
    result = try_load(CLS_NAME)
    if result is not None:
        return result

    # attempt to load with appended indices
    for i in range(10):
        result = try_load(f"{CLS_NAME}{i}")
        if result is not None:
            return result

    # failed to load any class
    raise RuntimeError(f"Could not load micro simulation from {path_to_micro_file}")


def create_simulation_class(
    log,
    micro_simulation_class,
    path_to_micro_file,
    num_ranks,
    conn=None,
    sim_class_name=None,
):
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
    if not hasattr(micro_simulation_class, "get_global_id"):
        raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "get_state"):
        raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "set_state"):
        raise ValueError("Invalid micro simulation class")
    if not hasattr(micro_simulation_class, "solve"):
        raise ValueError("Invalid micro simulation class")

    if sim_class_name is None:
        if not hasattr(create_simulation_class, "sim_id"):
            create_simulation_class.sim_id = 0
        else:
            create_simulation_class.sim_id += 1
        sim_class_name = f"MicroSimulation{create_simulation_class.sim_id}"

    result_cls = MicroSimulationClass(
        micro_simulation_class, path_to_micro_file, sim_class_name, num_ranks, conn, log
    )
    return result_cls
