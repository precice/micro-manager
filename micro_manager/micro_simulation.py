"""
This file provides a function which creates a class Simulation. This class inherits from the user-provided
class MicroSimulation. A global ID member variable is defined for the class Simulation, which ensures that each
created object is uniquely identifiable in a global setting.
"""

from abc import ABC, abstractmethod
import inspect
import importlib as ipl


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


class MicroSimulationWrapper(MicroSimulationInterface):
    def __init__(self, name, sim_cls, global_id, late_init):
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
    def __init__(self, sim_cls, name, log):
        self._sim_cls = sim_cls
        self._name = name
        self._log = log

    @property
    def name(self):
        return self._name

    def __call__(self, gid, *, late_init=False):
        return MicroSimulationWrapper(
            self._name,
            self._sim_cls,
            gid,
            late_init,
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
    CLS_NAME = "MicroSimulation"
    return getattr(ipl.import_module(path_to_micro_file, CLS_NAME), CLS_NAME)


def create_simulation_class(
    log,
    micro_simulation_class,
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
        micro_simulation_class, sim_class_name, log
    )
    return result_cls
