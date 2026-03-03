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
    InitializeTask,
    OutputTask,
    SolveTask,
    SetStateTask,
    GetStateTask,
)


class MicroSimulationInterface(ABC):
    """
    Abstract base class for micro simulations. Users should inherit from this class
    when creating their micro simulation and implement all abstract methods.

    The methods ``initialize`` and ``output`` are optional — override them only if
    your simulation needs them. The Micro Manager checks ``requires_initialize()``
    and ``requires_output()`` to decide whether to call them.

    Example usage::

        from micro_manager import MicroSimulationInterface

        class MicroSimulation(MicroSimulationInterface):
            def __init__(self, sim_id: int) -> None:
                self._sim_id = sim_id

            def initialize(self, initial_data: dict | None = None) -> dict | None:
                pass

            def solve(self, macro_data: dict, dt: float) -> dict:
                return {}

            def get_state(self) -> object:
                return None

            def set_state(self, state: object) -> None:
                pass

            def get_global_id(self) -> int:
                return self._sim_id

            def set_global_id(self, global_id: int) -> None:
                self._sim_id = global_id

            def output(self) -> None:
                pass
    """

    def initialize(self, *args, **kwargs) -> dict | None:
        """
        Initialize the micro simulation. Called once before the coupling loop starts.
        This method is optional. Override it if your simulation requires initialization.

        Parameters
        ----------
        initial_data : dict, optional
            Initial data passed from the Micro Manager.

        Returns
        -------
        dict or None
            Optional initial output data to be used in the adaptivity calculation.
        """
        pass

    @abstractmethod
    def solve(self, micro_sim_input: dict, dt: float) -> dict:
        """
        Solve the micro simulation for one time step.

        Parameters
        ----------
        micro_sim_input : dict
            Input data from the macro simulation.
        dt : float
            Time step size.

        Returns
        -------
        micro_sim_output : dict
            Output data to be passed to the macro simulation.
        """
        pass

    @abstractmethod
    def get_state(self) -> object:
        """
        Return the current state of the micro simulation for checkpointing.

        Returns
        -------
        state : object
            The current state of the micro simulation.
        """
        pass

    @abstractmethod
    def set_state(self, state: object) -> None:
        """
        Set the state of the micro simulation from a checkpoint.

        Parameters
        ----------
        state : object
            The state to restore.
        """
        pass

    @abstractmethod
    def get_global_id(self) -> int:
        """
        Return the global ID of this micro simulation instance.

        Returns
        -------
        global_id : int
            Global ID of the micro simulation.
        """
        pass

    @abstractmethod
    def set_global_id(self, global_id: int) -> None:
        """
        Set the global ID of this micro simulation instance.

        Parameters
        ----------
        global_id : int
            Global ID to assign.
        """
        pass

    def output(self) -> None:
        """
        Optional output method called after each solve step.
        Override this method if your simulation needs to write output at each step.
        """
        pass

    def requires_initialize(self) -> bool:
        """
        Return True if this simulation class overrides the ``initialize`` method.
        The Micro Manager calls this to determine whether initialization is needed.

        Returns
        -------
        requires_initialize : bool
            True if ``initialize`` is overridden, False otherwise.
        """
        return type(self).initialize is not MicroSimulationInterface.initialize

    def requires_output(self) -> bool:
        """
        Return True if this simulation class overrides the ``output`` method.
        The Micro Manager calls this to determine whether output is needed.

        Returns
        -------
        requires_output : bool
            True if ``output`` is overridden, False otherwise.
        """
        return type(self).output is not MicroSimulationInterface.output


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

    def requires_initialize(self) -> bool:
        return self._instance.requires_initialize()

    def requires_output(self) -> bool:
        return self._instance.requires_output()


class MicroSimulationRemote(MicroSimulationInterface):
    def __init__(self, gid, late_init, num_ranks, conn, cls_path, sim_cls):
        self._cls_path = cls_path
        self._gid = gid
        self._num_ranks = num_ranks
        self._conn = conn
        self._sim_cls = sim_cls

        construct_cls = ConstructLateTask if late_init else ConstructTask
        for worker_id in range(self._num_ranks):
            task = construct_cls.send_args(self._gid, self._cls_path)
            self._conn.send(worker_id, task)

        for worker_id in range(self._num_ranks):
            self._conn.recv(worker_id)

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

    def requires_initialize(self) -> bool:
        return self._sim_cls.initialize is not MicroSimulationInterface.initialize

    def requires_output(self) -> bool:
        return self._sim_cls.output is not MicroSimulationInterface.output


class MicroSimulationWrapper(MicroSimulationInterface):
    """
    If only a single rank is in use: will contain the micro sim instance.
    Otherwise, it will delegate method calls to workers and not contain state.
    """

    def __init__(self, name, sim_cls, cls_path, global_id, late_init, num_ranks, conn):
        self._impl = None

        if num_ranks > 1 and conn is not None:
            self._impl = MicroSimulationRemote(
                global_id, late_init, num_ranks, conn, cls_path, sim_cls
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

    def requires_initialize(self) -> bool:
        return self._impl.requires_initialize()

    def requires_output(self) -> bool:
        return self._impl.requires_output()

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

    def check_initialize(
        self, test_instance: MicroSimulationInterface, test_input: dict
    ) -> tuple[bool, bool]:
        """
        Check whether the micro simulation class implements ``initialize``.

        Since ``load_backend_class`` guarantees that ``self._sim_cls`` always
        inherits from ``MicroSimulationInterface``, we can rely on
        ``requires_initialize()`` directly. No ``issubclass`` guard is needed.

        Parameters
        ----------
        test_instance : object
            An instance of the micro simulation class used for signature probing.
        test_input : dict
            Sample input data used to probe whether ``initialize`` accepts arguments.

        Returns
        -------
        check_result : tuple[bool, bool]
            (has_initialize, requires_initial_data)
        """
        if not test_instance.requires_initialize():
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

        return True, has_args

    def check_output(self) -> bool:
        """
        Check whether the micro simulation class implements ``output``.

        Since ``load_backend_class`` guarantees that ``self._sim_cls`` always
        inherits from ``MicroSimulationInterface``, we can rely on
        ``requires_output()`` directly at the class level.

        Returns
        -------
        check_result : bool
            True if the micro simulation class overrides the ``output`` method.
        """
        return self._sim_cls.output is not MicroSimulationInterface.output


def _wrap_non_interface_class(cls: type, path_to_micro_file: str) -> type:
    """
    Dynamically create a class that inherits from MicroSimulationInterface
    and delegates all method calls to the provided class.

    This ensures that load_backend_class always returns a class that adheres
    to MicroSimulationInterface, even for pybind11 classes or legacy classes
    that do not explicitly inherit from it.

    Parameters
    ----------
    cls : type
        The original micro simulation class (e.g. loaded via pybind11).
    path_to_micro_file : str
        Path string used for the deprecation warning message.

    Returns
    -------
    type
        A new class inheriting from MicroSimulationInterface that wraps cls.
    """
    import warnings

    warnings.warn(
        "The MicroSimulation class in '{}' does not inherit from MicroSimulationInterface. "
        "Please update your class definition to: "
        "class MicroSimulation(MicroSimulationInterface). "
        "In a future version this will become an error.".format(path_to_micro_file),
        DeprecationWarning,
        stacklevel=3,
    )

    # Determine whether the original class provides initialize / output
    has_initialize = callable(getattr(cls, "initialize", None))
    has_output = callable(getattr(cls, "output", None))

    # Build the class body: __init__ and mandatory interface methods
    class_body = """
def __init__(self, global_id):
    self._wrapped = wrapped_cls(global_id)

def solve(self, micro_sim_input, dt):
    return self._wrapped.solve(micro_sim_input, dt)

def get_state(self):
    return self._wrapped.get_state()

def set_state(self, state):
    return self._wrapped.set_state(state)

def get_global_id(self):
    return self._wrapped.get_global_id()

def set_global_id(self, global_id):
    self._wrapped.set_global_id(global_id)

def __getattr__(self, name):
    return getattr(self._wrapped, name)
"""

    # Only add initialize override if the wrapped class actually has it,
    # so that requires_initialize() returns True for those classes.
    if has_initialize:
        class_body += """
def initialize(self, *args, **kwargs):
    return self._wrapped.initialize(*args, **kwargs)
"""

    # Only add output override if the wrapped class actually has it,
    # so that requires_output() returns True for those classes.
    if has_output:
        class_body += """
def output(self):
    return self._wrapped.output()
"""

    class_dict = {}
    exec(class_body, {"wrapped_cls": cls, "__builtins__": __builtins__}, class_dict)

    wrapper_cls = type(
        "CompatibilityWrapper_{}".format(cls.__name__),
        (MicroSimulationInterface,),
        class_dict,
    )
    return wrapper_cls


def load_backend_class(path_to_micro_file: str) -> type:
    """
    Load the MicroSimulation class from the given module path.

    Always returns a class that inherits from MicroSimulationInterface.
    If the loaded class does not inherit from it (e.g. pybind11 classes or
    legacy classes), it is wrapped in a dynamically created adapter class
    that delegates all calls to the original and correctly implements
    requires_initialize() and requires_output().

    Parameters
    ----------
    path_to_micro_file : str
        Dotted module path to the micro simulation file.

    Returns
    -------
    type
        A class inheriting from MicroSimulationInterface.
    """
    CLS_NAME = "MicroSimulation"
    cls = getattr(ipl.import_module(path_to_micro_file, CLS_NAME), CLS_NAME)

    try:
        inherits = issubclass(cls, MicroSimulationInterface)
    except TypeError:
        # pybind11 extension types may not support issubclass — wrap them
        inherits = False

    if not inherits:
        cls = _wrap_non_interface_class(cls, path_to_micro_file)

    return cls


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
