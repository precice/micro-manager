from micro_manager.micro_simulation import (
    MicroSimulationClass,
    MicroSimulationWrapper,
    MicroSimulationInterface,
)


class ModelWrapper(MicroSimulationInterface):
    """
    Stateless Model Wrapper, will delegate any method call to the main compute instance.
    This is used to replace instances in the main simulation container.
    """

    def __init__(self, global_id, backend):
        self._global_id = global_id
        self._backend = backend

    def set_global_id(self, global_id):
        self._global_id = global_id

    def get_global_id(self) -> int:
        return self._global_id

    def solve(self, macro_data, dt):
        return self._backend.solve(macro_data, dt)

    def get_state(self):
        return self._backend.get_state()

    def set_state(self, state):
        self._backend.set_state(state)

    def initialize(self, *args, **kwargs):
        return self._backend.initialize(*args, **kwargs)

    def output(self):
        return self._backend.output()

    @property
    def __class__(self):
        return self._backend.__class__

    @property
    def attachments(self):
        return self._backend.attachments

    @attachments.setter
    def attachments(self, value):
        self._backend.attachments = value

    @property
    def name(self):
        return self._backend.name


class ModelManager:
    """
    Manages all used micro simulation models. Stores their classes and checks whether they may
    use model instancing. To generate instances use the get_instance method regardless of model instancing,
    as the ModelManager handles either case.
    """

    def __init__(self):
        self._registered_classes: list[MicroSimulationClass] = []
        self._stateless_map: dict[MicroSimulationClass, bool] = dict()
        self._backend_map: dict[MicroSimulationClass, MicroSimulationWrapper] = dict()

    def register(self, micro_sim_cls: MicroSimulationClass, stateless: bool):
        """
        Register a micro simulation class to create an instance of it later.

        Parameters
        ----------
        micro_sim_cls : MicroSimulationClass
            Micro simulation class to register.
        stateless: bool
            Is the simulation class stateless.
        """
        if micro_sim_cls in self._registered_classes:
            return

        self._registered_classes.append(micro_sim_cls)
        self._stateless_map[micro_sim_cls] = stateless

        if stateless:
            self._backend_map[micro_sim_cls] = micro_sim_cls(
                len(self._registered_classes) - 1
            )

    def get_instance(
        self, gid: int, micro_sim_cls: MicroSimulationClass, *, late_init: bool = False
    ) -> MicroSimulationInterface:
        """
        Creates an instance of the requested class. If the class should be initialized later,
        the request will be delegated to the micro simulation object (in case it supports it).

        Parameters
        ----------
        gid: int
            Global Simulation ID
        micro_sim_cls: MicroSimulationClass
            Requested micro simulation class
        late_init: bool
            Should the simulation be initialized later?

        Returns
        -------
        micro_sim : MicroSimulationInterface
            Instance of the requested micro simulation class, either delegator or compute instance
        """
        if micro_sim_cls not in self._registered_classes:
            raise RuntimeError("Trying to create instance of unknown class!")

        if self._stateless_map[micro_sim_cls]:
            return ModelWrapper(
                gid,
                self._backend_map[micro_sim_cls],
            )
        else:
            return micro_sim_cls(gid, late_init=late_init)
