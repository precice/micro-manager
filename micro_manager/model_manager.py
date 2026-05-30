from typing import Optional, Any

from .tools.logging_wrapper import Logger
from .config import Config
from .micro_simulation import (
    MicroSimulationClass,
    MicroSimulationInterface,
    load_backend_class,
    create_simulation_class,
)


class ModelWrapper(MicroSimulationInterface):
    """
    Stateless Model Wrapper, will delegate any method call to the main compute instance.
    This is used to replace instances in the main simulation container.
    """

    def __init__(self, global_id: int, backend: MicroSimulationInterface):
        self._global_id = global_id
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def set_global_id(self, global_id: int):
        self._global_id = global_id

    def get_global_id(self) -> int:
        return self._global_id

    def destroy(self):
        # we do not want to delete our compute instance
        pass

    @property
    def __class__(self):
        return self._backend.__class__


class ModelManager:
    """
    Manages all used micro simulation models. Stores their classes and checks whether they may
    use model instancing. To generate instances use the get_instance method regardless of model instancing,
    as the ModelManager handles either case.
    """

    def __init__(self, logger: Logger):
        self._registered_classes: list[MicroSimulationClass] = []
        self._stateless_map: dict[MicroSimulationClass, bool] = dict()
        self._backend_map: dict[MicroSimulationClass, MicroSimulationInterface] = dict()
        self._logger: Logger = logger

    def load_models(self, config: Config, n_workers: int, conn_workers: Optional[Any]):
        stateless_flags = config.micro_stateless_flags()
        for idx, model_file in enumerate(config.micro_file_names()):
            try:
                base_model = load_backend_class(model_file)
                sim_cls = create_simulation_class(
                    self._logger, base_model, model_file, n_workers, conn_workers
                )
                self.register(sim_cls, stateless_flags[idx])
            except Exception as e:
                self._logger.log_info_rank_zero(
                    f"Failed to load model class with error: {e}"
                )

        if (
            len(self._registered_classes) != len(stateless_flags)
            or len(self._registered_classes) == 0
        ):
            raise RuntimeError("Not all models were loaded. Stopping!")

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

    @property
    def num_models(self) -> int:
        return len(self._registered_classes)

    def get_cls_by_name(self, name: str) -> MicroSimulationClass:
        """
        Returns the class determined by its name

        Parameters
        ----------
        name: str
            name of registered class

        Returns
        -------
        sim_class: MicroSimulationClass
            Class given by name
        """
        cls_names = [cls.name for cls in self._registered_classes]
        idx = cls_names.index(name)
        return self._registered_classes[idx]

    def get_cls_by_idx(self, idx: int) -> MicroSimulationClass:
        """
        Returns the class determined by its index

        Parameters
        ----------
        idx: int
            index of registered class

        Returns
        -------
        sim_class: MicroSimulationClass
            Class given by index
        """
        return self._registered_classes[idx]

    def get_idx_of_sim(self, sim: MicroSimulationInterface) -> int:
        """
        Returns the model index of the provided simulation object

        Parameters
        ----------
        sim: MicroSimulationInterface
            simulation object to be checked

        Returns
        -------
        idx: int
            model index
        """
        return next(
            (
                idx
                for idx, cls in enumerate(self._registered_classes)
                if cls.name == sim.name
            )
        )

    def is_stateless(self, name: str) -> bool:
        """
        Returns whether the class given by its name is stateless.

        Parameters
        ----------
        name: str
            name of registered class

        Returns
        -------
        is_stateless: bool
            true if class is stateless
        """
        cls = self.get_cls_by_name(name)
        return self._stateless_map[cls]

    def get_instance_by_name(
        self, gid: int, name: str, *, late_init: bool = False
    ) -> MicroSimulationInterface:
        """
        Creates an instance of the requested class determined by its name. If the class should be initialized later,
        the request will be delegated to the micro simulation object (in case it supports it).

        Parameters
        ----------
        gid: int
            Global Simulation ID
        name: str
            Requested micro simulation class
        late_init: bool
            Should the simulation be initialized later?

        Returns
        -------
        micro_sim : MicroSimulationInterface
            Instance of the requested micro simulation class, either delegator or compute instance
        """
        return self.get_instance(gid, self.get_cls_by_name(name), late_init=late_init)

    def get_instance_by_idx(
        self, gid: int, idx: int, *, late_init: bool = False
    ) -> MicroSimulationInterface:
        """
        Creates an instance of the requested class determined by its index. If the class should be initialized later,
        the request will be delegated to the micro simulation object (in case it supports it).

        Parameters
        ----------
        gid: int
            Global Simulation ID
        idx: int
            Index of requested micro simulation class
        late_init: bool
            Should the simulation be initialized later?

        Returns
        -------
        micro_sim : MicroSimulationInterface
            Instance of the requested micro simulation class, either delegator or compute instance
        """
        return self.get_instance(
            gid, self._registered_classes[idx], late_init=late_init
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
