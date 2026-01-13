class ModelWrapper:
    """
    Stateless Model Wrapper
    """

    def __init__(self, global_id, backend, attach_init, attach_output):
        self._global_id = global_id
        self._backend = backend

        if attach_init:
            self.initialize = backend.initialize
        if attach_output:
            self.output = backend.output

    def get_global_id(self) -> int:
        return self._global_id

    def solve(self, macro_data, dt):
        return self._backend.solve(macro_data, dt)

    def get_state(self):
        return self._backend.get_state()

    def set_state(self, state):
        self._backend.set_state(state)

    @property
    def __class__(self):
        return self._backend.__class__


class ModelManager:
    def __init__(self):
        self._registered_classes = []
        self._stateless_map = dict()
        self._backend_map = dict()
        self._has_init_map = dict()
        self._has_output_map = dict()

    def register(self, micro_sim_cls, stateless):
        if micro_sim_cls in self._registered_classes:
            return

        self._registered_classes.append(micro_sim_cls)
        self._stateless_map[micro_sim_cls] = stateless

        if stateless:
            self._backend_map[micro_sim_cls] = micro_sim_cls(-1)

        self._has_init_map[micro_sim_cls] = False
        if hasattr(micro_sim_cls, "initialize") and callable(
            getattr(micro_sim_cls, "initialize")
        ):
            self._has_init_map[micro_sim_cls] = True

        self._has_output_map[micro_sim_cls] = False
        if hasattr(micro_sim_cls, "output") and callable(
            getattr(micro_sim_cls, "output")
        ):
            self._has_output_map[micro_sim_cls] = True

    def get_instance(self, gid, micro_sim_cls, *, late_init=False):
        if micro_sim_cls not in self._registered_classes:
            raise RuntimeError("Trying to create instance of unknown class!")

        if self._stateless_map[micro_sim_cls]:
            return ModelWrapper(
                gid,
                self._backend_map[micro_sim_cls],
                self._has_init_map[micro_sim_cls],
                self._has_output_map[micro_sim_cls],
            )
        else:
            return micro_sim_cls(gid, late_init=late_init)
