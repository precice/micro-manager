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


class ConstructTask(Task):
    def __init__(self, gid, cls_path):
        super().__init__(ConstructTask.initializer, gid=gid, cls_path=cls_path)

    @staticmethod
    def initializer(gid, cls_path, state_data):
        if cls_path not in state_data["sim_classes"]:
            state_data["sim_classes"][cls_path] = state_data["load_function"](cls_path)
        cls = state_data["sim_classes"][cls_path]

        if gid in state_data["sim_classes"]:
            del state_data["sim_classes"][gid]
        state_data["sim_instances"][gid] = cls(gid)
        return None


class ConstructLateTask(Task):
    def __init__(self, gid, cls_path):
        super().__init__(ConstructLateTask.initializer, gid=gid, cls_path=cls_path)

    @staticmethod
    def initializer(gid, cls_path, state_data):
        if cls_path not in state_data["sim_classes"]:
            state_data["sim_classes"][cls_path] = state_data["load_function"](cls_path)
        cls = state_data["sim_classes"][cls_path]

        if gid in state_data["sim_classes"]:
            del state_data["sim_classes"][gid]
        state_data["sim_instances"][gid] = cls(-1)
        return None


class SolveTask(Task):
    def __init__(self, gid, sim_input, dt):
        super().__init__(SolveTask.solve, gid=gid, sim_input=sim_input, dt=dt)

    @staticmethod
    def solve(gid, sim_input, dt, state_data):
        sim_output = state_data["sim_instances"][gid].solve(sim_input, dt)
        return sim_output


class GetStateTask(Task):
    def __init__(self, gid):
        super().__init__(GetStateTask.get, gid=gid)

    @staticmethod
    def get(gid, state_data):
        return state_data["sim_instances"][gid].get_state()


class SetStateTask(Task):
    def __init__(self, gid, state):
        super().__init__(SetStateTask.set, gid=gid, state=state)

    @staticmethod
    def set(gid, state, state_data):
        state_data["sim_instances"][gid].set_state(state)
        return None


class InitializeTask(Task):
    def __init__(self, gid, *args, **kwargs):
        super().__init__(InitializeTask.initialize, *args, gid=gid, **kwargs)

    @staticmethod
    def initialize(gid, state_data, *args, **kwargs):
        return state_data["sim_instances"][gid].initialize(*args, **kwargs)


class OutputTask(Task):
    def __init__(self, gid):
        super().__init__(OutputTask.output, gid=gid)

    @staticmethod
    def output(gid, state_data):
        return state_data["sim_instances"][gid].output()

class ShutdownTask(Task):
    def __init__(self):
        super().__init__(ShutdownTask.shutdown)

    @staticmethod
    def shutdown(state_data):
        raise RuntimeError("Stopping Worker")


class RegisterAllTask(Task):
    def __init__(self, load_function):
        super().__init__(RegisterAllTask.register, load_function=load_function)

    @staticmethod
    def register(state_data, load_function):
        task_dict = dict()
        task_dict[ConstructTask.__name__] = ConstructTask
        task_dict[ConstructLateTask.__name__] = ConstructLateTask
        task_dict[SolveTask.__name__] = SolveTask
        task_dict[GetStateTask.__name__] = GetStateTask
        task_dict[SetStateTask.__name__] = SetStateTask
        task_dict[InitializeTask.__name__] = InitializeTask
        task_dict[OutputTask.__name__] = OutputTask
        task_dict[ShutdownTask.__name__] = ShutdownTask
        state_data["tasks"] = task_dict
        state_data["sim_classes"] = dict()
        state_data["sim_instances"] = dict()
        state_data["load_function"] = load_function
        return None


def handle_task(state_data, task_descriptor):
    name, args, kwargs = task_descriptor
    task = state_data["tasks"][name](*args, **kwargs)
    # print(f"handling task: {name} args={args} kwargs={kwargs}")
    return task(state_data)
