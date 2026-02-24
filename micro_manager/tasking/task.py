class Task:
    """
    This is the general task interface.
    Each task is callable and will be provided with a global state object (state_data).

    Inheriting classes may define functions to be executed that require more args than just the state object.
    These will be passed on as args and kwargs. Args and kwargs are bound to the task object upon construction.
    Thus, each call to a task object only requires the state object.

    Inheriting classes need to call super.__init__ with the function to be called and the args and kwargs to be bound.
    """

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, state_data: dict):
        return self.fn(*self.args, state_data=state_data, **self.kwargs)

    @classmethod
    def send_args(cls, *args, **kwargs):
        """
        Used to get a representation of the task without the need to pickle the task.
        """
        return cls.__name__, args, kwargs


class ConstructTask(Task):
    """
    Construction Task: Given a gid and micro simulation class path, it will construct an instance and store it
    in the state object under ['sim_instances'][gid]. If the desired class has not yet been loaded, then this
    will be done by the 'load_function' prior to construction.
    """

    def __init__(self, gid, cls_path):
        super().__init__(ConstructTask.initializer, gid=gid, cls_path=cls_path)

    @staticmethod
    def initializer(gid, cls_path, state_data):
        if cls_path not in state_data["sim_classes"]:
            import os
            import sys
            from pathlib import Path

            ms_dir = os.path.abspath(str(Path(cls_path).resolve().parent))
            sys.path.append(ms_dir)
            _, file_name = os.path.split(os.path.abspath(str(Path(cls_path).resolve())))

            state_data["sim_classes"][cls_path] = state_data["load_function"](file_name)
        cls = state_data["sim_classes"][cls_path]

        if gid in state_data["sim_classes"]:
            del state_data["sim_classes"][gid]
        state_data["sim_instances"][gid] = cls(gid)
        return None


class ConstructLateTask(Task):
    """
    Similar to ConstructTask, it will construct an instance and store it. However, it will pass -1 as the gid to the
    instance to allow for late initialization, if the micro simulation supports it.
    """

    def __init__(self, gid, cls_path):
        super().__init__(ConstructLateTask.initializer, gid=gid, cls_path=cls_path)

    @staticmethod
    def initializer(gid, cls_path, state_data):
        if cls_path not in state_data["sim_classes"]:
            import os
            import sys
            from pathlib import Path

            ms_dir = os.path.abspath(str(Path(cls_path).resolve().parent))
            sys.path.append(ms_dir)
            _, file_name = os.path.split(os.path.abspath(str(Path(cls_path).resolve())))

            state_data["sim_classes"][cls_path] = state_data["load_function"](file_name)
        cls = state_data["sim_classes"][cls_path]

        if gid in state_data["sim_classes"]:
            del state_data["sim_classes"][gid]
        state_data["sim_instances"][gid] = cls(-1)
        return None


class SolveTask(Task):
    """
    Given a gid, input data and the current dt, the SolveTask will call the solve function of its respective
    simulation object, that is stored in the state object under ['sim_instances'][gid].
    """

    def __init__(self, gid, sim_input, dt):
        super().__init__(SolveTask.solve, gid=gid, sim_input=sim_input, dt=dt)

    @staticmethod
    def solve(gid, sim_input, dt, state_data):
        sim_output = state_data["sim_instances"][gid].solve(sim_input, dt)
        return sim_output


class GetStateTask(Task):
    """
    Given a gid, the GetStateTask will call the get_state function of its respective
    simulation object, that is stored in the state object under ['sim_instances'][gid].
    """

    def __init__(self, gid):
        super().__init__(GetStateTask.get, gid=gid)

    @staticmethod
    def get(gid, state_data):
        return state_data["sim_instances"][gid].get_state()


class SetStateTask(Task):
    """
    Given a gid and a state the SetStateTask will call the set_state function of its respective
    simulation object, that is stored in the state object under ['sim_instances'][gid].
    """

    def __init__(self, gid, state):
        super().__init__(SetStateTask.set, gid=gid, state=state)

    @staticmethod
    def set(gid, state, state_data):
        state_data["sim_instances"][gid].set_state(state)

        # if gid was changed, we want to move it to the right location
        check_gid = state_data["sim_instances"][gid].get_global_id()
        if check_gid != gid:
            state_data["sim_instances"][check_gid] = state_data["sim_instances"][gid]
            del state_data["sim_instances"][gid]
            return check_gid

        return gid


class InitializeTask(Task):
    """
    Given a gid and arbitrary arguments the InitializeTask will call the initialize function of its respective
    simulation object, that is stored in the state object under ['sim_instances'][gid].
    All arguments will be passed along.
    """

    def __init__(self, gid, *args, **kwargs):
        super().__init__(InitializeTask.initialize, *args, gid=gid, **kwargs)

    @staticmethod
    def initialize(gid, state_data, *args, **kwargs):
        return state_data["sim_instances"][gid].initialize(*args, **kwargs)


class OutputTask(Task):
    """
    Given a gid, the OutputTask will call the output function of its respective
    simulation object, that is stored in the state object under ['sim_instances'][gid].
    """

    def __init__(self, gid):
        super().__init__(OutputTask.output, gid=gid)

    @staticmethod
    def output(gid, state_data):
        return state_data["sim_instances"][gid].output()


class ShutdownTask(Task):
    """
    The ShutdownTask will raise an exception in order to exit out of the work loop with in the worker_main.
    """

    def __init__(self):
        super().__init__(ShutdownTask.shutdown)

    @staticmethod
    def shutdown(state_data):
        raise RuntimeError("Stopping Worker")


class RegisterAllTask(Task):
    """
    Sets up the local worker state and registers all potentially used tasks on the workers side.
    By doing so, less pickling needs to be done during operation. Only the task name and data need to be transferred.
    Workers can then locally re-construct the task based on the registered tasks.

    Each worker has a state object (state_data). It is provided to each task when it is called.
    """

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
