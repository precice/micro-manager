
class Task:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, state_data: dict):
        return self.fn(*self.args, state_data=state_data, **self.kwargs)

class ConstructTask(Task):
    def __init__(self, gid, sim_cls):
        super().__init__(ConstructTask.initializer, gid=gid, sim_cls=sim_cls)

    @staticmethod
    def initializer(gid, sim_cls, state_data):
        state_data[gid] = sim_cls(gid)
        return None

class SolveTask(Task):
    def __init__(self, gid, sim_input, dt):
        super().__init__(SolveTask.solve, gid=gid, sim_input=sim_input, dt=dt)

    @staticmethod
    def solve(gid, sim_input, dt, state_data):
        sim_output = state_data[gid].solve(sim_input, dt)
        return sim_output

class GetStateTask(Task):
    def __init__(self, gid):
        super().__init__(GetStateTask.get, gid=gid)

    @staticmethod
    def get(gid, state_data):
        return state_data[gid].get_state()

class SetStateTask(Task):
    def __init__(self, gid, state):
        super().__init__(SetStateTask.set, gid=gid, state=state)

    @staticmethod
    def set(gid, state, state_data):
        state_data[gid].set_state(state)
        return None

class InitializeTask(Task):
    def __init__(self, gid, *args, **kwargs):
        super().__init__(InitializeTask.initialize, *args, gid=gid, **kwargs)

    @staticmethod
    def initialize(gid, state_data, *args, **kwargs):
        return state_data[gid].initialize(*args, **kwargs)

class OutputTask(Task):
    def __init__(self, gid):
        super().__init__(OutputTask.output, gid=gid)

    @staticmethod
    def output(gid, state_data):
        return state_data[gid].output()