from executorlib import SingleNodeExecutor, SlurmJobExecutor

def generate_executor(config):
    type = config.get_tasks_type()
    ranks = config.get_tasks_max_ranks()
    cache_dir = config.get_tasks_cache_dir()

    if type == "single":
        return SingleNodeExecutor(max_cores=ranks, cache_directory=cache_dir)
    elif type == "hpc":
        return SlurmJobExecutor(max_cores=ranks, cache_directory=cache_dir)
    else:
        raise ValueError("Invalid task type")
