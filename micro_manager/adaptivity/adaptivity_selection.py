from .global_adaptivity import GlobalAdaptivityCalculator
from .local_adaptivity import LocalAdaptivityCalculator
from .adaptivity import AdaptivityCalculator


def create_adaptivity_calculator(
    config,
    sim_container,
    profiler,
    logger,
    mpi,
    micro_problem_cls,
    model_manager,
) -> AdaptivityCalculator:
    adaptivity_type = config.adaptivity_type()

    if adaptivity_type == "local":
        return LocalAdaptivityCalculator(
            config,
            sim_container,
            logger,
            mpi,
            micro_problem_cls,
            model_manager,
        )

    if adaptivity_type == "global":
        return GlobalAdaptivityCalculator(
            config,
            sim_container,
            profiler,
            logger,
            mpi,
            micro_problem_cls,
            model_manager,
        )

    raise ValueError("Unknown adaptivity type")
