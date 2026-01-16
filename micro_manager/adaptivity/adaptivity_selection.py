from .global_adaptivity import GlobalAdaptivityCalculator
from .global_adaptivity_lb import GlobalAdaptivityLBCalculator
from .local_adaptivity import LocalAdaptivityCalculator
from .adaptivity import AdaptivityCalculator


def create_adaptivity_calculator(
    config,
    local_number_of_sims,
    global_number_of_sims,
    global_ids_of_local_sims,
    participant,
    logger,
    rank,
    comm,
    micro_problem_cls,
    model_manager,
    use_lb,
) -> AdaptivityCalculator:
    adaptivity_type = config.get_adaptivity_type()

    if adaptivity_type == "local":
        return LocalAdaptivityCalculator(
            config,
            local_number_of_sims,
            logger,
            rank,
            comm,
            micro_problem_cls,
            model_manager,
        )

    if adaptivity_type == "global":
        cls = GlobalAdaptivityCalculator
        if use_lb:
            cls = GlobalAdaptivityLBCalculator

        return cls(
            config,
            global_number_of_sims,
            global_ids_of_local_sims,
            participant,
            logger,
            rank,
            comm,
            micro_problem_cls,
            model_manager,
        )

    raise ValueError("Unknown adaptivity type")
