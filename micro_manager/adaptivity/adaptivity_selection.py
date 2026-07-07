from micro_manager.config import Config
from micro_manager.simulation_container import SimulationContainer
from micro_manager.tools.profiling import Profiler
from micro_manager.tools.logging_wrapper import Logger
from micro_manager.tools.mpi_handler import MPIHandler
from micro_manager.micro_simulation import MicroSimulationClass
from micro_manager.model_manager import ModelManager
from micro_manager.adaptivity.adaptivity_interface import AdaptivityInterface
from micro_manager.adaptivity.adaptivity import NoOpAdaptivity
from micro_manager.adaptivity.local_adaptivity import LocalAdaptivityCalculator
from micro_manager.adaptivity.global_adaptivity import GlobalAdaptivityCalculator


def create_adaptivity_calculator(
    config: Config,
    sim_container: SimulationContainer,
    profiler: Profiler,
    logger: Logger,
    mpi: MPIHandler,
    micro_problem_cls: MicroSimulationClass,
    model_manager: ModelManager,
) -> AdaptivityInterface:
    if not config.enable_adaptivity():
        return NoOpAdaptivity(sim_container)

    adaptivity_type = config.adaptivity_type()
    args = (
        config,
        sim_container,
        profiler,
        logger,
        mpi,
        micro_problem_cls,
        model_manager,
    )
    match adaptivity_type:
        case "local":
            return LocalAdaptivityCalculator(*args)
        case "global":
            return GlobalAdaptivityCalculator(*args)

    raise ValueError("Unknown adaptivity type")
