import argparse
import os

from .config import Config
from .micro_manager import MicroManagerCoupling

try:
    from .snapshot.snapshot import MicroManagerSnapshot

    is_snapshot_possible = True
except ImportError:
    is_snapshot_possible = False


def main():

    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "config_file", type=str, help="Path to the JSON config file of the manager."
    )
    parser.add_argument(
        "--snapshot", action="store_true", help="compute offline snapshot database"
    )

    args = parser.parse_args()
    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    if not args.snapshot:
        manager = MicroManagerCoupling(config_file_path)
    else:
        if not is_snapshot_possible:
            raise ImportError(
                "The Micro Manager snapshot computation requires the h5py package."
            )
        else:
            manager = MicroManagerSnapshot(config_file_path)

    manager.initialize()

    manager.solve()
