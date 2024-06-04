import argparse
import os

from .config import Config
from .micro_manager import MicroManager


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "config_file", type=str, help="Path to the JSON config file of the manager."
    )

    args = parser.parse_args()
    config_file_path = args.config_file
    if not os.path.isabs(config_file_path):
        config_file_path = os.getcwd() + "/" + config_file_path

    manager = MicroManager(config_file_path)

    manager.initialize()

    manager.solve()
