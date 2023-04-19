"""
Script to run the Micro Manager
"""

from micro_manager import MicroManager
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", help="Path to the micro manager configuration file")
args = parser.parse_args()

manager = MicroManager(args.config)

manager.initialize()

manager.solve()
