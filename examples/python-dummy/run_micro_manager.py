"""
Script to run the Micro Manager
"""

from argparse import ArgumentParser

from micro_manager import MicroManager

parser = ArgumentParser()
parser.add_argument("--config", help="Path to the micro manager configuration file")
args = parser.parse_args()

manager = MicroManager(args.config)

manager.solve()
