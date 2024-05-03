"""
Script to run Snapshot computation
"""

from argparse import ArgumentParser

from micro_manager import SnapshotComputation

parser = ArgumentParser()
parser.add_argument(
    "--config", help="Path to the snapshot computation configuration file"
)
args = parser.parse_args()

snapshot = SnapshotComputation(args.config)

snapshot.solve()
