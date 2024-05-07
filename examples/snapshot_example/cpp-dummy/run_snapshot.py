"""
Script to run the Micro Manager
"""

from argparse import ArgumentParser

from snapshot import SnapshotComputation

parser = ArgumentParser()
parser.add_argument(
    "--config", required=True, help="Path to the micro manager configuration file"
)
args = parser.parse_args()

snapshot_object = SnapshotComputation(args.config)

snapshot_object.solve()
