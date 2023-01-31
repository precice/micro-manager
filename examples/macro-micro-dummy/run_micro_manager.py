"""
Script to run the Micro Manager
"""

from micro_manager import MicroManager
import os

manager = MicroManager(os.path.join(os.path.dirname(__file__), "micro-manager-config.json"))

manager.initialize()

manager.solve()
