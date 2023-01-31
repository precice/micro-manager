"""
Script to run the Micro Manager
"""
import os
print(os.listdir())

from micro_manager import MicroManager

manager = MicroManager(os.path.join(os.path.dirname(__file__), "micro-manager-config.json"))

manager.initialize()

manager.solve()
