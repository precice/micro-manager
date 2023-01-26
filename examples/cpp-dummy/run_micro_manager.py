"""
Script to run the Micro Manager
"""
import os
print(os.listdir())

from micro_manager import MicroManager

manager = MicroManager("./micro-manager-config.json")

manager.initialize()

manager.solve()
