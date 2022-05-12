"""
Script to run the Micro Manager
"""

from micromanager import MicroManager

manager = MicroManager("micro-manager-config.json")

manager.run()
