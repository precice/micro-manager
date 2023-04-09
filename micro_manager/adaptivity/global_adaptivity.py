"""
Functionality for adaptive initialization and control of micro simulations
"""
import numpy as np
import sys
from copy import deepcopy
from adaptivity import AdaptivityCalculator


class GlobalAdaptivityCalculator(AdaptivityCalculator):
    