"""
A collection of miscellaneous functions that are used in various parts of the codebase.
"""

from typing import Union, Optional
import numpy as np


def divide_in_parts(number, parts):
    if parts <= 0:
        raise ValueError("Number of parts must be greater than zero")

    quotient, remainder = divmod(number, parts)
    result = [quotient + 1] * remainder + [quotient] * (parts - remainder)
    return result


def clamp_in_range(
    value: Union[int, np.array],
    min_value: Union[int, np.array],
    max_value: Union[int, np.array],
) -> Union[int, np.array]:
    """
    Clamps value between min_value and max_value inclusively.

    Parameters
    ----------
    value : [int, np.array]
        clamping target
    min_value : [int, np.array]
        minimum values
    max_value : [int, np.array]
        maximum values

    Returns
    -------
    value : [int, np.array]
        clamped results
    """
    return np.maximum(min_value, np.minimum(value, max_value))
