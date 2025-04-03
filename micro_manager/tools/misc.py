"""
A collection of miscellaneous functions that are used in various parts of the codebase.
"""


def divide_in_parts(number, parts):
    if parts <= 0:
        raise ValueError("Number of parts must be greater than zero")

    quotient, remainder = divmod(number, parts)
    result = [quotient + 1] * remainder + [quotient] * (parts - remainder)
    return result
