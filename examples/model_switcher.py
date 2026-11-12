import numpy as np


def switching_function(resolution, location, t, input, prev_output):
    # conditions:
    # 1. use resolution 0 after t==2
    # 2. try resolution 1 or 2 in domain outside ([0,0,0], [10,10,10])

    # res 0
    # valid input range: all
    # valid output range: all

    # res 1
    # valid input range: non-negative values
    # scalar output valid within [0, 10]

    # res 2
    # valid input range: [0, 10]
    # scalar output valid within [0, 5]
    output = 0

    # check inputs
    res_1_valid = input["macro-scalar-data"] >= 0
    res_2_valid = 0 <= input["macro-scalar-data"] <= 10
    # check outputs
    if prev_output is not None:
        res_1_valid &= 0 <= prev_output["micro-scalar-data"] <= 10
        res_2_valid &= 0 <= prev_output["micro-scalar-data"] <= 5

    if resolution == 0:
        if t > 2:
            output = 0
        elif res_1_valid and np.any(location > np.array([10, 10, 10])):
            output = 1

    elif resolution == 1:
        if t > 2 or not res_1_valid:
            output = -1
        elif res_2_valid:
            output = 1
        else:
            output = 0

    elif resolution == 2:
        if t > 2 or not res_2_valid:
            output = -1
        else:
            output = 0

    return output
