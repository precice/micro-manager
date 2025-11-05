import numpy as np


def switching_function(resolutions, locations, t, prev_output, active):
    output = np.zeros_like(resolutions)
    mask_loc = locations[:, 0] % 2 == 0
    mask_res = resolutions == 0
    output[mask_loc * mask_res] = 1
    return output
