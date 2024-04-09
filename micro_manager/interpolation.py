import numpy as np
from scipy.interpolate import griddata as gd


def interpolate(logger, known_coords: list, inter_coord: list , data: list)-> dict:
    """
    Interpolate parameters at a given vertex

    Args:
        logger : logger object
            Logger of the micro manager
        known_coords : list
            List of vertices where the data is known
        inter_coord : list
            Vertex where the data is to be interpolated
        data : list
            List of dicts in which keys are names of data and the values are the data.

    Returns:
        result: dict
            Interpolated data at inter_coord
    """
    result = dict()
    assert len(known_coords) == len(data), "Number of known coordinates and data points do not match"

    for params in data[0].keys():
        # Attempt to interpolate the data
        try:
            result[params] = gd(known_coords, [d[params] for d in data], inter_coord, method='linear').tolist()
            # Extrapolation is replaced by taking a nearest neighbor
            if np.isnan(result[params]).any():
                nearest_neighbor_pos = np.argmin(np.linalg.norm(np.array(known_coords) - np.array(inter_coord), axis=1))
                nearest_neighbor = known_coords[nearest_neighbor_pos]
                logger.info("Interpolation failed at macro vertex {}. Taking value of closest neighbor at macro vertex {}".format(params, inter_coord, nearest_neighbor))
                return data[nearest_neighbor_pos]
            return result
        # If interpolation fails, take the value of the nearest neighbor
        except Exception:
                nearest_neighbor_pos = np.argmin(np.linalg.norm(np.array(known_coords) - np.array(inter_coord), axis=1))
                nearest_neighbor = known_coords[nearest_neighbor_pos]
                logger.info("Interpolation failed at macro vertex {}. Taking value of closest neighbor at macro vertex {}".format(params, inter_coord, nearest_neighbor))
                return data[nearest_neighbor_pos]
 