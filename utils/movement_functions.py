from scipy.stats import mode
from .frames import *

import numpy as np


def extract_bird_and_wall_coordinates(json_filepath, frame_count):
    """
    Extracts coordinates of the bird and wall from a json file. If multiple
    bird or wall instances are detected in a single frame, selects the most
    confident one. If no bird or wall is detected in a frame, returns NaN.

    Parameters
    ----------
    json_filepath : str
        Path to json file.
    frame_count : int
        Number of frames to analyse.

    Returns
    -------
    bird_x, bird_y, wall_x, wall_y : tuple
        Tuple of arrays containing bird and wall position data.
    """
    # Load json file as dict
    dict_full = load_json_to_dict(json_filepath)
    # Initialize arrays
    bird_x = np.zeros(frame_count)
    bird_y = np.zeros(frame_count)
    wall_x = np.zeros(frame_count)
    wall_y = np.zeros(frame_count)
    # Loop over frames
    for index in range(frame_count):
        # Pick dict corresponding to current frame
        key = f"frame{index + 1}"
        dict_tmp = dict_full[key]
        # Select bird and wall instances from the full dictionary
        bird_list = []
        wall_list = []
        for key, value in dict_tmp.items():
            if value.get("class") == "bird":
                bird_list.append(value)
            elif value.get("class") == "wall":
                wall_list.append(value)
        # Loop over all bird detections and select the most confident one
        if len(bird_list) > 0:
            prev_conf = 0
            for bird in bird_list:
                curr_conf = bird["confidence"]
                if curr_conf > prev_conf:
                    bird_x[index] = int((bird["x1"] + bird["x2"]) / 2)
                    bird_y[index] = int((bird["y1"] + bird["y2"]) / 2)
                prev_conf = curr_conf
        # If no bird was detected, assign NaNs
        else:
            bird_x[index] = np.nan
            bird_y[index] = np.nan
        # Loop over all wall detections and select the most confident one
        if len(wall_list) > 0:
            prev_conf = 0
            for wall in wall_list:
                curr_conf = wall["confidence"]
                if curr_conf > prev_conf:
                    wall_x[index] = int((wall["x1"] + wall["x2"]) / 2)
                    wall_y[index] = int((wall["y1"] + wall["y2"]) / 2)
                prev_conf = curr_conf
        # If no wall was detected, assign NaNs
        else:
            wall_x[index] = np.nan
            wall_y[index] = np.nan
    return bird_x, bird_y, wall_x, wall_y


def sliding_mean(x: np.array, window_size: int = 15):
    """
    Returns the sliding average for an array.
    
    Parameters
    ----------
    x : np.array
        Array with shape (n,) containing input data.
    window_size : int
        Sliding mean window size. Must be odd.
    
    Returns
    -------
    means : np.array
        Array with shape (n,) containing the sliding averages.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    half_window = window_size // 2
    # Pad the array using reflection at the edges
    padded_x = np.pad(x, pad_width = half_window, mode = 'reflect')
    # Compute the moving average using a sliding window
    means = np.convolve(padded_x, np.ones(window_size) / window_size, mode = 'valid')
    return means


def sliding_mode(x: np.array, window_size: int = 31):
    """
    Returns the sliding mode for an array.
    
    Parameters
    ----------
    x : np.array
        Array with shape (n,) containing input data.
    window_size : int
        Sliding mode window size. Must be odd.
    
    Returns
    -------
    modes : np.array
        Array with shape (n,) containing the sliding modes.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    half_window = window_size // 2
    # Pad the array at the edges
    x_padded = np.pad(x, half_window, mode = 'edge')
    modes = []
    # Loop over array elements and pick modes
    for i in range(x.shape[0]):
        i_center = i + half_window
        window = x_padded[i_center - half_window : i_center + half_window]
        m = mode(window, keepdims = False).mode
        modes.append(m)
    return np.array(modes)


def determine_side(bird_x: np.array, wall_x: np.array):
    """
    Returns a list indicating the side of the cage that the bird is in
    (0 = homeside, 1 = exploration side).

    Parameters
    ----------
    bird_x : np.array
        Array containing the bird x-coordinates.
    wall_x : np.array
        Array containing the wall x-coordinates.

    Returns
    -------
    side : np.array
        Array which indicates the side.
    """
    assert bird_x.shape == wall_x.shape, "Input arrays must have the same shape."
    n = bird_x.shape[0]
    side = np.empty(n)
    for i in range(n):
        # Bird is in home side
        if bird_x[i] >= wall_x[i]:
            side[i] = 0
        # Bird is in exploration side
        else:
            side[i] = 1
    return side


def count_hops(action: np.array, side: np.array):
    """
    Counts the number of hops in home and exploration sides.

    Parameters
    ----------
    action : np.array
        Array containing bird actions.
    side : np.array
        Array indicationg which side of the cage the bird is on.

    Returns
    -------
    hops_home, hops_expl : tuple
        Tuple of ints. The number of hops in home and exploration sides.
    """
    count_home = 0
    count_expl = 0
    prev = action[0]
    for index, curr in enumerate(action[1:]):
        if curr != prev:
            if side[index - 1] == 0:
                count_home += 1
            else:
                count_expl += 1
            prev = curr
    return count_home, count_expl


def compute_distance(x: np.ndarray, y: np.ndarray):
    """
    Returns the total distance travelled.
    
    Parameters
    ----------
    x : np.ndarray
        x-coordinates.
    y : np.ndarray
        y-coordinates.

    Returns
    -------
    d : float
        Total distance travelled.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    return np.sum(d)


def compute_speed(x: np.ndarray, y: np.ndarray, dt: float):
    """
    Returns the speed given coordinate arrays.

    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates.
    y : np.ndarray
        Array of y-coordinates.
    dt : float
        Timestep size.

    Returns
    -------
    v : np.ndarray
        Array of speed values.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    return d / dt


def impute_data(x):
    """
    Uses linear imputation to fill in missing values.

    Parameters
    ----------
    x : np.ndarray
        Array of integer values.

    Returns
    -------
    x_copy : np.ndarray
        Imputed array of integer values.
    """
    x_copy = x.copy()
    nans = np.isnan(x_copy)
    f = lambda z: z.nonzero()[0]
    x_copy[nans] = np.interp(f(nans), f(~nans), x_copy[~nans])
    x_copy = np.array([int(x) for x in x_copy])
    return x_copy