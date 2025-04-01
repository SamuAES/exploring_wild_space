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


def sliding_average(x: np.array, window_size: int = 15):
    """
    Returns the sliding average for an array.
    
    Parameters
    ----------
    x : np.array
        Array with shape (n,) containing input data.
    window_size : int
        Sliding average window size. Must be odd.
    
    Returns
    -------
    res : np.array
        Array with shape (n,) containing the sliding averages.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    half_window = window_size // 2
    # Pad the array using reflection at the edges
    padded_x = np.pad(x, pad_width = half_window, mode = 'reflect')
    # Compute the moving average using a sliding window
    res = np.convolve(padded_x, np.ones(window_size) / window_size, mode='valid')
    return res


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


def count_threshold_crossings(x: np.ndarray, threshold: float):
    """
    Returns the number of times the values in 'x' go above 'threshold'.
    
    Parameters
    ----------
    x : np.ndarray
        Array of values.
    threshold : float
        Threshold value.

    Returns
    -------
    count : int
        Number of crossings.
    """
    above_threshold = x > threshold 
    crossings = np.diff(above_threshold.astype(int))
    return np.count_nonzero(crossings == 1)


def count_threshold_crossings_sidewise(v, x, wall_position, threshold):
    """
    Returns the number of times the values in 'x' go above 'threshold'
    in both sides of the cage.
    
    Parameters
    ----------
    v : np.ndarray
        Array of speed values.
    x : np.ndarray
        Array of position (x-coordinate) values.
    wall_position : float
        Position (x-coordinate) of the wall.
    threshold : float
        Threshold value.

    Returns
    -------
    counts : tuple
        Number of crossings on home side and exploration side.
    """
    n = v.shape[0]
    above_threshold_home = np.empty(n)
    above_threshold_expl = np.empty(n)
    for ind, v_val in enumerate(v):
        if v_val > threshold:
            if x[ind] > wall_position:
                above_threshold_home[ind] = 1
                above_threshold_expl[ind] = 0
            else:
                above_threshold_home[ind] = 0
                above_threshold_expl[ind] = 1
        else:
            above_threshold_home[ind] = 0
            above_threshold_expl[ind] = 0
    crossings_home = np.diff(above_threshold_home.astype(int))
    crossings_expl = np.diff(above_threshold_expl.astype(int))
    return np.count_nonzero(crossings_home == 1), np.count_nonzero(crossings_expl == 1)


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