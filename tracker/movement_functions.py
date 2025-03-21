import numpy as np

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
        Array with shape (n - (window_size/2) - 0.5,) containing the sliding averages.
        Edge elements are omitted for now.
    """
    assert window_size % 2 == 1, "Window size must be odd."
    return np.array([np.mean(x[i:i + window_size]) for i in range(len(x) - window_size + 1)])

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