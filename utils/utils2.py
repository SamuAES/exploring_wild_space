import numpy as np
from scipy.stats import mode

def get_best_detection(frame_data, class_name):
    """
    Finds the detection with the highest confidence for a given class name in a frame.

    Parameters
    ----------
    frame_data : dict
        Dictionary containing detection data for a single frame.
        Keys are detection IDs, values are dictionaries with 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'.
    class_name : str
        The name of the class to search for (e.g., 'bird', 'wall', 'stick', 'fence').

    Returns
    -------
    dict or None
        A dictionary containing the bounding box coordinates ('x1', 'y1', 'x2', 'y2')
        and 'confidence' of the highest confidence detection for the specified class.
        Returns None if no detections of that class are found.
    """
    detections = []
    for key, value in frame_data.items():
        if value.get("class") == class_name:
            detections.append(value)

    if not detections:
        return None

    # Sort by confidence in descending order
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    # Return the highest confidence detection
    return detections[0]


def identify_and_number_exploration_perches(frame_data, wall_x, max_perches=5):
    """
    Identifies sticks on the exploration side (left of the wall), selects the most confident ones,
    and numbers them from left to right.

    Parameters
    ----------
    frame_data : dict
        Dictionary containing detection data for a single frame.
    wall_x : float or None
        The x-coordinate of the center of the wall. If None, cannot determine exploration side.
    max_perches : int, optional
        The maximum number of perches expected on the exploration side, by default 5.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a numbered exploration perch
        and contains 'number', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y'.
        Returns an empty list if wall_x is None or no valid perches are found.
    """
    if wall_x is None:
        return []

    stick_detections = []
    for key, value in frame_data.items():
        if value.get("class") == "stick":
            center_x = (value['x1'] + value['x2']) / 2
            # Check if the stick is on the exploration side (left of the wall)
            if center_x < wall_x:
                 # Add center coordinates for sorting
                value['center_x'] = center_x
                value['center_y'] = (value['y1'] + value['y2']) / 2
                stick_detections.append(value)

    if not stick_detections:
        return []

    # Sort by confidence (descending) to prune extras
    stick_detections.sort(key=lambda x: x['confidence'], reverse=True)
    confident_sticks = stick_detections[:max_perches]

    # Sort the confident sticks by their center x-coordinate (ascending) for numbering
    confident_sticks.sort(key=lambda x: x['center_x'])

    # Assign numbers and prepare the final list
    numbered_perches = []
    for i, perch in enumerate(confident_sticks):
        numbered_perches.append({
            'number': i + 1,
            'x1': perch['x1'],
            'y1': perch['y1'],
            'x2': perch['x2'],
            'y2': perch['y2'],
            'center_x': perch['center_x'],
            'center_y': perch['center_y']
        })

    return numbered_perches


def find_bird_location(bird_bbox, exploration_perches, fence_detected, horizontal_threshold=50):
    """
    Determines the bird's location (specific perch, ground, fence, other, or missing).

    Parameters
    ----------
    bird_bbox : dict or None
        Bounding box of the bird {'x1', 'y1', 'x2', 'y2'} or None if not detected.
    exploration_perches : list
        List of numbered exploration perch dictionaries from identify_and_number_exploration_perches.
    fence_detected : bool
        True if a 'fence' object was detected in the frame with sufficient confidence.
    horizontal_threshold : int, optional
        Maximum horizontal distance (pixels) to consider the bird on a perch, by default 50.

    Returns
    -------
    str
        The determined location: 'perch1', 'perch2', ..., 'ground', 'fence', 'other', 'missing'.
    """
    if bird_bbox is None:
        return 'missing'

    if fence_detected:
        return 'fence'

    bird_center_x = (bird_bbox['x1'] + bird_bbox['x2']) / 2
    bird_center_y = (bird_bbox['y1'] + bird_bbox['y2']) / 2

    if not exploration_perches: # Handle case where no exploration perches were identified
         # Basic floor check: if bird is low, assume floor, otherwise 'other'
         # This needs a better definition, maybe based on frame height? Assuming a threshold for now.
         # A more robust approach might involve checking if it's on the right side and low.
         # For now, if no perches, can't determine floor reliably relative to perches.
        return 'other' # Or potentially check if bird_center_y > some_absolute_threshold

    # Check if bird is on the ground (below all identified exploration perches)
    min_perch_y2 = min(p['y2'] for p in exploration_perches)
    if bird_center_y > min_perch_y2: # Consider adding a buffer?
        return 'ground'

    min_dist = float('inf')
    found_perch_number = None

    for p in exploration_perches:
        # Check for vertical overlap (bird's center y is within perch's vertical span)
        if p['y1'] <= bird_center_y <= p['y2']:
            # Calculate horizontal distance based on perch number
            if p['number'] in [2, 3]:  # Special logic for perches 2 and 3
                if bird_center_y <= p['center_y']: # If bird center y is above perch center y (Notice the y-coordinate is inverted -> top frame has y=0)
                    dist = abs(bird_center_x - p['x1']) # Compare bird center to perch left edge
                else:
                    dist = abs(bird_center_x - p['x2']) # Compare bird center to perch right edge
            else:  # Standard logic for perches 1, 4, 5
                dist = abs(bird_center_x - p['center_x']) # Compare bird center to perch center

            # Check if this is the closest perch within the threshold
            if dist < horizontal_threshold and dist < min_dist:
                min_dist = dist
                found_perch_number = p['number']

    if found_perch_number is not None:
        return f'perch{found_perch_number}'
    else:
        return 'other' # Bird is not on fence, ground, or any exploration perch


def calculate_sections(bird_bbox, wall_x, exploration_perches):
    """
    Calculates the horizontal (left/right) and vertical (top/middle/bottom) section
    the bird is in.

    Parameters
    ----------
    bird_bbox : dict or None
        Bounding box of the bird {'x1', 'y1', 'x2', 'y2'} or None.
    wall_x : float or None
        The x-coordinate of the center of the wall or None.
    exploration_perches : list
        List of numbered exploration perch dictionaries.

    Returns
    -------
    dict
        A dictionary with 'horizontal': ('left', 'right', or 'unknown') and
        'vertical': ('top', 'middle', 'bottom', or 'unknown').
    """
    sections = {'horizontal': 'unknown', 'vertical': 'unknown'}

    if bird_bbox is None:
        return sections # Cannot determine sections without bird

    bird_center_x = (bird_bbox['x1'] + bird_bbox['x2']) / 2
    bird_center_y = (bird_bbox['y1'] + bird_bbox['y2']) / 2

    # Determine horizontal section
    if wall_x is not None:
        if bird_center_x < wall_x:
            sections['horizontal'] = 'left'
        else:
            sections['horizontal'] = 'right'

    # Determine vertical section (relative to exploration perches if available)
    if exploration_perches:
        min_y1 = min(p['y1'] for p in exploration_perches)
        max_y2 = max(p['y2'] for p in exploration_perches)
        perch_height = max_y2 - min_y1

        if perch_height > 0: # Avoid division by zero if perches have no height span
            # 0-30% perch length -> bottom
            # 30-70% perch length -> middle
            # 70-100% perch length -> top
            # Note: y-coordinates increase downwards
            top_boundary = min_y1 + perch_height * 0.3
            bottom_boundary = min_y1 + perch_height * 0.7

            if bird_center_y < top_boundary:
                sections['vertical'] = 'top'
            elif bird_center_y < bottom_boundary:
                sections['vertical'] = 'middle'
            else:
                sections['vertical'] = 'bottom'
        # else: vertical section remains 'unknown' if perch_height is 0

    # If no exploration perches, vertical section remains 'unknown'
    # A fallback using absolute coordinates could be added here if needed.

    return sections


# --- Smoothing and Imputation Functions (copied from utils/movement_functions.py) ---

def sliding_mean(x: np.array, window_size: int = 15):
    """
    Returns the sliding average for an array. Handles NaNs by ignoring them in the mean calculation
    for the window, but keeps NaNs in the output if the window is all NaNs.

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
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float) # Ensure it's a numpy array of floats

    assert window_size % 2 == 1, "Window size must be odd."
    half_window = window_size // 2

    # Use pandas rolling mean which handles NaNs gracefully
    import pandas as pd
    series = pd.Series(x)
    means = series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

    # Pad the edges using reflection (pandas doesn't do this directly for rolling)
    # We'll calculate the edges separately using reflected padding
    padded_x = np.pad(x, pad_width=half_window, mode='reflect')
    edge_means = np.convolve(np.where(np.isnan(padded_x), 0, padded_x), np.ones(window_size), mode='valid')
    counts = np.convolve(~np.isnan(padded_x), np.ones(window_size), mode='valid')
    
    # Avoid division by zero
    valid_counts_mask = counts > 0
    edge_means[valid_counts_mask] /= counts[valid_counts_mask]
    edge_means[~valid_counts_mask] = np.nan # Set to NaN where count is zero

    # Fill the edges of the pandas result
    means[:half_window] = edge_means[:half_window]
    means[-half_window:] = edge_means[-half_window:]

    return means


def sliding_mode(x: np.array, window_size: int = 31):
    """
    Returns the sliding mode for an array. Handles NaNs by ignoring them.

    Parameters
    ----------
    x : np.array
        Array with shape (n,) containing input data.
    window_size : int
        Sliding mode window size. Must be odd.

    Returns
    -------
    modes : np.array
        Array with shape (n,) containing the sliding modes. Returns NaN for windows with no valid data.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x) # Ensure it's a numpy array

    assert window_size % 2 == 1, "Window size must be odd."
    half_window = window_size // 2
    n = len(x)
    modes = np.full(n, np.nan) # Initialize with NaNs

    # Pad the array at the edges
    # Using 'edge' padding might be okay for mode, or reflect? Let's stick to edge for simplicity.
    x_padded = np.pad(x, half_window, mode='edge')

    for i in range(n):
        i_center = i + half_window
        window = x_padded[i_center - half_window : i_center + half_window + 1] # Correct window slicing
        valid_window = window[~np.isnan(window)] # Filter out NaNs

        if valid_window.size > 0:
            # Calculate mode using scipy.stats.mode
            mode_result = mode(valid_window, keepdims=False)
            # Handle cases where mode returns multiple values (take the first)
            if np.isscalar(mode_result.mode):
                 modes[i] = mode_result.mode
            elif len(mode_result.mode) > 0:
                 modes[i] = mode_result.mode[0]
            # else: modes[i] remains NaN if mode is empty (shouldn't happen if valid_window.size > 0)
        # else: modes[i] remains NaN if window has no valid data

    return modes


def impute_data(x):
    """
    Uses linear interpolation to fill in missing NaN values.

    Parameters
    ----------
    x : np.ndarray
        Array potentially containing NaN values.

    Returns
    -------
    np.ndarray
        Imputed array. If all values are NaN, returns the original array.
        If only one non-NaN value exists, fills NaNs with that value.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float) # Ensure float array for interpolation

    x_copy = x.copy()
    nans = np.isnan(x_copy)
    
    if not np.any(~nans): # All values are NaN
        return x_copy
        
    if np.sum(~nans) == 1: # Only one non-NaN value
        fill_value = x_copy[~nans][0]
        x_copy[nans] = fill_value
        return x_copy

    # Use np.interp for linear interpolation
    indices = lambda z: z.nonzero()[0]
    x_copy[nans] = np.interp(indices(nans), indices(~nans), x_copy[~nans])
    
    # Optional: Convert back to int if original data was likely integer-based,
    # but be careful as interpolation introduces floats. Returning float is safer.
    # x_copy = np.array([int(val) for val in x_copy]) # Uncomment if integer output is desired and appropriate

    return x_copy
