import numpy as np
from scipy.stats import mode
import pandas as pd
from tqdm import tqdm

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


def identify_and_number_exploration_perches(frame_data, wall_x):
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

    stick_detections_left = []
    stick_detections_right = []

    for key, value in frame_data.items():
        if value.get("class") == "stick":
            center_x = (value['x1'] + value['x2']) / 2
            # Check if the stick is on the exploration side (left of the wall)
            value['center_x'] = center_x
            value['center_y'] = (value['y1'] + value['y2']) / 2
            if center_x < wall_x:
                stick_detections_left.append(value)
            else:
                stick_detections_right.append(value)

    if not stick_detections_left:
        return []

    # Number of perches on left side
    left_perches=5
    # Number of perches on right side
    right_perches=3

    # Sort by confidence (descending) to prune extras
    stick_detections_left.sort(key=lambda x: x['confidence'], reverse=True)
    confident_sticks_left = stick_detections_left[:left_perches]

    stick_detections_right.sort(key=lambda x: x['confidence'], reverse=True)
    confident_sticks_right = stick_detections_right[:right_perches]

    # Sort the confident sticks by their center x-coordinate (ascending) for numbering
    confident_sticks_left.sort(key=lambda x: x['center_x'])
    confident_sticks_right.sort(key=lambda x: x['center_x'])

    # Assign numbers and prepare the final list
    numbered_perches = []
    for i, perch in enumerate(confident_sticks_left + confident_sticks_right):
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
        if fence_detected['confidence'] > 0.8: # Assuming a confidence threshold for fence detection
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
                if bird_bbox['y2'] <= p['center_y']: # If bird center y is above perch center y (Notice the y-coordinate is inverted -> top frame has y=0)
                    dist = abs(bird_center_x - p['x1']) # Compare bird center to perch left edge
                else:
                    dist = abs(bird_center_x - p['x2']) # Compare bird center to perch right edge
            else:  # Standard logic for perches 1, 4, 5, 6, 7, 8
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
        y1_average = np.mean([p['y1'] for p in exploration_perches])
        y2_average = np.mean([p['y2'] for p in exploration_perches])
        perch_height = np.abs(y2_average - y1_average)

        if perch_height > 0: # Avoid division by zero if perches have no height span
            # 0-35% perch length -> bottom
            # 35-65% perch length -> middle
            # 65-100% perch length -> top
            # Note: y-coordinates increase downwards
            top_boundary = y1_average + perch_height * 0.32
            bottom_boundary = y1_average + perch_height * 0.66

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
    modes = [] # Initialize with NaNs

    for i in tqdm(range(n), desc="Smoothing with sliding mode", mininterval=1):
        # Handle edge cases where the window extends beyond the array bounds
        if i < window_size or i >= n - window_size:
            window = x[max(0, i - half_window) : min(n, i + half_window + 1)]
        
        else:
            i_center = i + half_window
            window = x[i_center - half_window : i_center + half_window + 1] # Correct window slicing

        # Calculate mode using pandas df.mode()
        mode_result = pd.DataFrame({'values':window}).mode(numeric_only=False)
        mode_result = mode_result['values'].to_numpy() # Convert to numpy array
        # Handle cases where mode returns multiple values (take the first)
        if len(mode_result) > 0:
            modes.append(mode_result[0])
        else:
            modes.append(np.nan)
        
    return np.array(modes)


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
