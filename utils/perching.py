import numpy as np


def identify_and_number_perches(frame_data, wall_x):
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
        if value.get("class") == "stick" and value.get("confidence") >= 0.7: # Set confidence threshold
            center_x = (value['x1'] + value['x2']) / 2
            # Check if the stick is on the exploration side (left of the wall)
            value['center_x'] = center_x
            value['center_y'] = (value['y1'] + value['y2']) / 2
            if center_x < wall_x:
                stick_detections_left.append(value)
            else:
                stick_detections_right.append(value)

    
    # Sort by confidence (descending) to prune extras
    stick_detections_left.sort(key=lambda x: x['confidence'], reverse=True)
    stick_detections_right.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Remove duplicates by removing perches that are too close to each other (less than threshold)
    stick_detections_left = remove_close_perches(stick_detections_left)
    stick_detections_right = remove_close_perches(stick_detections_right)

    # Number of perches on left side is 5
    left_perches=5
    # Number of perches on right side is 3
    right_perches=3

    # Select the most confident sticks
    confident_sticks_left = stick_detections_left[:left_perches]
    confident_sticks_right = stick_detections_right[:right_perches]

    # Sort the confident sticks by their center x-coordinate (ascending) for numbering
    confident_sticks_left.sort(key=lambda x: x['center_x'])
    confident_sticks_right.sort(key=lambda x: x['center_x'])

    # Assign numbers and prepare the final list
    numbered_perches = []
    for i, perch in enumerate(confident_sticks_left):
        numbered_perches.append({
            'number': i + 1, # Numbers 1 to 5 for left side
            'x1': perch['x1'],
            'y1': perch['y1'],
            'x2': perch['x2'],
            'y2': perch['y2'],
            'center_x': perch['center_x'],
            'center_y': perch['center_y']
        })

    for i, perch in enumerate(confident_sticks_right):
        numbered_perches.append({
            'number': i + 6,  # Numbers 6 to 8 for right side
            'x1': perch['x1'],
            'y1': perch['y1'],
            'x2': perch['x2'],
            'y2': perch['y2'],
            'center_x': perch['center_x'],
            'center_y': perch['center_y']
        })

    return numbered_perches

def remove_close_perches(stick_detections: list, threshold: int = 4):
    """
    Removes perches that are too close to each other based on a specified threshold.
    If two perches are closer than the threshold, the one with the lower confidence score is removed.

    Parameters
    ----------
    stick_detections : list
        List of dictionaries containing perch coordinates and confidence scores.
    threshold : int, optional
        The minimum distance between perches to consider them as separate, by default 4.

    Returns
    -------
    list
        A filtered list of stick detections with close perches removed and sorted by confidence in descending order.
    """
    stick_detections.sort(key=lambda x: x['confidence'], reverse=True)
    filtered_sticks = []
    for i, stick in enumerate(stick_detections):
        is_too_close = False
        for j in range(i):
            if abs(stick['center_x'] - stick_detections[j]['center_x']) < threshold:
                is_too_close = True
                break
        if not is_too_close:
            filtered_sticks.append(stick)
    
    return filtered_sticks



def initialize_coordinate_arrays(numbered_perches):
    """
    Initializes arrays to store the x and y coordinates of the perches.
    The arrays are initialized with NaN values and the first row is filled with the coordinates
    of the first frame.
    The arrays are designed to hold the coordinates of 10 previous frames for each perch.
    The function assumes that there are 8 perches.

    Parameters
    ----------
    numbered_perches : list
        A list of dictionaries containing the coordinates of the perches.
    
    Returns
    -------
    tuple
        A tuple containing six arrays: perches_x1, perches_y1, perches_x2, perches_y2,
        perches_x_center, and perches_y_center. Each array has a shape of (10, 8).
    """
    number_of_perches = 8
    if len(numbered_perches) != number_of_perches:
        raise ValueError("Expected 8 perches, got {}".format(len(numbered_perches)))

    perches_x1 = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch 
    perches_x1[:,:] = np.nan # initialize as NaNs
    perches_x1[0,:] = [perch['x1'] for perch in numbered_perches] # coordinates of the first frame

    perches_y1 = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch 
    perches_y1[:,:] = np.nan # initialize as NaNs
    perches_y1[0,:] = [perch['y1'] for perch in numbered_perches] # coordinates of the first frame  

    perches_x2 = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch 
    perches_x2[:,:] = np.nan # initialize as NaNs
    perches_x2[0,:] = [perch['x2'] for perch in numbered_perches] # coordinates of the first frame

    perches_y2 = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch 
    perches_y2[:,:] = np.nan # initialize as NaNs
    perches_y2[0,:] = [perch['y2'] for perch in numbered_perches] # coordinates of the first frame

    perches_x_center = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch
    perches_x_center[:,:] = np.nan # initialize as NaNs
    perches_x_center[0,:] = [perch['center_x'] for perch in numbered_perches] # coordinates of the first frame

    perches_y_center = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch
    perches_y_center[:,:] = np.nan # initialize as NaNs
    perches_y_center[0,:] = [perch['center_y'] for perch in numbered_perches] # coordinates of the first frame
    
    return perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center


def update_perch_coordinates(perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center, numbered_perches, ind):
    """
    Updates the coordinates of the perches in the arrays with the new coordinates from the current frame.
    Matches each new perch to the closest historical perch based on center coordinates and updates that column.
    Fills columns for unmatched historical perches with NaN for the current frame.
    The function assumes that there are 8 perches historically.
    The arrays are designed to hold the coordinates of 10 previous frames for each perch.

    Parameters
    ----------
    perches_x1 : np.ndarray
    perches_y1 : np.ndarray
    perches_x2 : np.ndarray
    perches_y2 : np.ndarray
    perches_x_center : np.ndarray
    perches_y_center : np.ndarray
    numbered_perches : list
        A list of dictionaries containing the coordinates of the perches of the current frame.
    ind : int
        Current frame index.

    Returns
    -------
    tuple
        A tuple containing the updated arrays: perches_x1, perches_y1, perches_x2, perches_y2,
        perches_x_center, and perches_y_center. Each array has a shape of (10, 8).
    """
    # Compute moving averages from the previous coordinates for matching
    x_center_moving_avgs = np.nanmean(perches_x_center, axis=0)
    y_center_moving_avgs = np.nanmean(perches_y_center, axis=0)

    # Initialize arrays for the current frame's coordinates with NaN
    current_x1 = np.full(8, np.nan)
    current_y1 = np.full(8, np.nan)
    current_x2 = np.full(8, np.nan)
    current_y2 = np.full(8, np.nan)
    current_x_center = np.full(8, np.nan)
    current_y_center = np.full(8, np.nan)

    # --- Matching Logic ---
    # Use a copy of historical averages that we can mark as "used"
    available_historical_indices = list(range(8))
    matches = [] # Store potential matches (distance, current_perch_idx, historical_idx)

    for i, current_perch in enumerate(numbered_perches):
        if not available_historical_indices:
            break # No more historical perches to match to

        min_dist_sq = float('inf')
        best_historical_idx = -1

        # Calculate distance squared to available historical perch centers
        for hist_idx in available_historical_indices:
            # Handle potential NaN in moving averages if a perch wasn't seen recently
            if np.isnan(x_center_moving_avgs[hist_idx]) or np.isnan(y_center_moving_avgs[hist_idx]):
                continue
            dist_sq = (current_perch['center_x'] - x_center_moving_avgs[hist_idx])**2 + \
                      (current_perch['center_y'] - y_center_moving_avgs[hist_idx])**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_historical_idx = hist_idx

        # Add the best match found for this current perch (if any)
        # Add a threshold check if needed (e.g., if min_dist_sq < some_max_distance_sq:)
        if best_historical_idx != -1:
             matches.append((min_dist_sq, i, best_historical_idx))

    # Sort matches by distance to assign the closest ones first (greedy assignment)
    matches.sort()

    assigned_historical_indices = set()
    assigned_current_indices = set()

    for dist_sq, current_idx, historical_idx in matches:
        # Ensure both current perch and historical perch haven't been assigned yet
        if current_idx not in assigned_current_indices and historical_idx not in assigned_historical_indices:
            perch_to_assign = numbered_perches[current_idx]

            # Assign all coordinates to the matched historical column
            current_x1[historical_idx] = perch_to_assign['x1']
            current_y1[historical_idx] = perch_to_assign['y1']
            current_x2[historical_idx] = perch_to_assign['x2']
            current_y2[historical_idx] = perch_to_assign['y2']
            current_x_center[historical_idx] = perch_to_assign['center_x']
            current_y_center[historical_idx] = perch_to_assign['center_y']

            # Mark as assigned
            assigned_historical_indices.add(historical_idx)
            assigned_current_indices.add(current_idx)


    # Update the historical arrays at the current index (ind)
    # Columns corresponding to historical perches that weren't matched
    # will correctly get NaN from the initialized 'current_*' arrays.
    current_row_index = ind % 10
    perches_x1[current_row_index, :] = current_x1
    perches_y1[current_row_index, :] = current_y1
    perches_x2[current_row_index, :] = current_x2
    perches_y2[current_row_index, :] = current_y2
    perches_x_center[current_row_index, :] = current_x_center
    perches_y_center[current_row_index, :] = current_y_center

    return perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center


def bird_on_fence(frame:dict):
    """
    Check if the bird is on the fence in the current frame.
    If the bird is on the fence, it returns True, otherwise False.

    Parameters
    ----------
    frame : dict
        A dictionary containing the coordinates of the bird and perches.

    Returns
    -------
    bool
        True if the bird is on the fence, False otherwise.
    """
    for key, value in frame.items():
        if value.get("class") == "fence":
            if value.get("confidence") >= 0.5:
                return True
    return False
    
def bird_on_perch_2_or_3(bird_x, bird_y, perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center):
    """
    Check if the bird is above or below perch3 center y coordinate. If above, use x1 to determine is the bird on perch 2 or perch 3.
    If below, use x2 to determine is the bird on perch 2 or perch 3.

    Parameters
    ----------
    perches_x1 : np.ndarray
        Array of x1 coordinates for the perches.
    perches_y1 : np.ndarray
        Array of y1 coordinates for the perches.
    perches_x2 : np.ndarray
        Array of x2 coordinates for the perches.
    perches_y2 : np.ndarray
        Array of y2 coordinates for the perches.
    perches_x_center : np.ndarray
        Array of x center coordinates for the perches.
    perches_y_center : np.ndarray
        Array of y center coordinates for the perches.

    Returns
    -------
    int
        Number of the perch the bird is on (2 or 3).
    """
    p2_x1_moving_avg = np.nanmean(perches_x1[:,1])
    p2_x2_moving_avg = np.nanmean(perches_x2[:,1])

    p3_x1_moving_avg = np.nanmean(perches_x1[:,2])
    p3_x2_moving_avg = np.nanmean(perches_x2[:,2])
    
    p3_center_y_moving_avg = np.nanmean(perches_y_center[:,2])
    

    if bird_y < p3_center_y_moving_avg: # Bird is above perch 3
        p2_dist = abs(bird_x - p2_x1_moving_avg)
        p3_dist = abs(bird_x - p3_x1_moving_avg)
        if p2_dist < p3_dist:
            return 2
        else:
            return 3
    else: # Bird is below perch 3
        p2_dist = abs(bird_x - p2_x2_moving_avg)
        p3_dist = abs(bird_x - p3_x2_moving_avg)
        if p2_dist < p3_dist:
            return 2
        else:
            return 3
        

def find_bird_on_perch(px_moving_avgs, py_moving_avgs, bird_x, bird_y, fence_status):
    """
    Determine the perch number on which the bird is located based on its x and y coordinates.
    The function checks if the bird is on the cage, on a perch, or on the floor.
    If the bird is on the cage, it returns -1.
    If the bird is on a perch, it returns the perch number (1-8).
    If the bird is on the floor, it returns -2.
    If the bird is not on any perch or the cage, it returns 0.
    Parameters
    ----------
    px_moving_avgs : np.ndarray
    py_moving_avgs : np.ndarray
    bird_x : float
    bird_y : float
    fence_status : np.ndarray
        Array of cage status values (True/False) for each frame.
    """

    framecount = np.count_nonzero(~np.isnan(fence_status)) # how many non-nan values we have in cage_status

    # if more than 75% of recorded cage statuses are True, the bird is on the cage
    if np.nansum(fence_status)/framecount>0.75: 
        return -1

    # if both the x and y coordinates match, the bird as sitting on a perch
    dist = np.abs(px_moving_avgs-bird_x)
    min_dist = np.min(dist)
    perch_index = np.argmin(dist) # find perch index

    # check x-coordinate
    if min_dist<50:
        # if bird is lower than the perch, it is on the floor
        if bird_y > py_moving_avgs[perch_index]:
            return -2
        # otherwise, it is on a perch
        else:
            return perch_index+1

    # next, if the bird is not perching or on the cage, check if it is on the floor
    # if it's lower than the highest perch, we can assume it's on the floor
    if bird_y > np.nanmin(py_moving_avgs):
        return -2

    # finally, if none of the conditions are met, return 0=other
    return 0

def compute_perch_durations(status, fps):
    result = np.zeros(8)
    for i in range(8):
        result[i] = np.sum(status==i+1)/fps
    return result

def compute_ground_and_fence(status, fps):
    fence = np.sum(status==-1)/fps
    ground = np.sum(status==-2)/fps
    return ground, fence