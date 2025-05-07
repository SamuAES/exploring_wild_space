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
        if value.get("class") == "stick" and value.get("confidence") >= 0.4: # Set confidence threshold
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
    
    # Number the perches from left to right
    # Exploration side perches are numbered from 1 to 5
    # Home side perches are numbered from 6 to 8
    exploration_perches = [{'number': i + 1, **perch} for i, perch in enumerate(confident_sticks_left)]
    home_perches = [{'number': i + 6, **perch} for i, perch in enumerate(confident_sticks_right)]

    # return numbered_perches
    return exploration_perches, home_perches

def remove_close_perches(stick_detections: list, threshold: int = 20):
    """
    Removes perches that are too close to each other based on a specified threshold.
    If two perches are closer than the threshold, the one with the lower confidence score is removed.

    Parameters
    ----------
    stick_detections : list
        List of dictionaries containing perch coordinates and confidence scores.
    threshold : int, optional
        The minimum distance between perches to consider them as separate, by default 20.

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


def initialize_coordinate_arrays(exploration_perches, home_perches):
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
    number_of_exploration_perches = 5
    if len(exploration_perches) != number_of_exploration_perches:
        raise ValueError("Expected 5 perches, got {}".format(len(exploration_perches)))

    perches_x1 = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch 
    perches_x1[:,:] = np.nan # initialize as NaNs
   
    perches_y1 = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch 
    perches_y1[:,:] = np.nan # initialize as NaNs
   
    perches_x2 = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch 
    perches_x2[:,:] = np.nan # initialize as NaNs
   
    perches_y2 = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch 
    perches_y2[:,:] = np.nan # initialize as NaNs
    
    perches_x_center = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch
    perches_x_center[:,:] = np.nan # initialize as NaNs
   
    perches_y_center = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch
    perches_y_center[:,:] = np.nan # initialize as NaNs
    
    # Exploration side perch coordinates of the first frame
    # Note: The exploration side perches are numbered from 1 to 5
    for i, perch in enumerate(exploration_perches):
        perches_x1[0,i] = perch['x1']
        perches_y1[0,i] = perch['y1']
        perches_x2[0,i] = perch['x2']
        perches_y2[0,i] = perch['y2']
        perches_x_center[0,i] = perch['center_x']
        perches_y_center[0,i] = perch['center_y']

    # Home side perch coordinates of the first frame
    # Note: The home side perches are numbered from 6 to 8
    if len(home_perches) == 3:
        for i, perch in enumerate(home_perches):
            j = i + number_of_exploration_perches
            perches_x1[0,j] = perch['x1']
            perches_y1[0,j] = perch['y1']
            perches_x2[0,j] = perch['x2']
            perches_y2[0,j] = perch['y2']
            perches_x_center[0,j] = perch['center_x']
            perches_y_center[0,j] = perch['center_y']

    return perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center


# --- Helper function for matching a list of current perches to a range of historical slots ---
def _assign_matches_to_slots(current_perches_list, historical_indices_range, 
                                all_historical_avg_x, all_historical_avg_y,
                                target_coord_arrays_tuple):
    
    available_hist_indices_in_range = list(historical_indices_range)
    potential_matches = [] # Stores (distance_sq, current_perch_original_list_idx, matched_historical_array_idx)

    for i, current_perch_data in enumerate(current_perches_list):
        if not available_hist_indices_in_range: # No more historical slots in this range to match to
            break 

        min_dist_sq = float('inf')
        best_hist_idx_for_current_perch = -1

        # Find the best historical slot in the given range for the current_perch_data
        for hist_idx_in_absolute_array in available_hist_indices_in_range:
            # Check if historical average for this slot is valid
            if np.isnan(all_historical_avg_x[hist_idx_in_absolute_array]) or \
                np.isnan(all_historical_avg_y[hist_idx_in_absolute_array]):
                continue
            
            dist_sq = (current_perch_data['center_x'] - all_historical_avg_x[hist_idx_in_absolute_array])**2 + \
                        (current_perch_data['center_y'] - all_historical_avg_y[hist_idx_in_absolute_array])**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_hist_idx_for_current_perch = hist_idx_in_absolute_array
        
        if best_hist_idx_for_current_perch != -1:
            # Store original index 'i' from current_perches_list and the matched absolute historical index
            potential_matches.append((min_dist_sq, i, best_hist_idx_for_current_perch))

    # Sort all potential matches by distance (greedy assignment)
    potential_matches.sort()

    assigned_hist_indices_absolute = set() # Tracks which historical slots (0-7) have been assigned
    assigned_current_perch_original_indices = set() # Tracks which perches from current_perches_list are assigned

    for dist_sq, current_perch_original_idx, matched_hist_idx_absolute in potential_matches:
        # Ensure both the current perch and the historical slot haven't been assigned yet in this call
        if current_perch_original_idx not in assigned_current_perch_original_indices and \
            matched_hist_idx_absolute not in assigned_hist_indices_absolute:
            
            perch_to_assign = current_perches_list[current_perch_original_idx]
            
            # Assign coordinates to the matched historical slot in the current frame's data
            target_coord_arrays_tuple[0][matched_hist_idx_absolute] = perch_to_assign['x1']
            target_coord_arrays_tuple[1][matched_hist_idx_absolute] = perch_to_assign['y1']
            target_coord_arrays_tuple[2][matched_hist_idx_absolute] = perch_to_assign['x2']
            target_coord_arrays_tuple[3][matched_hist_idx_absolute] = perch_to_assign['y2']
            target_coord_arrays_tuple[4][matched_hist_idx_absolute] = perch_to_assign['center_x']
            target_coord_arrays_tuple[5][matched_hist_idx_absolute] = perch_to_assign['center_y']
            
            assigned_hist_indices_absolute.add(matched_hist_idx_absolute)
            assigned_current_perch_original_indices.add(current_perch_original_idx)

def update_perch_coordinates(perches_x1, perches_y1, perches_x2, perches_y2, 
                             perches_x_center, perches_y_center, 
                             exploration_perches_current, home_perches_current, 
                             ind, track_home_perches):
    """
    Updates the coordinates of the perches in the arrays with new coordinates from the current frame.
    Exploration perches (historical slots 0-4) are matched with 'exploration_perches_current'.
    Home perches (historical slots 5-7) are matched with 'home_perches_current' only if 'track_home_perches' is True.
    Matches each new perch to the closest historical perch within its group (exploration/home)
    based on center coordinates and updates that column. For historical perch slots that are not
    matched with a current perch, their coordinates from the previous frame are carried forward.
    The arrays are designed to hold the coordinates of 10 previous frames for each of the 8 potential perches.

    Parameters
    ----------
    perches_x1 : np.ndarray
    perches_y1 : np.ndarray
    perches_x2 : np.ndarray
    perches_y2 : np.ndarray
    perches_x_center : np.ndarray
    perches_y_center : np.ndarray
        Historical coordinate arrays, shape (10, 8).
    exploration_perches_current : list
        A list of dictionaries for exploration perches (up to 5) detected in the current frame.
        Expected to have 'center_x', 'center_y', 'x1', 'y1', 'x2', 'y2'.
    home_perches_current : list
        A list of dictionaries for home perches (up to 3) detected in the current frame.
        Expected to have 'center_x', 'center_y', 'x1', 'y1', 'x2', 'y2'.
    ind : int
        Current frame index.
    track_home_perches : bool
        If True, home perches are updated. If False, home perch coordinates remain NaN.

    Returns
    -------
    tuple
        A tuple containing the updated historical arrays: perches_x1, perches_y1, perches_x2, perches_y2,
        perches_x_center, and perches_y_center.
    """
    # Compute moving averages from the previous coordinates for matching
    x_center_moving_avgs = np.nanmean(perches_x_center, axis=0)
    y_center_moving_avgs = np.nanmean(perches_y_center, axis=0)

    # Determine the index for the previous frame's data in the circular buffer
    # Add buffer_size (10) before modulo to handle ind=0 correctly, ensuring a positive index
    buffer_size = perches_x1.shape[0] # Should be 10
    previous_row_index = (ind - 1 + buffer_size) % buffer_size

    # Initialize arrays for the current frame's coordinates by carrying over values
    # from the previous time step for each perch.
    # .copy() is important to avoid modifying the historical array directly if no new match is found.
    current_x1_frame = perches_x1[previous_row_index, :].copy()
    current_y1_frame = perches_y1[previous_row_index, :].copy()
    current_x2_frame = perches_x2[previous_row_index, :].copy()
    current_y2_frame = perches_y2[previous_row_index, :].copy()
    current_x_center_frame = perches_x_center[previous_row_index, :].copy()
    current_y_center_frame = perches_y_center[previous_row_index, :].copy()
    
    # Package current frame arrays into a tuple for the helper function
    current_coords_tuple = (current_x1_frame, current_y1_frame, current_x2_frame, current_y2_frame, 
                            current_x_center_frame, current_y_center_frame)
    
    # --- Match and assign exploration perches (target historical slots 0-4) ---
    if exploration_perches_current: # Check if the list is not empty
        _assign_matches_to_slots(exploration_perches_current, range(5), 
                                 x_center_moving_avgs, y_center_moving_avgs,
                                 current_coords_tuple)

    # --- Match and assign home perches (target historical slots 5-7) ---
    if track_home_perches and len(home_perches_current) > 0: # Check flag and if list is not empty
        _assign_matches_to_slots(home_perches_current, range(5, 8), 
                                 x_center_moving_avgs, y_center_moving_avgs,
                                 current_coords_tuple)
    
    # If not tracking home perches, or if home_perches_current is empty,
    # slots 5-7 of current_coords_tuple will remain NaN from initialization.

    # Update the historical arrays with the current frame's data
    current_row_index = ind % 10
    perches_x1[current_row_index, :] = current_x1_frame
    perches_y1[current_row_index, :] = current_y1_frame
    perches_x2[current_row_index, :] = current_x2_frame
    perches_y2[current_row_index, :] = current_y2_frame
    perches_x_center[current_row_index, :] = current_x_center_frame
    perches_y_center[current_row_index, :] = current_y_center_frame

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



def find_bird_on_perch(px_moving_avgs, py2_moving_avgs, bird_x, bird_y, fence_status, wall_x, track_home_perches):
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

    # Check wich side of the wall the bird is on
    exploration_side = False
    home_side = False
    
    if bird_x < wall_x: # Bird is on the exploration side
        px_coords = px_moving_avgs[:5] # only use the first 5 perches
        py2_coords = py2_moving_avgs[:5]
        exploration_side = True
    else: # Bird is on the home side
        # Check if we are tracking home perches
        if not track_home_perches:
            return 0
        px_coords = px_moving_avgs[5:] # only use the last 3 perches
        py2_coords = py2_moving_avgs[5:]
        home_side = True

    # Check if perch coordinate data is available for the current side
    if np.all(np.isnan(px_coords)):
        # No valid perch x-coordinates on this side. Bird cannot be on a perch here.
        # Check for floor status.
        if np.all(np.isnan(py2_coords)):
            return 0 # No perch or floor y-coordinate data available for this side
        # py2_coords has some valid data, check floor against it
        elif bird_y > np.nanmax(py2_coords): # This np.nanmax is now safer
            return -2 # Floor
        return 0 # Other (not on floor based on available py2_coords)

    # if both the x and y coordinates match, the bird is sitting on a perch
    dist = np.abs(px_coords-bird_x)
    min_dist = np.nanmin(dist)
    perch_index = np.nanargmin(dist) # find perch index

    # check x-coordinate
    if min_dist<50:
        # if bird is lower than the perch, it is on the floor
        if bird_y > py2_coords[perch_index]:
            return -2
        # otherwise, it is on a perch
        else:
            if exploration_side:
                return perch_index+1
            elif home_side:
                return perch_index+1 + 5

    # next, if the bird is not perching or on the cage, check if it is on the floor
    # if it's lower than the highest perch, we can assume it's on the floor
    if bird_y > np.nanmax(py2_coords):
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