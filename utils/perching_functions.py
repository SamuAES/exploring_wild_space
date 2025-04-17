import numpy as np
import copy
import pandas as pd
import re

def initialize_coordinates(video_name):

    p_xs = np.empty((10,8)) # arrays of 10 previous x-coordinates for each perch 
    p_xs[:,:] = np.nan # initialize as NaNs
    p_ys = np.empty((10,8)) # arrays of 10 previous y-coordinates for each perch 
    p_ys[:,:] = np.nan # initialize as NaNs

    # fill in first x-coordinates from manual annotations
    master_file = "data/masterfile_20202021_LOOPY(Coordinates_xPerches).csv"
    df = pd.read_csv(f"{master_file}") # relative filepath
    df = df[df["VIDEO TITLE"]==video_name]
    coords = np.array([df[f"X PERCH{i+1}"] for i in range(8)]).T[0] # get coordinates
    # check if annotation goes from left to right or right to left; flip them if right to left
    if coords[0]>coords[1] or coords[1]>coords[2] or coords[0]>coords[3] or coords[2]>coords[4]: # several conditions in case there are nans (kind of a hack)
        p_xs[0,:] = np.flip(coords)
    else:
        p_xs[0,:] = coords

    # for tracking 5perches, make a set of visible perch numbers
    five_perches_set = set()
    for i in range(5):
        #print(p_xs[0,i])
        if not np.isnan(p_xs[0,i]):
            five_perches_set.add(i+1)
    
    return p_xs, p_ys, five_perches_set


def extract_coordinates(frame):
    """
    Removes boxes with lowest probability if there are more than 1 bird and more than 8 perches

    Returns
    -------
    bird_x: (float) Middle of x-coordinates of most confident bird sighting
    bird_y: (float) Middle of y-coordinates of most confident bird sighting
    stick_xs: (numpy.array) Middles of x-coordinates of 8 most confident perch sightings
    stick_ys: (numpy.array) Lowest y-coordinates of 8 most confident perch sightings
    wall_x: (float) Middle of x-coordinates of most confident wall sighting
    """
    bird_probs = []
    stick_probs = []
    wall_probs = []
    on_cage = False
    for key in frame: # loop through items
        if frame[key]["class"] == "stick":
            if frame[key]["confidence"]>0.5:
                stick_probs.append((key, frame[key]["confidence"])) # append probability to list if confidence threshold of 50% is met
        elif frame[key]["class"] == "fence":
            if frame[key]["confidence"]>0.5:
                on_cage = True # flip on_cage to True if bird is on the fence with probability higher than threshold
    
    # create arrays for sorting
    dtype = [("id", object), ("probability", float)]
    stick_array = np.array(stick_probs, dtype=dtype)

    # sort arrays to get most confident bird and wall and 8 most confident sticks later
    sorted_sticks = np.sort(stick_array, order="probability")

    # if perches not found, return empty lists
    if len(sorted_sticks)==0:
        return [], [], on_cage

    # calculate perch coordinates
    stick_ids = np.array([s[0] for s in sorted_sticks])
    stick_xs = np.array([(frame[stick_id]["x1"]+frame[stick_id]["x2"])/2 for stick_id in stick_ids])
    stick_ys = np.array([frame[stick_id]["y2"] for stick_id in stick_ids])
    
    # remove perch coordinates that are too close to other perches (less than threshold)
    aa, bb = np.meshgrid(stick_xs, stick_xs)
    dist = np.abs(aa-bb)+np.eye(len(stick_xs))*10000 # distance matrix with diagonal changed to 10000 (so it doesn't interfere)
    min_dist = np.min(dist, axis=0) # smallest distance of each perch to the others
    xs_ind_drop = np.argwhere(min_dist<4).T[0] # indices below the threshold
    xs_ind_keep = np.argwhere(min_dist>=4).T[0] # indices above the threshold
    xs_ind = np.sort(np.concatenate((xs_ind_keep, xs_ind_drop[:1]))) # combine indices: the ones to keep and only one of the similar ones

    stick_xs = stick_xs[xs_ind]
    stick_ys = stick_ys[xs_ind]

    return stick_xs[:8], stick_ys[:8], on_cage # perch coordinates are sorted by confidence: return only first 8


def update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, n):

    px_moving_avgs = np.nanmean(p_xs, axis=0)
    py_moving_avgs = np.nanmean(p_ys, axis=0)

    # if the same number (or less) of new coordinates than moving avgs -> add new coordinates to closest avgs
    # calculate distance matrix between current coordinates and previous moving averages to decide which perch goes where
    aa, bb = np.meshgrid(px_moving_avgs, new_p_xs)
    dist = np.abs(aa-bb)
    # find closest ones
    perch_ind = np.argmin(dist, axis=1)
    # assign the value of the closest new coordinate to each previous perch
    for p_x, p_y, p_ind in zip(new_p_xs, new_p_ys, perch_ind):
        p_xs[n,p_ind] = p_x
        p_ys[n,p_ind] = p_y # y coordinate too

    # if more new perches than moving averages: TODO
    
    return p_xs, p_ys, px_moving_avgs, py_moving_avgs

def find_bird_on_perch(px_moving_avgs, py_moving_avgs, bird_x, bird_y, cage_status):

    framecount = np.count_nonzero(~np.isnan(cage_status)) # how many non-nan values we have in cage_status

    # if more than 75% of recorded cage statuses are True, the bird is on the cage
    if np.nansum(cage_status)/framecount>0.75: 
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

"""
logic:
- if on cage -> cage = -1
- elif on perch -> perch = 1-8
- elif lower than perches -> floor = -2
- else -> other = 0
"""

# FEATURES
# these functions could easily be combined into just one
def compute_perch_durations(status, fps):
    result = np.zeros(8)
    for i in range(8):
        result[i] = np.sum(status==i+1)/fps
    return result

def compute_ground_and_fence(status, fps):
    fence = np.sum(status==-1)/fps
    ground = np.sum(status==-2)/fps
    return ground, fence


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

if __name__=="__main__":

    # development tests, feel free to ignore. several of these are broken currently

    fname = "HE21320_210621_21JJ15_exploration_IB"
    print(initialize_coordinates(fname))

    test_result = np.array([0,-1,-1,0,2,3,3,4,4,2,2,3,3,4,4,5,5,5,6,4,4,5,6,7,8,0,1,1,1,1,7,6,5,5,5,6])
    perch1, perch2, perch3, perch4, perch5, perch6, perch7, perch8 = compute_perch_durations(test_result, 30)
    print(perch1, perch2)
    g, f = compute_ground_and_fence(test_result, 30)
    print(g, f)

    a = {0: {'class': 'bird', 'confidence': 0.9089881181716919, 'x1': 1144, 'y1': 296, 'x2': 1280, 'y2': 416},
        1: {'class': 'bird', 'confidence': 0.2559300899505615, 'x1': 131, 'y1': 55, 'x2': 178, 'y2': 648},
        2: {'class': 'stick', 'confidence': 0.7828576564788818, 'x1': 714, 'y1': 59, 'x2': 759, 'y2': 651},
        3: {'class': 'stick', 'confidence': 0.7416277527809143, 'x1': 837, 'y1': 70, 'x2': 863, 'y2': 613},
        4: {'class': 'stick', 'confidence': 0.7400134801864624, 'x1': 412, 'y1': 53, 'x2': 443, 'y2': 682},
        5: {'class': 'stick', 'confidence': 0.6996856331825256, 'x1': 607, 'y1': 81, 'x2': 631, 'y2': 611},
        6: {'class': 'stick', 'confidence': 0.5938723921775818, 'x1': 982, 'y1': 23, 'x2': 1030, 'y2': 707},
        7: {'class': 'stick', 'confidence': 0.665341854095459, 'x1': 371, 'y1': 71, 'x2': 390, 'y2': 542},
        8: {'class': 'stick', 'confidence': 0.7302467584609985, 'x1': 1066, 'y1': 49, 'x2': 1108, 'y2': 617},
        9: {'class': 'stick', 'confidence': 0.4596527636051178, 'x1': 372, 'y1': 148, 'x2': 390, 'y2': 621}}

    new_bird_x, new_bird_y, new_p_xs, new_p_ys = extract_coordinates(a)
    print(new_bird_x, new_bird_y)
    print(new_p_xs)
    print(new_p_ys)

    video_number = "HE21341_300621_21EK21"
    video_name = f"{video_number}_exploration_IB"
    p_xs, p_ys, w_xs = initialize_coordinates(video_name)
    p_xs, p_ys, px_moving_avgs, py_moving_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, 0)
    print(p_xs)
    print(p_ys)


    b = [11,22,33]
    c = [34,9,19]
    bb, cc = np.meshgrid(b, c)
    print(bb)
    print(cc)
    dist = np.abs(bb-cc)
    print(dist)
    perch_ind = np.argmin(dist, axis=1)
    print(perch_ind)