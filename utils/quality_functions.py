import numpy as np


def detect_movement(previous_coordinates, current_coordinates, new_coordinates):
    '''
    Some bugs still
    '''
    diff = np.abs(previous_coordinates - current_coordinates)
    # If there are more than two perches that have moved horizontally by 10 pixels 
    if len(diff[diff > 10]) > 1:
        return True
    # In the case when the camera moves so much that yolo does not detect all the perches and 
    # thus does not update the perch coordinates
    if np.count_nonzero(~np.isnan(previous_coordinates)) > len(new_coordinates) + 1:
        return True

    return False

def perches_within_threshold(perch_coodinates, threshold=50):
    '''
    Returns True when there are at least two perches separated by less than
    the threshold value
    '''
    aa, bb = np.meshgrid(perch_coodinates, perch_coodinates)

    dist = np.abs(aa-bb)+np.eye(len(perch_coodinates))*10000

    min_dist = np.nanmin(dist, axis=0)

    if np.nanmin(min_dist) < threshold:
        return True
    return False

def too_many_perches_on_exploration_side(perch_coordinates, wall_coordiate, n=5): # Unused
    if np.isnan(wall_coordiate):
        return False
    return n < len(perch_coordinates[perch_coordinates < wall_coordiate])

def perch_assignment_failure(assigned_perches, perch_coordinates): # Unused
    for perch in perch_coordinates:
        if perch not in assigned_perches:
            return True
    
    return False

def perch_deviates_from_average(average_perch_coordinates, new_perch_coordinates, threshold=30): #Unused
    dist = np.abs(average_perch_coordinates - new_perch_coordinates)

    return len(dist[dist > 30]) > 0

def bird_inbetween_sections(frame, bird_y_coordinate, threshold=30):
    '''
    Returns True if the bird is found on the threshold of either the top or bottom sections
    '''
    #Get the average y1 and y2 coordinates of the perches.
    # y1 is top and y2 is bottom, y1 < y2
    perch_y1 = 0
    perch_y2 = 0
    perch_count = 0
    for box in frame.values():
        if box['class'] == 'stick':
            perch_y1 += box['y1']
            perch_y2 += box['y2']
            perch_count += 1
    
    if perch_count == 0:
        # Raise an exception if no perches are found
        raise ValueError("No perches found in the frame.")
    perch_y1 /= perch_count
    perch_y2 /= perch_count

    # Calculate the middle lower and upper y-coordinates of the perch
    # 0-30% perch length -> bottom
    # 30-70% perch length -> middle
    # 70-100% perch length -> top
    middle_upper_y = perch_y1 + (perch_y2 - perch_y1) * 0.3
    middle_lower_y = perch_y1 + (perch_y2 - perch_y1) * 0.7

    if bird_y_coordinate is None:
        return False
    
    if middle_lower_y+threshold >= bird_y_coordinate >middle_lower_y -threshold:
        return True
    
    if middle_upper_y+threshold >= bird_y_coordinate > middle_upper_y - threshold:
        return True
    return False
    
def bird_between_perch_2_3(px_moving_avgs, py_moving_avgs, bird_x, bird_y, cage_status):
    '''
    Returns True when the bird is found between perch two and three
    '''
    framecount = np.count_nonzero(~np.isnan(cage_status)) # how many non-nan values we have in cage_status

    # if more than 75% of recorded cage statuses are True, the bird is on the cage
    if np.nansum(cage_status)/framecount>0.75: 
        return False
    
    
    if px_moving_avgs[1] < bird_x < px_moving_avgs[2]:
        if py_moving_avgs[1] >  bird_y and bird_y < py_moving_avgs[2]:
            return True
    return False