import numpy as np

def assign_section(perches_y1, perches_y2, bird_x, bird_y, wall_x):
    """
    Assign the section of the cage where the bird is located based on its coordinates and
    the middle wall coordinates.
    
    Parameters
    ----------
    perches_y1 : np.ndarray
        Array of size (10,8) y-coordinates for the top of the perches.
    perches_y2 : np.ndarray
        Array of size (10,8) y-coordinates for the bottom of the perches.
    bird_x : float
        Middle x-coordinate of the bird.
    bird_y : float
        Middle Y-coordinate of the bird.
    wall_x : float
        Middle x-coordinate of the wall.

    Returns
    -------
    dict
        Dictionary with the section of the cage where the bird is located.
        Keys are 'section_x' with value 'T_new'/'T_home' and 'section_y'
        with value 'top'/'middle'/'bottom'.
    """
    #
    # Get the average y1 and y2 coordinates of the perches.

    # y1 is top and y2 is bottom, y1 < y2
    perches_y1_avg = np.nanmean(perches_y1, axis=0).mean()
    perches_y2_avg = np.nanmean(perches_y2, axis=0).mean()

    # Calculate the middle lower and upper y-coordinates of the perch
    # 0-30% perch length -> bottom
    # 30-70% perch length -> middle
    # 70-100% perch length -> top
    middle_upper_y = perches_y1_avg + (perches_y2_avg - perches_y1_avg) * 0.3
    middle_lower_y = perches_y1_avg + (perches_y2_avg - perches_y1_avg) * 0.7

    result = {}

    if bird_y is None:
        pass
    elif bird_y > middle_lower_y:
        result['section_y'] = 'bottom'
    elif middle_lower_y >= bird_y > middle_upper_y:
        result['section_y'] = 'middle'
    else:
        result['section_y'] = 'top'

    if wall_x is None or bird_x is None:
        pass
    elif bird_x < wall_x:
        result['section_x'] = 'T_new'
    else:
        result['section_x'] = 'T_home'

    return result

def count_frames_by_section(result):
    """
    Count the number of frames for each section.
    
    Parameters
    ----------
    result : dict
        Dictionary containing bounding box information for a single frame.
    
    Returns
    -------
    dict
        Dictionary with the count for each section.
    """
    frame_counts = {'top': 0, 'middle': 0, 'bottom': 0, 'left': 0, 'right': 0}
    for value in result.values():
        if 'section_y' in value:
            frame_counts[value['section_y']] += 1
        if 'section_x' in value:
            frame_counts[value['section_x']] += 1
            break # only count the first detected bird
    
    return frame_counts

def convert_frame_counts_to_time(frame_counts, fps=30):
    """
    Convert frame counts to time in seconds.
    
    Parameters
    ----------
    frame_counts : dict
        Dictionary with the count for each section.
    fps : int, optional
        Frames per second, by default 30
    
    Returns
    -------
    dict
        Dictionary with the time (in seconds) for each section.
    """
    time_by_section = {section: round(count / fps, 2) for section, count in frame_counts.items()}
    return time_by_section


