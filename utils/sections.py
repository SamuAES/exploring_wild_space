
def assign_section(frame, bird_x, bird_y, wall_x):
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

    result = {}

    if bird_y is None:
        pass
    elif bird_y > middle_lower_y:
        result['section_y'] = 'bottom'
    elif middle_lower_y >= bird_y > middle_upper_y:
        result['section_y'] = 'middle'
    else:
        result['section_y'] = 'top'

    if wall_x is None:
        pass
    elif bird_x < wall_x:
        result['section_x'] = 'left'
    else:
        result['section_x'] = 'right'


    return result


def assign_section_ys(frame):
    """
    Assign sections to birds based on y-coordinate averages.
    
    Parameters
    ----------
    result : dict
        Dictionary containing bounding box information.
    
    Returns
    -------
    dict
        Updated dictionary with section_y information.
    """
    updated_result = {}
    for key, value in frame.items():
        if value['class'] == 'bird':
            y_avg = (value['y1'] + value['y2']) / 2
            if y_avg < 240:
                value['section_y'] = 'top'
            elif 240 <= y_avg < 420:
                value['section_y'] = 'middle'
            else:
                value['section_y'] = 'bottom'
        updated_result[key] = value
    return updated_result

def assign_section_xs(result):
    """
    Assign sections to birds based on x-coordinate averages relative to the wall x-coordinate.
    
    Parameters
    ----------
    result : dict
        Dictionary containing bounding box information.
    
    Returns
    -------
    dict
        Updated dictionary with section_x information.
    """
    updated_result = {}
    for key, value in result.items():
        if value['class'] == 'bird':
            wall_x = next((value['x1'] + value['x2']) / 2 for key, value in result.items() if value['class'] == 'wall')
            x_avg = (value['x1'] + value['x2']) / 2
            if x_avg < wall_x:
                value['section_x'] = 'left'
            else:
                value['section_x'] = 'right'
        updated_result[key] = value
    return updated_result


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


