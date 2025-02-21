
def assign_section_ys(result):
    """
    Assign sections to birds based on y-coordinate averages.
    
    Parameters
    ----------
    result : dict
        Dictionary containing bounding box information.
    
    Returns
    -------
    dict
        Updated dictionary with section information.
    """
    updated_result = {}
    for key, value in result.items():
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
    frame_counts = {'top': 0, 'middle': 0, 'bottom': 0}
    for value in result.values():
        if 'section_y' in value:
            frame_counts[value['section_y']] += 1
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


