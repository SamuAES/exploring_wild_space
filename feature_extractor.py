from utils.frames import *
from utils.perching_functions import *
from utils.movement_functions import *
from utils.sections import *

from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import cv2



def extract(video_id: str, video_directory: str, window_size: int = 3):
    """
    Extracts all features from specified video.

    Parameters
    ----------
    video_id : str
        The ID or the number of a video.
    model_path : str
        Path to a YOLO model.
    frame_count : int
        Number of frames to analyze. If not specified, all frames are analyzed.
    window_size : int
        Moving average window size. Must be odd.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing features.
    """
    ##################################################################################################################
    # TODO: Store frame count, fps, frame widht+height, and video name in the json file. Then we don't need to load the video here anymore.
    video_path = f"{video_directory}/{video_id}_exploration_IB.mp4"
    # Get video
    vcap = load_video(video_path)
    # Get frame count and fps of video
    frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    ##################################################################################################################



    # Raw data is saved in a json file
    json_filepath = f"data/raw_data/{video_id}_exploration_IB.json"
    data_full = load_json_to_dict(json_filepath)
    # Extract bird and wall position from json file
    bird_x, bird_y, wall_x, wall_y = extract_bird_and_wall_coordinates(json_filepath, frame_count)
    # Apply sliding average and data imputation to bird and wall positions
    bird_x = sliding_average(impute_data(bird_x), window_size)
    bird_y = sliding_average(impute_data(bird_y), window_size)
    wall_x = sliding_average(impute_data(wall_x), window_size)
    wall_y = sliding_average(impute_data(wall_y), window_size)

    
    
    # Extract data related to perches and sections
    df_perches, df_sections = extract_from_data(data_full, video_id, frame_count, fps)
    # Compute total travelled distance
    d = compute_distance(bird_x, bird_y)
    # Compute speed of the bird
    v = compute_speed(bird_x, bird_y, 1 / fps)

    # TODO: Count hops based on bird status instead of speed
    # Compute number of "hops" on home and exploration sides
    threshold = 200
    hops_home, hops_expl = count_threshold_crossings_sidewise(v, bird_x, np.mean(wall_x), threshold)

    # Join data frames
    df_out = df_perches.join(df_sections)    
    # Add remaining feature to DataFrame
    df_out["d"] = d
    df_out["hops_home"] = hops_home
    df_out["hops_expl"] = hops_expl

    return df_out

def extract_from_data(data:dict, video_id:str, nframes:int, fps:int):
    """
    Processes video in 'video_path' using model in 'model_path' and saves
    results (per frame coordinate information) to a .csv file or returns 
    the results as a DataFrame.

    Parameters
    ----------
    video_id : str
        ID of the video.
    model_path : str
        Path to the YOLO model.
    nframes : int
        Number of frames to process.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing location coordinates for the bird and the wall.
    """
    video_name = f"{video_id}_exploration_IB"
    
    
    ######################################
    # Initialization relateed to perches #
    ######################################
    # Create empty arrays for storing intermediate results
    perching_result = np.zeros(nframes) # final result: 1-8 for each of 8 perches, -1 for cage, -2 for floor, 0 for other
    all_px_avgs = np.zeros((nframes, 8))
    all_bird_xs = np.zeros(nframes)
    all_bird_ys = np.zeros(nframes)
    all_wall_xs = np.zeros(nframes)
    # Initialize perch and wall coordinates
    # The set tracks which perches (1-5) have been visited
    p_xs, p_ys, w_xs, five_perches_set = initialize_coordinates(video_name)
    # Initialize array of fence for the last 10 frames
    cage_status = np.empty(10)
    cage_status[:] = np.nan
    # Previous known bird location (center of its x coordinates)
    bird_x = np.nan
    # Previous known bird location (center of its y coordinates)
    bird_y = np.nan

    # For computing 5perches:
    five_perches_timer = 0 # for tracking time spent of left side    

    ############################################
    # Initialization related to movement count #
    ############################################
    bird_x_array = np.zeros(nframes)
    bird_y_array = np.zeros(nframes)
    wall_x_array = np.zeros(nframes)
    wall_y_array = np.zeros(nframes)
    numsticks_array = np.zeros(nframes)

    ######################################
    # Initialization related to sections #
    ######################################
    total_frame_counts = {'top': 0, 'middle': 0, 'bottom': 0, 'left': 0, 'right': 0}


    ####################
    # Loop over frames #
    ####################
    # TODO: Check how the bounding box coordinates are defined
    for ind, frame in enumerate(tqdm(data.values(), desc="Frames analysed", total=nframes, mininterval=1)):

        ###################################
        # Infering information on perches #
        ###################################
        # Get coordinates
        new_bird_x, new_bird_y, new_p_xs, new_p_ys, new_w_x, on_cage = extract_coordinates(frame)
        # If no perches are found (camera has fallen), continue loop without doing anything
        if len(new_p_xs) == 0:
            continue
        # Update x-coordinate of bird if it is found
        if new_bird_x is not None:
            bird_x = new_bird_x
            bird_y = new_bird_y
        # Update coordinates of each perch into moving arrays
        p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, ind%10)
        # Update coordinates of middle wall
        w_xs, wx_avg = update_wall_coordinates(w_xs, new_w_x, ind%10)
        # Update cage status
        cage_status[ind%10] = on_cage
        # Find if the bird is on a perch/fence, and which one
        bird_action = find_bird_on_perch(px_avgs, py_avgs, bird_x, bird_y, cage_status)
        # Update property arrays
        perching_result[ind] = bird_action
        all_bird_xs[ind] = bird_x
        all_bird_ys[ind] = bird_y
        all_px_avgs[ind] = px_avgs
        all_wall_xs[ind] = wx_avg
        # Computation of 5perches:
        # Remove perch number from reference set (does nothing if number is not present)
        five_perches_set.discard(bird_action)
        # If all five have been visited, the set is empty and timer stops
        if len(five_perches_set) > 0:
            # If bird is on the left side, increase frame counter by one (can be replaced with the 'side' feature)
            if bird_x < wx_avg:
                five_perches_timer += 1

        #####################
        # Infering sections #
        #####################
        # Assign sections
        sections = assign_section(frame, new_bird_x, new_bird_y, wx_avg)
        for section in sections.values():
            total_frame_counts[section] += 1
        #labeled_result = assign_section_ys(frame, new_bird_x, new_bird_y, wx_avg)
        #labeled_result = assign_section_xs(labeled_result)
        # Count frames by section
        #frame_counts = count_frames_by_section(labeled_result)
        # Update total frame counts
        #for section, count in frame_counts.items():
        #    total_frame_counts[section] += count


    #################################
    # Gathering results for perches #
    #################################
    # result codes:
    # -2 = on the floor
    # -1 = on the cage
    # 0 = other
    # 1-8 = on perch 1-8

    # Compute features. Boris helpdoc only asks for perches 1-5, but here are 6-8 as well
    perch1, perch2, perch3, perch4, perch5, perch6, perch7, perch8 = compute_perch_durations(perching_result, fps)
    ground, fence = compute_ground_and_fence(perching_result, fps)
    five_perches = five_perches_timer/fps
    # Turn features into dataframe
    features_perches = {
            "perch1": perch1,
            "perch2": perch2,
            "perch3": perch3,
            "perch4": perch4,
            "perch5": perch5,
            "5perches": five_perches,
            "fence": fence,
            "ground": ground
    }
    df_perches_out = pd.DataFrame(data = features_perches, index = [0])

    ##############################
    # Gathering results sections #
    ##############################
    # Convert frame counts to time
    time_by_section = convert_frame_counts_to_time(total_frame_counts)
    df_sections = pd.DataFrame(time_by_section, index=[0])

    return df_perches_out, df_sections


if __name__ == "__main__":
    video_id = "HE21355_090721_21NB21"
    video_directory = "data/original_videos"
    results = extract(video_id, video_directory)
    print(results)

