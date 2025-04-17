from utils.frames import *
from utils.perching_functions import *
from utils.movement_functions import *
from utils.sections import *
from utils.quality_functions import *

from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from IPython.display import display



def extract(video_id: str, video_directory: str, window_size_mean: int = 3, window_size_mode: int = 31):
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
    window_size_mean : int
        Sliding mean window size. Must be odd.
    window_size_mode : int
        Sliding mode window size. Must be odd.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing features.
    """
    ##################################################################################################################
    # TODO: Store frame count, fps, frame widht+height, and video name in the json file. Then we don't need to load the video here anymore.
    # video_path = f"{video_directory}/{video_id}_exploration_IB.mp4"
    # # Get video
    # vcap = load_video(video_path)
    # # Get frame count and fps of video
    # frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = vcap.get(cv2.CAP_PROP_FPS)
    ##################################################################################################################

    # Raw data is saved in a json file
    json_filepath = f"data/raw_data/{video_id}_exploration_IB.json"
    data_full = load_json_to_dict(json_filepath)
    frame_count = int(data_full["frame_count"])
    fps = int(data_full["fps"])
    data_full = data_full["frames"]
    

    # Extract bird and wall position from json file
    bird_x, bird_y, wall_x, wall_y = extract_bird_and_wall_coordinates(data_full, frame_count)
    # Apply sliding average and data imputation to bird and wall positions
    bird_x = sliding_mean(impute_data(bird_x), window_size_mean)
    bird_y = sliding_mean(impute_data(bird_y), window_size_mean)
    wall_x = sliding_mean(impute_data(wall_x), window_size_mean)
    wall_y = sliding_mean(impute_data(wall_y), window_size_mean)
    
    # Extract data related to perches and sections
    df_perches, df_sections, actions, quality_metrics = extract_from_data(data_full, video_id, frame_count, fps, bird_x, bird_y, wall_x)
    
    # Remove elements which correspond to unknown action <- what, why?
    action_mask = actions != 0
    actions = actions[action_mask]
    # Array indicating the side of the cage that the bird is in
    sides = determine_side(bird_x, wall_x)[action_mask]
    # Take sliding mode of actions array
    actions = sliding_mode(actions, window_size_mode)
    # Count the number of hops in home and exploration sides
    hops_home, hops_expl = count_hops(actions, sides)

    # Compute total travelled distance
    d = compute_distance(bird_x, bird_y)

    # Join DataFrames
    df_out = df_perches.join(df_sections)
    # Add remaining features to DataFrame
    df_out["d"] = d
    df_out["hops_home"] = hops_home
    df_out["hops_expl"] = hops_expl

    #print(len(bird_x))
    #print(bird_x)
    #print(bird_x[0])

    #print(sides)
    #print(len(sides))
    #print(sides[0])

    return df_out, quality_metrics


def extract_from_data(data:dict, video_id:str, nframes:int, fps:int, bird_x:np.array, bird_y:np.array, wall_x:np.array):
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
    # Initialize perch and wall coordinates
    # The set tracks which perches (1-5) have been visited
    p_xs, p_ys, five_perches_set = initialize_coordinates(video_name)
    # Initialize array of fence for the last 10 frames
    cage_status = np.empty(10)
    cage_status[:] = np.nan

    # For computing 5perches:
    five_perches_timer = 0 # for tracking time spent of left side    

    ####################
    # Quality measures #
    ####################
    camera_movement = False
    perch_coordinates_movement = np.nan
    number_of_manually_annotated_perches = np.count_nonzero(~np.isnan(p_xs[0, :5]))
    close_perches = perches_within_threshold(p_xs[0, :5])
    inbetween_zones = 0
    inbetween_perches_2_3 = 0

    ######################################
    # Initialization related to sections #
    ######################################
    total_frame_counts = {'top': 0, 'middle': 0, 'bottom': 0, 'left': 0, 'right': 0}


    ####################
    # Loop over frames #
    ####################
    for ind, frame in enumerate(tqdm(data.values(), desc="Frames analysed", total=nframes, mininterval=1)):

        ###################################
        # Infering information on perches #
        ###################################
        # Get coordinates
        new_bird_x = bird_x[ind]
        new_bird_y = bird_y[ind]
        new_wall_x = wall_x[ind]
        new_p_xs, new_p_ys, on_cage = extract_coordinates(frame)

        # If no perches are found (camera has fallen), continue loop without doing anything
        if len(new_p_xs) == 0:
            camera_movement = True
            continue
        # Update coordinates of each perch into moving arrays
        p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, ind%10)

        # Update cage status
        cage_status[ind%10] = on_cage
        # Find if the bird is on a perch/fence, and which one
        bird_action = find_bird_on_perch(px_avgs, py_avgs, new_bird_x, new_bird_y, cage_status)
        
        # check if perch is 2 or 3 and apply special logic
        if bird_action == 2 or bird_action == 3:
            exploration_perches = identify_and_number_exploration_perches(frame, new_wall_x)
            p2 = exploration_perches[1] # perch2
            p3 = exploration_perches[2] # perch3

            # Notice that y = 0 in the top and increses as you go down the frame so y1 < y2.
            # Check if new_bird_y is above the middle of stick3
            if new_bird_y < p3['center_y']:
                dist2 = abs(new_bird_x - p2['x1'])
                dist3 = abs(new_bird_x - p3['x1'])
                if dist2 < dist3:
                    bird_action = 2
                else:
                    bird_action = 3
            # Check if bird_y is below middle of stick3
            elif new_bird_y >= p3['center_y']:
                dist2 = abs(new_bird_x - p2['x2'])
                dist3 = abs(new_bird_x - p3['x2'])
                if dist2 < dist3:
                    bird_action = 2
                else:
                    bird_action = 3

        # Detect camera movement
        if ind+1 % 100 == 0 and not camera_movement:
            if np.isnan(perch_coordinates_movement):
                perch_coordinates_movement = px_avgs.copy()
            camera_movement = detect_movement(perch_coordinates_movement, px_avgs, new_p_xs)
            perch_coordinates_movement = px_avgs.copy()
        # Detect if bird is on the threshold between two different sections
        if bird_inbetween_sections(frame, new_bird_y):
            inbetween_zones += 1
        # Detect if the bird is between perches 2 and 3
        if bird_between_perch_2_3(px_avgs, py_avgs, new_bird_x, new_bird_y, cage_status):
            inbetween_perches_2_3 += 1

        # Update property arrays
        perching_result[ind] = bird_action
        all_px_avgs[ind] = px_avgs
        # Computation of 5perches:
        # Remove perch number from reference set (does nothing if number is not present)
        five_perches_set.discard(bird_action)
        # If all five have been visited, the set is empty and timer stops
        if len(five_perches_set) > 0:
            # If bird is on the left side, increase frame counter by one
            if new_bird_x < new_wall_x:
                five_perches_timer += 1

        #####################
        # Infering sections #
        #####################
        # Assign sections

        # For debug:
        # display({"bird_x": new_bird_x, "bird_y": new_bird_y, "wall_x_avg": wx_avg})
        
        if pd.isna(new_wall_x):
            # If wall_x_avg is NaN, skip this frame <- should not happen with imputed values
            continue

        sections = assign_section(frame, new_bird_x, new_bird_y, new_wall_x)
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

    ###################
    # Quality results #
    ###################

    quality_values = {
        "camera_movement": camera_movement,
        "perch_count": number_of_manually_annotated_perches,
        "close_perches": close_perches,
        "bird_inbetween_zones": inbetween_zones/30,
        "bird_inbetween_perches": inbetween_perches_2_3/30
    }
    quality_metrics = pd.DataFrame(data=quality_values, index=[0])

    df_sections = pd.DataFrame(time_by_section, index=[0])

    return df_perches_out, df_sections, perching_result, quality_metrics


if __name__ == "__main__":
    #video_id = "CAGE_020720_HA70343"
    #video_id = "CAGE_030720_HA70344"
    video_id = "CAGE_030720_HA70345"

    #video_id = "CAGE_050721_HA70384" # Large camera movement
    #video_id = "CAGE_220520_HA70337" # Small camera movement
    #video_id = "HE21359_100721_21OW7" # Very slight camera movement over time
    #video_id = "HE21365_110721_21NB23" # Small camera movement
    #video_id = "CAGE_100720_HA70355"
    #vidoe_id = "CAGE_200520_HA70335"
    
    video_directory = "data/original_videos"
    results, quality_metrics = extract(video_id, video_directory)
    print(quality_metrics)

