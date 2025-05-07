from utils.movement_functions import extract_bird_and_wall_coordinates, sliding_mean, impute_data, compute_distance, count_hops, determine_side, sliding_mode
from utils.perching import identify_and_number_perches, initialize_coordinate_arrays, update_perch_coordinates, find_bird_on_perch, bird_on_fence, bird_on_perch_2_or_3, compute_perch_durations, compute_ground_and_fence
from utils.sections import assign_section, assign_section_from_wall, convert_frame_counts_to_time
from utils.quality_functions import detect_movement, bird_inbetween_sections, bird_between_perch_2_3, perches_within_threshold

import numpy as np
import pandas as pd
from tqdm import tqdm
import os # Added import for os module


# Extracts features from a given JSON file and saves them to a CSV file.
#################################################################################################
# feature_name | type       | description                                                       #
# ----------------------------------------------------------------------------------------------#
# latency	    duration    first time entering new area                                        #
# 5perches	    duration	time from entering to visit the 5th perch                           #
# ground	    duration	time spent on ground                                                #
# perch1	    duration	time spent on perch1	                                            #   
# perch2	    duration	time spent on perch2                                                #
# perch3	    duration	time spent on perch3                                                # 
# perch4	    duration	time spent on perch4                                                #
# perch5	    duration	time spent on perch5                                                # 
# movements	    event	    number of hops and flights in novel area                            #
# back_home	    event	    first time when going back to home side after entering novel area   #
# T_new	        duration	total time spent in new area                                        #
# move_home	    event	    number of movements in home side whole period                       #
# Top	        duration	Top part cage                                                       #
# Middle	    duration	Middle part cage                                                    #
# Bottom	    duration	Bottom part cage                                                    #
# Fence	        duration	Bird against mesh                                                   #
#################################################################################################

def extract_features(data_raw:dict,
                     window_size_mean:int,
                     window_size_mode:int,
                     fps:int,
                     frame_count:int,
                     ) -> pd.DataFrame:

    """
    Extracts features from the raw data and returns a DataFrame with the results.
    
    Parameters
    ----------
    data_raw : dict
        Dictionary containing the raw data.
    window_size_mean : int
        Size of the sliding window for averaging.
    window_size_mode : int
        Size of the sliding window for mode calculation.
    fps : int
        Frames per second of the video.
    frame_count : int
        Number of frames in the video.
    bird_x : np.array
        Array containing the x-coordinates of the bird.
    bird_y : np.array
        Array containing the y-coordinates of the bird.
    wall_x : np.array
        Array containing the x-coordinates of the wall.
    wall_y : np.array
        Array containing the y-coordinates of the wall.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame containing the extracted features.
    quality_metrics : pd.DataFrame
        DataFrame containing the quality metrics.
    """

    # Extract bird and wall position from json file
    bird_x, bird_y, wall_x, wall_y, wall_y1, wall_y2 = extract_bird_and_wall_coordinates(data_raw["frames"], frame_count)
    # Apply sliding average and data imputation to bird and wall positions
    bird_x = sliding_mean(impute_data(bird_x), window_size_mean)
    bird_y = sliding_mean(impute_data(bird_y), window_size_mean)
    wall_x = sliding_mean(impute_data(wall_x), window_size_mean)
    wall_y = sliding_mean(impute_data(wall_y), window_size_mean)
    wall_y1 = sliding_mean(impute_data(wall_y1), window_size_mean)
    wall_y2 = sliding_mean(impute_data(wall_y2), window_size_mean)


    ##########################
    # Perches initialization #
    ##########################
    # Check 30 first frames and try to identify perches.
    # Requirement is that we find 5 distinct perches on left side of wall and 3 on the right side.  
    try_n = 1
    while try_n < 30:
        try: 
            initial_perch_positions = identify_and_number_perches(data_raw['frames'][f"frame{try_n}"], wall_x[try_n-1])
            perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center = initialize_coordinate_arrays(initial_perch_positions)
            break
        except ValueError as e:
            print(f"Unable to identify all 8 perches from frame{try_n}: {e}. Retrying...")
            try_n += 1
            continue
    if try_n == 30:
        raise ValueError("Unable to identify all 8 perches after 30 frames. Please check the data.")

    # Create empty arrays for storing intermediate results
    perching_result = np.zeros(frame_count) # final result: 1-8 for each of 8 perches, -1 for fence, -2 for floor, 0 for other

    # Initialize perch and wall coordinates
    # The set tracks which perches (1-5) have been visited
    five_perches_set = {1, 2, 3, 4, 5}

    # For computing 5perches:
    five_perches_timer = 0

    # Initialize array of fence for the last 10 frames
    fence_status = np.empty(10)
    fence_status[:] = np.nan

    ###########################
    # Sections initialization #
    ###########################
    total_frame_counts = {'top': 0, 'middle': 0, 'bottom': 0, 'T_new': 0, 'T_home': 0}
    first_entry_frame = -1 # Track the frame when the bird first enters the new area 
    back_home_frame = -1 # Track the frame when the bird first goes back to the home side
    entered_exploration = False

    ###################################
    # Quality measures initialization #
    ###################################
    camera_movement = False
    perch_coordinates_movement = np.nan
    number_of_manually_annotated_perches = np.count_nonzero(~np.isnan(perches_x_center[0, :5]))
    close_perches = perches_within_threshold(perches_x_center[0, :5])
    inbetween_zones = 0
    inbetween_perches_2_3 = 0

    ####################
    # Loop over frames #
    ####################
    for ind, frame in enumerate(tqdm(data_raw['frames'].values(), desc="Analyzing frames", total=frame_count, mininterval=1)):

        ###########
        # Perches #
        ###########
        # Get coordinates
        new_bird_x = bird_x[ind]
        new_bird_y = bird_y[ind]
        new_wall_x = wall_x[ind]
        new_wall_y1 = wall_y1[ind]
        new_wall_y2 = wall_y2[ind]
        
        numbered_perches = identify_and_number_perches(frame, new_wall_x)

        # Update coordinates of perches
        perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center = update_perch_coordinates(
            perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center, numbered_perches, ind)
        
        # Moving averages
        px_moving_avgs = np.nanmean(perches_x_center, axis=0)
        py_moving_avgs = np.nanmean(perches_y_center, axis=0)
        py2_moving_avgs = np.nanmean(perches_y2, axis=0)
        
        # Update fence status
        fence_status[ind%10] = bird_on_fence(frame)

        # Find if the bird is on a perch/fence, and which one
        bird_action = find_bird_on_perch(px_moving_avgs, py2_moving_avgs, new_bird_x, new_bird_y, fence_status)
        
        # check if perch is 2 or 3 and apply special logic
        if bird_action == 2 or bird_action == 3:
            bird_action = bird_on_perch_2_or_3(new_bird_x, new_bird_y, perches_x1, perches_y1, perches_x2, perches_y2, perches_x_center, perches_y_center)

        # Update property arrays
        perching_result[ind] = bird_action
        
        # Computation of 5perches:
        # Remove perch number from reference set (does nothing if number is not present)
        five_perches_set.discard(bird_action)
        # If all five have been visited, the set is empty and timer stops
        if len(five_perches_set) > 0:
            # If bird is on the left side, increase frame counter by one
            if new_bird_x < new_wall_x:
                five_perches_timer += 1

        ############
        # Sections #
        ############
        # Assign sections
        # sections = assign_section(perches_y1, perches_y2, new_bird_x, new_bird_y, new_wall_x)
        sections = assign_section_from_wall(new_bird_x, new_bird_y, new_wall_x, new_wall_y1, new_wall_y2)
        for section in sections.values():
            total_frame_counts[section] += 1
            # Check if bird is in the new area
            if section == 'T_new':
                if first_entry_frame == -1:
                    first_entry_frame = ind+1 # Record first entry frame index
                    entered_exploration = True # Mark that bird has been in exploration side
            
            # Check for back_home condition (first transition L->R after entering L)
            elif section == 'T_home': 
                if entered_exploration and back_home_frame == -1:
                    back_home_frame = ind+1

        ####################
        # Quality measures #
        ####################
        # Detect camera movement
        if (ind + 1) % 100 == 0 and not camera_movement:
            if np.all(np.isnan(perch_coordinates_movement)):
                perch_coordinates_movement = px_moving_avgs.copy()
            camera_movement = detect_movement(perch_coordinates_movement, px_moving_avgs, perches_x_center)
            perch_coordinates_movement = px_moving_avgs.copy()
        # Detect if bird is on the threshold between two different sections
        if bird_inbetween_sections(frame, new_bird_y):
            inbetween_zones += 1
        # Detect if the bird is between perches 2 and 3
        if bird_between_perch_2_3(px_moving_avgs, py_moving_avgs, new_bird_x, new_bird_y, fence_status):
            inbetween_perches_2_3 += 1

    ####################
    # Perching results #
    ####################

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
    df_perches = pd.DataFrame(data = features_perches, index = [0])


    ####################
    # Sections results #
    ####################
    # Convert frame counts to time
    time_by_section = convert_frame_counts_to_time(total_frame_counts)
    df_sections = pd.DataFrame(time_by_section, index=[0])
    
    # Latency
    df_sections['latency'] = first_entry_frame / fps if first_entry_frame != -1 else np.nan

    # Back Home (Event: 1 if occurred, 0 if not)
    df_sections['back_home'] = back_home_frame / fps if back_home_frame != -1 else np.nan


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


    
    # Remove elements which correspond to unknown action <- what, why?
    action_mask = perching_result != 0
    actions = perching_result[action_mask]
    # Array indicating the side of the cage that the bird is in
    sides = determine_side(bird_x, wall_x)[action_mask]
    # Take sliding mode of actions array
    actions = sliding_mode(actions, window_size_mode)
    # Count the number of hops in home and exploration sides
    hops_home, hops_expl = count_hops(actions, sides)

    # Compute total travelled distance
    #d = compute_distance(bird_x, bird_y)

    # Join DataFrames
    df_out = df_perches.join(df_sections)
    # Add remaining features to DataFrame
    #df_out["d"] = d
    df_out["move_home"] = hops_home
    df_out["movements"] = hops_expl

    return df_out, quality_metrics


def save_features_to_csv(features_df: pd.DataFrame, 
                         quality_df: pd.DataFrame, 
                         base_filename: str, 
                         output_dir: str):
    """
    Saves the extracted features and quality metrics DataFrames to CSV files.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing the extracted features.
    quality_df : pd.DataFrame
        DataFrame containing the quality metrics.
    base_filename : str
        The base name for the output files (e.g., derived from the input video/JSON name).
    output_dir : str
        The directory where the CSV files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct file paths
    features_filepath = os.path.join(output_dir, f"{base_filename}_features.csv")
    quality_filepath = os.path.join(output_dir, f"{base_filename}_quality.csv")

    # Save DataFrames to CSV
    try:
        features_df.to_csv(features_filepath, index=False)
        print(f"Saved features to: {features_filepath}")
    except Exception as e:
        print(f"Error saving features to {features_filepath}: {e}")

    try:
        quality_df.to_csv(quality_filepath, index=False)
        print(f"Saved quality metrics to: {quality_filepath}")
    except Exception as e:
        print(f"Error saving quality metrics to {quality_filepath}: {e}")


