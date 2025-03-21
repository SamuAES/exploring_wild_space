from frames import load_video, read_video
from perching_functions import *
from movement_functions import *

import pandas as pd
import numpy as np
import warnings
import cv2

def extract(video_id: str, model_path: str, frame_count: int = np.inf, w: int = 3):
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
    w : int
        Moving average windows size. Must be odd.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing features.
    """
    video_name = f"{video_id}_exploration_IB"
    video_path = f"../../videos/{video_id}_exploration_IB.mp4"
    # Get video
    vcap = load_video(video_path)
    # Get frame count and fps of video
    max_frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    # Check frame count
    if frame_count > max_frame_count:
        msg = f"Video {video_id} contains only {max_frame_count} frames."
        warnings.warn(msg)
        frame_count = max_frame_count

    # TODO: The data can either be extract from video on the fly, or read from disk
    df_raw = extract_from_video(video_id, model_path, frame_count)
    #df_raw = pd.read_csv(f"../../data_yolo_nano_v2/{video_id}_exploration_IB.csv")[:frame_count]
    # Apply data imputation and sliding average
    df = preprocess_data(df_raw, w)
    # Compute total travelled distance
    d = compute_distance(df["birdX"], df["birdY"])
    # Compute speed of the bird
    v = compute_speed(df["birdX"], df["birdY"], 1 / fps)
    # Compute number of "hops" on home and exploration sides
    # TODO: Adjust the threshold to a suitable value (determines
    # how much the bird needs to move for the 'hop' to be counted)
    threshold = 200
    hops_home, hops_expl = count_threshold_crossings_sidewise(v, df["birdX"], np.mean(df["wallX"]), threshold)

    # Extract features related to perches
    results = read_video(video_capture = vcap, model_path = model_path)
    df_out = extract_perches(results, video_name, fps, frame_count)

    # Add remaining feature to DataFrame
    df_out["d"] = d
    df_out["hops_home"] = hops_home
    df_out["hops_expl"] = hops_expl

    return df_out

def extract_from_video(video_id, model_path, nframes):
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
    video_path = f"../../videos/{video_id}_exploration_IB.mp4"
    video_capture = load_video(video_path)
    generator = read_video(video_capture = video_capture,
                           model_path = model_path)
    
    # Initialize arrays
    bird_x = np.zeros(nframes)
    bird_y = np.zeros(nframes)
    wall_x = np.zeros(nframes)
    wall_y = np.zeros(nframes)
    numsticks = np.zeros(nframes)

    # Loop over frames
    # TODO: Check how the bounding box coordinates are defined
    for ind, frame in enumerate(generator):

        if ind == nframes:
            break

        bird = [dict for dict in frame.values() if dict["class"] == "bird"]
        if len(bird) == 1:
            bird_x[ind] = int((bird[0]["x1"] + bird[0]["x2"]) / 2)
            bird_y[ind] = int((bird[0]["y1"] + bird[0]["y2"]) / 2)
        else:
            bird_x[ind] = np.nan
            bird_y[ind] = np.nan

        wall = [dict for dict in frame.values() if dict["class"] == "wall"]
        if len(wall) == 1:
            wall_x[ind] = int((wall[0]["x1"] + wall[0]["x2"]) / 2)
            wall_y[ind] = int((wall[0]["y1"] + wall[0]["y2"]) / 2)
        else:
            wall_x[ind] = np.nan
            wall_y[ind] = np.nan

        sticsk = [dict for dict in frame.values() if dict["class"] == "stick"]
        numsticks[ind] = len(sticsk)

    features = {"birdX": bird_x,
                "birdY": bird_y,
                "wallX": wall_x,
                "wallY": wall_y,
                "numsticks": numsticks}
    
    df_out = pd.DataFrame(data = features)

    #df_out.to_csv(f"../../data_yolo_nano_v2/{video_id}_exploration_IB_test.csv", index = False, float_format = "%.0f")

    return df_out

def preprocess_data(df: pd.DataFrame, w: int):
    """
    Processes data in 'df' by imputing missing values and applying a moving average.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing unprocessed data.
    w : int
        Moving average window size. Must be odd.
    
    Returns : pd.DataFrame
        New DataFrame containing processed data.
    """
    # Select which columns to process
    columns = ["birdX", "birdY", "wallX", "wallY"]
    df_new = pd.DataFrame()
    for col in columns:
        data = df[col].to_numpy()
        data = sliding_average(impute_data(data), w)
        df_new[col] = data
    return df_new

def extract_perches(results, video_name, fps, frame_count):
    # Create empty arrays for storing intermediate results
    perching_result = np.zeros(frame_count) # final result: 1-8 for each of 8 perches, -1 for cage, -2 for floor, 0 for other
    all_px_avgs = np.zeros((frame_count, 8))
    all_bird_xs = np.zeros(frame_count)
    all_bird_ys = np.zeros(frame_count)
    all_wall_xs = np.zeros(frame_count)
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

    i = 0
    for result in results:
        # Get coordinates
        new_bird_x, new_bird_y, new_p_xs, new_p_ys, new_w_x, on_cage = extract_coordinates(result)
        # If no perches are found (camera has fallen), continue loop without doing anything
        if len(new_p_xs) == 0:
            continue
        # Update x-coordinate of bird if it is found
        if new_bird_x is not None:
            bird_x = new_bird_x
            bird_y = new_bird_y
        # Update coordinates of each perch into moving arrays
        p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, i%10)
        # Update coordinates of middle wall
        w_xs, wx_avg = update_wall_coordinates(w_xs, new_w_x, i%10)
        # Update cage status
        cage_status[i%10] = on_cage
        # Find if the bird is on a perch/fence, and which one
        bird_action = find_bird_on_perch(px_avgs, py_avgs, bird_x, bird_y, cage_status)
        # Update property arrays
        perching_result[i] = bird_action
        all_bird_xs[i] = bird_x
        all_bird_ys[i] = bird_y
        all_px_avgs[i] = px_avgs
        all_wall_xs[i] = wx_avg
        # Computation of 5perches:
        # Remove perch number from reference set (does nothing if number is not present)
        five_perches_set.discard(bird_action)
        # If all five have been visited, the set is empty and timer stops
        if len(five_perches_set) > 0:
            # If bird is on the left side, increase frame counter by one (can be replaced with the 'side' feature)
            if bird_x < wx_avg:
                five_perches_timer += 1
        i += 1
        if i == frame_count: break

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
    features = {
            "perch1": perch1,
            "perch2": perch2,
            "perch3": perch3,
            "perch4": perch4,
            "perch5": perch5,
            "5perches": five_perches,
            "fence": fence,
            "ground": ground
    }
    return pd.DataFrame(data = features, index = [0])

video_id = "CAGE_220520_HA70339"
model_path = "custom_yolo11n_v2.pt"
frame_count = 5 # How many frames to analyse

res = extract(video_id, model_path, frame_count)
res.to_csv(f"{video_id}_features2.csv", header = True, index = False)