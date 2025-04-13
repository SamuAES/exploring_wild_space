# Import necessary libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import json # For loading data

# Import custom utility functions
# Note: utils2.py is now inside the utils folder
from utils.utils2 import (
    get_best_detection,
    identify_and_number_exploration_perches,
    find_bird_location,
    calculate_sections,
    sliding_mean,
    sliding_mode,
    impute_data
)
# Note: Removed old utils imports, cv2, display



# Define constants if needed, e.g., thresholds
HORIZONTAL_THRESHOLD_PERCH = 50

def load_json_to_dict(filepath):
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None


def extract_features(video_id: str, data_dir: str = "data", window_size_mean: int = 3, window_size_mode: int = 31):
    """
    Extracts bird behaviour features from pre-processed JSON data.

    Parameters
    ----------
    video_id : str
        The identifier for the video (e.g., "CAGE_020720_HA70343").
    data_dir : str, optional
        The base directory containing 'raw_data', by default "data".
    window_size_mode : int, optional
        Sliding mode window size for smoothing location data. Must be odd. By default 31.

    Returns
    -------
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing the calculated features for the video.
        Returns None if data loading fails.
    """
    json_filepath = f"{data_dir}/raw_data/{video_id}_exploration_IB.json"
    raw_data = load_json_to_dict(json_filepath)
    if raw_data is None:
        return None

    # Extract metadata and frame data
    try:
        frame_count = int(raw_data["frame_count"])
        fps = int(raw_data["fps"])
        # frame_width = int(raw_data["frame_width"]) # Available if needed
        # frame_height = int(raw_data["frame_height"]) # Available if needed
        frames_data = raw_data["frames"]
        if len(frames_data) != frame_count:
             print(f"Warning: Number of frames in data ({len(frames_data)}) does not match metadata frame_count ({frame_count}). Using length of data.")
             frame_count = len(frames_data)
    except KeyError as e:
        print(f"Error: Missing metadata key in JSON: {e}")
        return None
    except ValueError:
        print("Error: Invalid metadata format (frame_count, fps).")
        return None

    if fps <= 0:
        print("Error: FPS must be positive.")
        return None

    # --- Loop through frames and collect data ---
    # Initialize variables to store coordinates
    bird_coordinates = {"x1": [], "y1": [], "x2": [], "y2": []}
    wall_coordinates = {"x1": [], "y1": [], "x2": [], "y2": []}
    fence_detections = []
    perch_coordinates = {"perch1": {"x1": [], "y1": [], "x2": [], "y2": [], 'center_x': [], 'center_y': []},
                        "perch2": {"x1": [], "y1": [], "x2": [], "y2": [], 'center_x': [], 'center_y': []},
                        "perch3": {"x1": [], "y1": [], "x2": [], "y2": [], 'center_x': [], 'center_y': []},
                        "perch4": {"x1": [], "y1": [], "x2": [], "y2": [], 'center_x': [], 'center_y': []},
                        "perch5": {"x1": [], "y1": [], "x2": [], "y2": [], 'center_x': [], 'center_y': []}}
    
    last_known_wall_x = None # Initialize to None, will be updated when wall detected
    for i in tqdm(range(frame_count), desc="Collecting coordinates"):
        frame_key = f"frame{i + 1}"
        frame_data = frames_data.get(frame_key, {}) # Get data for current frame, empty dict if missing

        # Get Detections
        bird_detection = get_best_detection(frame_data, 'bird')
        wall_detection = get_best_detection(frame_data, 'wall')
        fence_detections.append(True if get_best_detection(frame_data, 'fence') else False)

        # Store Coordinates
        bird_coordinates["x1"].append(bird_detection['x1'] if bird_detection else np.nan)
        bird_coordinates["y1"].append(bird_detection['y1'] if bird_detection else np.nan)
        bird_coordinates["x2"].append(bird_detection['x2'] if bird_detection else np.nan)
        bird_coordinates["y2"].append(bird_detection['y2'] if bird_detection else np.nan)

        wall_coordinates["x1"].append(wall_detection['x1'] if wall_detection else np.nan)
        wall_coordinates["y1"].append(wall_detection['y1'] if wall_detection else np.nan)
        wall_coordinates["x2"].append(wall_detection['x2'] if wall_detection else np.nan)
        wall_coordinates["y2"].append(wall_detection['y2'] if wall_detection else np.nan)

        # Identify Perches
        # Pass current_wall_x which might be None if wall never detected
        if wall_detection:
            current_wall_x = (wall_detection['x1'] + wall_detection['x2']) / 2
            last_known_wall_x = current_wall_x
        else:
            current_wall_x = last_known_wall_x # Use last known if current is missing
        
        exploration_perches = identify_and_number_exploration_perches(frame_data, current_wall_x)
        # Store Perch Coordinates
        # Note: exploration_perches is a list of dicts with keys 'number', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y'
        for perch in exploration_perches:
            perch_num = perch['number']
            perch_coordinates[f"perch{perch_num}"]["x1"].append(perch['x1'])
            perch_coordinates[f"perch{perch_num}"]["y1"].append(perch['y1'])
            perch_coordinates[f"perch{perch_num}"]["x2"].append(perch['x2'])
            perch_coordinates[f"perch{perch_num}"]["y2"].append(perch['y2'])
            perch_coordinates[f"perch{perch_num}"]["center_x"].append(perch['center_x'])
            perch_coordinates[f"perch{perch_num}"]["center_y"].append(perch['center_y'])
        
    # --- Smoothe coordinates ---
    smoothed_bird_coordinates = {}
    for key in bird_coordinates:
        smoothed_bird_coordinates[key] = sliding_mean(np.array(bird_coordinates[key]), window_size_mean)

    smoothed_wall_coordinates = {}
    for key in wall_coordinates:
        smoothed_wall_coordinates[key] = sliding_mean(np.array(wall_coordinates[key]), window_size_mean)
    
    smoothed_perch_coordinates = {}
    for perch_num in perch_coordinates:
        smoothed_perch_coordinates[perch_num] = {}
        for key in perch_coordinates[perch_num]:
            smoothed_perch_coordinates[perch_num][key] = sliding_mean(np.array(perch_coordinates[perch_num][key]), window_size_mean)

    # --- Loop through smoothed coordinates ---
    locations = []
    sides = []
    vertical_sections = []
    visited_perches_set = set()
    five_perches_timer_frames = 0
    first_entry_frame = -1 # Use -1 to indicate not yet entered
    back_home_frame = -1   # Use -1 to indicate not yet returned
    entered_exploration = False
    last_known_wall_x = None

    print(f"Processing {frame_count} frames for video {video_id}...")
    # Use range(frame_count) to iterate based on metadata count
    for i in tqdm(range(frame_count), desc="Analyzing frames"):
    
        # --- Get Detections ---
        bird_bbox = {'x1': smoothed_bird_coordinates['x1'][i],
                     'y1': smoothed_bird_coordinates['y1'][i],
                     'x2': smoothed_bird_coordinates['x2'][i],
                     'y2': smoothed_bird_coordinates['y2'][i]}

        wall_bbox = {'x1': smoothed_wall_coordinates['x1'][i],
                     'y1': smoothed_wall_coordinates['y1'][i],
                     'x2': smoothed_wall_coordinates['x2'][i],
                     'y2': smoothed_wall_coordinates['y2'][i]}
        
        if wall_bbox['x1'] is not None and wall_bbox['x2'] is not None:
            current_wall_x = (wall_bbox['x1'] + wall_bbox['x2']) / 2
            last_known_wall_x = current_wall_x
        else:
            current_wall_x = last_known_wall_x # Use last known if current is missing
        
        fence_detected = fence_detections[i]

        exploration_perches = []
        for perch_num in range(1, 6):
            perch = {
                'number': perch_num,
                'x1': smoothed_perch_coordinates[f'perch{perch_num}']['x1'][i],
                'y1': smoothed_perch_coordinates[f'perch{perch_num}']['y1'][i],
                'x2': smoothed_perch_coordinates[f'perch{perch_num}']['x2'][i],
                'y2': smoothed_perch_coordinates[f'perch{perch_num}']['y2'][i],
                'center_x': smoothed_perch_coordinates[f'perch{perch_num}']['center_x'][i],
                'center_y': smoothed_perch_coordinates[f'perch{perch_num}']['center_y'][i]
            }
            exploration_perches.append(perch)

        # Skip frame if wall position is needed but never found
        if current_wall_x is None and i > 0: # Allow first frame potentially
             print(f"Warning: Skipping frame {i+1} due to missing wall detection and no prior known position.")
             # Append default/missing values for this frame
             locations.append('missing')
             sides.append('unknown')
             vertical_sections.append('unknown')
             continue # Skip rest of processing for this frame
        elif current_wall_x is None and i == 0:
             print(f"Warning: Wall not detected in first frame {i+1}. Perch/section identification might be inaccurate.")
             # Proceed, but perch/section results might be empty/unknown


        # --- Determine Location and Sections ---
        current_location = find_bird_location(bird_bbox, exploration_perches, fence_detected, HORIZONTAL_THRESHOLD_PERCH)
        sections = calculate_sections(bird_bbox, current_wall_x, exploration_perches)
        current_side = sections['horizontal']
        current_vertical = sections['vertical']

        # --- Store Frame Results ---
        locations.append(current_location)
        sides.append(current_side)
        vertical_sections.append(current_vertical)

        # --- Update State Variables ---
        if current_location.startswith('perch'):
            try:
                perch_num = int(current_location.replace('perch', ''))
                if 1 <= perch_num <= 5: # Only count exploration perches
                    visited_perches_set.add(perch_num)
            except ValueError:
                pass # Should not happen if find_bird_location works correctly

        if current_side == 'left':
            if first_entry_frame == -1:
                first_entry_frame = i # Record first entry frame index
            entered_exploration = True # Mark that bird has been in exploration side
            # Increment 5 perches timer only if not all 5 visited yet and bird is on left
            if len(visited_perches_set) < 5:
                five_perches_timer_frames += 1
        elif current_side == 'right':
            # Check for back_home condition (first transition L->R after entering L)
            if entered_exploration and back_home_frame == -1 and i > 0 and sides[i-1] == 'left':
                 back_home_frame = i

    # --- Apply Smoothing (Optional) ---
    location_map = {'missing':0, 'perch1':1, 'perch2':2, 'perch3':3, 'perch4':4, 'perch5':5, 'other':6, 'ground':7, 'fence':8}
    reverse_location_map = dict(map(reversed, location_map.items()))
    numeric_locations = np.array([location_map[loc] for loc in locations])
    smoothed_numeric_locations = sliding_mode(numeric_locations, window_size_mode)
    smoothed_locations = [reverse_location_map[num] for num in smoothed_numeric_locations]

    # --- Calculate Features ---
    print("Calculating final features...")
    feature_results = {}

    # Latency
    feature_results['latency'] = first_entry_frame / fps if first_entry_frame != -1 else np.nan

    # Back Home (Event: 1 if occurred, 0 if not)
    feature_results['back_home'] = 1 if back_home_frame != -1 else 0

    # Time-based features
    total_frames = len(smoothed_locations)
    feature_results['T_new'] = sum(1 for s in sides if s == 'left') / fps
    feature_results['T_home'] = sum(1 for s in sides if s == 'right') / fps
    feature_results['Top'] = sum(1 for vs in vertical_sections if vs == 'top') / fps
    feature_results['Middle'] = sum(1 for vs in vertical_sections if vs == 'middle') / fps
    feature_results['Bottom'] = sum(1 for vs in vertical_sections if vs == 'bottom') / fps
    feature_results['ground'] = sum(1 for loc in smoothed_locations if loc == 'ground') / fps
    feature_results['fence'] = sum(1 for loc in smoothed_locations if loc == 'fence') / fps

    for i in range(1, 6): # Perches 1 to 5
        feature_results[f'perch{i}'] = sum(1 for loc in smoothed_locations if loc == f'perch{i}') / fps

    # 5 Perches Time
    feature_results['5perches'] = five_perches_timer_frames / fps

    # Movement Counts (Hops)
    movements_expl = 0
    movements_home = 0
    if total_frames > 1:
        # Use smoothed_locations for hop counting
        prev_loc = smoothed_locations[0]
        prev_side = sides[0] # Use original side for attributing hops
        for i in range(1, total_frames):
            curr_loc = smoothed_locations[i]
            curr_side = sides[i]
            # Count hop if location changed (and not missing/unknown?)
            if curr_loc != prev_loc and prev_loc not in ['missing', 'other'] and curr_loc not in ['missing', 'other']:
                 # Attribute hop to the side where it started (previous frame's side)
                 if prev_side == 'left':
                     movements_expl += 1
                 elif prev_side == 'right':
                     movements_home += 1
            prev_loc = curr_loc
            prev_side = curr_side # Update previous side as well

    feature_results['movements'] = movements_expl # Exploration side movements
    feature_results['move_home'] = movements_home # Home side movements

    # Convert results to DataFrame
    df_out = pd.DataFrame([feature_results])

    return df_out


# Example usage:
if __name__ == "__main__":
    # Select the video ID to process
    video_id = "CAGE_030720_HA70345"

    # Specify the base data directory
    data_directory = "data"

    # Extract features
    # Pass window_size_mode if you want to change the default for smoothing
    feature_df = extract_features(video_id, data_dir=data_directory)

    # Print the results
    if feature_df is not None:
        print(f"\n--- Features for {video_id} ---")
        # Print DataFrame nicely
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(feature_df)
    else:
        print(f"Feature extraction failed for {video_id}.")
