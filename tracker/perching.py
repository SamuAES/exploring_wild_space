from frames import load_video, read_video
import pandas as pd
import numpy as np
import copy
from perching_functions import *
import cv2

#video_number = "CAGE_020720_HA70343"
#video_number = "CAGE_030720_HA70344"
#video_number = "CAGE_030720_HA70345"
#video_number = "CAGE_050721_HA70384" # camera movement
#video_number = "CAGE_220520_HA70339" # bird on the floor
video_number = "CAGE_220520_HA70337"
#video_number = "HE21362_100721_21JJ32"

video_name = f"{video_number}_exploration_IB"
video_filepath = f"../../videos/{video_number}_exploration_IB.mp4" # relative filepath, adjust accordingly
model_path = "custom_yolo11n_v2.pt"

vcap = load_video(video_filepath)
results = read_video(video_capture=vcap, model_path=model_path)

real_framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
limit = 500
framecount = limit #real_framecount
fps = vcap.get(cv2.CAP_PROP_FPS)

# create empty arrays for storing intermediate results
perching_result = np.zeros(framecount) # final result: 1-8 for each of 8 perches, -1 for cage, -2 for floor, 0 for other
all_px_avgs = np.zeros((framecount,8))
all_bird_xs = np.zeros(framecount)
all_bird_ys = np.zeros(framecount)
all_wall_xs = np.zeros(framecount)

p_xs, p_ys, w_xs, five_perches_set = initialize_coordinates(video_name) # initialize perch and wall coordinates
# the set tracks which perches (1-5) have been visited

cage_status = np.empty(10) # initialize array of fence for the last 10 frames
cage_status[:] = np.nan

bird_x = np.nan # previous known bird location (center of its x coordinates)
bird_y = np.nan # previous known bird location (center of its y coordinates)

# for computing 5perches:
five_perches_timer = 0 # for tracking time spent of left side

i = 0
for result in results:

    # get coordinates
    new_bird_x, new_bird_y, new_p_xs, new_p_ys, new_w_x, on_cage = extract_coordinates(result)

    # if no perches are found (camera has fallen), continue loop without doing anything
    if len(new_p_xs)==0:
        continue

    # update x-coordinate of bird if it is found
    if new_bird_x is not None:
        bird_x = new_bird_x
        bird_y = new_bird_y

    # update coordinates of each perch into moving arrays
    p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, i%10)
    # update coordinates of middle wall
    w_xs, wx_avg = update_wall_coordinates(w_xs, new_w_x, i%10)
    # update cage status
    cage_status[i%10] = on_cage

    print(px_avgs)
    print(py_avgs)
    print(wx_avg)
    print(cage_status)
    

    # find if the bird is on a perch/fence, and which one
    bird_action = find_bird_on_perch(px_avgs, py_avgs, bird_x, bird_y, cage_status)
    # update property arrays
    perching_result[i] = bird_action
    all_bird_xs[i] = bird_x
    all_bird_ys[i] = bird_y
    all_px_avgs[i] = px_avgs
    all_wall_xs[i] = wx_avg

    # computation of 5perches:
    # remove perch number from reference set (does nothing if number is not present)
    five_perches_set.discard(bird_action)
    # if all five have been visited, the set is empty and timer stops
    if len(five_perches_set)>0:
        # if bird is on the left side, increase frame counter by one (can be replaced with the 'side' feature)
        if bird_x<wx_avg:
            five_perches_timer += 1


    print(bird_action)

    i+=1
    print(i)
    print(1/30*i, "s")
    if i==framecount: break

# result codes:
# -2 = on the floor
# -1 = on the cage
# 0 = other
# 1-8 = on perch 1-8

# compute features
# Boris helpdoc only asks for perches 1-5, but here are 6-8 as well
perch1, perch2, perch3, perch4, perch5, perch6, perch7, perch8 = compute_perch_durations(perching_result, fps)
ground, fence = compute_ground_and_fence(perching_result, fps)
five_perches = five_perches_timer/fps

# turn features into dataframe and save as csv file
# to be combined with other features in the future
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
feat_df = pd.DataFrame(data=features, index=[0])
feat_df.to_csv(f"../../data/features_{video_number}.csv")

exit()

# turn frame-by-frame results into dataframe and save
# only necessary for creating demo videos or further analysis
data = {
        "perch 1 x coordinate": all_px_avgs[:,0],
        "perch 2 x coordinate": all_px_avgs[:,1],
        "perch 3 x coordinate": all_px_avgs[:,2],
        "perch 4 x coordinate": all_px_avgs[:,3],
        "perch 5 x coordinate": all_px_avgs[:,4],
        "perch 6 x coordinate": all_px_avgs[:,5],
        "perch 7 x coordinate": all_px_avgs[:,6],
        "perch 8 x coordinate": all_px_avgs[:,7],
        "bird x coordinate": all_bird_xs,
        "bird action/perch number": perching_result,
        "wall x coordinate": all_wall_xs
}

data_df = pd.DataFrame(data=data)
data_df.to_csv(f"../../data/data_{video_number}.csv")

"""
TODO:
minor improvements:
- when camera/perch is tilted, perch proximity threshold should depend on the width of the perch
    - width of bounding box + 50px maybe?
- handling case when perches that weren't initially there appear (through e.g. camera movement)
- thresholds shouldn't be hardcoded, switch to a fraction of frame width/height
- exception handling when 0 perches are found (e.g. camera falls) -> video should be flagged as bad 
features:
- maybe combine functions into one
"""

