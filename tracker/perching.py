from frames import load_video, read_video
import pandas as pd
import numpy as np
import copy
from perching_functions import initialize_p_coordinates, extract_coordinates, update_perch_coordinates, find_bird_on_perch

# Relative filepaths
# CAGE_020720_HA70343_exploration_IB
# CAGE_030720_HA70344_exploration_IB
# CAGE_030720_HA70345_exploration_IB

video_number = "CAGE_020720_HA70343"
#video_number = "CAGE_030720_HA70344"
#video_number = "CAGE_030720_HA70345"

video_name = f"{video_number}_exploration_IB"
fpo_id = f"{video_number}_exploration_FPO_IB" # used for extracting Loopy data, differs from filepath name by "FPO"
video_filepath = f"../../videos/{video_number}_exploration_IB.mp4"
model_path = "custom_yolo11n.pt"

# # load and find "fpo" information from Loopy output
# df = pd.read_csv("../../fpo.csv")
# df = df[df.ID==fpo_id] # find correct video
# fpo = df.oid.values # oid: 0=cage, 1=other, 2=perch
# fpo_str = df.name.values # probably not necessary
# print(fpo)
# print(fpo_str)
# print(len(fpo), len(fpo_str))

vcap = load_video(video_filepath)
results = read_video(video_capture=vcap, model_path=model_path)

framecount = 6000
perching_result = np.zeros(framecount) # final result: 0 if not perching, 1-8 for each of 8 perches
all_px_avgs = np.zeros((framecount,8))
all_bird_xs = np.zeros(framecount)
all_bird_ys = np.zeros(framecount)

p_xs, p_ys = initialize_p_coordinates(video_name) # initialize perch coordinates

bird_x = np.nan # previous known bird location (center of its x coordinates)
bird_y = np.nan # previous known bird location (center of its y coordinates)

i = 0
for result in results:

    new_bird_x, new_bird_y, new_p_xs, new_p_ys = extract_coordinates(result) # get coordinates

    # update x-coordinate of bird if it is found
    if new_bird_x is not None:
        bird_x = new_bird_x
        bird_y = new_bird_y

    # update coordinates of each perch into moving arrays
    p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, i%10)

    print(px_avgs)
    print(py_avgs)
    

    # find which perch the bird is sitting on
    perch_number = find_bird_on_perch(px_avgs, py_avgs, bird_x, bird_y)
    perching_result[i] = perch_number
    all_bird_xs[i] = bird_x
    all_bird_ys[i] = bird_y
    all_px_avgs[i] = px_avgs
    # print(p_avgs)

    i+=1
    print(i)
    print(1/30*i, "s")
    if i==100: break

np.savetxt(f"perches_{video_number}.txt", all_px_avgs)
np.savetxt(f"bird_{video_number}.txt", all_bird_xs)
np.savetxt(f"status_{video_number}.txt", perching_result)

print(perching_result[:100])

"""
TODO:
- clean up perch and bird info (no duplicates) X
- fix fpo data (possibly) X
- distinguishing between perches: track feet + instance segmentation + fpo data
    - decide how to tell the bird is on a perch
        - check if bird is on the ground (y-coordinate)
        - check if bird is on the cage mesh
- which perch is which -> assign closest to previous moving average X
- use manual annotations to check perch ids X
- create presentation base
- when camera is tilted, perch proximity threshold should depend on the width of the perch

"""


