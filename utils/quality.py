import numpy as np
import pandas as pd
import cv2

from perching_functions import *
from perching_functions import *
from frames import load_video, read_video


def read_perch_coordinates(video_name):

    # fill in first x-coordinates from manual annotations
    master_file = "../data/masterfile_20202021_LOOPY(Coordinates_xPerches).csv"
    df = pd.read_csv(f"{master_file}") # relative filepath
    df = df[df["VIDEO TITLE"]==video_name]
    coords = np.array([df[f"X PERCH{i+1}"] for i in range(8)]).T[0] # get coordinates
    if coords[0]>coords[1] and coords[1]>coords[2] and coords[0]>coords[3] and coords[2]>coords[4]: # several conditions in case there are nans (kind of a hack)
        coords = np.flip(coords)
    
    return coords

def update_perch_count(perch_count: np.array, annotated_perch_coord: np.array, perch_coords: np.array) -> np.array:
    """
    Increments the individual values in the perch_count array by one when the x coordinates of the perches
    in perch_coords match the coordinates in annothated_perch_coord 
    
    Parameters
    ----------
    perch_count : numpy array
        numpy array containing the current count of frames that contain the 8 perches
    
    annotated_perch_count : numpy array
        numpy array containing the annotated x coordinates of the perches
    
    perch_count : numpy array
        numpy array containing the x coordinates of the perches from the current frame
    
    Returns
    -------
    numpy array :
        The updated numpy array containing the count 
    """
    if len(perch_coords) < 8:
        diff = 8 - len(perch_coords)
        perch_coords = np.append(perch_coords, np.full(diff, 10000))

    aa, bb = np.meshgrid(annotated_perch_coord, perch_coords)
    dist = np.abs(aa-bb)

    perch_ind = np.argmin(dist, axis=0)

    for i, x in enumerate(perch_ind):
        if dist[x][i] < 30: #arbitrary threshold
            perch_count[i] = perch_count[i] + 1
    
    return perch_count

def detect_movement(p_xs: np.array, new_p_xs: np.array) -> np.array:
    """
    Detects the camera movement by comparing the x coordinates of the perches from the current 
    frame against the moving average
    
    Parameters
    ----------
    p_xs : numpy array
        numpy array containing the 10 most current x coordinates of the 8 perches
    
    new_p_xs : numpy array
        numpy array containing the x coordinates of the 8 perches of the current frame
    
    Returns
    -------
    Bool :
        True if camera movement is detected, False otherwise 
    """
    px_moving_avgs = np.nanmean(p_xs, axis=0)

    aa, bb = np.meshgrid(px_moving_avgs, new_p_xs)
    dist = np.abs(aa-bb)

    perch_ind = np.argmin(dist, axis=0)

    perch_movement = 0

    for i, x in enumerate(perch_ind):
        if dist[x][i] > 5: # arbitrary threshold
            perch_movement = perch_movement + 1
    
    if perch_movement > 1: 
        return True
    return False



if __name__ == "__main__":
    #video_number = "CAGE_020720_HA70343"
    #video_number = "CAGE_030720_HA70344"
    #video_number = "CAGE_030720_HA70345"
    video_number = "CAGE_050721_HA70384" # camera movement
    #video_number = "CAGE_220520_HA70339" # bird on the floor
    #video_number = "CAGE_220520_HA70337"
    #video_number = "HE21362_100721_21JJ32"

    video_name = f"{video_number}_exploration_IB"
    video_filepath = f"../../videos/{video_number}_exploration_IB.mp4" # relative filepath, adjust accordingly
    model_path = "../yolo/custom_yolo11n_v2.pt"

    vcap = load_video(video_filepath)
    results = read_video(video_capture=vcap, model_path=model_path)

    real_framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = 5400
    framecount = limit #real_framecount
    fps = vcap.get(cv2.CAP_PROP_FPS)

    annotated_perch_coords = read_perch_coordinates(video_name)

    perch_count = np.zeros(8)

    perching_result = np.zeros(framecount) # final result: 1-8 for each of 8 perches, -1 for cage, -2 for floor, 0 for other
    all_px_avgs = np.zeros((framecount,8))
    all_movement = np.zeros(framecount)

    p_xs, p_ys, w_xs, five_perches_set = initialize_coordinates(video_name) # initialize perch and wall coordinates

    i = 0
    for frame in results:

        new_bird_x, new_bird_y, new_p_xs, new_p_ys, new_w_x, on_cage = extract_coordinates(frame)

        # if no perches are found (camera has fallen), continue loop without doing anything
        if len(new_p_xs)==0:
            continue

        perch_count = update_perch_count(perch_count, annotated_perch_coords, new_p_xs)

        p_xs, p_ys, px_avgs, py_avgs = update_perch_coordinates(p_xs, p_ys, new_p_xs, new_p_ys, i%10)

        if detect_movement(p_xs, new_p_xs):
            all_movement[i] = 1
            print("movement")
            print(i)
            print(1/30*i, "s")

        all_px_avgs[i] = px_avgs

        i+=1
        if i % 150 == 0:
            print(i/30, "s")
        if i==framecount: break

    print(perch_count/ framecount)

    data = {
            "perch 1 x coordinate": all_px_avgs[:,0],
            "perch 2 x coordinate": all_px_avgs[:,1],
            "perch 3 x coordinate": all_px_avgs[:,2],
            "perch 4 x coordinate": all_px_avgs[:,3],
            "perch 5 x coordinate": all_px_avgs[:,4],
            "perch 6 x coordinate": all_px_avgs[:,5],
            "perch 7 x coordinate": all_px_avgs[:,6],
            "perch 8 x coordinate": all_px_avgs[:,7],
            "movement": all_movement,
            "perch 1 count": perch_count[0]/ framecount,
            "perch 2 count": perch_count[1]/ framecount,
            "perch 3 count": perch_count[2]/ framecount,
            "perch 4 count": perch_count[3]/ framecount,
            "perch 5 count": perch_count[4]/ framecount,
            "perch 6 count": perch_count[5]/ framecount,
            "perch 7 count": perch_count[6]/ framecount,
            "perch 8 count": perch_count[7]/ framecount,
    }
    data_df = pd.DataFrame(data=data)
    data_df.to_csv(f"../data/data_{video_number}.csv")





