from os.path import basename
from ultralytics import YOLO
import json
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

def load_video(video_filepath:str):
    """
    Parameters
    ----------
    video_filepath : str
        Relative path to videofile.

    Returns
    -------
    cv2.VideoCapture object
    """
    vcap = cv2.VideoCapture(video_filepath)
    if vcap.isOpened():
        print("video:", basename(video_filepath))
        print("frame count:", vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frame width:", vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("frame height:", vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("fps:", vcap.get(cv2.CAP_PROP_FPS))
    else:
        raise ValueError("VideoCapture did not succeed")
    cv2.destroyAllWindows()
    return vcap

def read_video(video_capture:cv2.VideoCapture, model_path:str):
    """
    Parameters
    ----------
    video_capture : cv2.VideoCapture object
    
    model : str
        Relative path to YOLO model.

    Returns
    -------
    Generator that yields a dictionary per frame.
    """
    model = YOLO(model_path)
    while True:
        retval, frame = video_capture.read() # Read one frame from video
        if not retval:
            break
        results = model(frame, stream=True, verbose=False) # Run YOLO detection on frame
        
        for result in results:
            class_names = result.names # class names
            boxes = []
            
            # iterate over each box
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0] # get coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert from tensor to int
                boxes.append({"class":class_names[int(box.cls)],
                            "confidence":float(box.conf),
                            "x1":x1,
                            "y1":y1,
                            "x2":x2,
                            "y2":y2})
                
            yield boxes
                
    

def read_video_and_save_frames_to_json(video_filepath:str, save_path:str, model_path:str, max_frames:int = np.inf):    
    video_capture = load_video(video_filepath)
    generator = read_video(video_capture, model_path)
    n = 1
    frames = {}
    for frame in tqdm(generator, desc="Frames", total=max_frames, mininterval=1):
        frames[f"frame{n}"] = frame
        if n == max_frames+1:
            break
        n += 1
    with open(save_path, "w") as json_file:
        json.dump(frames, json_file, indent=4)

def load_json_to_dict(json_filepath:str):
    with open(json_filepath, "r") as json_file:
        raw_dict = json.load(json_file)
    return raw_dict


def process_box(box:dict, row_index:int, num:int):
    df = pd.DataFrame(box, index=[row_index])
    df.columns = [str(num) + col for col in df.columns]
    return df

def process_raw_data(data:dict):
    # Initialize empty dataframes
    bird_df = pd.DataFrame()
    wall_df = pd.DataFrame()
    fence_df = pd.DataFrame()
    perch_df = pd.DataFrame()


    for row_index, frame in enumerate(data.values(), start=1):
        # Initialize empty dataframes for frame
        bird_frame_df = pd.DataFrame(index=[row_index])
        wall_frame_df = pd.DataFrame(index=[row_index])
        fence_frame_df = pd.DataFrame(index=[row_index])
        perch_frame_df = pd.DataFrame(index=[row_index])
        
        # Running numbers to separate different boxes of the same class
        bird_num = 1
        wall_num = 1
        fence_num = 1
        perch_num = 1

        # Join boxes of same class to their own frame_df
        for box in frame:
            if box["class"] == "bird":
                bird_frame_df = bird_frame_df.join(process_box(box, row_index, bird_num))
                bird_num += 1

            elif box["class"] == "wall":
                wall_frame_df = wall_frame_df.join(process_box(box, row_index, wall_num))
                wall_num += 1

            elif box["class"] == "fence":
                fence_frame_df = fence_frame_df.join(process_box(box, row_index, fence_num))
                fence_num += 1

            elif box["class"] == "stick":
                perch_frame_df = perch_frame_df.join(process_box(box, row_index, perch_num))
                perch_num += 1

        # Add frame_df of each class to their own df as new row
        bird_df = pd.concat([bird_df, bird_frame_df])
        wall_df = pd.concat([wall_df, wall_frame_df])
        fence_df = pd.concat([fence_df, fence_frame_df])
        perch_df = pd.concat([perch_df, perch_frame_df])

    bird_df.index.name = "frame"
    wall_df.index.name = "frame"
    fence_df.index.name = "frame"
    perch_df.index.name = "frame"

    return bird_df, wall_df, fence_df, perch_df





if __name__=="__main__":
    video_path = "../data/original_videos/HE21362_100721_21JJ32_exploration_IB.mp4"
    save_path = "../data/raw_data/HE21362_100721_21JJ32_exploration_IB.json"
    video_id = "HE21362_100721_21JJ32"
    video_directory = "data/original_videos"
    model_path = "../yolo/custom_yolo11n_v2.pt"
    max_frames = 10 # How many frames to analyse if larger than video frame count then count all frames
    read_video_and_save_frames_to_json(video_path, save_path, model_path, max_frames)
    raw_dict = load_json_to_dict(save_path)
    bird_df, wall_df, fence_df, perch_df = process_raw_data(raw_dict)
    print(bird_df)




