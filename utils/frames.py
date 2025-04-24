from os.path import basename
from ultralytics import YOLO
import json
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

def load_video(video_filepath:str):
    """
    This function loads a video file using OpenCV and prints some information about the video.
    It returns a cv2.VideoCapture object that can be used to read frames from the video.

    Parameters
    ----------
    video_filepath : str
        Relative path to videofile.

    Returns
    -------
    cv2.VideoCapture object
    """
    vcap = cv2.VideoCapture(video_filepath)
    try:
        if not vcap.isOpened():
            raise ValueError("Could not open video file")
    except ValueError as e:
        print(e)
        return None
    
    cv2.destroyAllWindows()
    return vcap

def read_video(video_capture:cv2.VideoCapture, model_path:str):
    """
    This function reads frames from a video file using OpenCV and runs YOLO detection on each frame.
    It returns a generator that yields a dictionary containing the results of YOLO detection for each frame.
    The dictionary contains the following information:
    - class: the class of the box
    - confidence: the confidence score of the box
    - x1: the x-coordinate of the top-left corner of the box
    - y1: the y-coordinate of the top-left corner of the box
    - x2: the x-coordinate of the bottom-right corner of the box
    - y2: the y-coordinate of the bottom-right corner of the box

    Parameters
    ----------
    video_capture : cv2.VideoCapture object
    
    model : str
        Relative path to YOLO model.

    Returns
    -------
    generator
        A generator that yields a dictionary containing the results of YOLO detection for each frame.
    """
    model = YOLO(model_path)
    while True:
        retval, frame = video_capture.read() # Read one frame from video
        if not retval:
            break
        results = model(frame, stream=True, verbose=False) # Run YOLO detection on frame
        
        for result in results:
            class_names = result.names # class names
            boxes = {}
            
            # iterate over each box
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0] # get coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert from tensor to int
                boxes[i] = {"class":class_names[int(box.cls)],
                            "confidence":float(box.conf),
                            "x1":x1,
                            "y1":y1,
                            "x2":x2,
                            "y2":y2}
                
            yield boxes




def read_video_and_save_frames_to_json(video_filepath:str, save_path:str, model_path:str, max_frames = None):        
    """
    This function reads a video file, runs YOLO detection on each frame, and saves the results to a JSON file.
    The JSON file contains the following information:
    - video_filename: the name of the video file
    - frame_count: the number of frames in the video
    - frame_width: the width of the frames in the video
    - frame_height: the height of the frames in the video
    - fps: the frames per second of the video
    - frames: a dictionary containing the results of YOLO detection for each frame
        - frameX: a dictionary containing the boxes detected in frame X
            - class: the class of the box
            - confidence: the confidence score of the box
            - x1: the x-coordinate of the top-left corner of the box
            - y1: the y-coordinate of the top-left corner of the box
            - x2: the x-coordinate of the bottom-right corner of the box
            - y2: the y-coordinate of the bottom-right corner of the box

    Parameters
    ----------
    video_filepath : str
        Relative path to the video file.
    save_path : str
        Path where the JSON file will be saved.
    model_path : str
        Relative path to YOLO model.
    max_frames : int, optional
        Maximum number of frames to process. If None, all frames will be processed.

    Returns
    -------
    None
    """
    
    video_capture = load_video(video_filepath)
    if max_frames is None:
        max_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(max_frames, int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    generator = read_video(video_capture, model_path)
    
    n = 1

    filename = basename(video_filepath)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps =  video_capture.get(cv2.CAP_PROP_FPS)

    result = {
        "video_filename":filename,
        "frame_count":frame_count,
        "frame_width":frame_width,
        "frame_height":frame_height,
        "fps":fps,
        "frames":{}
        }

    for frame in tqdm(generator, desc="Frames", total=max_frames, mininterval=1):
        result["frames"][f"frame{n}"] = frame
        if n == max_frames+1:
            break
        n += 1
    with open(save_path, "w") as json_file:
        json.dump(result, json_file, indent=4)

def load_json_to_dict(json_filepath:str):
    """
    This function loads a JSON file and returns its contents as a dictionary.
    Parameters
    ----------
    json_filepath : str
        Relative path to the JSON file.
    Returns
    -------
    dict
        A dictionary containing the contents of the JSON file.
    """
    with open(json_filepath, "r") as json_file:
        raw_dict = json.load(json_file)
    return raw_dict


def process_box(box:dict, row_index:int, num:int):
    """
    This function processes a single box and returns a DataFrame containing the box information.
    The DataFrame contains the following columns:
    - class: the class of the box
    - confidence: the confidence score of the box
    - x1: the x-coordinate of the top-left corner of the box
    - y1: the y-coordinate of the top-left corner of the box
    - x2: the x-coordinate of the bottom-right corner of the box
    - y2: the y-coordinate of the bottom-right corner of the box
    Parameters
    ----------
    box : dict
        A dictionary containing the box information.
    row_index : int
        The index of the row in the DataFrame.
    num : int
        A number used to separate different boxes of the same class.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the box information.
    """
    df = pd.DataFrame(box, index=[row_index])
    df.columns = [col + "_" + str(num) for col in df.columns]
    return df

def process_raw_data(data:dict):
    """
    This function processes the raw data from the JSON file and returns four DataFrames:
    - bird_df: a DataFrame containing the information of all birds detected in the video
    - wall_df: a DataFrame containing the information of all walls detected in the video
    - fence_df: a DataFrame containing the information of all fences detected in the video
    - perch_df: a DataFrame containing the information of all perches detected in the video
    Parameters
    ----------
    data : dict
        A dictionary containing the raw data from the JSON file.
    Returns
    -------
    tuple
        A tuple containing four DataFrames:
        - bird_df: a DataFrame containing the information of all birds detected in the video
        - wall_df: a DataFrame containing the information of all walls detected in the video
        - fence_df: a DataFrame containing the information of all fences detected in the video
        - perch_df: a DataFrame containing the information of all perches detected in the video
    """
    # Initialize empty dataframes
    bird_df = pd.DataFrame()
    wall_df = pd.DataFrame()
    fence_df = pd.DataFrame()
    perch_df = pd.DataFrame()
    frame_count = len(data)

    for row_index, frame in tqdm(enumerate(data.values(), start=1), desc="Processing frames", total=frame_count, mininterval=1):
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
        for box in list(frame.values()):
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






