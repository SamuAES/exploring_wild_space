from os.path import basename
from ultralytics import YOLO
import cv2
from tqdm import tqdm

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
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
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
                
    



