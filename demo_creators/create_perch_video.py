from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

#video_number = "CAGE_020720_HA70343"
video_number = "CAGE_030720_HA70345"
#video_number = "CAGE_050721_HA70384"

video_filepath = f"../../videos/{video_number}_exploration_IB.mp4"

vcap = cv2.VideoCapture(video_filepath)

if vcap.isOpened(): 
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = vcap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the video
videowriter = cv2.VideoWriter(filename=f"../../videos/demos/perching_video_{video_number}.mp4",
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                        fps=fps,
                        frameSize=(int(width), int(height))
                        )

# load data
df = pd.read_csv(f"../../data/data_{video_number}.csv")
# transform back into numpy arrays
perches = df["perch 1 x coordinate"].values.astype(int)
for i in range(1,8):
    perches = np.vstack((perches, df[f"perch {i} x coordinate"].values.astype(int)))
perches = perches.T
status = np.loadtxt(f"status_{video_number}.txt").astype(int)
bird = np.loadtxt(f"bird_{video_number}.txt").astype(int)
wall = np.loadtxt(f"wall_{video_number}.txt").astype(int)

# framecount of sample video
limit = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

n=0
while n < limit-1:
    n += 1
    success, frame = vcap.read()

    if success:
        for p in perches[n]:
            cv2.line(frame, (p, 0), (p, 720), (255, 255, 0), thickness=3)
        cv2.line(frame, (wall[n], 0), (wall[n], 720), (0, 255, 255), thickness=5)
        cv2.line(frame, (bird[n], 0), (bird[n], 720), (0, 0, 255), thickness=3)
        if status[n]==-2:
            cv2.putText(frame, "floor", (150, 150), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=5)
        elif status[n]==-1:
            cv2.putText(frame, "cage", (150, 150), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=5)
        elif status[n]==0:
            cv2.putText(frame, "other", (150, 150), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=5)
        else:
            cv2.putText(frame, str(status[n]), (150, 150), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=5)
        videowriter.write(frame)
    

videowriter.release()
cv2.destroyAllWindows()


