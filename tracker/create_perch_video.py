from ultralytics import YOLO
import cv2
import numpy as np

video_number = "CAGE_020720_HA70343"
video_filepath = f"../../videos/{video_number}_exploration_IB.mp4"
model_path = "custom_yolo11n.pt"

model = YOLO(model_path)

vcap = cv2.VideoCapture(video_filepath)

if vcap.isOpened(): 
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = vcap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the video
videowriter = cv2.VideoWriter(filename=f"perching_video_{video_number}.mp4",
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                        fps=fps,
                        frameSize=(int(width), int(height))
                        )

# Use only 6000 frames for sample video.
perches = np.loadtxt(f"perches_{video_number}.txt").astype(int)
status = np.loadtxt(f"status_{video_number}.txt").astype(int)
bird = np.loadtxt(f"bird_{video_number}.txt").astype(int)
n=0
while n < 100:
    n += 1
    success, frame = vcap.read()

    if success:
        for p in perches[n]:
            cv2.line(frame, (p, 0), (p, 720), (255, 255, 0), thickness=3)
        cv2.line(frame, (bird[n], 0), (bird[n], 720), (0, 0, 255), thickness=3)
        cv2.line(frame, (bird[n], 0), (bird[n], 720), (0, 0, 255), thickness=3)
        cv2.putText(frame, str(status[n]), (150, 150), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 0), thickness=5)
        videowriter.write(frame)
    

videowriter.release()
cv2.destroyAllWindows()


