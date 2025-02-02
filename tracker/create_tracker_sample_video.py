from ultralytics import YOLO
import cv2

name = "CAGE_050721_HA70384_exploration_IB"

model = YOLO("custom_yolo11n.pt")

vcap = cv2.VideoCapture(f'videos/videos/{name}.mp4')

if vcap.isOpened(): 
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = vcap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the video
videowriter = cv2.VideoWriter(filename="custom_yolo11n_sample.mp4",
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                        fps=fps,
                        frameSize=(int(width), int(height))
                        )

# Use only 6000 frames for sample video.
n=0
while n < 6000:
    n += 1
    success, frame = vcap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        for r in results:
            annotated_frame = r.plot()
            videowriter.write(annotated_frame)

videowriter.release()
cv2.destroyAllWindows()


