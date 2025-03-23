from ultralytics import YOLO
import cv2

video_path = "data/videos/videos/HE21359_100721_21OW7_exploration_IB.mp4"

demo_video_path = "data/demo1.mp4"


model = YOLO("custom_yolo11n_v2.pt")

vcap = cv2.VideoCapture(video_path)

if vcap.isOpened(): 
    width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = vcap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the video
videowriter = cv2.VideoWriter(filename=demo_video_path,
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                        fps=fps,
                        frameSize=(int(width), int(height))
                        )

# Use only n frames for sample video.
n=0
while n < 2000:
    n += 1
    success, frame = vcap.read()

    if success:
        # Run YOLO inference on the frame
        results = model.track(frame)

        # Visualize the results on the frame
        for r in results:
            annotated_frame = r.plot()
            videowriter.write(annotated_frame)

videowriter.release()
cv2.destroyAllWindows()


