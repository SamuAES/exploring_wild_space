from ultralytics import YOLO

name = "CAGE_050721_HA70384_exploration_IB"

model = YOLO("custom_yolo11n.pt")
#results = model(f"videos/videos/{name}.mp4", stream=True)

# n=0
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.save(filename=f"test_results/{name}_result_frame{n}.jpg")  # save to disk
#     n+=1

metrics = model.val(data="data/data.yaml")
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category
