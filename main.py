from utils.frames import load_json_to_dict
from utils.features import extract_features

json_filepath = 'data/raw_data/CAGE_200520_HA70336_exploration_IB.json'
data_raw = load_json_to_dict(json_filepath)
video_filename = data_raw["video_filename"]
frame_count = int(data_raw["frame_count"])
fps = int(data_raw["fps"])

results_df, quality_metrics = extract_features(
    data_raw,
    window_size_mean=5,
    window_size_mode=35,
    fps=fps,
    frame_count=frame_count,
)

print(quality_metrics.head())
print(results_df.head())




