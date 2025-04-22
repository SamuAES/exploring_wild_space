from utils.frames import load_json_to_dict
from utils.features import bird_and_wall_positions, extract_features


# Extracts features from a given JSON file and saves them to a CSV file.
#################################################################################################
# feature_name | type       | description                                                       #
# ----------------------------------------------------------------------------------------------#
# latency	    duration    first time entering new area                                        #
# 5perches	    duration	time from entering to visit the 5th perch                           #
# ground	    duration	time spent on ground                                                #
# perch1	    duration	time spent on perch1	                                            #   
# perch2	    duration	time spent on perch2                                                #
# perch3	    duration	time spent on perch3                                                # 
# perch4	    duration	time spent on perch4                                                #
# perch5	    duration	time spent on perch5                                                # 
# movements	    event	    number of hops and flights in novel area                            #
# back_home	    event	    first time when going back to home side after entering novel area   #
# T_new	        duration	total time spent in new area                                        #
# move_home	    event	    number of movements in home side whole period                       #
# Top	        duration	Top part cage                                                       #
# Middle	    duration	Middle part cage                                                    #
# Bottom	    duration	Bottom part cage                                                    #
# Fence	        duration	Bird against mesh                                                   #
#################################################################################################


json_filepath = 'data/raw_data/CAGE_030720_HA70345_exploration_IB.json'
data_raw = load_json_to_dict(json_filepath)
video_filename = data_raw["video_filename"]
frame_count = int(data_raw["frame_count"])
fps = int(data_raw["fps"])


bird_x, bird_y, wall_x, wall_y = bird_and_wall_positions(
    data=data_raw["frames"],
    frame_count=frame_count,
    window_size_mean=3
)

results_df, quality_metrics = extract_features(
    data_raw,
    window_size_mean=3,
    window_size_mode=31,
    fps=fps,
    frame_count=frame_count,
    bird_x=bird_x,
    bird_y=bird_y,
    wall_x=wall_x,
    wall_y=wall_y
)

print(quality_metrics.head())
print(results_df.head())
