# check whether the file is included in the manual annotation file

import pandas as pd
import numpy as np
import json
from feature_extractor import extract_from_data

def validate_data(video_id, json_path, manual_csv_path='data/Manually annotated features.csv', fps=30):
    # Load the JSON data
    with open(json_path) as f:
        data = json.load(f)

    # Extract features
    extracted_features = extract_from_data(data, video_id=video_id, nframes=len(data), fps=fps)
    df_perches, df_sections = extracted_features

    # Load the manually annotated CSV file
    df_manual = pd.read_csv(manual_csv_path, sep='|', encoding='latin1')

    # Filter and rename columns
    try:
        ringnr = video_id.split("_")[0]
        if ringnr == 'CAGE':
            ringnr = video_id.split("_")[2]
    except:
        ringnr = video_id

    df_manual_filtered = df_manual[df_manual["Ringnr"] == ringnr][
        ['Top_time', 'Middle_time', 'Bottom_time', 'Time_exploration', 'Time_homeside']
    ]
    df_manual_filtered = df_manual_filtered.astype(float)
    df_manual_filtered.rename(columns={
        'Top_time': 'top',
        'Middle_time': 'middle',
        'Bottom_time': 'bottom',
        'Time_exploration': 'left',
        'Time_homeside': 'right'
    }, inplace=True)
    df_manual_filtered['Source'] = 'Manual'
    df_manual_filtered.set_index('Source', inplace=True)

    # Prepare detected data
    df_sections['Source'] = 'Detected'
    df_sections.set_index('Source', inplace=True)
    df_sections = df_sections.astype(float)

    # Combine the two DataFrames
    combined_df = pd.concat([df_manual_filtered, df_sections], ignore_index=False)

    # Calculate error
    error = (combined_df.loc['Manual'] - combined_df.loc['Detected'])

    return combined_df, error