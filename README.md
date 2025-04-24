# exploring_wild_space

This repository contains tools for analyzing animal behavior in videos, specifically focusing on exploration patterns in a novel environment setup. It uses object detection (YOLO) to identify the animal and environmental features (like perches) and then extracts behavioral metrics from the detection data.

## Setup

1.  **Clone the repository:**
    ````bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd exploring_wild_space
    ````

2.  **Create a virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies.
    ````bash
    python3 -m venv .venv
    ````

3.  **Activate the virtual environment:**
    *   On Linux/macOS:
        ````bash
        source .venv/bin/activate
        ````
    *   On Windows:
        ````bash
        .\.venv\Scripts\activate
        ````

4.  **Install requirements:**
    Install the required packages using pip. Ensure you have PyTorch installed according to your system's specifications (CPU/GPU). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
    ````bash
    pip install -r requirements.txt
    ````


## Usage: `exploring.ipynb` Notebook

The main workflow is contained within the `exploring.ipynb` Jupyter Notebook. This notebook performs two primary steps:

1.  **Object Detection:** Processes input videos to detect objects (e.g., bird, perches) frame by frame using a custom YOLO model. The raw detection data (bounding boxes, frame info) is saved to JSON files.
2.  **Feature Extraction:** Loads the raw JSON data and calculates various behavioral features and quality metrics, saving them into CSV files.

**Instructions:**

1.  **Activate Environment:** Make sure your virtual environment is activated (`source .venv/bin/activate` or `.\.venv\Scripts\activate`).
2.  **Prepare Data:** Place the video files you want to analyze into the `data/original_videos/` directory.
3.  **Launch Jupyter:** Start Jupyter Lab or Jupyter Notebook from your activated environment:
    ````bash
    jupyter lab
    # or
    jupyter notebook
    ````
4.  **Open Notebook:** Navigate to and open `exploring.ipynb`.
5.  **Configure Paths (Optional):**
    *   Verify or modify the `input_path` (pointing to your video(s) or directory) and `output_dir` (for JSON files, default `data/raw_data/`).
    *   Verify or modify `json_input_path` (default `data/raw_data/`) and `output_features_dir` (default `data/extracted_features/`).
6.  **Run Cells:** Execute the notebook cells sequentially from top to bottom.
    *   The first section runs object detection, creating JSON files in `data/raw_data/`.
    *   The second section runs feature extraction, creating CSV files (`_features.csv` and `_quality.csv` for each input, plus combined files) in `data/extracted_features/`.

## Extracted Features and Metrics

The notebook calculates the following features and quality metrics:

### Extracted Features

| Feature     | Unit         | Description                                                                 |
|-------------|--------------|-----------------------------------------------------------------------------|
| latency     | Duration (s) | Time until first entry into the novel (exploration) area.                   |
| 5perches    | Duration (s) | Time spent in the novel area until the 5th distinct perch (1-5) is visited. |
| ground      | Duration (s) | Total time spent on the ground.                                             |
| perch1      | Duration (s) | Total time spent on perch 1.                                                |
| perch2      | Duration (s) | Total time spent on perch 2.                                                |
| perch3      | Duration (s) | Total time spent on perch 3.                                                |
| perch4      | Duration (s) | Total time spent on perch 4.                                                |
| perch5      | Duration (s) | Total time spent on perch 5.                                                |
| movements   | Count        | Number of movements (hops/flights) detected in the novel area.              |
| back_home   | Duration (s) | Time until the bird first returns to the home area after entering novel area. |
| T_new       | Duration (s) | Total time spent in the novel (exploration) area.                           |
| T_home      | Duration (s) | Total time spent in the home area.                                          |
| move_home   | Count        | Number of movements (hops/flights) detected in the home area.               |
| top         | Duration (s) | Total time spent in the top section of the cage.                            |
| middle      | Duration (s) | Total time spent in the middle section of the cage.                         |
| bottom      | Duration (s) | Total time spent in the bottom section of the cage.                         |
| fence       | Duration (s) | Total time spent detected near the fence/mesh.                              |

### Quality Metrics

| Metric                   | Unit         | Description                                                                                 |
|--------------------------|--------------|---------------------------------------------------------------------------------------------|
| camera_movement        | Boolean      | Indicates if significant camera/perch coordinate movement was detected during analysis.     |
| perch_count            | Count        | Number of perches (out of 5) reliably identified in the novel area in initial frames.     |
| close_perches          | Boolean      | Indicates if any identified perches (1-5) are potentially too close together.               |
| bird_inbetween_zones   | Rate (ev/s)  | Rate at which the bird was detected in ambiguous vertical zone boundaries (events per sec). |
| bird_inbetween_perches | Rate (ev/s)  | Rate at which the bird was detected in ambiguous location between perches 2 & 3 (ev per sec). |

## Deactivating the Environment

When you're finished working, you can deactivate the virtual environment:
````bash
deactivate
````