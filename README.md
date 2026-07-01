# Video Analysis

## Motivation

Do you have an old dashcam? 
I had one and set it up recording out window all day. But I had no time to scrub through hours of mundane footage? 
So, I built this tool to solve exactly that problem. 

Whether you're trying to figure out *exactly* when a package was delivered today, or need to find the specific moment someone keyed your car parked out front, reviewing an entire day's video manually is tedious and impractical. 

This program automates the review process. It scans through your video files, analyzes frame-by-frame pixel changes, and runs YOLO object detection. It then produces a timeline graph identifying **what** appeared in the video and **when** (e.g., "Detected: car, bicycle, person"). Instead of watching 24 hours of empty streets, you can skip directly to the "interesting" frames and find your events in seconds.

![Analysis Graph Example](sample_output/graph_detections_timestamps.png)

*(Example: A timeline plotting timestamp vs. change score, highlighting exact moments objects were detected.)*

## Overview

This workspace currently centers on a small frame-change workflow built around
`video_analyzer.py` and the companion notebook `video_analyzer_test.ipynb`.
Together they let you identify the car moving through the video, rank the most
changed frames, export the results, and inspect the output interactively.

### Intermediate output

![Car moving through the video](sample_output/rank_01_frame_002558_changes_4729.jpg)

*Car moving through the video*

### YOLO Detections

The YOLO-based scripts correctly identified a bicycle in the sample
frame below.

![YOLO detections](sample_output/yolo_detections.png)

### `video_analyzer.py` and `dashcam_change_analysis.py`

The script scans a video frame by frame, compares each frame against the
previous one, and ranks frames by the number of changed pixels so the moving
car stands out in the results.

It provides helpers to:

- find the top changed frames
- export frame-change counts to CSV
- export the selected results to CSV
- save the selected frames as images

### `video_analyzer_test.ipynb` and `validate_dashcam_change_analysis.py`

The notebook is the interactive validation path for the same workflow. It is
useful when you want to explore the change-detection results, inspect the
derived data in pandas, and visualize the moving car across the most changed
frames.

## Setup

Install the Python dependencies used by the script and notebook environment.
For local development, make sure the packages required for OpenCV, NumPy,
Matplotlib, pandas, Jupyter, and video analysis are available in your Python
environment.

## Typical Workflow

1. Load a video into the analyzer.
2. Run the frame-change ranking logic.
3. Export the counts or top results if needed.
4. Open the notebook to inspect the results visually.

## Output

The workflow produces CSV summaries and saved frame images for the highest
change-score frames.


## Common Failure Cases

- Strong camera motion can dominate the pixel-difference score and create false positives.
- Motion blur can hide small objects or reduce IoU overlap between adjacent frames.
- Low light and noise can raise the difference mask even when the scene is stable.
- YOLO may miss distant or partially occluded objects, especially in motion-heavy frames.
- Simple IoU matching can misassociate objects when boxes overlap heavily or objects cross paths.