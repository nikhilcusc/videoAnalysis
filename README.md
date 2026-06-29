# Video Analysis

This workspace currently centers on a small frame-change workflow built around
`video_analyzer.py` and the companion notebook `video_analyzer_test.ipynb`.
Together they let you identify the car moving through the video, rank the most
changed frames, export the results, and inspect the output interactively.

## Example Output

![Car moving through the video](sample_output/rank_01_frame_002558_changes_4729.jpg)

## `video_analyzer.py`

The script scans a video frame by frame, compares each frame against the
previous one, and ranks frames by the number of changed pixels so the moving
car stands out in the results.

It provides helpers to:

- find the top changed frames
- export frame-change counts to CSV
- export the selected results to CSV
- save the selected frames as images

## `video_analyzer_test.ipynb`

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
