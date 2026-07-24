# Video Analysis TODO

## Gradio UI
- [ ] Build a Gradio app for the existing dashcam analysis pipeline.
- [ ] Add inputs for video upload/path, sampling settings, thresholds, output directory, and annotated-image toggle.
- [ ] Display the top changed frames, detection summary, and generated CSV path in the UI.

## Model Selection
- [ ] Add support for selecting between multiple YOLO models instead of a single default weight.
- [ ] Define a small model registry for built-in options such as `yolov8n.pt` and at least one additional model.
- [ ] Allow custom local weights paths as a fallback when the user wants to upload or point to their own model file.
- [ ] Surface the active model choice in both the Gradio UI and the CLI output.

## Pipeline Wiring
- [ ] Refactor model loading so the analysis code accepts a selected model name or path from either CLI or Gradio.
- [ ] Keep the CLI behavior working as a headless fallback for batch runs.
- [ ] Reuse the existing analysis summary and CSV export logic instead of duplicating it in the UI layer.

## Validation
- [ ] Test the Gradio flow with a sample video from `12Feb2022/`.
- [ ] Verify the alternate YOLO model produces results and can be switched at runtime.
- [ ] Confirm CSV and annotated image exports still land in `output/`.
