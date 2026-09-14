from __future__ import annotations

import hashlib
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashcam_change_analysis import (
    AnalysisConfig,
    build_model,
    compute_frame_change_score,
    detect_object_transitions,
    detections_to_dataframe,
    draw_annotations,
    extract_frames,
    load_video,
    run_yolo_detection,
)


EXAMPLE_VIDEOS = sorted(Path("exampleVids").glob("*.mp4"))
MODEL_OPTIONS = {
    "YOLOv8 nano": "models/yolov8n.pt",
    "YOLO11 small": "models/yolo11s.pt",
    "YOLO26 nano": "models/yolo26n.pt",
}


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as video_file:
        for chunk in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video(path: Path) -> tuple[Any, float]:
    capture, metadata = load_video(path)
    duration = metadata.frame_count / metadata.fps if metadata.fps else 0.0
    if duration > 180:
        capture.release()
        raise ValueError(f"Video duration is {duration:.1f}s; it must be 180s or shorter.")
    if metadata.height > 1080 or metadata.width > 1920:
        capture.release()
        raise ValueError(
            f"Video resolution is {metadata.width}x{metadata.height}; maximum is 1920x1080."
        )
    return capture, duration


@st.cache_data(show_spinner=False)
def _extract_cached(path_string: str, fingerprint: str, interval: int, resize_width: int) -> tuple[list[Any], Any]:
    del fingerprint
    return extract_frames(path_string, interval=interval, resize_width=resize_width)


@st.cache_resource(show_spinner="Loading YOLO model...")
def _load_model(model_name_or_path: str) -> Any:
    return build_model(model_name_or_path)


def _score_samples(samples: list[Any], diff_threshold: int) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for previous, current in zip(samples[:-1], samples[1:]):
        score, changed_pixels, changed_ratio, diff_mask, abs_diff = compute_frame_change_score(
            previous.frame, current.frame, diff_threshold=diff_threshold
        )
        scores.append(
            {
                "previous_frame_index": previous.frame_index,
                "frame_index": current.frame_index,
                "timestamp_seconds": current.timestamp_seconds,
                "change_score": score,
                "changed_pixels": changed_pixels,
                "changed_ratio": changed_ratio,
                "diff_mask": diff_mask,
                "abs_diff": abs_diff,
                "frame": current.frame,
                "previous_frame": previous.frame,
            }
        )
    return scores


def _top_peaks(scores: list[dict[str, Any]], count: int, spacing_seconds: float = 2.0) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in sorted(scores, key=lambda value: value["change_score"], reverse=True):
        if all(abs(item["timestamp_seconds"] - other["timestamp_seconds"]) >= spacing_seconds for other in selected):
            selected.append(item)
        if len(selected) == count:
            break
    return selected


def _object_label(detections: list[Any]) -> str:
    counts = Counter(detection.class_name for detection in detections)
    return ", ".join(f"{name}({count})" for name, count in sorted(counts.items())) or "none"


def _rgb(image: Any) -> Any:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _render_result(result: dict[str, Any], detections: list[Any], entered: list[Any], exited: list[Any], matches: list[Any]) -> None:
    annotated = draw_annotations(
        result["frame"], detections, entered, exited, matches
    )
    st.image(_rgb(annotated), caption="Annotated current frame", use_container_width=True)

    summary_columns = st.columns(3)
    summary_columns[0].write(f"**Detected**  \n{_object_label(detections)}")
    summary_columns[1].write(f"**Entered**  \n{_object_label(entered)}")
    summary_columns[2].write(f"**Exited**  \n{_object_label(exited)}")

    if detections:
        table = pd.DataFrame(detections_to_dataframe(detections))[
            ["class_id", "class_name", "confidence"]
        ]
        table["confidence"] = table["confidence"].round(3)
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No objects were detected above the selected confidence threshold.")


def main() -> None:
    st.set_page_config(page_title="Dashcam Auto Video watcher", page_icon="🎥", layout="wide")
    st.title("Dashcam Auto Video watcher")
    st.caption("Change detection with YOLO object detection")

    with st.sidebar:
        st.header("Video")
        source = st.radio("Input source", ["Example video", "Upload video"])
        uploaded_file = None
        if source == "Example video":
            if not EXAMPLE_VIDEOS:
                st.error("No example videos were found in exampleVids/.")
                return
            selected_video = st.selectbox("Choose a video", EXAMPLE_VIDEOS, format_func=lambda item: item.name)
            video_path = selected_video
        else:
            uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
            video_path = None

        st.header("Analysis")
        sample_interval = st.number_input("Sample interval", min_value=1, value=5, step=1)
        diff_threshold = st.number_input("Difference threshold", min_value=0, max_value=255, value=25, step=1)
        confidence_threshold = st.slider("YOLO confidence", 0.0, 1.0, 0.25, 0.01)
        iou_threshold = st.slider("IoU matching threshold", 0.0, 1.0, 0.30, 0.05)
        resize_width = st.number_input("Processing width", min_value=320, max_value=1920, value=960, step=32)
        model_label = st.selectbox("YOLO model", list(MODEL_OPTIONS), index=2)
        #custom_model = st.text_input("Custom model path (optional)")
        #model_path = custom_model.strip() or MODEL_OPTIONS[model_label]
        model_path = MODEL_OPTIONS[model_label]
        run_analysis = st.button("Run analysis", type="primary", use_container_width=True)

    if not run_analysis:
        st.info("Choose a video and configure the analysis in the sidebar, then select Run analysis.")
        return

    temporary_path: Path | None = None
    try:
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temporary_path = Path(temp_file.name)
            video_path = temporary_path
        if video_path is None:
            st.error("Select or upload a video before running analysis.")
            return

        config = AnalysisConfig(
            resize_width=int(resize_width),
            diff_threshold=int(diff_threshold),
            top_n=5,
            sample_interval=int(sample_interval),
            iou_threshold=float(iou_threshold),
            confidence_threshold=float(confidence_threshold),
            object_limit=25,
        )
        capture, duration = _validate_video(video_path)
        capture.release()
        _, metadata = load_video(video_path)
        st.success(f"Validated {metadata.width}x{metadata.height}, {duration:.1f}s, {metadata.fps:.2f} FPS")

        fingerprint = _file_fingerprint(video_path)
        with st.spinner("Extracting frames and calculating pixel changes..."):
            samples, _ = _extract_cached(
                str(video_path), fingerprint, config.sample_interval, config.resize_width or 960
            )
            scores = _score_samples(samples, config.diff_threshold)
        if not scores:
            st.error("The video does not contain enough sampled frames for comparison.")
            return

        top_three = _top_peaks(scores, 3)
        top_five = _top_peaks(scores, 5)
        model = _load_model(model_path)
        detection_cache: dict[int, list[Any]] = {}

        def detections_for(frame_index: int, frame: Any) -> list[Any]:
            if frame_index not in detection_cache:
                detection_cache[frame_index] = run_yolo_detection(
                    model, frame, confidence_threshold=config.confidence_threshold, object_limit=config.object_limit
                )
            return detection_cache[frame_index]

        with st.spinner("Running YOLO on change peaks..."):
            peak_detections = {item["frame_index"]: detections_for(item["frame_index"], item["frame"]) for item in top_five}

        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=[item["timestamp_seconds"] for item in scores],
            y=[item["change_score"] for item in scores],
            mode="lines+markers", name="Pixel change", line={"color": "#0f766e"}, marker={"size": 5},
        ))
        chart.add_trace(go.Scatter(
            x=[item["timestamp_seconds"] for item in top_five],
            y=[item["change_score"] for item in top_five],
            mode="markers+text",
            text=[f"Peak {index + 1}: {_object_label(peak_detections[item['frame_index']])}" for index, item in enumerate(top_five)],
            textposition="top center", name="Top peaks", marker={"color": "#dc2626", "size": 9},
        ))
        chart.update_layout(
            height=460,
            xaxis={"title": "Video time (seconds)", "range": [0, None]},
            yaxis_title="Change score",
            hovermode="x unified",
        )
        st.subheader("Change timeline")
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Most changed frames")
        for rank, result in enumerate(top_three, start=1):
            with st.expander(f"{rank}. Frame {result['frame_index']} at {result['timestamp_seconds']:.2f}s", expanded=rank == 1):
                current_detections = peak_detections.get(result["frame_index"], detections_for(result["frame_index"], result["frame"]))
                previous_detections = detections_for(result["previous_frame_index"], result["previous_frame"])
                entered, exited, matches = detect_object_transitions(
                    previous_detections, current_detections, config.iou_threshold
                )
                _render_result(result, current_detections, entered, exited, matches)

        summary = pd.DataFrame([
            {
                "timestamp_seconds": round(item["timestamp_seconds"], 2),
                "frame_index": item["frame_index"],
                "detected_objects": _object_label(peak_detections[item["frame_index"]]),
            }
            for item in top_five
        ])
        st.download_button("Download peak summary CSV", summary.to_csv(index=False), "dashcam_peaks.csv", "text/csv")
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        st.error(str(exc))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()