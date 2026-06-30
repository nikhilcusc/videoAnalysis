"""Dashcam frame-change and object-transition analysis.

This module combines simple pixel-level change scoring with YOLO object
 detection to help identify which objects appeared or disappeared between
 highly changed dashcam frames.

The implementation favors clear, testable building blocks:
- video loading and frame sampling
- pixel change scoring and top-N selection
- YOLO inference and detection normalization
- simple IoU-based association across frames
- drawing and saving visual summaries

Dependencies
------------
Install the required packages with:

    pip install opencv-python numpy ultralytics

If you want to work in a notebook, also install:

    pip install matplotlib pandas jupyter
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger(__name__)


def _load_cv2() -> Any:
    """Import OpenCV lazily and raise a clear error if it is unavailable."""

    try:
        return importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "OpenCV is required. Install it with: pip install opencv-python"
        ) from exc


def _load_numpy() -> Any:
    """Import NumPy lazily and raise a clear error if it is unavailable."""

    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "NumPy is required. Install it with: pip install numpy"
        ) from exc


def _load_yolo_class() -> Any:
    """Import the Ultralytics YOLO class lazily."""

    try:
        return importlib.import_module("ultralytics").YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Ultralytics is required. Install it with: pip install ultralytics"
        ) from exc


cv2 = _load_cv2()
np = _load_numpy()
YOLO = _load_yolo_class()

FrameArray = Any
MaskArray = Any
ModelType = Any


@dataclass(slots=True)
class VideoMetadata:
    """Basic video properties used during analysis."""

    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(slots=True)
class FrameSample:
    """A sampled frame and its source index."""

    frame_index: int
    timestamp_seconds: float
    frame: FrameArray


@dataclass(slots=True)
class Detection:
    """A normalized object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class MatchedDetection:
    """A matched detection pair between two frames."""

    previous: Detection
    current: Detection
    iou: float


@dataclass(slots=True)
class FrameChangeResult:
    """Result for one frame compared against its previous frame."""

    frame_index: int
    timestamp_seconds: float
    change_score: float
    changed_pixels: int
    changed_ratio: float
    diff_mask: MaskArray
    frame: FrameArray
    detections: list[Detection] = field(default_factory=list)
    entered_objects: list[Detection] = field(default_factory=list)
    exited_objects: list[Detection] = field(default_factory=list)
    persisted_objects: list[MatchedDetection] = field(default_factory=list)
    matched_objects: list[MatchedDetection] = field(default_factory=list)
    previous_frame_index: int | None = None


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration for frame change and object transition analysis."""

    resize_width: int | None = 960
    diff_threshold: int = 25
    top_n: int = 10
    sample_interval: int = 1
    iou_threshold: float = 0.3
    confidence_threshold: float = 0.25
    object_limit: int | None = None


@dataclass(slots=True)
class FrameChangeSummary:
    """Compact result describing the most changed frames and object transitions."""

    video_path: Path
    metadata: VideoMetadata
    results: list[FrameChangeResult]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure module logging for console usage."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_video(video_path: str | Path) -> tuple[Any, VideoMetadata]:
    """Open a video file and return its capture handle and metadata.

    Args:
        video_path: Path to a local video file.

    Returns:
        A tuple containing the opened ``cv2.VideoCapture`` and basic metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the video cannot be opened.
    """

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        fps = 30.0

    metadata = VideoMetadata(
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )
    return capture, metadata


def _resize_frame(frame: FrameArray, resize_width: int | None) -> FrameArray:
    """Resize a frame while preserving aspect ratio when requested."""

    if resize_width is None or resize_width <= 0:
        return frame

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0 or width == resize_width:
        return frame

    scale = resize_width / float(width)
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def extract_frames(
    video_path: str | Path,
    interval: int = 1,
    resize_width: int | None = 960,
    max_frames: int | None = None,
) -> tuple[list[FrameSample], VideoMetadata]:
    """Extract frames from a video at a fixed interval.

    Args:
        video_path: Input video file.
        interval: Sample every Nth frame, starting from the first frame.
        resize_width: Optional resize target for extracted frames.
        max_frames: Optional cap on the number of extracted frames.

    Returns:
        A list of sampled frames and the video metadata.
    """

    if interval <= 0:
        raise ValueError("interval must be greater than zero")

    capture, metadata = load_video(video_path)
    samples: list[FrameSample] = []

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame is None or frame.size == 0:
                LOGGER.warning("Skipping empty frame at index %s", frame_index)
                frame_index += 1
                continue

            if frame_index % interval == 0:
                resized = _resize_frame(frame, resize_width)
                timestamp_seconds = frame_index / metadata.fps
                samples.append(
                    FrameSample(
                        frame_index=frame_index,
                        timestamp_seconds=float(timestamp_seconds),
                        frame=resized,
                    )
                )
                if max_frames is not None and len(samples) >= max_frames:
                    break

            frame_index += 1
    finally:
        capture.release()

    return samples, metadata


def compute_frame_change_score(
    previous_frame: FrameArray,
    current_frame: FrameArray,
    diff_threshold: int = 25,
) -> tuple[float, int, float, MaskArray, MaskArray]:
    """Compute a pixel-level change score between two frames.

    The score is based on a thresholded absolute grayscale difference.

    Returns:
        A tuple containing the change score, changed pixel count, changed ratio,
        diff mask, and raw absolute difference image.
    """

    if previous_frame is None or current_frame is None:
        raise ValueError("Frames must not be None")
    if previous_frame.size == 0 or current_frame.size == 0:
        raise ValueError("Frames must not be empty")

    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if previous_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (previous_gray.shape[1], previous_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    previous_blurred = cv2.GaussianBlur(previous_gray, (5, 5), 0)
    current_blurred = cv2.GaussianBlur(current_gray, (5, 5), 0)

    abs_diff = cv2.absdiff(current_blurred, previous_blurred)
    _, diff_mask = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = int(cv2.countNonZero(diff_mask))
    total_pixels = int(diff_mask.shape[0] * diff_mask.shape[1])
    changed_ratio = float(changed_pixels / total_pixels) if total_pixels else 0.0
    change_score = float(changed_pixels * changed_ratio)
    return change_score, changed_pixels, changed_ratio, diff_mask, abs_diff


def select_top_changed_frames(
    samples: Sequence[FrameSample],
    top_n: int = 10,
    diff_threshold: int = 25,
    min_time_interval_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Select the top-N most changed sampled frames.

    The score compares each sample against its immediate predecessor.
    The first sample is skipped because there is no prior frame.
    Results are filtered so selected frames are at least
    ``min_time_interval_seconds`` apart.
    """

    if top_n <= 0:
        return []
    if len(samples) < 2:
        return []

    scored: list[dict[str, Any]] = []
    previous_sample = samples[0]
    for sample in samples[1:]:
        change_score, changed_pixels, changed_ratio, diff_mask, abs_diff = compute_frame_change_score(
            previous_sample.frame,
            sample.frame,
            diff_threshold=diff_threshold,
        )
        scored.append(
            {
                "frame_index": sample.frame_index,
                "timestamp_seconds": sample.timestamp_seconds,
                "change_score": change_score,
                "changed_pixels": changed_pixels,
                "changed_ratio": changed_ratio,
                "diff_mask": diff_mask,
                "abs_diff": abs_diff,
                "frame": sample.frame,
                "previous_frame_index": previous_sample.frame_index,
            }
        )
        previous_sample = sample

    scored.sort(key=lambda item: item["change_score"], reverse=True)

    selected: list[dict[str, Any]] = []
    for item in scored:
        timestamp = float(item["timestamp_seconds"])
        if all(
            abs(timestamp - float(existing["timestamp_seconds"])) >= min_time_interval_seconds
            for existing in selected
        ):
            selected.append(item)
        if len(selected) >= top_n:
            break

    return selected


def build_model(model_name_or_path: str) -> ModelType:
    """Load a YOLO model by Ultralytics model name or local weights path."""

    try:
        return YOLO(model_name_or_path)
    except Exception as exc:  # pragma: no cover - thin wrapper around library errors
        raise ValueError(f"Failed to load YOLO model: {model_name_or_path}") from exc


def run_yolo_detection(
    model: ModelType,
    frame: FrameArray,
    confidence_threshold: float = 0.25,
    object_limit: int | None = None,
) -> list[Detection]:
    """Run YOLO detection on a single frame and normalize the results."""

    if frame is None or frame.size == 0:
        raise ValueError("Frame must not be empty")

    try:
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
    except Exception as exc:  # pragma: no cover - library behavior
        raise RuntimeError("YOLO inference failed") from exc

    detections: list[Detection] = []
    if not results:
        return detections

    result = results[0]
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {})
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy()

    for index, bbox in enumerate(xyxy):
        class_id = int(cls[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=float(conf[index]),
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    if object_limit is not None:
        detections = detections[: max(0, object_limit)]
    return detections


def detections_to_dataframe(detections: Sequence[Detection]):
    """Convert detections into plain dictionaries for tabular display."""

    return [
        {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "x1": detection.bbox[0],
            "y1": detection.bbox[1],
            "x2": detection.bbox[2],
            "y2": detection.bbox[3],
        }
        for detection in detections
    ]


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute intersection-over-union for two bounding boxes."""

    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Bounding boxes must contain four values")

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def match_objects_between_frames(
    previous_detections: Sequence[Detection],
    current_detections: Sequence[Detection],
    iou_threshold: float = 0.3,
) -> list[MatchedDetection]:
    """Match objects between two frames using a greedy IoU association."""

    matches: list[MatchedDetection] = []
    used_current: set[int] = set()

    for previous in previous_detections:
        best_index: int | None = None
        best_iou = 0.0
        for current_index, current in enumerate(current_detections):
            if current_index in used_current:
                continue
            iou = bbox_iou(previous.bbox, current.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = current_index

        if best_index is not None and best_iou >= iou_threshold:
            used_current.add(best_index)
            matches.append(
                MatchedDetection(
                    previous=previous,
                    current=current_detections[best_index],
                    iou=best_iou,
                )
            )

    return matches


def detect_object_transitions(
    previous_detections: Sequence[Detection],
    current_detections: Sequence[Detection],
    iou_threshold: float = 0.3,
) -> tuple[list[Detection], list[Detection], list[MatchedDetection]]:
    """Detect entered, exited, and persisted objects between two frames."""

    matches = match_objects_between_frames(
        previous_detections=previous_detections,
        current_detections=current_detections,
        iou_threshold=iou_threshold,
    )
    matched_previous_ids = {id(match.previous) for match in matches}
    matched_current_ids = {id(match.current) for match in matches}

    entered_objects = [
        detection for detection in current_detections if id(detection) not in matched_current_ids
    ]
    exited_objects = [
        detection for detection in previous_detections if id(detection) not in matched_previous_ids
    ]
    return entered_objects, exited_objects, matches


def _color_for_index(index: int) -> tuple[int, int, int]:
    """Generate a stable color for repeated annotations."""

    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (255, 140, 0),
        (138, 43, 226),
        (0, 206, 209),
    ]
    return palette[index % len(palette)]


def draw_annotations(
    frame: FrameArray,
    detections: Sequence[Detection],
    entered_objects: Sequence[Detection] | None = None,
    exited_objects: Sequence[Detection] | None = None,
    matches: Sequence[MatchedDetection] | None = None,
    diff_mask: MaskArray | None = None,
    alpha: float = 0.35,
) -> FrameArray:
    """Draw boxes, labels, and change annotations on a frame."""

    if frame is None or frame.size == 0:
        raise ValueError("Frame must not be empty")

    annotated = frame.copy()
    if diff_mask is not None and diff_mask.size > 0:
        if diff_mask.shape[:2] != annotated.shape[:2]:
            diff_mask = cv2.resize(
                diff_mask,
                (annotated.shape[1], annotated.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        heatmap = cv2.applyColorMap(diff_mask, cv2.COLORMAP_JET)
        annotated = cv2.addWeighted(annotated, 1.0 - alpha, heatmap, alpha, 0)

    entered_set = {id(item) for item in (entered_objects or [])}
    exited_set = {id(item) for item in (exited_objects or [])}
    matched_lookup = {id(match.current): match for match in (matches or [])}

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = map(int, detection.bbox)
        if id(detection) in entered_set:
            color = (0, 200, 0)
            prefix = "ENTER"
        elif id(detection) in exited_set:
            color = (0, 0, 255)
            prefix = "EXIT"
        elif id(detection) in matched_lookup:
            color = _color_for_index(index)
            prefix = f"MATCH {matched_lookup[id(detection)].iou:.2f}"
        else:
            color = _color_for_index(index)
            prefix = "OBJ"

        label = f"{prefix} {detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_origin_y = max(0, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, text_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def save_visual_result(image: FrameArray, output_path: str | Path) -> Path:
    """Save an annotated frame or image to disk."""

    if image is None or image.size == 0:
        raise ValueError("Image must not be empty")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to write image: {path}")
    return path


def analyze_frame_pair(
    previous_sample: FrameSample,
    current_sample: FrameSample,
    model: ModelType,
    config: AnalysisConfig | None = None,
) -> FrameChangeResult:
    """Analyze a pair of adjacent frames for pixel change and object transitions."""

    config = config or AnalysisConfig()
    change_score, changed_pixels, changed_ratio, diff_mask, _ = compute_frame_change_score(
        previous_sample.frame,
        current_sample.frame,
        diff_threshold=config.diff_threshold,
    )
    previous_detections = run_yolo_detection(
        model=model,
        frame=previous_sample.frame,
        confidence_threshold=config.confidence_threshold,
        object_limit=config.object_limit,
    )
    current_detections = run_yolo_detection(
        model=model,
        frame=current_sample.frame,
        confidence_threshold=config.confidence_threshold,
        object_limit=config.object_limit,
    )
    entered_objects, exited_objects, matches = detect_object_transitions(
        previous_detections=previous_detections,
        current_detections=current_detections,
        iou_threshold=config.iou_threshold,
    )
    return FrameChangeResult(
        frame_index=current_sample.frame_index,
        timestamp_seconds=current_sample.timestamp_seconds,
        change_score=change_score,
        changed_pixels=changed_pixels,
        changed_ratio=changed_ratio,
        diff_mask=diff_mask,
        frame=current_sample.frame,
        detections=current_detections,
        entered_objects=entered_objects,
        exited_objects=exited_objects,
        persisted_objects=list(matches),
        matched_objects=list(matches),
        previous_frame_index=previous_sample.frame_index,
    )


def analyze_top_changed_frames(
    video_path: str | Path,
    model_name_or_path: str,
    config: AnalysisConfig | None = None,
) -> FrameChangeSummary:
    """Run the end-to-end pipeline over the most changed sampled frames."""

    config = config or AnalysisConfig()
    samples, metadata = extract_frames(
        video_path=video_path,
        interval=config.sample_interval,
        resize_width=config.resize_width,
    )
    if len(samples) < 2:
        raise ValueError("Not enough frames were extracted for analysis")

    model = build_model(model_name_or_path)
    top_frames = select_top_changed_frames(
        samples=samples,
        top_n=config.top_n,
        diff_threshold=config.diff_threshold,
    )
    sample_by_index = {sample.frame_index: sample for sample in samples}

    results: list[FrameChangeResult] = []
    for item in top_frames:
        previous_index = int(item["previous_frame_index"])
        current_index = int(item["frame_index"])
        previous_sample = sample_by_index.get(previous_index)
        current_sample = sample_by_index.get(current_index)
        if previous_sample is None or current_sample is None:
            LOGGER.warning("Skipping frame pair %s -> %s because a sample is missing", previous_index, current_index)
            continue
        results.append(analyze_frame_pair(previous_sample, current_sample, model, config))

    results.sort(key=lambda result: result.change_score, reverse=True)
    return FrameChangeSummary(
        video_path=Path(video_path),
        metadata=metadata,
        results=results,
    )


def summarize_result(result: FrameChangeResult) -> dict[str, Any]:
    """Convert a frame analysis result into a plain dictionary for display."""

    return {
        "frame_index": result.frame_index,
        "previous_frame_index": result.previous_frame_index,
        "timestamp_seconds": result.timestamp_seconds,
        "change_score": result.change_score,
        "changed_pixels": result.changed_pixels,
        "changed_ratio": result.changed_ratio,
        "detected_objects": [
            {
                "class_name": detection.class_name,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in result.detections
        ],
        "entered_objects": [
            {
                "class_name": detection.class_name,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in result.entered_objects
        ],
        "exited_objects": [
            {
                "class_name": detection.class_name,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
            }
            for detection in result.exited_objects
        ],
        "matched_objects": [
            {
                "previous": {
                    "class_name": match.previous.class_name,
                    "class_id": match.previous.class_id,
                    "confidence": match.previous.confidence,
                    "bbox": match.previous.bbox,
                },
                "current": {
                    "class_name": match.current.class_name,
                    "class_id": match.current.class_id,
                    "confidence": match.current.confidence,
                    "bbox": match.current.bbox,
                },
                "iou": match.iou,
            }
            for match in result.matched_objects
        ],
    }


def save_analysis_results(
    summary: FrameChangeSummary,
    output_dir: str | Path = "output/dashcam_change_analysis",
) -> list[Path]:
    """Save annotated change-analysis images for later inspection."""

    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for result in summary.results:
        annotated = draw_annotations(
            frame=result.frame,
            detections=result.detections,
            entered_objects=result.entered_objects,
            exited_objects=result.exited_objects,
            matches=result.matched_objects,
            diff_mask=result.diff_mask,
        )
        file_name = f"frame_{result.frame_index:06d}_change_{result.change_score:.0f}.jpg"
        saved_files.append(save_visual_result(annotated, output_folder / file_name))

    return saved_files


def print_summary(summary: FrameChangeSummary) -> None:
    """Log a concise text summary of the analysis results."""

    LOGGER.info(
        "Video %s | frames=%s fps=%.2f size=%sx%s results=%s",
        summary.video_path,
        summary.metadata.frame_count,
        summary.metadata.fps,
        summary.metadata.width,
        summary.metadata.height,
        len(summary.results),
    )
    for result in summary.results:
        LOGGER.info(
            "Frame %s (prev %s) | score=%.2f entered=%s exited=%s matches=%s changed=%s",
            result.frame_index,
            result.previous_frame_index,
            result.change_score,
            len(result.entered_objects),
            len(result.exited_objects),
            len(result.matched_objects),
            result.changed_pixels,
        )


def main() -> None:
    """Example command-line entry point."""

    setup_logging()

    video_path = Path("12Feb2022/VID_001.MOV")
    model_name_or_path = "yolov8n.pt"
    config = AnalysisConfig(
        resize_width=960,
        diff_threshold=25,
        top_n=5,
        sample_interval=5,
        iou_threshold=0.3,
        confidence_threshold=0.25,
        object_limit=25,
    )

    summary = analyze_top_changed_frames(video_path, model_name_or_path, config)
    print_summary(summary)
    save_analysis_results(summary)


if __name__ == "__main__":
    main()
